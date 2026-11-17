#ifndef _PHLOP_TIMING_COMMON_TIMER_HPP_
#define _PHLOP_TIMING_COMMON_TIMER_HPP_

#include <chrono>
#include <string>
#include <vector>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <unordered_map>

#include "phlop/os/io.hpp"
#include "phlop/os/mem.hpp"

#include <algorithm>

namespace phlop
{

// process resident memory at a point in time, see phlop/os/mem.hpp
struct MemSample
{
    static MemSample now()
    {
        auto const u = os::mem::usage();
        return {u.rss, u.peak};
    }

    std::uint64_t rss  = 0; // current resident set size in bytes
    std::uint64_t peak = 0; // process lifetime peak resident set size in bytes
};

// the calling thread's I/O accounting at a point in time, see phlop/os/io.hpp
struct IoSample
{
    static IoSample now()
    {
        IoSample s;
        s.ok = os::ThreadIo::local().read(s.counts, s.self_bytes);
        return s;
    }

    // exit - entry, less what phlop itself read in between: the entry sample,
    //  and any reads added to self_calls/self_bytes, eg perf counter reads
    os::IoCounts operator-(IoSample const& in) const
    {
        auto d = counts - in.counts;
        d.syscr -= (std::min)(d.syscr, in.self_calls);
        d.rchar -= (std::min)(d.rchar, in.self_bytes);
        return d;
    }

    os::IoCounts counts{};
    std::uint64_t self_bytes = 0; // read by phlop after this sample, see above
    std::uint64_t self_calls = 1; // this sample's own read
    bool ok                  = false;
};

struct SteadyClock
{
    static std::uint64_t now()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
};

namespace mpi
{
    struct Clock; // MPI_Wtime, defined in mpi_timer.hpp: the timers only name it here
} // namespace mpi



// clock values taken at points every process passed together, eg just after a
//  barrier, see phlop::mpi::sync_clock(). Written as "#clock sync=<t>,..." so
//  starts in different files can be aligned in post-processing. global: whether
//  the clock is already the same everywhere, 1/0, -1 unknown
struct ClockSyncs
{
    void add(std::uint64_t const t, int const g = -1)
    {
        syncs.emplace_back(t);
        if (g >= 0)
            global = g;
    }

    // "#clock sync=<t>,... global=<0|1>" or none
    std::vector<std::string> headers() const
    {
        if (syncs.empty())
            return {};
        std::string line = "#clock sync=";
        for (std::size_t i = 0; i < syncs.size(); ++i)
            line += (i ? "," : "") + std::to_string(syncs[i]);
        if (global >= 0)
            line += " global=" + std::to_string(global);
        return {line};
    }

    bool empty() const { return syncs.empty(); }

    void clear()
    {
        syncs.clear();
        global = -1;
    }

    std::vector<std::uint64_t> syncs;
    int global = -1;
};


template<typename Report>
struct RunTimerReportSnapshotT
{
    RunTimerReportSnapshotT(Report* s, Report* p, std::uint64_t const st, std::uint64_t const t)
        : self{s}
        , parent{p}
        , start{st}
        , time{t}
    {
        // no reserve: childs is replaced by the scope's own vector on creation
    }

    Report const* const self;
    Report const* const parent;

    std::uint64_t const start;
    std::uint64_t const time;
    std::vector<RunTimerReportSnapshotT*> childs;

    // PHLOP_SCOPE perf counts: level 1 perf_light, 2 perf_all, 0 none
    std::uint8_t level = 0;
    std::vector<std::uint64_t> counts;

    // PHLOP_SCOPE mem: resident memory on entry/exit
    bool has_mem = false;
    MemSample mem_in{}, mem_out{};

    // PHLOP_SCOPE io: the thread's I/O over the scope, exit - entry
    bool has_io = false;
    os::IoCounts io{};
};


struct BinaryTimerFileNode
{
    BinaryTimerFileNode(std::uint32_t const _fn_id, std::uint64_t const _start,
                        std::uint64_t const _time)
        : fn_id{_fn_id}
        , start{_start}
        , time{_time}
    {
    }

    std::uint32_t const fn_id;
    std::uint64_t const start;
    std::uint64_t const time;
    std::uint8_t level = 0;
    std::vector<std::uint64_t> counts{};
    bool has_mem = false;
    MemSample mem_in{}, mem_out{};
    bool has_io = false;
    os::IoCounts io{};

    std::vector<BinaryTimerFileNode> kinder{};
};


struct BinaryTimerFile
{
    template<typename Traces>
    BinaryTimerFile(Traces const& traces)
    {
        for (auto const& trace : traces)
            recurse_traces_for_keys(trace);
        for (auto const& trace : traces)
            recurse_traces_for_nodes(trace, add_node(roots, trace));
    }

    template<typename Trace>
    BinaryTimerFileNode& add_node(std::vector<BinaryTimerFileNode>& nodes, Trace const& c)
    {
        auto const id = static_cast<std::uint32_t>(key_ids[std::string{c->self->k}]);
        auto& node    = nodes.emplace_back(id, c->start, c->time);
        node.level    = c->level;
        node.counts   = c->counts;
        node.has_mem  = c->has_mem;
        node.mem_in   = c->mem_in;
        node.mem_out  = c->mem_out;
        node.has_io   = c->has_io;
        node.io       = c->io;
        return node;
    }

    template<typename Trace>
    void recurse_traces_for_nodes(Trace const& c, BinaryTimerFileNode& node)
    {
        for (std::size_t i = 0; i < c->childs.size(); ++i)
            recurse_traces_for_nodes(c->childs[i], add_node(node.kinder, c->childs[i]));
    }

    template<typename Trace>
    void recurse_traces_for_keys(Trace const& c)
    {
        assert(c);
        assert(c->self);
        std::string const s{c->self->k};
        if (!key_ids.count(s))
        {
            auto [it, b] = key_ids.emplace(s, key_ids.size());
            assert(b);
            auto const& [k, i] = *it;
            assert(!id_to_key.count(i));
            id_to_key.emplace(i, k);
        }
        for (std::size_t i = 0; i < c->childs.size(); ++i)
            recurse_traces_for_keys(c->childs[i]);
    }

    template<typename... Args>
    void _byte_write(std::ofstream& file, Args const&... args) const
    {
        (file << ... << args) << '\n';
    }

    // '\n' not std::endl: endl flushes, a write per line; closing flushes once
    // perf_headers: "#perf ..." lines naming the counts, see perf_counters.hpp
    void write(std::string const& filename, bool const force_strings = false,
               std::vector<std::string> const& headers      = {},
               std::vector<std::string> const& perf_headers = {}) const
    {
        std::ofstream f{filename, std::ios::binary};

        if (force_strings)
        {
            if (headers.size())
            {
                f << headers[0];
                for (std::size_t i = 1; i < headers.size(); ++i)
                    f << "," << headers[i];
                f << '\n';
            }
            for (auto const& line : perf_headers) // names the counts below
                f << line << '\n';
            for (auto const& root : roots)
                _write_strings(f, root);
        }
        else
        {
            for (auto const& line : perf_headers)
                f << line << '\n';
            for (auto const& [i, k] : id_to_key)
                _byte_write(f, i, " ", k);
            f << '\n';
            for (auto const& root : roots)
                _write(f, root);
        }

        f.close();
    }

    void _write_strings(std::ofstream& file, BinaryTimerFileNode const& node,
                        std::uint16_t const tabs = 0) const
    {
        for (std::size_t ti = 0; ti < tabs; ++ti)
            file << " ";
        file << id_to_key.at(node.fn_id) << node.time;
        _write_counts(file, node);
        _write_mem(file, node);
        _write_io(file, node);
        file << '\n';
        for (auto const& n : node.kinder)
            _write_strings(file, n, tabs + 1);
    }

    void _write(std::ofstream& file, BinaryTimerFileNode const& node,
                std::uint16_t const tabs = 0) const
    {
        file << tabs << " ";
        file << node.fn_id << " " << node.start << ":" << node.time;
        _write_counts(file, node);
        _write_mem(file, node);
        _write_io(file, node);
        file << '\n';
        for (auto const& n : node.kinder)
            _write(file, n, tabs + 1);
    }

    // " <level>:<c0>,<c1>,..." after the time, only for scopes with counts
    static void _write_counts(std::ofstream& file, BinaryTimerFileNode const& node)
    {
        if (node.counts.empty())
            return;
        file << " " << static_cast<unsigned>(node.level) << ":" << node.counts[0];
        for (std::size_t i = 1; i < node.counts.size(); ++i)
            file << "," << node.counts[i];
    }

    // " m:<rss_in>,<rss_out>,<peak_in>,<peak_out>" after any counts, only for mem scopes
    static void _write_mem(std::ofstream& file, BinaryTimerFileNode const& node)
    {
        if (node.has_mem)
            file << " m:" << node.mem_in.rss << "," << node.mem_out.rss << "," << node.mem_in.peak
                 << "," << node.mem_out.peak;
    }

    // " io:<rchar>,<wchar>,<syscr>,<syscw>,<read_bytes>,<write_bytes>,<cancelled_write_bytes>"
    //  the scope's exit - entry, after any mem, only for io scopes
    static void _write_io(std::ofstream& file, BinaryTimerFileNode const& node)
    {
        if (!node.has_io)
            return;
        auto const& io = node.io;
        file << " io:" << io.rchar << "," << io.wchar << "," << io.syscr << "," << io.syscw << ","
             << io.read_bytes << "," << io.write_bytes << "," << io.cancelled_write_bytes;
    }

    std::unordered_map<std::string, std::size_t> key_ids;
    std::unordered_map<std::size_t, std::string> id_to_key;
    std::vector<BinaryTimerFileNode> roots;
};


} // namespace phlop

#endif /*_PHLOP_TIMING_COMMON_TIMER_HPP_*/
