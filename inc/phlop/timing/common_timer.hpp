
#ifndef _PHLOP_TIMING_COMMON_TIMER_HPP_
#define _PHLOP_TIMING_COMMON_TIMER_HPP_

#include <chrono>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

namespace phlop
{

struct SteadyClock
{
    static std::uint64_t now()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
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
        childs.reserve(2);
    }

    Report const* const self;
    Report const* const parent;

    std::uint64_t const start;
    std::uint64_t const time;
    std::vector<RunTimerReportSnapshotT*> childs;
};


struct BinaryTimerFileNode
{
    BinaryTimerFileNode(std::uint16_t const& _fn_id, std::uint64_t const _start,
                        std::uint64_t const _time)
        : fn_id{_fn_id}
        , start{_start}
        , time{_time}
    {
    }

    std::uint16_t const fn_id;
    std::uint64_t const start;
    std::uint64_t const time;

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
            recurse_traces_for_nodes(trace, roots.emplace_back(key_ids[std::string{trace->self->k}],
                                                               trace->start, trace->time));
    }

    template<typename Trace>
    void recurse_traces_for_nodes(Trace const& c, BinaryTimerFileNode& node)
    {
        for (std::size_t i = 0; i < c->childs.size(); ++i)
            recurse_traces_for_nodes(
                c->childs[i], node.kinder.emplace_back(key_ids[std::string{c->childs[i]->self->k}],
                                                       c->childs[i]->start, c->childs[i]->time));
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
    void _byte_write(std::ofstream& file, Args const... args) const
    {
        std::stringstream ss;
        (ss << ... << args);
        auto s = ss.str();
        file.write(s.c_str(), s.size());
        file << std::endl;
    }

    void write(std::string const& filename, bool const force_strings = false,
               std::vector<std::string> const& headers = {}) const
    {
        std::ofstream f{filename, std::ios::binary};

        if (force_strings)
        {
            if (headers.size())
            {
                f << headers[0];
                for (std::size_t i = 1; i < headers.size(); ++i)
                    f << "," << headers[i];
                f << std::endl;
            }
            for (auto const& root : roots)
                _write_strings(f, root);
        }
        else
        {
            for (auto const& [i, k] : id_to_key)
                _byte_write(f, i, " ", k);
            f << std::endl;
            for (auto const& root : roots)
                _write(f, root);
        }
    }

    void _write_strings(std::ofstream& file, BinaryTimerFileNode const& node,
                        std::uint16_t const tabs = 0) const
    {
        for (std::size_t ti = 0; ti < tabs; ++ti)
            file << " ";
        file << id_to_key.at(node.fn_id) << node.time << std::endl;
        for (auto const& n : node.kinder)
            _write_strings(file, n, tabs + 1);
    }

    void _write(std::ofstream& file, BinaryTimerFileNode const& node,
                std::uint16_t const tabs = 0) const
    {
        file << tabs << " ";
        file << node.fn_id << " " << node.start << ":" << node.time << std::endl;
        for (auto const& n : node.kinder)
            _write(file, n, tabs + 1);
    }

    std::unordered_map<std::string, std::size_t> key_ids;
    std::unordered_map<std::size_t, std::string> id_to_key;
    std::vector<BinaryTimerFileNode> roots;
};


} // namespace phlop

#endif /*_PHLOP_TIMING_COMMON_TIMER_HPP_*/
