#ifndef _PHLOP_TIMING_SCOPE_TIMER_HPP_
#define _PHLOP_TIMING_SCOPE_TIMER_HPP_

#include <chrono>
#include <memory>
#include <vector>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string_view>
#include <unordered_map>

#include "phlop/macros/def/string.hpp"

namespace phlop
{

struct RunTimerReport;
struct RunTimerReportSnapshot;

namespace detail
{

    void inline write_timer_file();

} // namespace detail

struct ScopeTimerMan
{
    static ScopeTimerMan& INSTANCE();
    static constexpr std::string_view file_name_default = ".phlop_scope_times.bin";

    ScopeTimerMan()
        : timer_file{file_name_default}
    {
        reports.reserve(5);
        traces.reserve(5);
    }

    void init() { active = true; }

    void shutdown(bool write = true)
    {
        if (traces.size())
        {
            if (write)
                detail::write_timer_file();
            traces.clear();
            reports.clear();
        }
        _headers.clear();
        active = false;
    }

    auto& file_name(std::string const& fn)
    {
        timer_file = fn;
        return *this;
    }

    auto& force_strings(bool const b = true)
    {
        _force_strings = b;
        return *this;
    }

    template<typename... Args>
    auto& headers(Args const... args)
    {
        _headers.clear();
        (_headers.emplace_back(args), ...);
        return *this;
    }



    static void reset() { INSTANCE().shutdown(); }

    bool active         = false;
    bool _force_strings = false;
    std::string timer_file;
    std::vector<std::string> _headers;
    std::vector<RunTimerReport*> reports;
    std::vector<RunTimerReportSnapshot*> traces;
    RunTimerReport* report_stack_ptr = nullptr;
};


struct RunTimerReportSnapshot
{
    RunTimerReportSnapshot(RunTimerReport* s, RunTimerReport* p, std::uint64_t const& st,
                           std::uint64_t const& t)
        : self{s}
        , parent{p}
        , start{st}
        , time{t}
    {
        childs.reserve(2);
    }

    RunTimerReport const* const self;
    RunTimerReport const* const parent;

    std::uint64_t const start;
    std::uint64_t const time;
    std::vector<RunTimerReportSnapshot*> childs;
};

struct RunTimerReport
{
    std::string_view k, f;
    std::uint32_t l = 0;

    RunTimerReport(std::string_view const& _k, std::string_view const& _f, std::uint32_t const& _l)
        : k{_k}
        , f{_f}
        , l{_l}
    {
        ScopeTimerMan::INSTANCE().reports.emplace_back(this);
        snapshots.reserve(5);
    }

    ~RunTimerReport() {}

    auto operator()(std::size_t i) { return snapshots[i].get(); }
    auto size() { return snapshots.size(); }

    std::vector<std::shared_ptr<RunTimerReportSnapshot>> snapshots; // emplace back breaks pointers!
};




struct scope_timer
{
    scope_timer(RunTimerReport& _r);

    ~scope_timer();

    std::uint64_t static now()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    static scope_timer& root_parent_from(scope_timer& self)
    {
        if (self.pscope)
            return root_parent_from(*self.pscope);
        return self;
    }


    RunTimerReport& r;
    RunTimerReport* parent = ScopeTimerMan::INSTANCE().report_stack_ptr;
    RunTimerReport* child  = nullptr;

    scope_timer* pscope       = nullptr;
    std::uint64_t const start = now();

    std::vector<RunTimerReportSnapshot*> childs;
};


struct BinaryTimerFileNode
{
    BinaryTimerFileNode(std::uint16_t const& _fn_id, std::uint64_t const& _start,
                        std::uint64_t const& _time)
        : fn_id{_fn_id}
        , start{_start}
        , time{_time}
    {
    }

    std::uint16_t fn_id;
    std::uint64_t start;
    std::uint64_t time;

    std::vector<BinaryTimerFileNode> kinder{};
};

struct BinaryTimerFile
{
    BinaryTimerFile(bool fromStaticTraces = true)
    {
        if (fromStaticTraces)
        {
            for (auto const& trace : ScopeTimerMan::INSTANCE().traces)
                recurse_traces_for_keys(trace);
            for (auto const& trace : ScopeTimerMan::INSTANCE().traces)
                recurse_traces_for_nodes(trace,
                                         roots.emplace_back(key_ids[std::string{trace->self->k}],
                                                            trace->start, trace->time));
        }
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
        std::string s{c->self->k};
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

    void write(std::string const& filename) const
    {
        std::ofstream f{filename, std::ios::binary};

        if (ScopeTimerMan::INSTANCE()._force_strings)
        {
            if (ScopeTimerMan::INSTANCE()._headers.size())
            {
                f << ScopeTimerMan::INSTANCE()._headers[0];
                for (std::size_t i = 1; i < ScopeTimerMan::INSTANCE()._headers.size(); ++i)
                    f << "," << ScopeTimerMan::INSTANCE()._headers[i];
                f << std::endl;
            }
            for (auto const& root : roots)
                _write_strings(f, root);
        }
        else
        {
            for (auto const& [i, k] : id_to_key)
                _byte_write(f, i, " ", k);
            f << std::endl; // break between function id map and function times
            for (auto const& root : roots)
                _write(f, root);
        }
    }

    void _write_strings(std::ofstream& file, BinaryTimerFileNode const& node,
                        std::uint16_t tabs = 0) const
    {
        for (std::size_t ti = 0; ti < tabs; ++ti)
            file << " ";
        file << id_to_key.at(node.fn_id) << node.time << std::endl;
        for (auto const& n : node.kinder)
            _write_strings(file, n, tabs + 1);
    }

    void _write(std::ofstream& file, BinaryTimerFileNode const& node, std::uint16_t tabs = 0) const
    {
        for (std::size_t ti = 0; ti < tabs; ++ti)
            file << " ";
        file << node.fn_id << " " << node.start << ":" << node.time << std::endl;
        for (auto const& n : node.kinder)
            _write(file, n, tabs + 1);
    }


    std::unordered_map<std::string, std::size_t> key_ids; // only used on write
    std::unordered_map<std::size_t, std::string> id_to_key;
    std::vector<BinaryTimerFileNode> roots;
};


namespace detail
{

    void inline write_timer_file()
    {
        BinaryTimerFile{}.write(ScopeTimerMan::INSTANCE().timer_file);
    }


} // namespace detail


} // namespace phlop



#define PHLOP_SCOPE_TIMER(key)                                                                     \
    static phlop::RunTimerReport PHLOP_STR_CAT(ridx_, __LINE__){key, __FILE__, __LINE__};          \
    phlop::scope_timer PHLOP_STR_CAT(_scope_timer_, __LINE__){PHLOP_STR_CAT(ridx_, __LINE__)};     \
    phlop::ScopeTimerMan::INSTANCE().report_stack_ptr = &PHLOP_STR_CAT(ridx_, __LINE__);


#endif /*_PHLOP_TIMING_SCOPE_TIMER_HPP_*/
