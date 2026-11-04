#ifndef _PHLOP_TIMING_SCOPE_TIMER_HPP_
#define _PHLOP_TIMING_SCOPE_TIMER_HPP_

#if defined(_PHLOP_TIMER_EXPORTED_) && _PHLOP_TIMER_EXPORTED_ != 0
#error // can't use more than one timer type at time
#endif
#define _PHLOP_TIMER_EXPORTED_ 0

#include "phlop/macros/def/string.hpp"
#include "phlop/timing/common_timer.hpp"
#include "phlop/timing/scope.hpp"

#include <atomic>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string_view>
#include <type_traits>

#if !defined(PHLOP_SCOPE_TIMER)
#define PHLOP_SCOPE_TIMER(key)                                                                     \
    static phlop::RunTimerReport PHLOP_STR_CAT(ridx_, __LINE__){key, __FILE__, __LINE__};          \
    phlop::ScopeTimer<> PHLOP_STR_CAT(_ScopeTimer_, __LINE__){PHLOP_STR_CAT(ridx_, __LINE__)};     \
    phlop::ScopeTimerMan::INSTANCE().report_stack_ptr = &PHLOP_STR_CAT(ridx_, __LINE__);
#endif

// PHLOP_SCOPE("key", time, perf_light), see scope.hpp
#if !defined(PHLOP_SCOPE)
#define PHLOP_SCOPE(key, ...)                                                                      \
    static phlop::RunTimerReport PHLOP_STR_CAT(ridx_, __LINE__){key, __FILE__, __LINE__};          \
    phlop::Scope<PHLOP_SCOPE_FLAGS(__VA_ARGS__)> PHLOP_STR_CAT(_Scope_, __LINE__){                 \
        PHLOP_STR_CAT(ridx_, __LINE__)};                                                           \
    phlop::ScopeTimerMan::INSTANCE().report_stack_ptr = &PHLOP_STR_CAT(ridx_, __LINE__);
#endif

namespace phlop
{

struct RunTimerReport;
using RunTimerReportSnapshot = RunTimerReportSnapshotT<RunTimerReport>;

// forward declare so detail::_current_ScopeTimer<Clock> can use it as a pointer type
template<typename Clock = SteadyClock>
struct ScopeTimer;
struct ScopeTimerMan;

namespace detail
{
    void inline write_timer_file(); // defined below after ScopeTimerMan

#if defined(PHLOP_SCOPE_TIMER_SELF_STATS)
    // the timer's own cost per scope, off unless defined, see print_self_stats
    struct SelfStat
    {
        std::uint64_t count = 0, total = 0, max = 0;

        void add(std::uint64_t const t)
        {
            ++count;
            total += t;
            max = (std::max)(max, t); // parens: windows.h max macro
        }
        double mean() const
        {
            return count ? static_cast<double>(total) / static_cast<double>(count) : 0;
        }
    };

    inline SelfStat construct_stats, destruct_stats;
#endif

} // namespace detail


struct ScopeTimerMan
{
    static ScopeTimerMan& INSTANCE();
    static constexpr std::string_view file_name_default = ".phlop_scope_times.txt";

    ScopeTimerMan()
        : timer_file{file_name_default}
    {
        reports.reserve(25);
        traces.reserve(25);
    }


    // perf counters (PHLOP_PERF_LIGHT/ALL env) count the calling thread
    void init()
    {
        active = true;
        perf::Config::INSTANCE().init();
        memory(_memory.load(std::memory_order_relaxed));
    }

    void shutdown(bool const write = true)
    {
        if (traces.size())
        {
            if (write)
                detail::write_timer_file();
            traces.clear();
            reports.clear();
            report_stack_ptr = nullptr;
        }

        if (auto const s = perf::summary(); !s.empty())
            std::cerr << "phlop perf counters: " << s << '\n';
        perf::Config::INSTANCE().close();
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

    // sample memory for PHLOP_SCOPE("key", time, mem) scopes, also env PHLOP_SCOPE_MEM=1/0
    //  resident memory is per process, not per thread
    ScopeTimerMan& memory(bool const b = true)
    {
        bool on = b; // PHLOP_SCOPE_MEM=1/0, if set, wins over the code either way
        if (auto const env = std::getenv("PHLOP_SCOPE_MEM"))
            on = std::string_view{env} != "0";
        _memory.store(on, std::memory_order_relaxed);
        return *this;
    }

    template<typename... Args>
    auto& headers(Args const... args)
    {
        _headers.clear();
        (_headers.emplace_back(args), ...);
        return *this;
    }

    // ns per ScopeTimer constructor/destructor while active, each including
    // one clock read for the measurement itself. Mean is the typical cost, max
    // is usually a one off (page fault, first allocation, preemption)
    void print_self_stats() const
    {
#if defined(PHLOP_SCOPE_TIMER_SELF_STATS)
        auto const print = [](char const* const name, detail::SelfStat const& s) {
            std::cout << " " << name << " n=" << s.count << " mean=" << s.mean()
                      << "ns max=" << s.max << "ns\n";
        };
        std::cout << "PHLOP SCOPE TIMER STATS\n";
        print("construct", detail::construct_stats);
        print("destruct ", detail::destruct_stats);
#else
        std::cout << "PHLOP SCOPE TIMER STATS: build with -DPHLOP_SCOPE_TIMER_SELF_STATS\n";
#endif
    }

    static void reset(bool const active = true)
    {
        if (auto& self = INSTANCE(); self.active)
        {
            self.shutdown();
            if (active)
                self.init();
        }
    }

    bool active         = false;
    bool _force_strings = false;
    std::atomic<bool> _memory{false};
    std::string timer_file;
    std::vector<std::string> _headers;
    std::vector<RunTimerReport*> reports;
    std::vector<RunTimerReportSnapshot*> traces;
    RunTimerReport* report_stack_ptr = nullptr;
};


struct RunTimerReport
{
    std::string k, f;
    std::uint32_t l = 0;

    RunTimerReport(std::string_view const& _k, std::string_view const& _f, std::uint32_t const& _l)
        : k{_k}
        , f{_f}
        , l{_l}
    {
        ScopeTimerMan::INSTANCE().reports.emplace_back(this);
        snapshots.reserve(25);
    }

    ~RunTimerReport() {}

    auto operator()(std::size_t i) { return snapshots[i].get(); }
    auto size() { return snapshots.size(); }

    // sole owner: traces/childs hold raw pointers into these
    std::vector<std::unique_ptr<RunTimerReportSnapshot>> snapshots;
};


namespace detail
{
    template<typename Clock>
    inline ScopeTimer<Clock>* _current_ScopeTimer = nullptr;

    void inline write_timer_file()
    {
        auto& man = ScopeTimerMan::INSTANCE();
        BinaryTimerFile{man.traces}.write(man.timer_file, man._force_strings, man._headers,
                                          perf::Config::INSTANCE().file_headers());
    }

} // namespace detail


template<typename Clock>
struct ScopeTimer
{
    ScopeTimer(RunTimerReport& _r)
        : r{_r}
    {
        if (ScopeTimerMan::INSTANCE().active)
        {
            this->pscope                       = detail::_current_ScopeTimer<Clock>;
            detail::_current_ScopeTimer<Clock> = this;

#if defined(PHLOP_SCOPE_TIMER_SELF_STATS)
            detail::construct_stats.add(now() - _self_start);
#endif
        }
    }

    ~ScopeTimer()
    {
        if (ScopeTimerMan::INSTANCE().active)
        {
#if defined(PHLOP_SCOPE_TIMER_SELF_STATS)
            auto const self_start = now();
#endif
            detail::_current_ScopeTimer<Clock> = this->pscope;

            // time first, then the exit memory sample, then any bookkeeping allocation
            //  so neither the sample nor the snapshot are counted in the scope
            auto const end          = now();
            MemSample const mem_out = pending_mem ? MemSample::now() : MemSample{};

            auto& s = *r.snapshots.emplace_back(
                std::make_unique<RunTimerReportSnapshot>(&r, parent, start, end - start));

            if (this->pscope)
                pscope->childs.emplace_back(&s);

            s.childs = std::move(childs);
            s.level  = pending_level;
            s.counts = std::move(pending_counts);
            if (pending_mem)
            {
                s.has_mem = true;
                s.mem_in  = pending_mem_in;
                s.mem_out = mem_out;
            }

            if (parent == nullptr)
                ScopeTimerMan::INSTANCE().traces.emplace_back(&s);

            ScopeTimerMan::INSTANCE().report_stack_ptr = parent;

#if defined(PHLOP_SCOPE_TIMER_SELF_STATS)
            detail::destruct_stats.add(now() - self_start);
#endif
        }
    }

    static std::uint64_t now() { return Clock::now(); }

    static ScopeTimer& root_parent_from(ScopeTimer& self)
    {
        if (self.pscope)
            return root_parent_from(*self.pscope);
        return self;
    }

#if defined(PHLOP_SCOPE_TIMER_SELF_STATS)
    // first member: initialised before the others, so they're measured too
    std::uint64_t const _self_start = now();
#endif
    RunTimerReport& r;
    RunTimerReport* parent    = ScopeTimerMan::INSTANCE().report_stack_ptr;
    RunTimerReport* child     = nullptr;
    ScopeTimer* pscope        = nullptr;
    std::uint64_t const start = now();
    std::vector<RunTimerReportSnapshot*> childs;

    // set by an enclosing Scope before this destructs, moved into the snapshot
    std::uint8_t pending_level = 0;
    std::vector<std::uint64_t> pending_counts;
    bool pending_mem = false;
    MemSample pending_mem_in{};
};


template<std::uint32_t F, typename Clock = SteadyClock>
using Scope = ScopeT<F, ScopeTimer<Clock>, ScopeTimerMan>;


inline auto& scope_timer()
{
    return ScopeTimerMan::INSTANCE();
}

inline void shutdown_scope_timer()
{
    ScopeTimerMan::INSTANCE().shutdown();
}


} // namespace phlop



#endif /*_PHLOP_TIMING_SCOPE_TIMER_HPP_*/
