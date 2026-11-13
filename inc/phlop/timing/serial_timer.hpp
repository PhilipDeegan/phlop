#ifndef _PHLOP_TIMING_SERIAL_TIMER_HPP_
#define _PHLOP_TIMING_SERIAL_TIMER_HPP_

// The single threaded scope timer, templated on its clock: phlop:: (steady clock)
//  here, phlop::mpi:: (MPI_Wtime) in mpi_timer.hpp. Each is its own timer, own
//  file. ONLY SUPPORTS SERIAL OPERATIONS: scopes on other threads need para_timer.hpp
// No macros here, see scope_timer.hpp / mpi_scope_timer.hpp. The instances are in
//  src/phlop/timing/scope_timer.cpp, the one source for every timer

#include "phlop/macros/def/string.hpp"
#include "phlop/timing/nvtx.hpp"
#include "phlop/timing/common_timer.hpp"
#include "phlop/timing/scope.hpp"

#include <atomic>
#include <utility>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string_view>
#include <type_traits>

// the body of PHLOP_SCOPE for the serial timer in namespace ns, see scope_timer.hpp
#define PHLOP_DETAIL_SERIAL_SCOPE(ns, key, flags)                                                  \
    static ns::RunTimerReport PHLOP_STR_CAT(ridx_, __LINE__){key, __FILE__, __LINE__};             \
    PHLOP_NVTX_RANGE(PHLOP_STR_CAT(ridx_, __LINE__).k)                                             \
    ns::Scope<flags> PHLOP_STR_CAT(_Scope_, __LINE__){PHLOP_STR_CAT(ridx_, __LINE__)};


namespace phlop::detail
{
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

inline SelfStat construct_stats, destruct_stats; // all serial timers together
#endif
} // namespace phlop::detail


namespace phlop::impl::serial
{

template<typename Clock>
struct RunTimerReport;
template<typename Clock>
struct ScopeTimer;


template<typename Clock>
struct ScopeTimerNode
{
    ScopeTimerNode* pscope = nullptr;
    std::vector<RunTimerReportSnapshotT<RunTimerReport<Clock>>*> childs;
};


template<typename Clock_>
struct ScopeTimerMan
{
    using Clock    = Clock_;
    using Report   = RunTimerReport<Clock>;
    using Snapshot = RunTimerReportSnapshotT<Report>;

    // out of line, src/phlop/timing/scope_timer.cpp: one instance per process,
    //  even across shared libraries built with -fvisibility=hidden
    static ScopeTimerMan& INSTANCE();
    static ScopeTimerNode<Clock>*& current_scope(); // the innermost active scope

    ScopeTimerMan(std::string_view const fn)
        : timer_file{fn}
    {
        perf::Config::INSTANCE(); // constructed first so destroyed after this
        reports.reserve(25);
        traces.reserve(25);
    }


    // perf counters (PHLOP_PERF_LIGHT/ALL env) count the calling thread
    void init()
    {
        if (!std::exchange(active, true)) // a second init keeps the one acquire
            perf::Config::INSTANCE().acquire();
        memory(_memory.load(std::memory_order_relaxed), _memory_peak);
        io(_io.load(std::memory_order_relaxed));
        _peak_carry = 0;
    }

    void shutdown(bool const write = true)
    {
        // no scopes but clock syncs: still written, so an mpi rank without scopes
        //  is in the file set, and reported as such, rather than silently missing
        if (traces.size() or !_clock_syncs.empty())
        {
            if (write)
                write_file();
            traces.clear();
            reports.clear();
            report_stack_ptr = nullptr;
        }
        _clock_syncs.clear();

        if (auto const s = perf::summary(); !s.empty())
            std::cerr << "phlop perf counters: " << s << '\n';
        if (std::exchange(active, false))
            perf::Config::INSTANCE().release();
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
    //  peak resets the process peak on entry to each mem scope, so the peak is
    //  per scope, not "a new process high" (linux, else ignored), also env
    //  PHLOP_SCOPE_MEM=peak. Process wide: anything reading VmHWM/ru_maxrss sees it
    ScopeTimerMan& memory(bool const b = true, bool const peak = false)
    {
        bool on = b, pk = peak; // PHLOP_SCOPE_MEM=1/0/peak, if set, wins over the code
        if (auto const env = std::getenv("PHLOP_SCOPE_MEM"))
        {
            on = std::string_view{env} != "0";
            pk = std::string_view{env} == "peak";
        }
        _memory.store(on, std::memory_order_relaxed);
        _memory_peak = on and pk;
        return *this;
    }

    // sample the calling thread's I/O for PHLOP_SCOPE("key", time, io) scopes,
    //  also env PHLOP_SCOPE_IO=1/0, which wins over the code. linux, else ignored
    ScopeTimerMan& io(bool const b = true)
    {
        bool on = b;
        if (auto const env = std::getenv("PHLOP_SCOPE_IO"))
            on = std::string_view{env} != "0";
        _io.store(on, std::memory_order_relaxed);
        return *this;
    }

    template<typename... Args>
    auto& headers(Args const... args)
    {
        _headers.clear();
        (_headers.emplace_back(args), ...);
        return *this;
    }

    // see ClockSyncs, phlop::mpi::sync_clock()
    void clock_sync(std::uint64_t const t, int const global = -1) { _clock_syncs.add(t, global); }

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

    void write_file()
    {
        auto file_headers = perf::Config::INSTANCE().file_headers();
        for (auto& line : _clock_syncs.headers())
            file_headers.emplace_back(std::move(line));
        BinaryTimerFile{traces}.write(timer_file, _force_strings, _headers, file_headers);
    }

    bool active         = false;
    bool _force_strings = false;
    std::atomic<bool> _memory{false};
    std::atomic<bool> _io{false};
    bool _memory_peak         = false;
    std::uint64_t _peak_carry = 0; // peak erased by open mem scope resets, see scope.hpp
    std::string timer_file;
    std::vector<std::string> _headers;
    ClockSyncs _clock_syncs; // see clock_sync
    std::vector<Report*> reports;
    std::vector<Snapshot*> traces;
    Report* report_stack_ptr = nullptr;
};

// the instances, defined in src/phlop/timing/scope_timer.cpp
template<>
ScopeTimerMan<SteadyClock>& ScopeTimerMan<SteadyClock>::INSTANCE();
template<>
ScopeTimerNode<SteadyClock>*& ScopeTimerMan<SteadyClock>::current_scope();
template<>
ScopeTimerMan<mpi::Clock>& ScopeTimerMan<mpi::Clock>::INSTANCE();
template<>
ScopeTimerNode<mpi::Clock>*& ScopeTimerMan<mpi::Clock>::current_scope();


template<typename Clock>
struct RunTimerReport
{
    std::string k, f;
    std::uint32_t l = 0;

    RunTimerReport(std::string_view const& _k, std::string_view const& _f, std::uint32_t const& _l)
        : k{_k}
        , f{_f}
        , l{_l}
    {
        ScopeTimerMan<Clock>::INSTANCE().reports.emplace_back(this);
        snapshots.reserve(25);
    }

    auto operator()(std::size_t i) { return snapshots[i].get(); }
    auto size() { return snapshots.size(); }

    // sole owner: traces/childs hold raw pointers into these
    std::vector<std::unique_ptr<RunTimerReportSnapshotT<RunTimerReport>>> snapshots;
};


template<typename Clock>
struct ScopeTimer : ScopeTimerNode<Clock>
{
    using Man      = ScopeTimerMan<Clock>;
    using Report   = RunTimerReport<Clock>;
    using Snapshot = RunTimerReportSnapshotT<Report>;

    ScopeTimer(Report& _r)
        : r{_r}
    {
        if (Man::INSTANCE().active)
        {
            this->pscope                     = Man::current_scope();
            Man::current_scope()             = this;
            Man::INSTANCE().report_stack_ptr = &r; // restored to parent by the destructor

#if defined(PHLOP_SCOPE_TIMER_SELF_STATS)
            detail::construct_stats.add(now() - _self_start);
#endif
        }
    }

    ~ScopeTimer()
    {
        auto& man = Man::INSTANCE();
        if (man.active)
        {
#if defined(PHLOP_SCOPE_TIMER_SELF_STATS)
            auto const self_start = now();
#endif
            Man::current_scope() = this->pscope;

            // time first, then the exit memory sample, then any bookkeeping allocation
            //  so neither the sample nor the snapshot are counted in the scope
            auto const end        = now();
            IoSample const io_out = pending_io ? IoSample::now() : IoSample{}; // before mem's read
            MemSample mem_out     = pending_mem ? MemSample::now() : MemSample{};
            if (pending_peak) // carry this scope's peak to its parent, see scope.hpp
            {
                auto& carry  = man._peak_carry;
                mem_out.peak = (std::max)(mem_out.peak, carry);
                carry        = (std::max)(pending_carry, mem_out.peak);
            }

            auto& s = *r.snapshots.emplace_back(
                std::make_unique<Snapshot>(&r, parent, start, end - start));

            if (this->pscope)
                this->pscope->childs.emplace_back(&s);

            s.childs = std::move(this->childs);
            s.level  = pending_level;
            s.counts = std::move(pending_counts);
            if (pending_mem)
            {
                s.has_mem = true;
                s.mem_in  = pending_mem_in;
                s.mem_out = mem_out;
            }
            if (pending_io and io_out.ok)
            {
                s.has_io = true;
                s.io     = io_out - pending_io_in;
            }

            if (parent == nullptr)
                man.traces.emplace_back(&s);

            man.report_stack_ptr = parent;

#if defined(PHLOP_SCOPE_TIMER_SELF_STATS)
            detail::destruct_stats.add(now() - self_start);
#endif
        }
    }

    static std::uint64_t now() { return Clock::now(); }

#if defined(PHLOP_SCOPE_TIMER_SELF_STATS)
    // first member: initialised before the others, so they're measured too
    std::uint64_t const _self_start = now();
#endif
    Report& r;
    Report* parent            = Man::INSTANCE().report_stack_ptr;
    std::uint64_t const start = now();

    // set by an enclosing Scope before this destructs, moved into the snapshot
    std::uint8_t pending_level = 0;
    std::vector<std::uint64_t> pending_counts;
    bool pending_mem = false;
    MemSample pending_mem_in{};
    bool pending_peak           = false; // peak reset on entry
    std::uint64_t pending_carry = 0;
    bool pending_io             = false;
    IoSample pending_io_in{};
};


template<std::uint32_t F, typename Clock>
using Scope = ScopeT<F, ScopeTimer<Clock>, ScopeTimerMan<Clock>>;

} // namespace phlop::impl::serial


// phlop:: the serial timer with the steady clock
namespace phlop
{

using ScopeTimerMan  = impl::serial::ScopeTimerMan<SteadyClock>;
using RunTimerReport = impl::serial::RunTimerReport<SteadyClock>;
using ScopeTimer     = impl::serial::ScopeTimer<SteadyClock>;
template<std::uint32_t F>
using Scope = impl::serial::Scope<F, SteadyClock>;

inline auto& scope_timer()
{
    return ScopeTimerMan::INSTANCE();
}

inline void shutdown_scope_timer()
{
    ScopeTimerMan::INSTANCE().shutdown();
}

inline void reset_scope_timer()
{
    ScopeTimerMan::INSTANCE().reset();
}

} // namespace phlop


#endif /*_PHLOP_TIMING_SERIAL_TIMER_HPP_*/
