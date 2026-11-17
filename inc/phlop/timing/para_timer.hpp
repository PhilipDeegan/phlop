#ifndef _PHLOP_TIMING_PARA_TIMER_HPP_
#define _PHLOP_TIMING_PARA_TIMER_HPP_

// The multithread capable scope timer, templated on its clock: phlop::para::
//  (steady clock) here, phlop::mpi::para:: (MPI_Wtime) in mpi_timer.hpp. Per thread
//  storage, merged at shutdown. No per scope peak reset: PHLOP_SCOPE_MEM=peak is =1
// No macros here, see para_scope_timer.hpp / mpi_para_scope_timer.hpp. The
//  instances are in src/phlop/timing/scope_timer.cpp, the one source for every timer

#include "phlop/macros/def/string.hpp"
#include "phlop/timing/nvtx.hpp"
#include "phlop/timing/common_timer.hpp"
#include "phlop/timing/scope.hpp"

#include <mutex>
#include <utility>
#include <atomic>
#include <memory>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <string_view>
#include <unordered_set>

// the body of PHLOP_SCOPE for the para timer in namespace ns, see para_scope_timer.hpp
#define PHLOP_DETAIL_PARA_SCOPE(ns, key, flags)                                                    \
    static thread_local auto PHLOP_STR_CAT(ridx_, __LINE__)                                        \
        = std::make_shared<ns::RunTimerReport>(key, __FILE__, __LINE__);                           \
    static thread_local ns::ThreadLifeWatcher PHLOP_STR_CAT(_watcher_, __LINE__){                  \
        PHLOP_STR_CAT(ridx_, __LINE__)};                                                           \
    PHLOP_NVTX_RANGE(PHLOP_STR_CAT(ridx_, __LINE__)->k)                                            \
    ns::Scope<flags> PHLOP_STR_CAT(_Scope_, __LINE__){*PHLOP_STR_CAT(ridx_, __LINE__)};


namespace phlop::impl::para
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

    struct per_thread;

    // out of line, src/phlop/timing/scope_timer.cpp: one instance per process,
    //  even across shared libraries built with -fvisibility=hidden
    static ScopeTimerMan& INSTANCE();
    static per_thread& local(); // the calling thread's storage
    static ScopeTimerNode<Clock>*& current_scope(); // the calling thread's innermost scope

    ScopeTimerMan(std::string_view const fn)
        : timer_file{fn}
    {
        perf::Config::INSTANCE(); // constructed first so destroyed after this, see ~ScopeTimerMan
    }
    ~ScopeTimerMan()
    {
        if (active)
            shutdown();
    }

    // perf counters (PHLOP_PERF_LIGHT/ALL env): every thread counts itself,
    // opening its counters on its first counted scope
    void init()
    {
        if (!std::exchange(active, true)) // a second init keeps the one acquire
            perf::Config::INSTANCE().acquire();
        local().movable = true;
        memory(_memory.load(std::memory_order_relaxed));
        io(_io.load(std::memory_order_relaxed));
    }

    void shutdown(bool write = true)
    {
        local().move();
        local().movable = false;
        if (write)
            write_file();

        if (auto const s = perf::summary(); !s.empty())
            std::cerr << "phlop perf counters: " << s << '\n';
        if (std::exchange(active, false))
            perf::Config::INSTANCE().release();

        _headers.clear();
        _clock_syncs.clear();
        thread_storage.clear();
        thread_reports.clear();
    }

    auto& file_name(std::string const fn)
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
    //  resident memory is per process, not per thread. No peak reset (see scope.hpp):
    //  it would erase the peak other threads' scopes see, PHLOP_SCOPE_MEM=peak is =1.
    //  Scopes on one thread only: the serial timer, phlop:: / phlop::mpi::, has the reset
    ScopeTimerMan& memory(bool const b = true)
    {
        bool on = b; // PHLOP_SCOPE_MEM=1/0, if set, wins over the code either way
        if (auto const env = std::getenv("PHLOP_SCOPE_MEM"))
            on = std::string_view{env} != "0";
        _memory.store(on, std::memory_order_relaxed);
        return *this;
    }

    // sample each thread's own I/O for PHLOP_SCOPE("key", time, io) scopes,
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

    // see ClockSyncs, phlop::mpi::para::sync_clock()
    void clock_sync(std::uint64_t const t, int const global = -1)
    {
        std::unique_lock<std::mutex> lk(work_);
        _clock_syncs.add(t, global);
    }

    std::vector<std::string> clock_headers()
    {
        std::unique_lock<std::mutex> lk(work_);
        return _clock_syncs.headers();
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


    struct per_thread
    {
        per_thread()
        {
            if (auto& self = INSTANCE(); self.active)
                self.add(this);
        }
        per_thread(per_thread&& pt)
            : reports{std::move(pt.reports)}
            , traces{std::move(pt.traces)}
        {
            pt.movable = false;
            if (auto& self = INSTANCE(); self.active)
                self.add(this);
        }
        per_thread(per_thread const&) = delete;
        ~per_thread()
        {
            auto& self = INSTANCE();
            if (self.active && movable)
                move();
            self.rm(this);
        }

        void move()
        {
            INSTANCE().move(*this);
            movable = false;
        }

        std::vector<Report*> reports;
        std::vector<Snapshot*> traces;
        Report* report_stack_ptr = nullptr;
        bool movable             = true;
    };

    void move(per_thread& pt)
    {
        std::unique_lock<std::mutex> lk(work_);
        thread_storage.emplace_back(std::move(pt.reports), std::move(pt.traces));
    }
    void move(std::shared_ptr<Report>& report)
    {
        std::unique_lock<std::mutex> lk(work_);
        thread_reports.emplace_back(std::move(report));
    }

    void add(per_thread* thread)
    {
        std::unique_lock<std::mutex> lk(work_);
        threads.emplace(thread);
    }
    void rm(per_thread* thread)
    {
        std::unique_lock<std::mutex> lk(work_);
        threads.erase(thread);
    }

    void write_file()
    {
        std::vector<Snapshot*> all_traces;
        std::vector<per_thread*> alive;

        // Snapshot dead-thread storage and alive-thread pointers atomically so
        // a thread dying mid-collection can't cause a race on either container.
        {
            std::unique_lock<std::mutex> lk(work_);
            for (auto const& [reports, traces] : thread_storage)
                for (auto* t : traces)
                    all_traces.push_back(t);
            thread_storage.clear();
            alive.assign(threads.begin(), threads.end());
        }

        // Safe provided no new timer data is being written during shutdown.
        for (auto* pt : alive)
        {
            for (auto* t : pt->traces)
                all_traces.push_back(t);
            pt->traces.clear();
        }

        // no scopes but clock syncs: still written, so a rank without scopes is
        //  in the file set, and reported as such, rather than silently missing
        auto ch = clock_headers();
        if (all_traces.size() || ch.size())
        {
            auto file_headers = perf::Config::INSTANCE().file_headers();
            for (auto& line : ch)
                file_headers.emplace_back(std::move(line));
            BinaryTimerFile{all_traces}.write(timer_file, _force_strings, _headers, file_headers);
        }
    }


    bool active         = false;
    bool _force_strings = false;
    std::atomic<bool> _memory{false}; // read by every thread entering a mem scope
    std::atomic<bool> _io{false};     // read by every thread entering an io scope
    std::string timer_file = "";
    std::vector<std::string> _headers;
    ClockSyncs _clock_syncs; // see clock_sync

    std::mutex work_;
    std::vector<std::pair<std::vector<Report*>, std::vector<Snapshot*>>> thread_storage;
    std::vector<std::shared_ptr<Report>> thread_reports;
    std::unordered_set<per_thread*> threads; // all live threads
};

// the instances, defined in src/phlop/timing/scope_timer.cpp
template<>
ScopeTimerMan<SteadyClock>& ScopeTimerMan<SteadyClock>::INSTANCE();
template<>
ScopeTimerMan<SteadyClock>::per_thread& ScopeTimerMan<SteadyClock>::local();
template<>
ScopeTimerNode<SteadyClock>*& ScopeTimerMan<SteadyClock>::current_scope();
template<>
ScopeTimerMan<mpi::Clock>& ScopeTimerMan<mpi::Clock>::INSTANCE();
template<>
ScopeTimerMan<mpi::Clock>::per_thread& ScopeTimerMan<mpi::Clock>::local();
template<>
ScopeTimerNode<mpi::Clock>*& ScopeTimerMan<mpi::Clock>::current_scope();


template<typename Clock>
struct RunTimerReport
{
    std::string const k;
    std::string const f;
    std::uint32_t l = 0;

    RunTimerReport(std::string_view const& _k, std::string_view const& _f, std::uint32_t const& _l)
        : k{_k}
        , f{_f}
        , l{_l}
    {
        ScopeTimerMan<Clock>::local().reports.emplace_back(this);
        snapshots.reserve(5);
    }

    auto operator()(std::size_t i) { return snapshots[i].get(); }
    auto size() { return snapshots.size(); }

    // sole owner: traces/childs hold raw pointers into these
    std::vector<std::unique_ptr<RunTimerReportSnapshotT<RunTimerReport>>> snapshots;
};


// hands a thread's report to the timer as the thread exits, see PHLOP_DETAIL_PARA_SCOPE
template<typename Clock>
struct ThreadLifeWatcher
{
    ~ThreadLifeWatcher() { ScopeTimerMan<Clock>::INSTANCE().move(report); }

    std::shared_ptr<RunTimerReport<Clock>> report;
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
            this->pscope                   = Man::current_scope();
            Man::current_scope()           = this;
            Man::local().report_stack_ptr = &r; // restored to parent by the destructor
        }
    }

    ~ScopeTimer()
    {
        if (Man::INSTANCE().active)
        {
            Man::current_scope() = this->pscope;

            // time first, then the exit memory sample, then any bookkeeping allocation
            //  so neither the sample nor the snapshot are counted in the scope
            auto const end        = now();
            IoSample const io_out = pending_io ? IoSample::now() : IoSample{}; // before mem's read
            MemSample const mem_out = pending_mem ? MemSample::now() : MemSample{};

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

            auto& local = Man::local();
            if (parent == nullptr)
                local.traces.emplace_back(&s);

            local.report_stack_ptr = parent;
        }
    }

    static std::uint64_t now() { return Clock::now(); }

    Report& r;
    Report* parent            = Man::local().report_stack_ptr;
    std::uint64_t const start = now();

    // set by an enclosing Scope before this destructs, moved into the snapshot
    std::uint8_t pending_level = 0;
    std::vector<std::uint64_t> pending_counts;
    bool pending_mem = false;
    MemSample pending_mem_in{};
    bool pending_io = false;
    IoSample pending_io_in{};
};


template<std::uint32_t F, typename Clock>
using Scope = ScopeT<F, ScopeTimer<Clock>, ScopeTimerMan<Clock>>;

} // namespace phlop::impl::para


// phlop::para:: the multithread capable timer with the steady clock
namespace phlop::para
{

using ScopeTimerMan     = impl::para::ScopeTimerMan<SteadyClock>;
using RunTimerReport    = impl::para::RunTimerReport<SteadyClock>;
using ThreadLifeWatcher = impl::para::ThreadLifeWatcher<SteadyClock>;
using ScopeTimer        = impl::para::ScopeTimer<SteadyClock>;
template<std::uint32_t F>
using Scope = impl::para::Scope<F, SteadyClock>;

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

} // namespace phlop::para


#endif /*_PHLOP_TIMING_PARA_TIMER_HPP_*/
