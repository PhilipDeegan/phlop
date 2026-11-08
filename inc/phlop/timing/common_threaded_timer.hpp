#ifndef _PHLOP_TIMING_COMMON_THREADED_TIMER_HPP_
#define _PHLOP_TIMING_COMMON_THREADED_TIMER_HPP_


#include "phlop/timing/common_timer.hpp"
#include "phlop/timing/scope.hpp"

#include <iostream>

#include <mutex>
#include <atomic>
#include <memory>
#include <cstdlib>
#include <vector>
#include <functional>
#include <string_view>
#include <unordered_set>


namespace phlop::threaded
{

struct RunTimerReport;
using RunTimerReportSnapshot = RunTimerReportSnapshotT<RunTimerReport>;

template<typename Clock = SteadyClock>
struct ScopeTimer;

} // namespace phlop::threaded


namespace phlop::threaded::detail
{

void inline write_timer_file(); // defined below after ScopeTimerMan
auto const default_file_name = []() -> std::string { return ".phlop_scope_times.bin"; };

} // namespace phlop::threaded::detail


namespace phlop::threaded
{



struct ScopeTimerMan
{
    static ScopeTimerMan& INSTANCE();
    static constexpr std::string_view file_name_default = ".phlop_scope_times.bin";

    ScopeTimerMan(std::function<std::string()> namer = detail::default_file_name)
        : timer_file{namer()}
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
        active          = true;
        local().movable = true;
        perf::Config::INSTANCE().init();
        memory(_memory.load(std::memory_order_relaxed));
        io(_io.load(std::memory_order_relaxed));
    }

    void shutdown(bool write = true)
    {
        local().move();
        local().movable = false;
        if (write)
            detail::write_timer_file();

        if (auto const s = perf::summary(); !s.empty())
            std::cerr << "phlop perf counters: " << s << '\n';
        perf::Config::INSTANCE().close();

        _headers.clear();
        _clock_syncs.clear();
        _clock_global = -1;
        thread_storage.clear();
        thread_reports.clear();
        active = false;
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
    //  it would erase the peak other threads' scopes see, PHLOP_SCOPE_MEM=peak is =1
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

    // a clock value taken at a point every process passed together, eg just after a
    //  barrier, see phlop::mpi::sync_clock(). Written as "#clock sync=<t>,..." so
    //  starts in different files can be aligned in post-processing. global: whether
    //  the clock is already the same everywhere, 1/0, -1 unknown
    void clock_sync(std::uint64_t const t, int const global = -1)
    {
        std::unique_lock<std::mutex> lk(work_);
        _clock_syncs.emplace_back(t);
        if (global >= 0)
            _clock_global = global;
    }

    // "#clock sync=<t>,... global=<0|1>" or none, see clock_sync
    std::vector<std::string> clock_headers()
    {
        std::unique_lock<std::mutex> lk(work_);
        if (_clock_syncs.empty())
            return {};
        std::string line = "#clock sync=";
        for (std::size_t i = 0; i < _clock_syncs.size(); ++i)
            line += (i ? "," : "") + std::to_string(_clock_syncs[i]);
        if (_clock_global >= 0)
            line += " global=" + std::to_string(_clock_global);
        return {line};
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
            ScopeTimerMan::INSTANCE().move(*this);
            movable = false;
        }

        std::vector<RunTimerReport*> reports;
        std::vector<RunTimerReportSnapshot*> traces;
        RunTimerReport* report_stack_ptr = nullptr;
        bool movable                     = true;
    };

    // the calling thread's storage, defined in src/phlop/timing/common_threaded_timer.cpp
    //  not inline, so shared across shared libraries built with -fvisibility=hidden
    static per_thread& local();

    void move(per_thread& pt)
    {
        std::unique_lock<std::mutex> lk(work_);
        thread_storage.emplace_back(std::move(pt.reports), std::move(pt.traces));
    }
    void move(std::shared_ptr<RunTimerReport>& report)
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


    bool active         = false;
    bool _force_strings = false;
    std::atomic<bool> _memory{false}; // read by every thread entering a mem scope
    std::atomic<bool> _io{false};     // read by every thread entering an io scope
    std::string timer_file = "";
    std::vector<std::string> _headers;
    std::vector<std::uint64_t> _clock_syncs; // see clock_sync
    int _clock_global = -1;

    std::mutex work_;
    std::vector<std::pair<std::vector<RunTimerReport*>, std::vector<RunTimerReportSnapshot*>>>
        thread_storage;
    std::vector<std::shared_ptr<RunTimerReport>> thread_reports;
    std::unordered_set<per_thread*> threads; // all live threads
};


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
        ScopeTimerMan::local().reports.emplace_back(this);
        snapshots.reserve(5);
    }

    ~RunTimerReport() {}

    auto operator()(std::size_t i) { return snapshots[i].get(); }
    auto size() { return snapshots.size(); }

    // sole owner: traces/childs hold raw pointers into these
    std::vector<std::unique_ptr<RunTimerReportSnapshot>> snapshots;
};


struct ThreadLifeWatcher
{
    ~ThreadLifeWatcher() { ScopeTimerMan::INSTANCE().move(report); }

    std::shared_ptr<RunTimerReport> report;
};


// non-template link in the scope stack: the current scope per thread is held
//  out of line, see detail::current_scope(), so it is shared across shared libraries
struct ScopeTimerNode
{
    static ScopeTimerNode& root_parent_from(ScopeTimerNode& self)
    {
        if (self.pscope)
            return root_parent_from(*self.pscope);
        return self;
    }

    ScopeTimerNode* pscope = nullptr;
    std::vector<RunTimerReportSnapshot*> childs;
};


namespace detail
{
    // the innermost active scope on the calling thread, defined in
    // src/phlop/timing/common_threaded_timer.cpp
    ScopeTimerNode*& current_scope();

    inline void write_timer_file()
    {
        auto& man = ScopeTimerMan::INSTANCE();
        std::vector<RunTimerReportSnapshot*> all_traces;
        std::vector<ScopeTimerMan::per_thread*> alive;

        // Snapshot dead-thread storage and alive-thread pointers atomically so
        // a thread dying mid-collection can't cause a race on either container.
        {
            std::unique_lock<std::mutex> lk(man.work_);
            for (auto const& [reports, traces] : man.thread_storage)
                for (auto* t : traces)
                    all_traces.push_back(t);
            man.thread_storage.clear();
            alive.assign(man.threads.begin(), man.threads.end());
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
        auto clock_headers = man.clock_headers();
        if (all_traces.size() || clock_headers.size())
        {
            auto file_headers = perf::Config::INSTANCE().file_headers();
            for (auto& line : clock_headers)
                file_headers.emplace_back(std::move(line));
            phlop::BinaryTimerFile{all_traces}.write(man.timer_file, man._force_strings,
                                                     man._headers, file_headers);
        }
    }

} // namespace detail


template<typename Clock>
struct ScopeTimer : ScopeTimerNode
{
    ScopeTimer(RunTimerReport& _r)
        : r{_r}
    {
        if (ScopeTimerMan::INSTANCE().active)
        {
            this->pscope            = detail::current_scope();
            detail::current_scope() = this;
        }
    }

    ~ScopeTimer()
    {
        if (ScopeTimerMan::INSTANCE().active)
        {
            detail::current_scope() = this->pscope;

            // time first, then the exit memory sample, then any bookkeeping allocation
            //  so neither the sample nor the snapshot are counted in the scope
            auto const end        = now();
            IoSample const io_out = pending_io ? IoSample::now() : IoSample{}; // before mem's read
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
            if (pending_io and io_out.ok)
            {
                s.has_io = true;
                s.io     = io_out - pending_io_in;
            }

            if (parent == nullptr)
                ScopeTimerMan::local().traces.emplace_back(&s);

            ScopeTimerMan::local().report_stack_ptr = parent;
        }
    }

    static std::uint64_t now() { return Clock::now(); }

    RunTimerReport& r;
    RunTimerReport* parent    = ScopeTimerMan::local().report_stack_ptr;
    RunTimerReport* child     = nullptr;
    std::uint64_t const start = now();

    // set by an enclosing Scope before this destructs, moved into the snapshot
    std::uint8_t pending_level = 0;
    std::vector<std::uint64_t> pending_counts;
    bool pending_mem = false;
    MemSample pending_mem_in{};
    bool pending_io = false;
    IoSample pending_io_in{};
};


template<std::uint32_t F, typename Clock = SteadyClock>
using Scope = ScopeT<F, ScopeTimer<Clock>, ScopeTimerMan>;

} // namespace phlop::threaded

#endif /*_PHLOP_TIMING_COMMON_THREADED_TIMER_HPP_*/
