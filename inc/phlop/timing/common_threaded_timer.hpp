#ifndef _PHLOP_TIMING_COMMON_THREADED_TIMER_HPP_
#define _PHLOP_TIMING_COMMON_THREADED_TIMER_HPP_


#include "phlop/timing/common_timer.hpp"

#include <mutex>
#include <memory>
#include <vector>
#include <functional>
#include <string_view>
#include <unordered_set>


namespace phlop::threaded
{

struct RunTimerReport;
using RunTimerReportSnapshot = RunTimerReportSnapshotT<RunTimerReport>;

// forward declare so detail::_current_ScopeTimer<Clock> can hold a pointer
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
    }
    ~ScopeTimerMan()
    {
        if (active)
            shutdown();
    }

    void init()
    {
        active          = true;
        local().movable = true;
    }

    void shutdown(bool write = true)
    {
        local().move();
        local().movable = false;
        if (write)
            detail::write_timer_file();

        _headers.clear();
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

    template<typename... Args>
    auto& headers(Args const... args)
    {
        _headers.clear();
        (_headers.emplace_back(args), ...);
        return *this;
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
        per_thread() { ScopeTimerMan::INSTANCE().add(this); }
        per_thread(per_thread&& pt)
            : reports{std::move(pt.reports)}
            , traces{std::move(pt.traces)}
        {
            pt.movable = false;
            ScopeTimerMan::INSTANCE().add(this);
        }
        per_thread(per_thread const&) = delete;
        ~per_thread()
        {
            if (movable)
                move();
            ScopeTimerMan::INSTANCE().rm(this);
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

    static per_thread& local()
    {
        static thread_local per_thread* ptr;
        if (!ptr)
        {
            static thread_local per_thread inst;
            ptr = &inst;
        }
        return *ptr;
    }

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


    bool active            = false;
    bool _force_strings    = false;
    std::string timer_file = "";
    std::vector<std::string> _headers;

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

    std::vector<std::shared_ptr<RunTimerReportSnapshot>> snapshots;
};


struct ThreadLifeWatcher
{
    ~ThreadLifeWatcher() { ScopeTimerMan::INSTANCE().move(report); }

    std::shared_ptr<RunTimerReport> report;
};


namespace detail
{
    template<typename Clock>
    inline thread_local ScopeTimer<Clock>* _current_ScopeTimer = nullptr;

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

        if (all_traces.size())
            phlop::BinaryTimerFile{all_traces}.write(man.timer_file, man._force_strings,
                                                     man._headers);
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

            if (this->pscope)
                pscope->childs.reserve(pscope->childs.size() + 1);
        }
    }

    ~ScopeTimer()
    {
        if (ScopeTimerMan::INSTANCE().active)
        {
            detail::_current_ScopeTimer<Clock> = this->pscope;

            auto& s = *r.snapshots.emplace_back(
                std::make_shared<RunTimerReportSnapshot>(&r, parent, start, now() - start));

            if (this->pscope)
                pscope->childs.emplace_back(&s);

            s.childs = std::move(childs);

            if (parent == nullptr)
                ScopeTimerMan::local().traces.emplace_back(&s);

            ScopeTimerMan::local().report_stack_ptr = parent;
        }
    }

    static std::uint64_t now() { return Clock::now(); }

    static ScopeTimer& root_parent_from(ScopeTimer& self)
    {
        if (self.pscope)
            return root_parent_from(*self.pscope);
        return self;
    }

    RunTimerReport& r;
    RunTimerReport* parent    = ScopeTimerMan::local().report_stack_ptr;
    RunTimerReport* child     = nullptr;
    ScopeTimer* pscope        = nullptr;
    std::uint64_t const start = now();
    std::vector<RunTimerReportSnapshot*> childs;
};

} // namespace phlop::threaded

#endif /*_PHLOP_TIMING_COMMON_THREADED_TIMER_HPP_*/
