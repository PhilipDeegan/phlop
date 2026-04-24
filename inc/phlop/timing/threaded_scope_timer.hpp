#ifndef _PHLOP_TIMING_THREADED_SCOPE_TIMER_HPP_
#define _PHLOP_TIMING_THREADED_SCOPE_TIMER_HPP_

#include <mutex>
#include <memory>
#include <vector>
#include <string_view>

#include "phlop/macros/def/string.hpp"
#include "phlop/timing/common_timer.hpp"


namespace phlop::threaded
{

struct RunTimerReport;
using RunTimerReportSnapshot = phlop::RunTimerReportSnapshotT<RunTimerReport>;

// forward declare so detail::_current_scope_timer<Clock> can hold a pointer
template<typename Clock = phlop::SteadyClock>
struct scope_timer;

namespace detail
{
    void inline write_timer_file(); // defined below after ScopeTimerMan

} // namespace detail

struct ScopeTimerMan
{
    static ScopeTimerMan& INSTANCE();
    static constexpr std::string_view file_name_default = ".phlop_scope_times.bin";

    ScopeTimerMan()
        : timer_file{file_name_default}
    {
    }
    ~ScopeTimerMan() {}

    void init() { active = true; }

    void shutdown(bool write = true)
    {
        local().move();
        local().movable = false;
        if (thread_storage.size())
        {
            if (write)
                detail::write_timer_file();
        }
        _headers.clear();
        thread_storage.clear();
        thread_reports.clear();
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

    struct per_thread
    {
        per_thread() {}
        per_thread(per_thread&& pt)
            : reports{std::move(pt.reports)}
            , traces{std::move(pt.traces)}
        {
            pt.movable = false;
        }
        per_thread(per_thread const&) = delete;
        ~per_thread()
        {
            if (movable)
                move();
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

    std::mutex work_;
    std::vector<std::pair<std::vector<RunTimerReport*>, std::vector<RunTimerReportSnapshot*>>>
        thread_storage;
    std::vector<std::shared_ptr<RunTimerReport>> thread_reports;
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
    inline thread_local scope_timer<Clock>* _current_scope_timer = nullptr;

    inline void write_timer_file()
    {
        auto& man = ScopeTimerMan::INSTANCE();
        std::vector<RunTimerReportSnapshot*> all_traces;
        for (auto const& [reports, traces] : man.thread_storage)
            for (auto* t : traces)
                all_traces.push_back(t);
        phlop::BinaryTimerFile{all_traces}.write(man.timer_file, man._force_strings, man._headers);
    }

} // namespace detail


template<typename Clock>
struct scope_timer
{
    scope_timer(RunTimerReport& _r)
        : r{_r}
    {
        if (ScopeTimerMan::INSTANCE().active)
        {
            this->pscope                        = detail::_current_scope_timer<Clock>;
            detail::_current_scope_timer<Clock> = this;

            if (this->pscope)
                pscope->childs.reserve(pscope->childs.size() + 1);
        }
    }

    ~scope_timer()
    {
        if (ScopeTimerMan::INSTANCE().active)
        {
            detail::_current_scope_timer<Clock> = this->pscope;

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

    static scope_timer& root_parent_from(scope_timer& self)
    {
        if (self.pscope)
            return root_parent_from(*self.pscope);
        return self;
    }

    RunTimerReport& r;
    RunTimerReport* parent    = ScopeTimerMan::local().report_stack_ptr;
    RunTimerReport* child     = nullptr;
    scope_timer* pscope       = nullptr;
    std::uint64_t const start = now();
    std::vector<RunTimerReportSnapshot*> childs;
};


} // namespace phlop::threaded


#if !defined(PHLOP_SCOPE_TIMER)
#define PHLOP_SCOPE_TIMER(key)                                                                     \
    static thread_local auto PHLOP_STR_CAT(ridx_, __LINE__)                                        \
        = std::make_shared<phlop::threaded::RunTimerReport>(key, __FILE__, __LINE__);              \
    static thread_local phlop::threaded::ThreadLifeWatcher PHLOP_STR_CAT(_watcher_, __LINE__){     \
        PHLOP_STR_CAT(ridx_, __LINE__)};                                                           \
    phlop::threaded::scope_timer<> PHLOP_STR_CAT(_scope_timer_,                                    \
                                                 __LINE__){*PHLOP_STR_CAT(ridx_, __LINE__)};       \
    phlop::threaded::ScopeTimerMan::local().report_stack_ptr = PHLOP_STR_CAT(ridx_, __LINE__).get();
#endif




#endif /*_PHLOP_TIMING_THREADED_SCOPE_TIMER_HPP_*/
