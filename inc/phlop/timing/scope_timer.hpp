#ifndef _PHLOP_TIMING_SCOPE_TIMER_HPP_
#define _PHLOP_TIMING_SCOPE_TIMER_HPP_

#include <iostream>
#include <algorithm>
#include <string_view>

#include "phlop/timing/common_timer.hpp"
#include "phlop/macros/def/string.hpp"

namespace phlop
{

struct RunTimerReport;
using RunTimerReportSnapshot = RunTimerReportSnapshotT<RunTimerReport>;

// forward declare so detail::_current_scope_timer<Clock> can use it as a pointer type
template<typename Clock = SteadyClock>
struct scope_timer;

namespace detail
{
    void inline write_timer_file(); // defined below after ScopeTimerMan

    inline std::size_t max_construct_time = 0;
    inline std::size_t max_destruct_time  = 0;

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

    void init() { active = true; }

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

    void print_self_stats() const
    {
        std::cout << "PHLOP SCOPE TIMER STATS" << std::endl;
        std::cout << " max_construct_time " << detail::max_construct_time << std::endl;
        std::cout << " max_destruct_time  " << detail::max_destruct_time << std::endl;
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

    std::vector<std::shared_ptr<RunTimerReportSnapshot>> snapshots;
};


namespace detail
{
    template<typename Clock>
    inline scope_timer<Clock>* _current_scope_timer = nullptr;

    void inline write_timer_file()
    {
        auto& man = ScopeTimerMan::INSTANCE();
        BinaryTimerFile{man.traces}.write(man.timer_file, man._force_strings, man._headers);
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
            auto const begin                    = now();
            this->pscope                        = detail::_current_scope_timer<Clock>;
            detail::_current_scope_timer<Clock> = this;

            if (this->pscope)
                pscope->childs.reserve(pscope->childs.size() + 1);

            detail::max_construct_time = std::max(detail::max_construct_time, now() - begin);
        }
    }

    ~scope_timer()
    {
        if (ScopeTimerMan::INSTANCE().active)
        {
            auto const begin                    = now();
            detail::_current_scope_timer<Clock> = this->pscope;

            auto& s = *r.snapshots.emplace_back(
                std::make_shared<RunTimerReportSnapshot>(&r, parent, start, now() - start));

            if (this->pscope)
                pscope->childs.emplace_back(&s);

            s.childs = std::move(childs);

            if (parent == nullptr)
                ScopeTimerMan::INSTANCE().traces.emplace_back(&s);

            ScopeTimerMan::INSTANCE().report_stack_ptr = parent;

            detail::max_destruct_time = std::max(detail::max_destruct_time, now() - begin);
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
    RunTimerReport* parent    = ScopeTimerMan::INSTANCE().report_stack_ptr;
    RunTimerReport* child     = nullptr;
    scope_timer* pscope       = nullptr;
    std::uint64_t const start = now();
    std::vector<RunTimerReportSnapshot*> childs;
};


} // namespace phlop

#if !defined(PHLOP_SCOPE_TIMER)
#define PHLOP_SCOPE_TIMER(key)                                                                     \
    static phlop::RunTimerReport PHLOP_STR_CAT(ridx_, __LINE__){key, __FILE__, __LINE__};          \
    phlop::scope_timer<> PHLOP_STR_CAT(_scope_timer_, __LINE__){PHLOP_STR_CAT(ridx_, __LINE__)};   \
    phlop::ScopeTimerMan::INSTANCE().report_stack_ptr = &PHLOP_STR_CAT(ridx_, __LINE__);
#endif

#endif /*_PHLOP_TIMING_SCOPE_TIMER_HPP_*/
