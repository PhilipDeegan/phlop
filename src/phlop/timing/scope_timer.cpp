
#include "phlop/timing/scope_timer.hpp"

namespace phlop
{


ScopeTimerMan& ScopeTimerMan::INSTANCE()
{
    static ScopeTimerMan i;
    return i;
}

namespace detail
{
    static scope_timer* _current_scope_timer = nullptr;


} // namespace detail

scope_timer::scope_timer(RunTimerReport& _r)
    : r{_r}
{
    if (ScopeTimerMan::INSTANCE().active)
    {
        auto const begin = now();
        this->pscope     = detail::_current_scope_timer;

        if (this->pscope)
            pscope->childs.reserve(pscope->childs.size() + 1);

        detail::_current_scope_timer = this;

        detail::max_construct_time = std::max(detail::max_construct_time, now() - begin);
    }
}

scope_timer::~scope_timer()
{
    if (ScopeTimerMan::INSTANCE().active)
    {
        auto const begin             = now();
        detail::_current_scope_timer = this->pscope;

        auto& s = *r.snapshots.emplace_back( // allocated in construtor
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


} // namespace phlop
