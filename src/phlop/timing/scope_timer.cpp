
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
        this->pscope = detail::_current_scope_timer;

        if (this->pscope)
            pscope->childs.reserve(pscope->childs.size() + 1);

        detail::_current_scope_timer = this;
    }
}

scope_timer::~scope_timer()
{
    if (ScopeTimerMan::INSTANCE().active)
    {
        detail::_current_scope_timer = this->pscope;

        auto& s = *r.snapshots.emplace_back( // allocated in construtor
            std::make_shared<RunTimerReportSnapshot>(&r, parent, now() - start));

        if (this->pscope)
            pscope->childs.emplace_back(&s);

        s.childs = std::move(childs);

        if (parent == nullptr)
            ScopeTimerMan::INSTANCE().traces.emplace_back(&s);

        ScopeTimerMan::INSTANCE().report_stack_ptr = parent;
    }
}


} // namespace phlop
