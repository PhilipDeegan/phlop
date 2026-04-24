
#include "phlop/timing/threaded_scope_timer.hpp"

namespace phlop::threaded
{

ScopeTimerMan& ScopeTimerMan::INSTANCE()
{
    static ScopeTimerMan i;
    return i;
}

} // namespace phlop::threaded
