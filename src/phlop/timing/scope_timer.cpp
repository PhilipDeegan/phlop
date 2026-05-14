
#include "phlop/timing/scope_timer.hpp"

namespace phlop
{

ScopeTimerMan& ScopeTimerMan::INSTANCE()
{
    static ScopeTimerMan i;
    return i;
}

} // namespace phlop
