
#include "phlop/timing/common_threaded_timer.hpp"

namespace phlop::threaded
{

ScopeTimerMan& ScopeTimerMan::INSTANCE()
{
    static ScopeTimerMan i;
    return i;
}

} // namespace phlop::threaded
