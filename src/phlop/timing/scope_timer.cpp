
#include "phlop/timing/scope_timer.hpp"

namespace phlop
{

ScopeTimerMan& ScopeTimerMan::INSTANCE()
{
    static ScopeTimerMan i;
    return i;
}

ScopeTimerNode*& detail::current_scope()
{
    static ScopeTimerNode* current = nullptr;
    return current;
}

} // namespace phlop

// out of line so that every shared library uses the same instances, see perf_counters.hpp
//  only one of scope_timer.cpp / common_threaded_timer.cpp may be linked into a program
namespace phlop::perf
{

Config& Config::INSTANCE()
{
    static Config c;
    return c;
}

Group& local()
{
    static thread_local Group group;
    return group;
}

} // namespace phlop::perf
