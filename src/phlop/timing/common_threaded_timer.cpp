
#include "phlop/timing/common_threaded_timer.hpp"

namespace phlop::threaded
{

ScopeTimerMan& ScopeTimerMan::INSTANCE()
{
    static ScopeTimerMan i;
    return i;
}

ScopeTimerMan::per_thread& ScopeTimerMan::local()
{
    static thread_local per_thread* ptr;
    if (!ptr)
    {
        static thread_local per_thread inst;
        ptr = &inst;
    }
    return *ptr;
}

ScopeTimerNode*& detail::current_scope()
{
    static thread_local ScopeTimerNode* current = nullptr;
    return current;
}

} // namespace phlop::threaded

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
