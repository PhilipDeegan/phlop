#ifndef _PHLOP_TIMING_THREADED_SCOPE_TIMER_HPP_
#define _PHLOP_TIMING_THREADED_SCOPE_TIMER_HPP_

#if defined(_PHLOP_TIMER_EXPORTED_) && _PHLOP_TIMER_EXPORTED_ != 1
#error // can't use more than one timer type at time
#endif
#define _PHLOP_TIMER_EXPORTED_ 1

#include "phlop/macros/def/string.hpp"
#include "phlop/timing/common_threaded_timer.hpp"

#include <memory>

#define PHLOP_SCOPE_TIMER(key)                                                                     \
    static thread_local auto PHLOP_STR_CAT(ridx_, __LINE__)                                        \
        = std::make_shared<phlop::threaded::RunTimerReport>(key, __FILE__, __LINE__);              \
    static thread_local phlop::threaded::ThreadLifeWatcher PHLOP_STR_CAT(_watcher_, __LINE__){     \
        PHLOP_STR_CAT(ridx_, __LINE__)};                                                           \
    phlop::threaded::ScopeTimer<> PHLOP_STR_CAT(_scope_timer_,                                     \
                                                __LINE__){*PHLOP_STR_CAT(ridx_, __LINE__)};        \
    phlop::threaded::ScopeTimerMan::local().report_stack_ptr = PHLOP_STR_CAT(ridx_, __LINE__).get();


namespace phlop
{

inline auto& scope_timer()
{
    return threaded::ScopeTimerMan::INSTANCE();
}

inline void shutdown_scope_timer()
{
    threaded::ScopeTimerMan::INSTANCE().shutdown();
}

inline void reset_scope_timer()
{
    threaded::ScopeTimerMan::INSTANCE().reset();
}


} // namespace phlop


#endif /*_PHLOP_TIMING_THREADED_SCOPE_TIMER_HPP_*/
