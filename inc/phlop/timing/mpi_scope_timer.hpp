#ifndef _PHLOP_TIMING_MPI_SCOPE_TIMER_HPP_
#define _PHLOP_TIMING_MPI_SCOPE_TIMER_HPP_

#if defined(_PHLOP_TIMER_EXPORTED_) && _PHLOP_TIMER_EXPORTED_ != 2
#error // can't use more than one timer type at time
#endif
#define _PHLOP_TIMER_EXPORTED_ 2

#include "phlop/macros/def/string.hpp"
#include "phlop/timing/common_threaded_timer.hpp"

#include <memory>
#include <string>
#include <cstdint>

#include <mpi.h>


#define PHLOP_SCOPE_TIMER(key)                                                                     \
    static thread_local auto PHLOP_STR_CAT(ridx_, __LINE__)                                        \
        = std::make_shared<phlop::threaded::RunTimerReport>(key, __FILE__, __LINE__);              \
    static thread_local phlop::threaded::ThreadLifeWatcher PHLOP_STR_CAT(_watcher_, __LINE__){     \
        PHLOP_STR_CAT(ridx_, __LINE__)};                                                           \
    phlop::threaded::ScopeTimer<phlop::mpi::Clock> PHLOP_STR_CAT(_scope_timer_, __LINE__){         \
        *PHLOP_STR_CAT(ridx_, __LINE__)};                                                          \
    phlop::threaded::ScopeTimerMan::local().report_stack_ptr = PHLOP_STR_CAT(ridx_, __LINE__).get();



namespace phlop::mpi
{

static auto const scope_timer_file_namer = [](std::string const base = "phlop_scope_timers.") {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return base + std::to_string(rank) + ".bin";
};

struct Clock
{
    static std::uint64_t now() { return static_cast<std::uint64_t>(MPI_Wtime() * 1e9); }
};


} // namespace phlop::mpi

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

} // namespace phlop


#endif /*_PHLOP_TIMING_MPI_SCOPE_TIMER_HPP_*/
