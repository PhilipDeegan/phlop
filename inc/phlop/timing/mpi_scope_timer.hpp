#ifndef _PHLOP_TIMING_MPI_SCOPE_TIMER_HPP_
#define _PHLOP_TIMING_MPI_SCOPE_TIMER_HPP_

#if defined(_PHLOP_TIMER_EXPORTED_) && _PHLOP_TIMER_EXPORTED_ != 2
#error // can't use more than one timer type at time
#endif
#define _PHLOP_TIMER_EXPORTED_ 2

#include "phlop/macros/def/string.hpp"
#include "phlop/timing/nvtx.hpp"
#include "phlop/timing/common_threaded_timer.hpp"

#include <memory>
#include <string>
#include <cstdint>

#include <mpi.h>


#if !defined(PHLOP_SCOPE_TIMER)
#define PHLOP_SCOPE_TIMER(key)                                                                     \
    PHLOP_NVTX_RANGE(key)                                                                          \
    static thread_local auto PHLOP_STR_CAT(ridx_, __LINE__)                                        \
        = std::make_shared<phlop::threaded::RunTimerReport>(key, __FILE__, __LINE__);              \
    static thread_local phlop::threaded::ThreadLifeWatcher PHLOP_STR_CAT(_watcher_, __LINE__){     \
        PHLOP_STR_CAT(ridx_, __LINE__)};                                                           \
    phlop::threaded::ScopeTimer<phlop::mpi::Clock> PHLOP_STR_CAT(_scope_timer_, __LINE__){         \
        *PHLOP_STR_CAT(ridx_, __LINE__)};                                                          \
    phlop::threaded::ScopeTimerMan::local().report_stack_ptr = PHLOP_STR_CAT(ridx_, __LINE__).get();
#endif


// PHLOP_SCOPE("key", time, perf_light), see scope.hpp
#if !defined(PHLOP_SCOPE)
#define PHLOP_SCOPE(key, ...)                                                                      \
    PHLOP_NVTX_RANGE(key)                                                                          \
    static thread_local auto PHLOP_STR_CAT(ridx_, __LINE__)                                        \
        = std::make_shared<phlop::threaded::RunTimerReport>(key, __FILE__, __LINE__);              \
    static thread_local phlop::threaded::ThreadLifeWatcher PHLOP_STR_CAT(_watcher_, __LINE__){     \
        PHLOP_STR_CAT(ridx_, __LINE__)};                                                           \
    phlop::threaded::Scope<PHLOP_SCOPE_FLAGS(__VA_ARGS__), phlop::mpi::Clock> PHLOP_STR_CAT(       \
        _Scope_, __LINE__){*PHLOP_STR_CAT(ridx_, __LINE__)};                                       \
    phlop::threaded::ScopeTimerMan::local().report_stack_ptr = PHLOP_STR_CAT(ridx_, __LINE__).get();
#endif



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

// Collective over comm: a barrier, then this rank's clock, kept as a reference point
//  written into the timer file ("#clock sync=..."). MPI_Wtime may have a per process
//  origin and node clocks drift, so starts compare across ranks only once aligned:
//  print_variance_across subtracts each rank's first sync, and with two or more
//  (call once after MPI_Init and once before shutdown) scales out the drift too.
//  Barrier exit skew is ~log2(ranks) latencies. Raw timestamps are unchanged.
//  Also records MPI_WTIME_IS_GLOBAL. Call before MPI_Finalize, on every rank of comm
inline void sync_clock(MPI_Comm const comm = MPI_COMM_WORLD)
{
    MPI_Barrier(comm);
    auto const t = Clock::now();

    int global = -1;
    int* flag  = nullptr;
    int found  = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &flag, &found);
    if (found && flag)
        global = *flag ? 1 : 0;

    threaded::ScopeTimerMan::INSTANCE().clock_sync(t, global);
}


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

inline void reset_scope_timer()
{
    threaded::ScopeTimerMan::INSTANCE().reset();
}

} // namespace phlop


#endif /*_PHLOP_TIMING_MPI_SCOPE_TIMER_HPP_*/
