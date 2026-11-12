#ifndef _PHLOP_TIMING_MPI_TIMER_HPP_
#define _PHLOP_TIMING_MPI_TIMER_HPP_

// MPI_Wtime timers, per rank file names, sync_clock():
//  phlop::mpi::        serial, scopes on one thread per rank, see serial_timer.hpp
//  phlop::mpi::para::  multithread capable, see para_timer.hpp
// Separate timers from phlop:: / phlop::para::, own instances and files.
// No macros here, see mpi_scope_timer.hpp / mpi_para_scope_timer.hpp

#include "phlop/timing/serial_timer.hpp"
#include "phlop/timing/para_timer.hpp"

#include <string>
#include <cstdint>

#include <mpi.h>

namespace phlop::mpi
{

struct Clock
{
    static std::uint64_t now() { return static_cast<std::uint64_t>(MPI_Wtime() * 1e9); }
};

static auto const scope_timer_file_namer = [](std::string const base = "phlop_scope_timers.") {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return base + std::to_string(rank) + ".bin";
};

namespace detail
{
    // Collective over MPI_COMM_WORLD: a barrier, then this rank's clock, kept as a
    //  reference point written into the timer file ("#clock sync=..."). MPI_Wtime may
    //  have a per process origin and node clocks drift, so starts compare across ranks
    //  only once aligned: print_variance_across subtracts each rank's first sync, and
    //  with two or more (call once after MPI_Init and once before shutdown) scales out
    //  the drift too. Barrier exit skew is ~log2(ranks) latencies. Raw timestamps are
    //  unchanged. Also records MPI_WTIME_IS_GLOBAL. Call before MPI_Finalize, on every
    //  rank. World only: the files don't say which communicator a sync was over, so
    //  syncs on disjoint sub-communicators would be aligned as if they were one barrier
    template<typename Man>
    void sync_clock(Man& man)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        auto const t = Clock::now();

        int global = -1;
        int* flag  = nullptr;
        int found  = 0;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &flag, &found);
        if (found && flag)
            global = *flag ? 1 : 0;

        man.clock_sync(t, global);
    }
} // namespace detail


// phlop::mpi:: the serial timer with MPI_Wtime
using ScopeTimerMan  = impl::serial::ScopeTimerMan<Clock>;
using RunTimerReport = impl::serial::RunTimerReport<Clock>;
using ScopeTimer     = impl::serial::ScopeTimer<Clock>;
template<std::uint32_t F>
using Scope = impl::serial::Scope<F, Clock>;

inline auto& scope_timer()
{
    return ScopeTimerMan::INSTANCE();
}

inline void shutdown_scope_timer()
{
    ScopeTimerMan::INSTANCE().shutdown();
}

inline void reset_scope_timer()
{
    ScopeTimerMan::INSTANCE().reset();
}

// see detail::sync_clock
inline void sync_clock()
{
    detail::sync_clock(ScopeTimerMan::INSTANCE());
}

} // namespace phlop::mpi


// phlop::mpi::para:: the multithread capable timer with MPI_Wtime
namespace phlop::mpi::para
{

using ScopeTimerMan     = impl::para::ScopeTimerMan<Clock>;
using RunTimerReport    = impl::para::RunTimerReport<Clock>;
using ThreadLifeWatcher = impl::para::ThreadLifeWatcher<Clock>;
using ScopeTimer        = impl::para::ScopeTimer<Clock>;
template<std::uint32_t F>
using Scope = impl::para::Scope<F, Clock>;

inline auto& scope_timer()
{
    return ScopeTimerMan::INSTANCE();
}

inline void shutdown_scope_timer()
{
    ScopeTimerMan::INSTANCE().shutdown();
}

inline void reset_scope_timer()
{
    ScopeTimerMan::INSTANCE().reset();
}

// see mpi::detail::sync_clock
inline void sync_clock()
{
    detail::sync_clock(ScopeTimerMan::INSTANCE());
}

} // namespace phlop::mpi::para

#endif /*_PHLOP_TIMING_MPI_TIMER_HPP_*/
