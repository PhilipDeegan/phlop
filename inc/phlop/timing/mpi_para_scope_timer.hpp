#ifndef _PHLOP_TIMING_MPI_PARA_SCOPE_TIMER_HPP_
#define _PHLOP_TIMING_MPI_PARA_SCOPE_TIMER_HPP_

// PHLOP_SCOPE into phlop::mpi::para::, the multithread capable timer with MPI_Wtime.
//  Opt in: no per scope peak reset, PHLOP_SCOPE_MEM=peak is =1
// Picks where PHLOP_SCOPE / PHLOP_SCOPE_TIMER record, one per translation unit.
//  Every timer links src/phlop/timing/scope_timer.cpp

#if defined(_PHLOP_TIMER_EXPORTED_) && _PHLOP_TIMER_EXPORTED_ != 3
#error // PHLOP_SCOPE can only record into one timer per translation unit
#endif
#define _PHLOP_TIMER_EXPORTED_ 3

#include "phlop/timing/mpi_timer.hpp"

#if !defined(PHLOP_SCOPE_TIMER)
#define PHLOP_SCOPE_TIMER(key)                                                                     \
    PHLOP_DETAIL_PARA_SCOPE(phlop::mpi::para, key, phlop::scope_flags(phlop::scope_key::time))
#endif

// PHLOP_SCOPE("key", time, perf_light), see scope.hpp
#if !defined(PHLOP_SCOPE)
#define PHLOP_SCOPE(key, ...) PHLOP_DETAIL_PARA_SCOPE(phlop::mpi::para, key, PHLOP_SCOPE_FLAGS(__VA_ARGS__))
#endif

#endif /*_PHLOP_TIMING_MPI_PARA_SCOPE_TIMER_HPP_*/
