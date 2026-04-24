#ifndef _PHLOP_TIMING_MPI_SCOPE_TIMER_HPP_
#define _PHLOP_TIMING_MPI_SCOPE_TIMER_HPP_

#include <cstdint>
#include <string>

#include <mpi.h>

#define PHLOP_SCOPE_TIMER(key)                                                                     \
    static thread_local auto PHLOP_STR_CAT(ridx_, __LINE__)                                        \
        = std::make_shared<phlop::threaded::RunTimerReport>(key, __FILE__, __LINE__);              \
    static thread_local phlop::threaded::ThreadLifeWatcher PHLOP_STR_CAT(_watcher_, __LINE__){     \
        PHLOP_STR_CAT(ridx_, __LINE__)};                                                           \
    phlop::threaded::scope_timer<phlop::mpi::Clock> PHLOP_STR_CAT(_scope_timer_, __LINE__){        \
        *PHLOP_STR_CAT(ridx_, __LINE__)};                                                          \
    phlop::threaded::ScopeTimerMan::local().report_stack_ptr = PHLOP_STR_CAT(ridx_, __LINE__).get();

#include "phlop/timing/threaded_scope_timer.hpp"

namespace phlop::mpi
{

struct Clock
{
    static std::uint64_t now() { return static_cast<std::uint64_t>(MPI_Wtime() * 1e9); }
};

inline void init_scope_timer(std::string const& file
                             = std::string{phlop::threaded::ScopeTimerMan::file_name_default})
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    phlop::threaded::ScopeTimerMan::INSTANCE().file_name(file + "." + std::to_string(rank)).init();
}

} // namespace phlop::mpi


#endif /*_PHLOP_TIMING_MPI_SCOPE_TIMER_HPP_*/
