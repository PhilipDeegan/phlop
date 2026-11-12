
// The one source for every scope timer: phlop::, phlop::para::, phlop::mpi:: and
//  phlop::mpi::para::, whichever headers a program uses. Out of line so that every
//  shared library uses the same instances. No MPI needed: the mpi timers only differ
//  by their clock, which their headers use, never these definitions

#include "phlop/timing/serial_timer.hpp"
#include "phlop/timing/para_timer.hpp"

namespace phlop::impl::serial
{

template<>
ScopeTimerMan<SteadyClock>& ScopeTimerMan<SteadyClock>::INSTANCE()
{
    static ScopeTimerMan i{".phlop_scope_times.txt"};
    return i;
}

template<>
ScopeTimerNode<SteadyClock>*& ScopeTimerMan<SteadyClock>::current_scope()
{
    static ScopeTimerNode<SteadyClock>* current = nullptr;
    return current;
}

template<>
ScopeTimerMan<mpi::Clock>& ScopeTimerMan<mpi::Clock>::INSTANCE()
{
    static ScopeTimerMan i{".phlop_mpi_scope_times.txt"};
    return i;
}

template<>
ScopeTimerNode<mpi::Clock>*& ScopeTimerMan<mpi::Clock>::current_scope()
{
    static ScopeTimerNode<mpi::Clock>* current = nullptr;
    return current;
}

} // namespace phlop::impl::serial


namespace phlop::impl::para
{

template<>
ScopeTimerMan<SteadyClock>& ScopeTimerMan<SteadyClock>::INSTANCE()
{
    static ScopeTimerMan i{".phlop_para_scope_times.txt"};
    return i;
}

template<>
ScopeTimerMan<SteadyClock>::per_thread& ScopeTimerMan<SteadyClock>::local()
{
    static thread_local per_thread* ptr;
    if (!ptr)
    {
        static thread_local per_thread inst;
        ptr = &inst;
    }
    return *ptr;
}

template<>
ScopeTimerNode<SteadyClock>*& ScopeTimerMan<SteadyClock>::current_scope()
{
    static thread_local ScopeTimerNode<SteadyClock>* current = nullptr;
    return current;
}

template<>
ScopeTimerMan<mpi::Clock>& ScopeTimerMan<mpi::Clock>::INSTANCE()
{
    static ScopeTimerMan i{".phlop_mpi_para_scope_times.txt"};
    return i;
}

template<>
ScopeTimerMan<mpi::Clock>::per_thread& ScopeTimerMan<mpi::Clock>::local()
{
    static thread_local per_thread* ptr;
    if (!ptr)
    {
        static thread_local per_thread inst;
        ptr = &inst;
    }
    return *ptr;
}

template<>
ScopeTimerNode<mpi::Clock>*& ScopeTimerMan<mpi::Clock>::current_scope()
{
    static thread_local ScopeTimerNode<mpi::Clock>* current = nullptr;
    return current;
}

} // namespace phlop::impl::para


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
