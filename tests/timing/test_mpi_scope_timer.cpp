#include "phlop/timing/mpi_scope_timer.hpp"

#include "scope_timer_fns.hpp"

#include <iostream>

// serial ranks, the same scopes as test_scope_timer.cpp, one file per rank
int main(int argc, char** argv)
{
    std::cout << __FILE__ << std::endl;

    MPI_Init(&argc, &argv);

    auto const file = phlop::mpi::scope_timer_file_namer();
    phlop::mpi::scope_timer().file_name("bin/" + file).memory().io().init();
    phlop::mpi::sync_clock(); // reference points for cross rank starts, see sh/test_mpi.sh

    run_scope_timer_fns("bin/io." + file); // per rank, not shared

    phlop::mpi::sync_clock(); // collective, so before MPI_Finalize: drift correction
    phlop::mpi::scope_timer().shutdown();
    MPI_Finalize();
}
