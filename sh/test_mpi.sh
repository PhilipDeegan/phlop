#!/usr/bin/env bash
# mpi_scope_timer.hpp: one timer file per rank, each with the two
# phlop::mpi::sync_clock() points, and variance across ranks aligned on them.
# PHLOP_MPI_RANKS ranks, default 2. Needs mpicxx and mpirun, else skipped

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..
set -e

if ! command -v mpicxx > /dev/null || ! command -v mpirun > /dev/null; then
    echo "mpicxx/mpirun not found, mpi scope timer skipped"
    exit 0
fi
set -x

N=${PHLOP_MPI_RANKS:-2}
export PYTHONPATH=$PWD
# more ranks than cores: OpenMPI 4 and 5 (prrte) refuse by default, MPICH doesn't care
export OMPI_MCA_rmaps_base_oversubscribe=1
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe

mkdir -p bin
mpicxx -std=c++20 -O2 -Wall -Wextra -Werror -pthread -Iinc tests/timing/test_mpi_scope_timer.cpp \
    src/phlop/timing/common_threaded_timer.cpp -o bin/mpi_scope_timer_test
rm -f bin/phlop_scope_timers.*.bin
mpirun -n "$N" bin/mpi_scope_timer_test
[ "$(ls bin/phlop_scope_timers.*.bin | wc -l)" = "$N" ]
for f in bin/phlop_scope_timers.*.bin; do
    python3 tests/timing/test_scope_timer.py test_threaded_scope_timer -f "$f"
    grep -E "^#clock sync=[0-9]+,[0-9]+( global=[01])?$" "$f" # two sync_clock() calls
done
python3 -m phlop.timing.scope_timer variance "bin/phlop_scope_timers.*.bin" | tee bin/mpi_variance.txt
grep -q "starts from the clock sync" bin/mpi_variance.txt
