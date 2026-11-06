#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

py(){
    PYTHONPATH=$PWD python3 $@
}

py -Om phlop.run.test_cases -i tests -p
py -Om phlop.run.test_cases -i tests/_phlop/empty_test.py -p
py -Om phlop.run.test_cases -i tests -c 10
py -m phlop.run.test_cases -i tests -c 1 --rerun 2 --logging 2
py -O tests/all_concurrent.py

mkn test -p scope_timer
py -O tests/timing/test_scope_timer.py test_scope_timer -f "bin/scope_timer.txt"
py tests/timing/test_scope_timer.py test_mem_parse
mkn test -p threaded_scope_timer
py tests/timing/test_scope_timer.py test_threaded_scope_timer -f "bin/threaded_scope_timer.txt"
PHLOP_SCOPE_MEM=0 mkn test -p threaded_scope_timer
PHLOP_SCOPE_MEM=0 py tests/timing/test_scope_timer.py test_threaded_scope_timer -f "bin/threaded_scope_timer.txt"
if command -v mpicxx >/dev/null && command -v mpirun >/dev/null; then # not in CI
    mpicxx -std=c++20 -O2 -Iinc tests/timing/test_mpi_scope_timer.cpp \
        src/phlop/timing/common_threaded_timer.cpp -o bin/mpi_scope_timer_test
    rm -f bin/phlop_scope_timers.*.bin
    OMPI_MCA_rmaps_base_oversubscribe=1 mpirun -n 2 bin/mpi_scope_timer_test
    [ "$(ls bin/phlop_scope_timers.*.bin | wc -l)" = 2 ]
    for f in bin/phlop_scope_timers.*.bin; do
        py tests/timing/test_scope_timer.py test_threaded_scope_timer -f "$f"
    done
else
    echo "mpicxx/mpirun not found, mpi scope timer skipped"
fi
./sh/test_shared_libs.sh
mkn test -p scope
py tests/timing/test_scope_timer.py test_scope -f "bin/scope.txt" # no -O: asserts
mkn test -p threaded_scope
py tests/timing/test_scope_timer.py test_threaded_scope -f "bin/threaded_scope.txt"

# exit codes checked, memcheck errors are 99. The shell's builtin echo: ubuntu
#  26.04's /usr/bin/echo (rust coreutils) leaks, memcheck rightly fails it
if command -v valgrind > /dev/null; then
    py -Om phlop.run.valgrind -q sh -c "echo yes"
    py -Om phlop.run.valgrind -q --tool=massif sh -c "echo yes"
    py -Om phlop.run.valgrind --show --md - > /dev/null
fi

# needs a GPU, nsys's exit code is the program's
if command -v nsys > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
    py -Om phlop.run.nsys -q sh -c "exit 0"
    py -Om phlop.run.nsys --show --md - > /dev/null
fi
if command -v compute-sanitizer > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
    py -Om phlop.run.csan -q sh -c "exit 0" && exit 1 # 255: no CUDA calls, no results
fi
# no kernels, so nothing replayed: real profiles via PHLOP_NCU_TEST=1, see test_ncu.py
if command -v ncu > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
    py -Om phlop.run.ncu -q sh -c "exit 0"
fi

py -Om phlop.run.perf -e="--all-user" echo yes || echo "perf failed, assumed CI"

# install via ./sh/setup_pfm.sh
[ -d "tpp/pfm" ] && py -O tests/_phlop/app/pfm/test_pfm.py || echo "pfm missing, skipped"

for i in $(seq 1 4); do # verify non-zero exit codes
    PHLOP_FORCE_TEST_CASE_FAILURE=$i py -Om phlop.run.test_cases -i tests -c 10 -r test_fails && exit 1
done

exit 0
