#!/usr/bin/env bash
# -DPHLOP_NVTX: the scope timer tests built with NVTX ranges, run under nsys
# tracing NVTX only (CPU, no GPU work), and the ranges checked against phlop's
# own timer file: the same scopes, counts and nesting. Needs nsys and the
# nvtx3 headers (CUDA's include directory, or NVTX_INC), else skipped

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..
set -e

NVTX_INC=${NVTX_INC:-${CUDA_HOME:-/usr/local/cuda}/include}
if ! command -v nsys > /dev/null || [ ! -f "$NVTX_INC/nvtx3/nvToolsExt.h" ]; then
    echo "nsys or $NVTX_INC/nvtx3 not found, nvtx test skipped"
    exit 0
fi
set -x

CXX=${CXX:-g++}
OUT=bin/nvtx
mkdir -p $OUT
FLAGS="-std=c++20 -O2 -pthread -Iinc -I$NVTX_INC -DPHLOP_NVTX"

nvtx_run(){ # name, timer file the program writes, sources...
    local name=$1 times=$2
    shift 2
    $CXX $FLAGS "$@" -o $OUT/$name -ldl
    nsys profile --trace=nvtx --sample=none --cpuctxsw=none --export=sqlite \
        --force-overwrite=true -o $OUT/$name $OUT/$name > $OUT/$name.log 2>&1
    PYTHONPATH=$PWD python3 tests/timing/test_scope_timer.py test_nvtx \
        -f "$times" -s $OUT/$name.sqlite
}

nvtx_run scope_timer bin/scope_timer.txt \
    tests/timing/test_scope_timer.cpp src/phlop/timing/scope_timer.cpp
nvtx_run threaded_scope_timer bin/threaded_scope_timer.txt \
    tests/timing/test_threaded_scope_timer.cpp src/phlop/timing/common_threaded_timer.cpp
