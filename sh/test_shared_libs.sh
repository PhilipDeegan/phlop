#!/usr/bin/env bash
# phlop state shared across shared libraries, linux only: a core library with
# phlop's .cpp and a -fvisibility=hidden module, both loaded RTLD_LOCAL.
# Built with ASan, run with and without shutdown() for exit time ordering.
# Without PHLOP_PERF_LIGHT a fake one is used, counts are checked if the
# kernel allows perf_event_open

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..
set -ex

[ "$(uname)" = "Linux" ] || { echo "linux only, skipped"; exit 0; }

CXX=${CXX:-g++}
SRC=tests/timing/shared_libs
OUT=bin/shared_libs
mkdir -p $OUT
FLAGS="-std=c++20 -O1 -g -fsanitize=address -fno-omit-frame-pointer -pthread -Iinc"

$CXX $FLAGS -fPIC -shared $SRC/core.cpp src/phlop/timing/scope_timer.cpp -o $OUT/libcore.so
$CXX $FLAGS -fPIC -shared -fvisibility=hidden -fvisibility-inlines-hidden $SRC/module.cpp \
    -o $OUT/module.so -L$OUT -lcore -Wl,-rpath,"\$ORIGIN"
$CXX $FLAGS $SRC/main.cpp -o $OUT/main -ldl

export PHLOP_PERF_LIGHT=${PHLOP_PERF_LIGHT:-"fake/light;cycles=0x76,instructions=0xc0"}
export PHLOP_SCOPE_MEM=1

PHLOP_SHARED_LIBS_OUT=$OUT/times.txt $OUT/main $OUT/libcore.so $OUT/module.so
PYTHONPATH=$PWD python3 tests/timing/test_scope_timer.py test_shared_libs -f $OUT/times.txt

PHLOP_SHARED_LIBS_NO_SHUTDOWN=1 PHLOP_SHARED_LIBS_OUT=$OUT/times_no_shutdown.txt \
    $OUT/main $OUT/libcore.so $OUT/module.so
PYTHONPATH=$PWD python3 tests/timing/test_scope_timer.py test_shared_libs -f $OUT/times_no_shutdown.txt
