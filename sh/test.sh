#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CWD"/..

set -ex

py(){
    PYTHONPATH=$PWD python3 $@
}

py -Om phlop.run.test_cases -d tests -p
py -Om phlop.run.test_cases -d tests -c 10
py -O tests/all_concurrent.py

mkn test -p scope_timer
py -O tests/timing/test_scope_timer.py test_scope_timer -f scope_timer.txt

py -Om phlop.run.valgrind echo yes
py -Om phlop.run.valgrind --tool=massif echo yes

py -Om phlop.run.perf echo yes

