#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

py(){
    PYTHONPATH=$PWD python3 $@
}

py -Om phlop.run.test_cases -d tests -p
py -Om phlop.run.test_cases -d tests -c 10
py -m phlop.run.test_cases -d tests -c 1 --rerun 2 --logging 2
py -O tests/all_concurrent.py

mkn test -p scope_timer
py -O tests/timing/test_scope_timer.py test_scope_timer -f scope_timer.txt

py -Om phlop.run.valgrind echo yes
py -Om phlop.run.valgrind --tool=massif echo yes

py -Om phlop.run.perf -e="--all-user" echo yes || echo "perf failed, assumed CI"

# install via ./sh/setup_pfm.sh
[ -d "tpp/pfm" ] && py -O tests/_phlop/app/pfm/test_pfm.py || echo "pfm missing, skipped"

for i in $(seq 1 4); do # verify non-zero exit codes
    PHLOP_FORCE_TEST_CASE_FAILURE=$i py -Om phlop.run.test_cases -d tests -c 10 -r test_fails && exit 1
done

exit 0
