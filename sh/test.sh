#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CWD"/..

set -ex

PYTHONPATH=$PWD python3 -m phlop.run.test_cases -d tests -p
PYTHONPATH=$PWD python3 -m phlop.run.test_cases -d tests -c 10
