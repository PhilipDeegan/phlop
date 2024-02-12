#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CWD"/..

set -ex

PYTHONPATH=$PWD python3 -Om phlop.run.test_cases -d tests -p
PYTHONPATH=$PWD python3 -Om phlop.run.test_cases -d tests -c 10
PYTHONPATH=$PWD python3 -O tests/all_concurrent.py
