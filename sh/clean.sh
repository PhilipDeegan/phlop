#!/usr/bin/env bash
set -e
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CWD"/..

RM_RF=(
  __pycache__
)

for RM in ${RM_RF[@]}; do
  find . -name "${RM}" | xargs rm -rf
done
