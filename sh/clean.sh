#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

RM_RF=(
  __pycache__ .ruff_cache phlop.egg-info dist
)

for RM in ${RM_RF[@]}; do
  find . -name "${RM}" | xargs rm -rf
done
