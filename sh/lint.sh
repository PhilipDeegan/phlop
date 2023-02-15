#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CWD"/..

set -e

PY_FILES=$(find . -name "*.py")

for FILE in ${PY_FILES[@]}; do
  python3 -m black "$FILE"
  autoflake -i "$FILE"
  isort "$FILE"
done
