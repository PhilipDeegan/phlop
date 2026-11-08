#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


(
  cd "$CWD"/..

  set -e

  PY_FILES=$(find phlop tests sh setup.py -name "*.py")

  # locally the formatters fix files; in CI (GitHub sets CI=true) they would
  #  fix the runner's copy and pass, so check instead and fail on changes
  FIX=(-i) BLACK=()
  [ -n "$CI" ] && FIX=(--check-diff) BLACK=(--check --diff)

  python3 -m black "${BLACK[@]}" phlop tests
  pylint --errors-only phlop tests
  # isort phlop tests
  python3 -m ruff check phlop tests
  for FILE in ${PY_FILES[@]}; do

    autoflake "${FIX[@]}" "$FILE"

  done

) 1> >(tee $CWD/.lint.sh.out ) 2> >(tee $CWD/.lint.sh.err >&2 )

