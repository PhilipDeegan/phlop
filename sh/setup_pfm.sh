#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

[ ! -d "tpp/pfm" ] && (
    git clone https://github.com/wcohen/libpfm4 tpp/pfm --depth 4 --shallow-submodules --recursive
    cd tpp/pfm
    # libpfm builds with -Werror, new toolchains trip it (glibc 2.43 C23
    # const strpbrk in pfmlib_common.c): keep the warnings, not the error
    make DBG="-g -Wall -Wextra -Wno-unused-parameter"
)
