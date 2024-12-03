#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

[ ! -d "tpp/pfm" ] && (
    git clone git://perfmon2.git.sourceforge.net/gitroot/perfmon2/libpfm4 tpp/pfm
    cd tpp/pfm
    make
)
