#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

[ ! -d "tpp/pfm" ] && (
    git clone https://github.com/wcohen/libpfm4 tpp/pfm --depth 4 --shallow-submodules --recursive
    cd tpp/pfm
    make
)
