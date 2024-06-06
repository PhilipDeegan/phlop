# various plotting tools for PHARE development, NOT physics!
#

import argparse
import logging
import sys

import numpy as np
import phlop.timing.scope_timer as scope_timer

# from pathlib import Path


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_scope_timer(scope_timer_file=None):
    if scope_timer_file is None:  # assume cli
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--file", default=None, help="timer file")
        scope_timer_file = parser.parse_args().file
        if not scope_timer_file:
            parser.print_help()
            sys.exit(1)

    results = scope_timer.file_parser(scope_timer_file)
    np.testing.assert_equal(results(results.roots[0].k), "fn1")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fn = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        globals()[fn]()
    else:
        print("available functions:")
        fns = [k for k in list(globals().keys()) if k.startswith("test_")]
        for k in fns:
            print(" ", k)
