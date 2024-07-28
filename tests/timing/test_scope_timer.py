# various plotting tools for PHARE development, NOT physics!
#
import argparse
import logging
import sys

import numpy as np

from phlop.timing import scope_timer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_scope_timer(scope_timer_filepath=None):
    if scope_timer_filepath is None:  # assume cli
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--file", default=None, help="timer file")
        scope_timer_filepath = parser.parse_args().file
        if not scope_timer_filepath:
            parser.print_help()
            sys.exit(1)

    scope_timer_file = scope_timer.file_parser(scope_timer_filepath)
    # np.testing.assert_equal(scope_timer_file(scope_timer_file.roots[0].k), "fn1")
    scope_timer.write_scope_timings(scope_timer_file, outfile="times.txt")

    scope_timer.write_root_as_csv(
        scope_timer_file,
        "times.csv",
        ["fn", "dim", "layout", "alloc", "storage", "impl", "time", "norm_ppc"],
        "update,",
    )


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
