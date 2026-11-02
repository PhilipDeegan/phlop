# various plotting tools for PHARE development, NOT physics!
#
import argparse
import logging
import sys

import numpy as np

from phlop.timing import scope_timer

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
    np.testing.assert_equal(scope_timer_file(scope_timer_file.roots[0].k), "fn1")


def test_scope(scope_timer_filepath=None):
    """bin/scope.txt from tests/timing/test_scope.cpp, counts if the
    PHLOP_PERF_* env was set when it ran"""
    if scope_timer_filepath is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--file", default="bin/scope.txt", help="timer file")
        scope_timer_filepath = parser.parse_args().file

    stf = scope_timer.file_parser(scope_timer_filepath)
    main = stf.roots[0]
    np.testing.assert_equal(stf(main.k), "main")
    loops = [c for c in main.c if stf(c.k) == "fma_loop"]
    np.testing.assert_equal(len(loops), 3)
    np.testing.assert_equal([len(loop.c) for loop in loops], [2000] * 3)
    if not stf.perf:
        print("test_scope: no perf counts in file, time only")
        return
    for loop in loops:
        counts = stf.perf_counts(loop)
        assert counts and all(v >= 0 for v in counts.values()), counts
        for p in loop.c:
            pass_counts = stf.perf_counts(p)
            assert pass_counts["instructions"] > 4096 / 16, pass_counts  # widest SIMD
            assert pass_counts["cycles"] <= counts["cycles"]
    print(f"test_scope: counts ok, levels {[lvl.label for lvl in stf.perf.values()]}")


def test_threaded_scope(scope_timer_filepath=None):
    """bin/threaded_scope.txt from tests/timing/test_threaded_scope.cpp: thread
    t runs t + 1 units of work, each thread counts only itself"""
    if scope_timer_filepath is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--file", default="bin/threaded_scope.txt")
        scope_timer_filepath = parser.parse_args().file

    stf = scope_timer.file_parser(scope_timer_filepath)
    works = [r for r in stf.roots if stf(r.k) == "work"]
    mains = [r for r in stf.roots if stf(r.k) == "main"]
    np.testing.assert_equal((len(works), len(mains)), (4, 1))
    if not stf.perf:
        print("test_threaded_scope: no perf counts in file, time only")
        return
    instructions = sorted(stf.perf_counts(w)["instructions"] for w in works)
    unit = instructions[0]
    for k, n in enumerate(instructions, start=1):  # linear in work, per thread
        assert abs(n / unit - k) < 0.05, instructions
    main_counts = stf.perf_counts(mains[0])
    assert main_counts["instructions"] < unit, (main_counts, unit)  # not children's
    print(f"test_threaded_scope: counts ok, per thread instructions {instructions}")


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
