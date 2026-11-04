# various plotting tools for PHARE development, NOT physics!
#
import argparse
import logging
import os
import sys

import numpy as np

from phlop.timing import scope_timer

logger = logging.getLogger(__name__)


_LEVEL_ENVS = {1: "PHLOP_PERF_LIGHT", 2: "PHLOP_PERF_ALL"}


def _no_counts(stf, name):
    """True for a time only file. Fails if a requested level, PHLOP_PERF_* set,
    has no "#perf" header: its counts were not written"""
    missing = [
        env
        for lvl, env in _LEVEL_ENVS.items()
        if os.environ.get(env) and lvl not in stf.perf
    ]
    assert not missing, f"{name}: {missing} set but no counts written for them"
    if stf.perf:
        return False
    print(f"{name}: no perf counts in file, time only")
    return True


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

    # time only scopes have no memory fields
    assert not any(root.has_mem for root in scope_timer_file.roots)

    # fn3 is time only, its child fn_mem uses the mem flag and touches 64MiB
    fn3 = [r for r in scope_timer_file.roots if scope_timer_file(r.k) == "fn3"]
    np.testing.assert_equal(len(fn3), 1)
    np.testing.assert_equal(len(fn3[0].c), 1)
    fn_mem = fn3[0].c[0]
    np.testing.assert_equal(scope_timer_file(fn_mem.k), "fn_mem")
    np.testing.assert_equal(fn_mem.has_mem, True)
    np.testing.assert_array_less(0, fn_mem.r)
    np.testing.assert_array_less(
        32 * 1024 * 1024, fn_mem.dp, err_msg=f"r={fn_mem.r} dr={fn_mem.dr} p={fn_mem.p}"
    )
    np.testing.assert_array_less(fn_mem.dp - 1, fn_mem.p)


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
    if _no_counts(stf, "test_scope"):
        return
    for loop in loops:  # perf_all, level 2
        counts = stf.perf_counts(loop) if 2 in stf.perf else {}
        if 2 in stf.perf:
            assert counts and all(v >= 0 for v in counts.values()), counts
        if 1 not in stf.perf:
            continue
        for p in loop.c:  # perf_light, level 1
            pass_counts = stf.perf_counts(p)
            assert pass_counts["instructions"] > 4096 / 16, pass_counts  # widest SIMD
            if counts:
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
    if _no_counts(stf, "test_threaded_scope"):
        return
    instructions = []
    if 1 in stf.perf:  # work scopes are perf_light
        instructions = sorted(stf.perf_counts(w)["instructions"] for w in works)
        for k, n in enumerate(instructions, start=1):  # linear in work, per thread
            assert abs(n / instructions[0] - k) < 0.05, instructions
    if 2 in stf.perf:  # main is perf_all, counts the main thread only
        main_counts = stf.perf_counts(mains[0])
        assert main_counts["instructions"] >= 0, main_counts
        if instructions:
            assert main_counts["instructions"] < instructions[0], main_counts
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
