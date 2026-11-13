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


_FN_PEAK_SCOPES = {"fn_peak", "fn_peak_big", "fn_peak_small"}


def _check_fn_mem(stf, name, other_mem=()):
    """fn3 is a time only root scope with one child, fn_mem, a mem scope that
    touches 64MiB. Memory is enabled in code, PHLOP_SCOPE_MEM=0 turns it off.
    other_mem names any other mem scopes in the file. Returns True if memory
    was recorded"""
    fn3 = [r for r in stf.roots if stf(r.k) == "fn3"]
    np.testing.assert_equal(len(fn3), 1)
    np.testing.assert_equal(len(fn3[0].c), 1)
    fn_mem = fn3[0].c[0]
    np.testing.assert_equal(stf(fn_mem.k), "fn_mem")

    def walk(nodes):
        for n in nodes:
            yield n
            yield from walk(n.c)

    # fn_mem and other_mem are the only mem scopes, none other at any depth
    others = [
        stf(n.k)
        for n in walk(stf.roots)
        if n.has_mem and n is not fn_mem and stf(n.k) not in other_mem
    ]
    if others:  # not assert: sh/test.sh runs test_scope_timer with -O
        raise AssertionError(f"{name}: memory recorded on time only scopes {others}")

    if os.environ.get("PHLOP_SCOPE_MEM") == "0":  # forced off, overrides memory()
        np.testing.assert_equal(fn_mem.has_mem, False)
        print(f"{name}: PHLOP_SCOPE_MEM=0, no memory recorded")
        return False

    np.testing.assert_equal(fn_mem.has_mem, True)
    if not sys.platform.startswith("netbsd"):  # current rss unsupported there, 0
        np.testing.assert_array_less(0, fn_mem.r)
    np.testing.assert_array_less(
        32 * 1024 * 1024, fn_mem.dp, err_msg=f"r={fn_mem.r} dr={fn_mem.dr} p={fn_mem.p}"
    )
    np.testing.assert_array_less(fn_mem.dp, fn_mem.p)  # p is absolute, above baseline
    print(f"{name}: memory ok, peak +{scope_timer.format_bytes(fn_mem.dp)}")
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

    _check_io(scope_timer_file)

    # fn_peak and its children are the only mem scopes besides fn_mem
    if _check_fn_mem(scope_timer_file, "test_scope_timer", _FN_PEAK_SCOPES):
        _check_fn_peak(scope_timer_file)


def test_threaded_scope_timer(scope_timer_filepath=None):
    """bin/threaded_scope_timer.txt from tests/timing/test_threaded_scope_timer.cpp,
    its mem scope runs on a worker thread"""
    if scope_timer_filepath is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--file", default="bin/threaded_scope_timer.txt")
        scope_timer_filepath = parser.parse_args().file

    _check_fn_mem(
        scope_timer.file_parser(scope_timer_filepath), "test_threaded_scope_timer"
    )


def _check_fn_peak(stf):
    """fn_peak: fn_peak_big touches 256MiB then fn_peak_small 32MiB, below that high.
    With PHLOP_SCOPE_MEM=peak on linux the peak resets per scope, so fn_peak_small
    sees its own 32MiB and fn_peak still sees fn_peak_big's 256MiB carried up"""
    MiB = 1024 * 1024
    roots = [r for r in stf.roots if stf(r.k) == "fn_peak"]
    np.testing.assert_equal(len(roots), 1)
    fn_peak = roots[0]
    big, small = fn_peak.c
    np.testing.assert_equal(
        [stf(big.k), stf(small.k)], ["fn_peak_big", "fn_peak_small"]
    )
    np.testing.assert_equal([n.has_mem for n in (fn_peak, big, small)], [True] * 3)

    msg = "\n".join(
        f"{stf(n.k)}: r={n.r} dr={n.dr} p={n.p} dp={n.dp}"
        for n in (fn_peak, big, small)
    )
    np.testing.assert_array_less(128 * MiB, big.dp, err_msg=msg)
    np.testing.assert_array_less(128 * MiB, fn_peak.dp, err_msg=msg)

    peak_mode = os.environ.get("PHLOP_SCOPE_MEM") == "peak"
    reset = small.p - small.dp == small.r  # peak on entry is rss on entry
    if peak_mode and sys.platform.startswith("linux") and not reset:
        print("test_scope_timer: peak reset denied or unsupported, checks skipped")
    elif peak_mode and sys.platform.startswith("linux"):
        np.testing.assert_array_less(16 * MiB, small.dp, err_msg=msg)
        np.testing.assert_array_less(small.dp, 64 * MiB, err_msg=msg)
        print("test_scope_timer: peak reset per scope ok")
    elif sys.platform.startswith("linux"):  # no reset, below the earlier high
        np.testing.assert_array_less(small.dp, 16 * MiB, err_msg=msg)


def test_shared_libs(scope_timer_filepath=None):
    """from sh/test_shared_libs.sh: a module scope called inside a core library
    scope nests under it, so both libraries share phlop's per thread state.
    If the core library has counters, the module's scopes have them too"""
    if scope_timer_filepath is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--file", default="bin/shared_libs/times.txt")
        scope_timer_filepath = parser.parse_args().file

    stf = scope_timer.file_parser(scope_timer_filepath)
    names = sorted(stf(r.k) for r in stf.roots)
    np.testing.assert_equal(names, ["core_outer", "module_scope"])
    outer = next(r for r in stf.roots if stf(r.k) == "core_outer")
    np.testing.assert_equal([stf(c.k) for c in outer.c], ["module_scope"])

    module_scopes = [r for r in stf.roots if stf(r.k) == "module_scope"] + outer.c
    assert all(n.has_mem for n in module_scopes), "PHLOP_SCOPE_MEM=1 set"
    if 1 in stf.perf:  # the core library opened counters, the module must use them
        assert all(n.level == 1 and n.counts for n in module_scopes), module_scopes
    print(
        f"test_shared_libs: ok, perf counts {'yes' if 1 in stf.perf else 'unavailable'}"
    )


def test_mixed():
    """from tests/timing/mixed: the serial and para timers in one program, each
    only its own scopes, para's thread scope a root of its own"""
    serial = scope_timer.file_parser("bin/mixed_serial.txt")
    para = scope_timer.file_parser("bin/mixed_para.txt")
    np.testing.assert_equal([serial(r.k) for r in serial.roots], ["serial_outer"])
    np.testing.assert_equal(serial.roots[0].c, [])
    np.testing.assert_equal(
        sorted(para(r.k) for r in para.roots), ["para_outer", "para_thread"]
    )
    print("test_mixed: ok")


def _check_io(stf):
    """fn_io: 4 writes of 1MiB and one read of 1MiB, exactly, the scope's own
    samples and counter reads excluded. Linux only, PHLOP_SCOPE_IO=0 skips"""
    if (
        not sys.platform.startswith("linux")
        or os.environ.get("PHLOP_SCOPE_IO") == "0"
        or not os.access("/proc/thread-self/io", os.R_OK)
    ):
        print("test_scope_timer: io checks skipped")
        return
    MiB = 1024 * 1024
    (fn_io,) = [r for r in stf.roots if stf(r.k) == "fn_io"]
    if not fn_io.has_io:  # not assert: sh/test.sh runs test_scope_timer with -O
        raise AssertionError("fn_io has no io: token")
    io = fn_io.io
    expected = {"wchar": 4 * MiB, "syscw": 4, "rchar": MiB, "syscr": 1}
    np.testing.assert_equal({k: io[k] for k in expected}, expected, err_msg=str(io))
    if others := [stf(r.k) for r in stf.roots if r.has_io and stf(r.k) != "fn_io"]:
        raise AssertionError(f"io recorded on roots other than fn_io {others}")
    print("test_scope_timer: io ok", scope_timer.io_str(io))


def _nvtx_pairs(sqlite_path):
    """Counter of (range, innermost enclosing range or None) for the "phlop"
    domain's push/pop ranges in an nsys sqlite export, per thread"""
    import sqlite3
    from collections import Counter, defaultdict

    db = sqlite3.connect(sqlite_path)
    try:
        has_nvtx = db.execute(
            "select 1 from sqlite_master where type = 'table' and name = 'NVTX_EVENTS'"
        ).fetchone()
        row = (
            has_nvtx
            and db.execute(
                "select domainId from NVTX_EVENTS where eventType = 75 and text = 'phlop'"
            ).fetchone()
        )
        assert (
            row
        ), f"{sqlite_path}: no NVTX domain 'phlop', built without -DPHLOP_NVTX?"
        (domain,) = row
        rows = db.execute(
            "select e.start, e.end, e.globalTid, coalesce(e.text, s.value)"
            " from NVTX_EVENTS e left join StringIds s on s.id = e.textId"
            " where e.eventType = 59 and e.domainId = ? order by e.start, e.end desc",
            (domain,),
        ).fetchall()
    finally:
        db.close()
    by_thread = defaultdict(list)
    for start, end, tid, name in rows:
        by_thread[tid].append((start, end, name))
    pairs = Counter()
    for ranges in by_thread.values():
        stack = []  # open ranges, sorted by start: a stack per thread
        for start, end, name in ranges:
            while stack and stack[-1][1] <= start:
                stack.pop()
            pairs[(name, stack[-1][2] if stack else None)] += 1
            stack.append((start, end, name))
    return pairs


def test_nvtx(scope_timer_filepath=None, sqlite_path=None):
    """from sh/test_nvtx.sh: a -DPHLOP_NVTX build run under nsys. Every scope
    is an NVTX range in domain "phlop": the same (scope, parent) pairs, with
    the same counts, as the timer file of the same run"""
    from collections import Counter

    if scope_timer_filepath is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--file", required=True, help="timer file")
        parser.add_argument("-s", "--sqlite", required=True, help="nsys export")
        args = parser.parse_args()
        scope_timer_filepath, sqlite_path = args.file, args.sqlite

    stf = scope_timer.file_parser(scope_timer_filepath)
    expect = Counter()

    def walk(nodes, parent):
        for n in nodes:
            expect[(stf(n.k), parent)] += 1
            walk(n.c, stf(n.k))

    walk(stf.roots, None)
    got = _nvtx_pairs(sqlite_path)
    np.testing.assert_equal(dict(got), dict(expect))
    print(f"test_nvtx: {sum(got.values())} ranges match the timer file")


def test_mem_parse():
    """the m: token, alone and after a perf counts token, and the derived fields"""
    _, node = scope_timer.RunTimerNode.from_scope_timer("0 1 2:3 m:1000,1600,1500,2500")
    np.testing.assert_equal((node.k, node.s, node.t), (1, 2, 3))
    np.testing.assert_equal((node.r, node.dr, node.p, node.dp), (1000, 600, 2500, 1000))
    np.testing.assert_equal((node.level, node.counts), (0, None))

    _, node = scope_timer.RunTimerNode.from_scope_timer(
        "1 4 5:6 1:7,8 m:900,400,2000,2000"
    )
    np.testing.assert_equal((node.level, node.counts), (1, [7, 8]))
    np.testing.assert_equal((node.r, node.dr, node.p, node.dp), (900, -500, 2000, 0))

    _, node = scope_timer.RunTimerNode.from_scope_timer("0 1 2:3")
    np.testing.assert_equal(node.has_mem, False)

    # dedupe: rss deltas sum, peaks max, leaves without memory don't erase it
    a = scope_timer.RunTimerNode(0, 0, 1)
    b = scope_timer.RunTimerNode(0, 0, 1, r=10, dr=5, p=50, dp=20)
    c = scope_timer.RunTimerNode(0, 0, 1, r=20, dr=-2, p=60, dp=10)
    for order in ([a, b, c], [b, a, c], [c, b, a]):
        base = scope_timer.RunTimerNode(0, 0, 1)
        for n in order:
            scope_timer._merge_mem(base, n)
        np.testing.assert_equal((base.dr, base.p, base.dp), (3, 60, 20))
        np.testing.assert_equal(base.mem_n, 2)  # b and c, a has no memory
        np.testing.assert_equal(
            scope_timer.mem_str(base), "  x2 rss sum +3B peak max +20B"
        )
    np.testing.assert_equal(scope_timer.mem_str(b), "  rss +5B peak +20B")

    np.testing.assert_equal(
        scope_timer.format_bytes(-10 * 1024 * 1024, sign=True), "-10.00MiB"
    )
    np.testing.assert_equal(scope_timer.format_bytes(512, sign=True), "+512B")
    print("test_mem_parse: ok")


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
    if sys.platform.startswith("linux") and os.environ.get("PHLOP_SCOPE_IO") != "0":
        # thread t writes t + 1 times 64KiB, the main thread nothing itself
        # and no reads: phlop's own (io samples, counter reads, a thread's first
        # counter open) are subtracted
        writes = sorted(
            (w.io["syscw"], w.io["wchar"], w.io["syscr"], w.io["rchar"]) for w in works
        )
        np.testing.assert_equal(writes, [(k, k * 64 * 1024, 0, 0) for k in range(1, 5)])
        np.testing.assert_equal((mains[0].io["syscw"], mains[0].io["wchar"]), (0, 0))
        print("test_threaded_scope: per thread io ok")
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
