# tests/timing/test_scope_compare.py
#
# scope timer file regression checks, synthetic files: needs no counters

import unittest

from phlop.timing.scope_compare import (
    IncompatibleFiles,
    compare,
    main,
    parse_thresholds,
    report,
)
from phlop.timing.scope_timer import lines_parser

HEADER = """#perf 1 amd64_fam1ah_zen5/light cycles,instructions,fetches
2 leaf
1 solve
0 main

"""


def stf(solve, leafs, header=HEADER):
    """main > solve > leafs, solve and each leaf [cycles, instructions, fetches]"""
    body = "0 0 0:9000000\n1 1 0:5000000 1:" + ",".join(map(str, solve)) + "\n"
    for leaf in leafs:
        body += "2 2 0:1000000 1:" + ",".join(map(str, leaf)) + "\n"
    return lines_parser((header + body).splitlines(keepends=True))


BASE = stf([3_000_000, 9_000_000, 900], [[1_000_000, 2_000_000, 100]] * 2)


class ScopeCompareTest(unittest.TestCase):
    def test_same_file_is_clean(self):
        c = compare(BASE, BASE)
        self.assertFalse(c.regressions)
        self.assertFalse(c.improvements)
        paths = {ch.path for ch in c.changes}
        # own instructions: solve 9M - 2 * 2M = 5M, leafs merged 4M
        self.assertEqual(paths, {("main", "solve"), ("main", "solve", "leaf")})

    def test_leaf_regression_stays_in_leaf(self):
        # leafs +5% instructions, solve's inclusive count grows by the same
        new = stf([3_000_000, 9_200_000, 900], [[1_000_000, 2_100_000, 100]] * 2)
        c = compare(BASE, new)
        self.assertEqual([r.path for r in c.regressions], [("main", "solve", "leaf")])
        self.assertAlmostEqual(c.regressions[0].pct, 5.0)

        inclusive = compare(BASE, new, own=False)
        self.assertEqual(
            {r.path for r in inclusive.regressions},
            {("main", "solve"), ("main", "solve", "leaf")},
        )

    def test_threshold_and_improvement(self):
        new = stf([3_000_000, 9_000_000, 900], [[1_000_000, 1_900_000, 100]] * 2)
        c = compare(BASE, new, {"instructions": 1})
        self.assertEqual([i.path for i in c.improvements], [("main", "solve", "leaf")])
        loose = compare(BASE, new, {"instructions": 10})
        self.assertFalse(loose.improvements)

    def test_min_count_and_calls(self):
        new = stf([3_000_000, 9_000_000, 900], [[1_000_000, 2_000_000, 100]] * 3)
        c = compare(BASE, new, min_count=4_500_000)
        self.assertEqual({ch.path for ch in c.changes}, {("main", "solve")})
        self.assertEqual(c.calls_changed, [(("main", "solve", "leaf"), 2, 3)])

    def test_added_removed(self):
        new = stf([3_000_000, 9_000_000, 900], [])
        c = compare(BASE, new)
        self.assertEqual(c.only_base, [("main", "solve", "leaf")])
        self.assertFalse(c.only_new)

    def test_incompatible(self):
        other = stf(
            [3_000_000, 9_000_000, 900],
            [],
            HEADER.replace("amd64_fam1ah_zen5", "arm_n2"),
        )
        with self.assertRaises(IncompatibleFiles):
            compare(BASE, other)

    def test_time_gate(self):
        c = compare(BASE, BASE, {"time": 10}, min_count=0)
        self.assertTrue(all(ch.name == "time" for ch in c.changes))
        own = {ch.path: ch.base for ch in c.changes}
        self.assertEqual(own[("main",)], 4_000_000)  # 9ms - solve's 5ms

    def test_report_and_thresholds(self):
        new = stf([3_000_000, 9_200_000, 900], [[1_000_000, 2_100_000, 100]] * 2)
        lines = []
        report(compare(BASE, new), printer=lines.append)
        self.assertTrue(lines[0].startswith("REGRESSION  main/solve/leaf instructions"))
        self.assertIn("1 regressions", lines[-1])
        self.assertEqual(parse_thresholds(["cycles=5"]), {"cycles": 5.0})
        with self.assertRaises(ValueError):
            parse_thresholds(["cycles"])

    def test_main_exit_codes(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            base, bad = Path(d) / "base.txt", Path(d) / "bad.txt"
            base.write_text(
                HEADER + "0 0 0:9000000\n1 1 0:5000000 1:3000000,9000000,900\n"
            )
            bad.write_text(HEADER.replace("light", "ipc") + "0 0 0:9000000\n")
            self.assertEqual(main([str(base), str(base)]), 0)
            self.assertEqual(main([str(base), str(bad)]), 2)
            self.assertEqual(main([str(base), str(base), "-t", "x"]), 2)


def text(solve, leafs, solve_ns=5_000_000):
    body = f"0 0 0:9000000\n1 1 0:{solve_ns} 1:" + ",".join(map(str, solve)) + "\n"
    for leaf in leafs:
        body += "2 2 0:1000000 1:" + ",".join(map(str, leaf)) + "\n"
    return HEADER + body


def stf_ns(solve_ns):
    return lines_parser(
        text([3_000_000, 9_000_000, 900], [LEAF] * 2, solve_ns).splitlines()
    )


LEAF = [1_000_000, 2_000_000, 100]


class ScopeCompareRunsTest(unittest.TestCase):
    """sides of runs x ranks, see as_runs and load"""

    def test_ranks_sum_counts_max_time(self):
        rank0, slow = stf_ns(5_000_000), stf_ns(7_000_000)
        c = compare(
            [rank0, slow], [rank0, slow], {"instructions": 1, "time": 1}, min_count=0
        )
        by = {(ch.path, ch.name): ch.base for ch in c.changes}
        self.assertEqual(by[(("main", "solve"), "instructions")], 2 * 5_000_000)
        self.assertEqual(by[(("main", "solve"), "time")], 7_000_000 - 2 * 1_000_000)
        self.assertEqual(c.calls_changed, [])
        self.assertEqual(c.ranks, 2)

    def test_rank_counts_must_match(self):
        one = stf_ns(5_000_000)
        with self.assertRaisesRegex(IncompatibleFiles, "rank counts"):
            compare([one, one], [one, one, one, one])  # between sides
        with self.assertRaisesRegex(IncompatibleFiles, "rank counts"):
            compare([[one, one], [one]], [[one, one]])  # between runs of a side

    def test_noise_widens_the_limit(self):
        # base runs' own solve time 3ms and 3.3ms: 10% noise, min 3ms
        fast, slow, new = stf_ns(5_000_000), stf_ns(5_300_000), stf_ns(5_240_000)
        gate = {"time": 5}
        one = compare([[fast]], [[new]], gate, min_count=0)  # +8%, no noise
        self.assertEqual([r.path for r in one.regressions], [("main", "solve")])
        both = compare([[fast], [slow]], [[new]], gate, min_count=0)
        solve = next(ch for ch in both.changes if ch.path == ("main", "solve"))
        self.assertAlmostEqual(solve.base_noise, 10.0)
        self.assertFalse(both.regressions)
        self.assertEqual(both.runs, (2, 1))

    def test_unstable_counts_warn(self):
        a = stf([3_000_000, 9_000_000, 900], [LEAF] * 2)
        b = stf([3_000_000, 9_500_000, 900], [LEAF] * 2)
        c = compare([[a], [b]], [[a]])
        self.assertTrue(any("varies" in w for w in c.warnings))

    def test_scopes_not_in_every_run(self):
        a = stf([3_000_000, 9_000_000, 900], [LEAF] * 2)
        b = stf([3_000_000, 9_000_000, 900], [])
        c = compare([[a], [b]], [[a]])
        self.assertTrue(any("not in every run" in w for w in c.warnings))
        self.assertNotIn(("main", "solve", "leaf"), {ch.path for ch in c.changes})

    def test_load_dir_and_tar(self):
        import tarfile
        import tempfile
        from pathlib import Path

        from phlop.timing.scope_compare import load

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "base"
            for run in ("r0", "r1"):
                timings = root / run / ".phare" / "timings"
                timings.mkdir(parents=True)
                for rank in range(3):
                    (timings / f"rank.{rank}.txt").write_text(
                        text([3_000_000, 9_000_000, 900], [LEAF] * 2)
                    )
                (root / run / ".phare" / "summary.txt").write_text("not a timer file")
            runs = load(root, "rank.*.txt")
            self.assertEqual([len(r) for r in runs], [3, 3])

            tar = Path(d) / "base.tar.gz"
            with tarfile.open(tar, "w:gz") as t:
                t.add(root, arcname="base")
            self.assertEqual([len(r) for r in load(tar, "rank.*.txt")], [3, 3])

            self.assertEqual(main([str(root), str(tar), "-p", "rank.*.txt"]), 0)
            self.assertEqual(
                main([str(root), str(Path(d) / "nope"), "-p", "rank.*.txt"]), 2
            )
            with self.assertRaisesRegex(
                ValueError, "summary.txt: not a scope timer file"
            ):
                load(root)  # default *.txt also finds summary.txt


def mem_text(peaks, kept=0):
    """main > solve x len(peaks), solve's rss 1GiB in, peak 1GiB in, peak out
    as given"""
    body = "0 0 0:9000000\n"
    gib = 1 << 30
    for peak in peaks:
        body += f"1 1 0:1000000 m:{gib},{gib + kept},{gib},{peak}\n"
    return lines_parser((HEADER + body).splitlines())


class ScopeCompareMemTest(unittest.TestCase):
    def test_mem_values(self):
        gib = 1 << 30
        base = mem_text([gib + (1 << 20), gib + (2 << 20)], kept=1 << 20)
        c = compare(base, base, {"mem_peak": 1}, all_events=True)
        by = {ch.name: ch.base for ch in c.changes if ch.path == ("main", "solve")}
        self.assertEqual(by["mem_peak"], gib + (2 << 20))  # max of the merged
        self.assertEqual(by["mem_spike"], 2 << 20)
        self.assertEqual(by["mem_kept"], 2 << 20)  # summed
        self.assertNotIn(
            "mem_peak", {ch.name for ch in c.changes if ch.path == ("main",)}
        )

    def test_mem_regression_and_worst_rank(self):
        gib = 1 << 30
        base = mem_text([gib + (1 << 20)])
        big = mem_text([gib + (200 << 20)])  # +19% peak
        gate = {"mem_peak": 2}
        c = compare([base, base], [base, big], gate)
        self.assertEqual(
            [(r.path, r.name) for r in c.regressions], [(("main", "solve"), "mem_peak")]
        )
        self.assertFalse(compare([base, base], [base, base], gate).regressions)
        lines = []
        report(c, printer=lines.append)
        self.assertTrue(any("GiB" in line and "REGRESSION" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
