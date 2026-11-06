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


if __name__ == "__main__":
    unittest.main()
