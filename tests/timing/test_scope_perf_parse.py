# tests/timing/test_scope_perf_parse.py
#
# PHLOP_SCOPE timer files with perf counts, synthetic: needs no counters

import unittest

from phlop.timing.scope_timer import lines_parser

LINES = """#perf 1 amd64_fam1ah_zen5/light cycles,instructions,fetches
#perf 2 amd64_fam1ah_zen5/ipc cycles,instructions,ops,branches,br_misp
2 inner
1 outer
0 main

0 0 100:9000
1 1 200:5000 2:4000,8000,8000,1000,10
2 2 300:1000 1:1000,3000,6
2 2 1400:1000 1:1000,1000,0
1 2 6000:500
""".splitlines(keepends=True)


class ScopePerfParseTest(unittest.TestCase):
    def setUp(self):
        self.stf = lines_parser(LINES)
        self.main = self.stf.roots[0]
        self.outer = self.main.c[0]

    def test_header_and_tree(self):
        self.assertEqual(self.stf.perf[1].set_name, "light")
        self.assertEqual(self.stf.perf[2].names[-1], "br_misp")
        self.assertEqual([self.stf(c.k) for c in self.main.c], ["outer", "inner"])
        self.assertEqual(len(self.outer.c), 2)
        self.assertIsNone(self.main.counts)  # time only
        self.assertIsNone(self.main.c[1].counts)

    def test_counts(self):
        self.assertEqual(self.stf.perf_counts(self.outer)["instructions"], 8000)
        inner = self.outer.c[0]
        self.assertEqual(
            self.stf.perf_counts(inner),
            {"cycles": 1000, "instructions": 3000, "fetches": 6},
        )

    def test_self_counts_only_subtract_same_level(self):
        # inner scopes are perf_light, outer perf_all: nothing to subtract
        self.assertEqual(self.stf.perf_self_counts(self.outer)["cycles"], 4000)

    def test_metrics_use_event_set_formulas(self):
        self.assertAlmostEqual(self.stf.perf_metrics(self.outer, ctx={})["ipc"], 2.0)
        inner = self.outer.c[0]
        m = self.stf.perf_metrics(inner, ctx={})
        self.assertAlmostEqual(m["instructions_per_fetch"], 500)
        self.assertIsNone(
            self.stf.perf_metrics(self.outer.c[1], ctx={})["instructions_per_fetch"]
        )

    def test_old_files_without_counts_still_parse(self):
        old = [line for line in LINES if not line.startswith("#perf")]
        old = [" ".join(line.split(" ")[:3]).rstrip() + "\n" for line in old]
        stf = lines_parser(old)
        self.assertEqual(stf.perf, {})
        self.assertEqual(stf.roots[0].c[0].t, 5000)


if __name__ == "__main__":
    unittest.main()
