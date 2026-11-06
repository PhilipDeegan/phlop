# tests/timing/test_scope_perf_parse.py
#
# PHLOP_SCOPE timer files with perf counts, synthetic: needs no counters

import unittest
import unittest.mock

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


SAME_LEVEL = """#perf 1 amd64_fam1ah_zen5/light cycles,instructions,fetches
#perf 2 amd64_fam1ah_zen5/ipc cycles,instructions,ops,branches,br_misp
#ctx slots=5
1 child
0 parent

0 0 0:9000 1:1000,3000,9
1 1 10:100 1:300,900,4
1 1 20:100 1:200,600,1
1 1 30:100 2:100,100,100,10,1
""".splitlines(keepends=True)


class ScopePerfParseTest(unittest.TestCase):
    def setUp(self):
        self.stf = lines_parser(LINES)
        self.main = self.stf.roots[0]
        self.outer = self.main.c[0]

    def test_kernel_label(self):
        stf = lines_parser([LINES[0].replace("/light", "/light+kernel")] + LINES[1:])
        self.assertTrue(stf.perf[1].kernel)
        self.assertFalse(stf.perf[2].kernel)
        self.assertEqual(stf.perf[1].set_name, "light")
        metrics = stf.perf_metrics(stf.roots[0].c[0].c[0])  # inner, perf_light
        self.assertAlmostEqual(metrics["ipc"], 3.0)

        from phlop.timing.scope_timer import print_scope_perf

        lines = []
        with unittest.mock.patch(
            "phlop.timing.scope_timer.file_parser", return_value=stf
        ):
            print_scope_perf("unused", dedupe=False, printer=lines.append)
        self.assertTrue(any("[light +kernel]" in line for line in lines))

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

    def test_self_counts_skip_other_levels(self):
        # inner scopes are perf_light, outer perf_all: nothing to subtract
        self.assertEqual(
            self.stf.perf_self_counts(self.outer), self.stf.perf_counts(self.outer)
        )

    def test_self_counts_subtract_same_level(self):
        same = lines_parser(SAME_LEVEL)
        parent = same.roots[0]
        self.assertEqual(
            same.perf_self_counts(parent),
            {
                "cycles": 1000 - 300 - 200,
                "instructions": 3000 - 900 - 600,
                "fetches": 9 - 4 - 1,
            },
        )
        # a perf_all child is not subtracted from the perf_light parent
        self.assertEqual(len(parent.c), 3)

    def test_metrics_use_event_set_formulas(self):
        self.assertAlmostEqual(self.stf.perf_metrics(self.outer, ctx={})["ipc"], 2.0)
        inner = self.outer.c[0]
        m = self.stf.perf_metrics(inner, ctx={})
        self.assertAlmostEqual(m["instructions_per_fetch"], 500)
        self.assertIsNone(
            self.stf.perf_metrics(self.outer.c[1], ctx={})["instructions_per_fetch"]
        )

    def test_ctx_from_file_is_used(self):
        same = lines_parser(SAME_LEVEL)
        self.assertEqual(same.ctx, {"slots": 5})
        self.assertIsNone(self.stf.ctx)  # no #ctx line

    def test_old_files_without_counts_still_parse(self):
        old = [line for line in LINES if not line.startswith("#perf")]
        old = [" ".join(line.split(" ")[:3]).rstrip() + "\n" for line in old]
        stf = lines_parser(old)
        self.assertEqual(stf.perf, {})
        self.assertEqual(stf.roots[0].c[0].t, 5000)

    def test_none_label_for_zero_fetches(self):
        labels = {}
        inner_no_fetch = self.outer.c[1]
        self.stf.metrics_for(1, self.stf.perf_counts(inner_no_fetch), {}, labels)
        self.assertEqual(labels["instructions_per_fetch"], "none (no RAM fetches)")

    def test_group_scopes_merges_siblings_and_sums_counts(self):
        from phlop.timing.scope_timer import group_scopes

        (main,) = group_scopes(self.stf, self.stf.roots)
        outer, inner = main.c
        self.assertEqual((outer.key, outer.n), ("outer", 1))
        self.assertEqual((inner.key, inner.n, inner.t), ("inner", 1, 500))
        (merged,) = outer.c  # two inner scopes under outer
        self.assertEqual((merged.n, merged.t, merged.n_counted), (2, 2000, 2))
        self.assertEqual(
            merged.counts, {"cycles": 2000, "instructions": 4000, "fetches": 6}
        )

    def test_print_scope_perf_dedupe_and_verbose(self):
        import tempfile

        from phlop.timing.scope_timer import print_scope_perf

        with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
            f.writelines(LINES)
            f.flush()
            lines = []
            print_scope_perf(f.name, verbose=True, printer=lines.append)
        text = "\n".join(lines)
        self.assertIn("inner x2 0.002ms (mean 0.001ms) [light]", text)
        self.assertIn("instructions_per_fetch=667", text)  # 4000 / 6, summed
        self.assertIn("counts: cycles=2,000 instructions=4,000 fetches=6", text)

    def test_no_accuracy_warnings_for_host_sets(self):
        self.assertEqual(self.stf.perf_warnings(), [])

    def test_generic_and_forced_files_warn(self):
        import tempfile

        from phlop.timing.scope_timer import print_scope_perf

        generic = lines_parser(
            ["#perf 1 generic/light cycles,instructions,branches,br_misp\n", "0 main\n"]
            + ["\n", "0 0 0:100 1:1000,2000,100,1\n"]
        )
        (w,) = generic.perf_warnings()
        self.assertIn("generic events", w)
        self.assertIn("potentially inaccurate", w)
        self.assertIn("phlop/app/pfm/sets.py", w)  # how to resolve
        self.assertEqual(generic.perf_metrics(generic.roots[0])["ipc"], 2)

        forced = LINES[:2] + ["#ctx stall_slot_counts_cycles=0,pmu_forced=1\n"]
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
            f.writelines(forced + LINES[2:])
            f.flush()
            lines = []
            print_scope_perf(f.name, printer=lines.append)
        self.assertTrue(lines[0].startswith("WARNING: "), lines[0])
        self.assertIn("LIBPFM_FORCE_PMU", lines[0])
        self.assertIn("potentially wrong", lines[0])

    def test_malformed_ctx_values_are_skipped(self):
        stf = lines_parser(["#ctx slots=5,sve_bits=unknown,=3,bare\n"] + LINES[2:])
        self.assertEqual(stf.ctx, {"slots": 5})
        self.assertEqual(stf(stf.roots[0].k), "main")

    def test_cli_verbose_flag(self):
        from phlop.timing.scope_timer import cli_args_parser

        args = cli_args_parser().parse_args(["perf", "f.txt", "-v", "--no-dedupe"])
        self.assertEqual((args.tool, args.verbose, args.dedupe), ("perf", True, False))


if __name__ == "__main__":
    unittest.main()
