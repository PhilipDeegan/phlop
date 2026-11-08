# tests/timing/test_scope_colors.py
#
# severity colors and the printers using them, synthetic files: no counters

import tempfile
import unittest
from pathlib import Path

from phlop.timing import severity
from phlop.timing.scope_timer import (
    align_clocks,
    dedupe_leafs,
    lines_parser,
    print_scope_perf,
    print_variance_across,
)

PERF = "#perf 1 amd64_fam1ah_zen5/light cycles,instructions,fetches\n"


def scope_file(root_t, leaf_t, start=0, counts=None, perf=PERF):
    """root > leaf, leaf counts [cycles, instructions, fetches] if given"""
    leaf = f"1 1 {start + 10}:{leaf_t}"
    if counts:
        leaf += " 1:" + ",".join(map(str, counts))
    return (
        perf if counts else ""
    ) + f"0 root\n1 leaf\n\n0 0 {start}:{root_t}\n{leaf}\n"


class SeverityTest(unittest.TestCase):
    def test_level(self):
        self.assertEqual(severity.level(None, 1, 2), severity.OK)
        self.assertEqual(severity.level(0.5, 1, 2), severity.OK)
        self.assertEqual(severity.level(1, 1, 2), severity.WARN)
        self.assertEqual(severity.level(3, 1, 2), severity.BAD)

    def test_colors_reset_and_marks(self):
        on = severity.Colors(enabled=True)
        self.assertTrue(on("x", severity.BAD).endswith("\033[0m"))
        self.assertEqual(on("x", severity.OK), "x")
        off = severity.Colors(enabled=False)
        self.assertEqual(off("x", severity.WARN), "x (!)")
        self.assertEqual(off("x", severity.BAD), "x (!!)")
        self.assertIn("(!!) bad", off.legend("warn", "bad"))


class VarianceTest(unittest.TestCase):
    def variance(self, *files, **kwargs):
        with tempfile.TemporaryDirectory() as d:
            for i, text in enumerate(files):
                (Path(d) / f"t.{i}.txt").write_text(text)
            lines = []
            print_variance_across(f"{d}/t.*.txt", printer=lines.append, **kwargs)
        return lines

    def line(self, lines, key):
        return next(line for line in lines if line.strip().startswith(key + " "))

    def test_relative_not_absolute(self):
        # root 10s +-0.5%: fine. leaf 1ms ranging 0.1-1.9ms: bad
        lines = self.variance(
            scope_file(10_000_000_000, 1_000_000),
            scope_file(10_050_000_000, 100_000),
            scope_file(9_950_000_000, 1_900_000),
        )
        self.assertIn("marks:", lines[0])
        self.assertNotIn("(!", self.line(lines, "root"))
        self.assertIn("spread 180.0% (!!)", self.line(lines, "leaf"))

    def test_no_variance_is_not_bad(self):
        lines = self.variance(scope_file(10**9, 10**6), scope_file(10**9, 10**6))
        self.assertFalse(any("(!" in line for line in lines[1:]))

    def test_one_file(self):
        lines = self.variance(scope_file(10**9, 10**6))
        self.assertEqual(len(lines), 1)
        self.assertIn("need 2 or more", lines[0])

    def test_starts_relative_to_each_file(self):
        # different boot times / launches, same shape: no start spread
        lines = self.variance(
            scope_file(10**9, 10**6, start=5),
            scope_file(10**9, 10**6, start=10**12),
        )
        # absolute starts would spread ~1e12ns
        self.assertIn("start +0.000ms spread 0.000ms", self.line(lines, "leaf"))

    def test_clock_sync_aligns_starts(self):
        # rank 1's clock is 1e12 ahead and starts 2ms later after the barrier:
        #  per file origins hide it, the sync shows it
        a = "#clock sync=100,1000000100 global=0\n" + scope_file(10**9, 10**6, 200)
        b = "#clock sync=1000000000100,1001000000100 global=0\n"
        b += scope_file(10**9, 10**6, 10**12 + 2_000_200)
        lines = self.variance(a, b)
        self.assertIn("starts from the clock sync", lines[0])
        self.assertIn("spread 2.000ms", self.line(lines, "leaf"))

    def test_clock_sync_only_if_every_file_has_one(self):
        a = "#clock sync=100\n" + scope_file(10**9, 10**6, 200)
        lines = self.variance(a, scope_file(10**9, 10**6, 10**12))
        self.assertIn("starts from each file's first scope", lines[0])
        self.assertIn("start +0.000ms spread 0.000ms", self.line(lines, "leaf"))

    def test_clock_sync_counts_must_match(self):
        # 2 syncs vs 1: not the same points, starts stay per file
        a = "#clock sync=100,1000000100\n" + scope_file(10**9, 10**6, 200)
        b = "#clock sync=10000\n" + scope_file(10**9, 10**6, 10**12)
        lines = self.variance(a, b)
        self.assertIn("starts from each file's first scope", lines[0])
        self.assertIn("start +0.000ms spread 0.000ms", self.line(lines, "leaf"))

    def test_clock_drift_two_point(self):
        # rank 1's clock runs 10ppm fast: its span between syncs is longer
        a = lines_parser(
            ("#clock sync=0,1000000000\n" + scope_file(10, 1, 500_000_000)).split("\n")
        )
        b = lines_parser(
            ("#clock sync=7,1000020007\n" + scope_file(10, 1, 500_010_007)).split("\n")
        )
        c = lines_parser(
            ("#clock sync=0,1000010000\n" + scope_file(10, 1, 500_005_000)).split("\n")
        )
        self.assertEqual(b.clock_syncs, [7, 1000020007])
        self.assertFalse(b.clock_global)
        self.assertTrue(align_clocks([a, b, c]))
        # median span 1000010000: every root lands on the same instant
        starts = {f.roots[0].s for f in (a, b, c)}
        self.assertEqual(starts, {500_005_000})

    def test_clock_global_common_origin(self):
        # MPI_WTIME_IS_GLOBAL: one origin, the 2ms between syncs is barrier skew
        a = lines_parser(
            ("#clock sync=1000 global=1\n" + scope_file(10, 1, 5000)).split("\n")
        )
        b = lines_parser(
            ("#clock sync=3000 global=1\n" + scope_file(10, 1, 5000)).split("\n")
        )
        self.assertTrue(align_clocks([a, b]))
        self.assertEqual([a.roots[0].s, b.roots[0].s], [4000, 4000])

    def test_short_scopes_not_colored(self):
        lines = self.variance(scope_file(10**9, 100), scope_file(10**9, 900))
        self.assertIn("spread 160.0%,", self.line(lines, "leaf"))

    def test_work_vs_speed(self):
        # same instructions, 1.5x cycles: the machine, not the work
        machine = self.variance(
            scope_file(10**9, 10**6, counts=[10**6, 10**6, 1]),
            scope_file(10**9, 15 * 10**5, counts=[15 * 10**5, 10**6, 1]),
        )
        leaf = self.line(machine, "leaf")
        self.assertIn("work 0.0%,", leaf)
        self.assertIn("speed 40.0% (!!)", leaf)
        # 1.5x instructions at the same cycles per instruction: imbalance
        imbalance = self.variance(
            scope_file(10**9, 10**6, counts=[10**6, 10**6, 1]),
            scope_file(10**9, 15 * 10**5, counts=[15 * 10**5, 15 * 10**5, 1]),
        )
        leaf = self.line(imbalance, "leaf")
        self.assertIn("work 40.0% (!!)", leaf)
        self.assertTrue(leaf.endswith("speed 0.0%"))

    def test_different_event_sets_time_only(self):
        other = PERF.replace("light", "ipc")
        lines = self.variance(
            scope_file(10**9, 10**6, counts=[1, 1, 1]),
            scope_file(10**9, 10**6, counts=[1, 1, 1], perf=other),
        )
        self.assertNotIn("work", self.line(lines, "leaf"))


MIB = 1 << 20


def mem_file(peak_in, peak_out, rss_in=10 * MIB, rss_out=10 * MIB):
    """root (time only) > io (mem): "m:<rss_in>,<rss_out>,<peak_in>,<peak_out>" """
    mem = f"m:{rss_in},{rss_out},{peak_in},{peak_out}"
    return f"0 root\n1 io\n\n0 0 0:1000\n1 1 0:100 {mem}\n"


class MemVarianceTest(unittest.TestCase):
    def variance(self, *files, **kwargs):
        from phlop.timing.scope_timer import print_mem_variance_across

        with tempfile.TemporaryDirectory() as d:
            for i, text in enumerate(files):
                (Path(d) / f"rank.{i}.txt").write_text(text)
            lines = []
            print_mem_variance_across(f"{d}/rank.*.txt", printer=lines.append, **kwargs)
        return lines

    def test_imbalance_names_the_worst_file(self):
        lines = self.variance(
            mem_file(100 * MIB, 100 * MIB),
            mem_file(100 * MIB, 100 * MIB),
            mem_file(100 * MIB, 200 * MIB),  # +100MiB spike, peak 2x the median
        )
        self.assertIn("peak imbalance", lines[0])
        self.assertEqual(lines[2], "root")  # time only, keeps the tree
        io = lines[3]
        self.assertIn("peak 100.00MiB max 200.00MiB +100.0% (!!) rank.2.txt", io)
        self.assertIn("spike 100.00MiB (!!) rank.2.txt", io)
        self.assertIn("kept +0B max +0B", io)

    def test_balanced_is_not_flagged(self):
        lines = self.variance(
            mem_file(100 * MIB, 105 * MIB, rss_out=12 * MIB),
            mem_file(100 * MIB, 110 * MIB, rss_out=14 * MIB),
        )
        io = lines[3]
        self.assertIn("+2.3% rank.1.txt", io)  # 110 / 107.5 - 1
        self.assertNotIn("(!", io)
        self.assertIn("kept +3.00MiB max +4.00MiB", io)

    def test_peak_mode_in_header(self):
        lifetime = self.variance(
            mem_file(100 * MIB, 100 * MIB), mem_file(100 * MIB, 100 * MIB)
        )
        self.assertIn("process high water mark", lifetime[1])
        # PHLOP_SCOPE_MEM=peak: peak on entry is rss on entry
        per_scope = mem_file(10 * MIB, 30 * MIB)
        lines = self.variance(per_scope, per_scope)
        self.assertEqual(lines[1], "peaks per scope, PHLOP_SCOPE_MEM=peak")
        self.assertIn("spike 20.00MiB (!!)", lines[3])  # 20 of a 30MiB peak

    def test_only_mem_scopes_and_ancestors(self):
        text = (
            "0 a\n1 b\n2 c\n\n0 0 0:9\n1 1 0:1\n0 0 0:9\n1 2 0:1 m:1,1,1,1\n0 2 0:1\n"
        )
        lines = self.variance(text, text)
        self.assertEqual([line.split(" peak")[0] for line in lines[2:]], ["a", "  c"])
        self.assertEqual(
            self.variance(scope_file(9, 1), scope_file(9, 1))[0][:13], "no mem scopes"
        )

    def test_main_memvariance(self):
        from phlop.timing.scope_timer import main

        with tempfile.TemporaryDirectory() as d:
            for i in range(2):
                (Path(d) / f"rank.{i}.txt").write_text(mem_file(MIB, MIB))
            self.assertEqual(main(["memvariance", f"{d}/rank.*.txt"]), 0)
            self.assertEqual(main(["variance", f"{d}/rank.*.txt", "--warn", "1"]), 0)


class DedupeCountsTest(unittest.TestCase):
    def test_counts_summed_or_dropped(self):
        text = (
            PERF + "0 root\n1 leaf\n\n0 0 0:100\n1 1 0:10 1:1,2,3\n1 1 0:10 1:4,5,6\n"
        )
        stf = lines_parser(text.splitlines(keepends=True))
        (leaf,) = dedupe_leafs(stf, stf.roots[0])
        self.assertEqual(leaf.counts, [5, 7, 9])

        text = PERF + "0 root\n1 leaf\n\n0 0 0:100\n1 1 0:10 1:1,2,3\n1 1 0:10\n"
        stf = lines_parser(text.splitlines(keepends=True))
        (leaf,) = dedupe_leafs(stf, stf.roots[0])
        self.assertIsNone(leaf.counts)


class PerfFlagsTest(unittest.TestCase):
    def test_flagged_metrics_marked(self):
        # 10 instructions per RAM fetch: memory bound, the light set flags it
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.txt"
            path.write_text(scope_file(10**9, 10**6, counts=[10**6, 10**5, 10**4]))
            lines, verbose = [], []
            print_scope_perf(str(path), printer=lines.append)
            print_scope_perf(str(path), verbose=True, printer=verbose.append)
        leaf = next(line for line in lines if "leaf" in line)
        self.assertIn("instructions_per_fetch=10 (!)", leaf)
        self.assertIn("ipc=0.1", leaf)
        self.assertNotIn("ipc=0.1 (!)", leaf)  # light has no ipc hint
        self.assertTrue(lines[-1].startswith("flagged (!)"))
        self.assertTrue(any("instructions_per_fetch: under 100" in v for v in verbose))


if __name__ == "__main__":
    unittest.main()
