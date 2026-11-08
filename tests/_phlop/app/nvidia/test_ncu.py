# tests/_phlop/app/nvidia/test_ncu.py
#
# the ncu csv parser, verdicts and reports on synthetic imports with known
# answers. A real profile only with PHLOP_NCU_TEST=1: ncu replays each kernel
# ~30 times, holding a GPU that may also drive the desktop

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phlop.app.nvidia import ncu
from phlop.app.nvidia import ncu_report as report

HERE = Path(__file__).parent
M = ncu.METRICS
NVTX = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
STALL = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"
HEAD = ["ID", "Process ID", "Process Name", NVTX, "Kernel Name", "Block Size",
        "Grid Size", "device__attribute_display_name", *M.values(),
        STALL.format("long_scoreboard"), STALL.format("wait"),
        STALL.format("selected"), "launch__occupancy_limit_registers",
        "launch__occupancy_limit_warps"]  # fmt: skip

BASE = {"sm_pct": 10, "mem_pct": 10, "dram_pct": 5, "l2_pct": 5, "l1_pct": 5, "dram_bw": 1e9,
            "occupancy": 90, "occupancy_max": 100, "registers": 32, "waves": 10, "sms": 84,
            "sectors": 100, "sectors_ideal": 100, "wavefronts": 0, "wavefronts_ideal": 0,
            "l1_hit": 50, "l2_hit": 50}  # fmt: skip
# name -> (time ns, metrics over BASE, nvtx)
KERNELS = {
    "void ns::stream<float>(int, float *)": (2e6, {"mem_pct": 90, "dram_pct": 88}, "a"),
    "gather(int, int const *, float *)": (
        3e6, {"mem_pct": 93, "l2_pct": 93, "sectors": 800}, "b"),
    "fma(int, float *)": (1e6, {"sm_pct": 95}, ""),
    "tiny()": (5e3, {"waves": 0.1}, ""),
    "spill(float *)": (1e6, {"occupancy": 20, "occupancy_max": 25, "registers": 255,
                                 "wavefronts": 500, "wavefronts_ideal": 100}, ""),
}  # fmt: skip


def _cell(v):
    return f"{v:,.2f}" if isinstance(v, float) and v >= 1000 else str(v)


def write_import(base, kernels=KERNELS, launches=1, scale=1.0, process="app"):
    """<base>.ncu-repz (empty) and its two csv imports, as ncu writes them"""
    Path(base + ".ncu-repz").write_bytes(b"")
    rows, rules, i = [], [], 0
    for name, (t, over, nvtx) in kernels.items():
        m = {**BASE, **over, "time": t * scale}
        for _ in range(launches):
            cell = f'1  "<default domain>:{nvtx}:none:none:none:none:none:none" '
            rows.append([str(i), "42", process, cell if nvtx else "", name,
                         "(256, 1, 1)", "(1024, 1, 1)", "Test GPU",
                         *(_cell(float(m[k])) for k in M), "40", "2", "1",
                         "4", "16" if name != "spill(float *)" else "2"])  # fmt: skip
            if "gather" in name:
                rules.append([str(i), "UncoalescedGlobalAccess", "OPT",
                              "This workload has uncoalesced global accesses. More.", "82.5"])  # fmt: skip
                rules.append([str(i), "CPIStall", "OPT", "Stalled. On loads.", "60"])
            rules.append([str(i), "HighPipeUtilization", "OPT", "Pipes idle.", "90"])
            rules.append([str(i), "SOLBottleneck", "INF", "Information.", ""])
            rules.append([str(i), "Tiny", "OPT", "Nothing to gain.", "0.2"])
            i += 1
    with open(base + ".raw.csv", "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(HEAD)
        w.writerow([""] * 8 + ["ns"] + [""] * (len(HEAD) - 9))
        w.writerows(rows)
    with open(base + ".rules.csv", "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(
            ["ID", "Rule Name", "Rule Type", "Rule Description", "Estimated Speedup"]
        )
        w.writerows(rules)
    return base + ".ncu-repz"


class NcuParseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rep = ncu.load(write_import(os.path.join(self.tmp, "ncu.r1.p1")))
        self.k = {k.name: k for k in self.rep.kernels}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_kernels(self):
        self.assertEqual(self.rep.device, "Test GPU")
        self.assertEqual(self.rep.process, "app")
        self.assertEqual(
            [k.name for k in self.rep.kernels],
            ["gather", "ns::stream<float>", "fma", "spill", "tiny"],
        )
        self.assertEqual(self.k["ns::stream<float>"].nvtx, "a")
        self.assertEqual(self.k["gather"].time, 3e6)  # "3,000,000.00"
        self.assertEqual(self.k["gather"].stalls[0], ("long_scoreboard", 40))
        self.assertNotIn("selected", dict(self.k["gather"].stalls))

    def test_verdicts(self):
        bound = {n: k.bound for n, k in self.k.items()}
        self.assertEqual(bound["ns::stream<float>"], "memory (DRAM)")
        self.assertEqual(bound["gather"], "memory (L2)")
        self.assertEqual(bound["fma"], "compute")
        self.assertEqual(bound["tiny"], "latency (small grid)")
        self.assertEqual(bound["spill"], "latency")

    def test_ratios(self):
        self.assertAlmostEqual(self.k["gather"].coalescing, 8)
        self.assertIsNone(self.k["gather"].bank_conflicts)  # no shared memory
        self.assertAlmostEqual(self.k["spill"].bank_conflicts, 5)
        self.assertEqual(self.k["spill"].limiter, "warps")

    def test_rules(self):
        g = self.k["gather"]
        self.assertEqual(
            [r.name for r in g.rules][:2],
            ["HighPipeUtilization", "UncoalescedGlobalAccess"],
        )
        self.assertNotIn("SOLBottleneck", [r.name for r in g.rules])  # INF
        # memory bound: idle compute pipes and load stalls restate the verdict
        self.assertEqual([r.name for r in report.rules(g)], ["UncoalescedGlobalAccess"])
        self.assertEqual(
            [r.name for r in report.rules(self.k["fma"])], ["HighPipeUtilization"]
        )

    def test_text(self):
        text = report.text([self.rep])
        self.assertIn("app on Test GPU, 5 kernels", text)
        self.assertIn("gather: global accesses move 8.0x", text)
        self.assertIn("ncu UncoalescedGlobalAccess (est. 82%): This workload has"
                      " uncoalesced global accesses.", text)  # fmt: skip
        self.assertIn("ns::stream<float>: DRAM at 88% of peak", text)
        self.assertIn("fma: SM at 95% of peak", text)
        self.assertIn("tiny: the grid fills 0.10 waves of the 84 SMs", text)
        self.assertIn("spill: shared memory needs 5.0x", text)
        self.assertIn("spill: theoretical occupancy 25%, limited by warps", text)
        self.assertIn("spill: warps mostly stall on long_scoreboard", text)
        md = report.markdown([self.rep])
        self.assertIn("| 3ms | memory (L2) |", md)
        self.assertIn('x-axis ["gather", "ns::streamfloat"', md)  # no <> in mermaid

    def test_launches_merge(self):
        rep = ncu.load(write_import(os.path.join(self.tmp, "ncu.r2.p1"), launches=3))
        self.assertEqual(len(rep.kernels), 5)
        self.assertEqual(rep.kernels[0].launches, 3)
        self.assertEqual(rep.time, 3 * sum(t for t, _, _ in KERNELS.values()))
        self.assertIn("gather x3 [b]", report.text([rep]))


class NcuRanksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_straggler_and_resolve(self):
        base = os.path.join(self.tmp, "ncu.mpi-1")
        for rank, scale in ((0, 1.0), (1, 2.0)):
            write_import(f"{base}.rank{rank}", scale=scale)
        reps = ncu.results("mpi-1", self.tmp)
        self.assertEqual([r.rank for r in reps], [0, 1])
        text = report.text(reps)
        # 2 ranks: against the other rank, not a median including itself
        self.assertIn(
            "straggler: gather takes 2.00x the other ranks' median on rank 1", text
        )
        self.assertIn("ncu rank 1:", text)
        self.assertEqual(len(ncu.resolve(None, self.tmp)), 2)  # the latest run

    def test_balanced_and_compare(self):
        a = [ncu.load(write_import(os.path.join(self.tmp, "ncu.a.p1")))]
        b = [ncu.load(write_import(os.path.join(self.tmp, "ncu.b.p1"), scale=1.5))]
        c = [ncu.load(write_import(os.path.join(self.tmp, "ncu.c.p1"), scale=1.05))]
        self.assertIn("balanced", report.text(a + c))
        text = report.compare(a, b, ("ra", "rb"))
        self.assertIn("+50.0%", text)
        self.assertIn("total", text)

    def test_log_problem(self):
        Path(self.tmp, "ncu.x.p1.log").write_text(
            "==WARNING== No kernels were profiled.\nAvailable Kernels:\n1. saxpy\n2. strided\n"
        )
        self.assertIn("saxpy, strided", ncu.log_problem(self.tmp, "x"))
        Path(self.tmp, "ncu.y.p1.log").write_text("==ERROR== ERR_NVGPUCTRPERM - x\n")
        self.assertIn(
            "NVreg_RestrictProfilingToAdminUsers", ncu.log_problem(self.tmp, "y")
        )

    def test_names(self):
        self.assertEqual(
            ncu.short_name("void a::b<int (*)(int)>(int, float *)"),
            "a::b<int (*)(int)>",
        )
        self.assertEqual(
            ncu._nvtx(
                '7  "<default domain>:outer:none:none:none:none:none:none" "dom:in:ner:none:none:none:none:none:none" '
            ),
            "in:ner",
        )


def _gpu():
    if os.environ.get("PHLOP_NCU_TEST") != "1":
        return False
    if not (shutil.which("nvcc") and shutil.which("ncu")):
        return False
    if not shutil.which("nvidia-smi"):
        return False
    # bench_gpu.cu includes nvtx3, nvcc's own include directory has it
    nvcc_inc = Path(shutil.which("nvcc")).resolve().parent.parent / "include"
    if not (nvcc_inc / "nvtx3" / "nvToolsExt.h").is_file():
        return False
    return (
        subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, check=False
        ).returncode
        == 0
    )


@unittest.skipUnless(_gpu(), "PHLOP_NCU_TEST=1, nvcc, ncu and a GPU")
class NcuRealRunTest(unittest.TestCase):
    def test_profile(self):
        tmp = tempfile.mkdtemp()
        try:
            exe = os.path.join(tmp, "bench_gpu")
            subprocess.run(
                ["nvcc", "-O2", "-lineinfo", "-o", exe, str(HERE / "bench_gpu.cu")],
                check=True,
            )
            out = os.path.join(tmp, "out")
            res = subprocess.run(
                [sys.executable, "-m", "phlop.run.ncu", "-q", "-o", out, exe, "1"],
                capture_output=True, text=True, check=False,
            )  # fmt: skip
            self.assertEqual(res.returncode, 0, res.stderr)
            k = {k.name: k for k in ncu.results(None, out)[0].kernels}
            self.assertEqual(k["strided"].bound, "memory (L2)")
            self.assertGreater(k["strided"].coalescing, 4)
            self.assertEqual(k["saxpy"].bound, "memory (DRAM)")
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
