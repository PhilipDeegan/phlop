# tests/_phlop/app/nvidia/test_nsys.py
#
# interval maths, the sqlite parser and reports on synthetic profiles with
# known answers, then real runs if nvcc, nsys and a GPU are available

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phlop.app.nvidia import nsys
from phlop.app.nvidia import nsys_report as report

HERE = Path(__file__).parent
MS = 1_000_000

SCHEMA = """
create table StringIds (id integer primary key, value text);
create table ANALYSIS_DETAILS (globalVid int, duration int, startTime int, stopTime int);
create table TARGET_INFO_GPU (id int, name text);
create table META_DATA_CAPTURE (name text, value text);
create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, demangledName int,
  shortName int, gridX int, gridY int, gridZ int, blockX int, blockY int, blockZ int,
  registersPerThread int, staticSharedMemory int, dynamicSharedMemory int);
create table CUPTI_ACTIVITY_KIND_MEMCPY (start int, end int, copyKind int,
  srcKind int, dstKind int, bytes int);
create table CUPTI_ACTIVITY_KIND_RUNTIME (start int, end int, nameId int);
create table NVTX_EVENTS (start int, end int, eventType int, text text, textId int);
create table MPI_RANKS (globalTid int, rank int);
create table MPI_COLLECTIVES_EVENTS (start int, end int, textId int);
"""


def make_profile(path, rank=0, compute_ms=40, nccl=((40, 50),), pageable=True):
    """a profile with known numbers, times in ms:
    kernel 'gemm' [0, compute_ms), NCCL kernels as given, an idle gap
    [60, 90) inside NVTX 'dataload', one H2D copy [90, 100)"""
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    names = ["gemm", "ncclDevKernel_AllReduce_Sum_f32_RING_LL", "cudaDeviceSynchronize_v3020",
             "MPI_Allreduce", "MPI_Init", "dataload", "step"]  # fmt: skip
    db.executemany("insert into StringIds values (?, ?)", list(enumerate(names)))
    db.execute("insert into ANALYSIS_DETAILS values (0, ?, 0, 0)", (200 * MS,))
    db.execute("insert into TARGET_INFO_GPU values (0, 'Test GPU')")
    db.executemany(
        "insert into META_DATA_CAPTURE values (?, ?)",
        [("PROCESS_0:COMMAND", "./train"), ("PROCESS_0:ARGUMENT_0", "--fast")],
    )
    kern = "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, 128, 1, 1, 256, 1, 1, 32, 0, 0)"
    db.execute(kern, (0, compute_ms * MS, 0, 0))
    for s, e in nccl:
        db.execute(kern, (s * MS, e * MS, 1, 1))
    db.execute(
        "insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, 1, ?, 2, ?)",
        (90 * MS, 100 * MS, 0 if pageable else 1, 1 << 30),
    )
    db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, 2)", (0, 50 * MS))
    db.executemany(
        "insert into NVTX_EVENTS values (?, ?, 59, ?, ?)",
        [
            (0, 100 * MS, None, 6),
            (55 * MS, 95 * MS, "dataload", None),
            (1, 2, None, None),
        ],
    )
    db.execute("insert into NVTX_EVENTS values (5, null, 34, 'a mark', null)")
    db.execute("insert into MPI_RANKS values (0, ?)", (rank,))
    db.executemany(
        "insert into MPI_COLLECTIVES_EVENTS values (?, ?, ?)",
        [(40 * MS, 50 * MS, 3), (0, 5 * MS, 4)],
    )
    db.commit()
    db.close()
    return path


class IntervalsTest(unittest.TestCase):
    def test_merge_and_overlap(self):
        m = nsys.merge([(5, 8), (0, 2), (1, 3), (7, 10)])
        self.assertEqual(m, [(0, 3), (5, 10)])
        self.assertEqual(nsys.length(m), 8)
        self.assertEqual(nsys.overlap(m, [(2, 6)]), 2)
        self.assertEqual(nsys.gaps(m), [(2, 3)])

    def test_coverage(self):
        m = nsys.merge([(0, 10), (20, 30), (40, 50)])
        cov = nsys.Coverage(m)
        for lo, hi in [
            (0, 50),
            (5, 25),
            (12, 18),
            (25, 45),
            (-5, 100),
            (30, 40),
            (9, 41),
        ]:
            self.assertEqual(cov(lo, hi), nsys.overlap(m, [(lo, hi)]), (lo, hi))
        self.assertEqual(nsys.Coverage([])(0, 10), 0)


class ParseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p = nsys.parse(make_profile(os.path.join(self.tmp, "nsys.x.rank0.sqlite")))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_numbers(self):
        p = self.p
        self.assertEqual((p.cmd, p.rank, p.gpus), ("./train --fast", 0, ["Test GPU"]))
        self.assertEqual((p.first, p.last), (0, 100 * MS))
        self.assertEqual(p.compute, 40 * MS)
        self.assertEqual(p.nccl, 10 * MS)
        self.assertEqual(p.nccl_overlap, 0)
        self.assertEqual(p.busy, 60 * MS)
        self.assertEqual(p.idle, 40 * MS)
        self.assertEqual(p.mpi_time, 10 * MS)  # MPI_Init isn't communication
        self.assertEqual(p.wait, 20 * MS)
        self.assertEqual(p.kernels[0].name, "gemm")
        self.assertTrue(p.kernels[1].nccl)
        self.assertEqual(p.api[0].name, "cudaDeviceSynchronize")

    def test_nvtx_and_gaps(self):
        p = self.p
        nvtx = {r.name: r for r in p.nvtx}
        self.assertEqual(nvtx["step"].gpu, 60 * MS)
        self.assertEqual(nvtx["dataload"].gpu, 5 * MS)  # the copy, from 90
        self.assertNotIn("a mark", nvtx)
        self.assertEqual((p.gaps[0].length, p.gaps[0].nvtx), (40 * MS, "dataload"))

    def test_hints(self):
        h = "\n".join(report.hints(self.p))
        self.assertIn("GPU idle 40.0%", h)
        self.assertIn("in NVTX 'dataload'", h)
        self.assertIn("pageable", h)
        self.assertIn("communication is exposed", h)
        self.assertIn("synchronise", h)

    def test_overlapped_nccl_no_hint(self):
        path = make_profile(
            os.path.join(self.tmp, "o.sqlite"), nccl=((10, 30),), pageable=False
        )
        p = nsys.parse(path)
        self.assertEqual(p.nccl_overlap, 20 * MS)
        h = "\n".join(report.hints(p))
        self.assertNotIn("exposed", h)
        self.assertNotIn("pageable", h)

    def test_text_and_markdown(self):
        text = report.text([self.p])
        self.assertIn("top kernel gemm", text)
        self.assertIn("H2D pageable 1.0GiB", text)
        md = report.markdown([self.p])
        self.assertIn("pie title GPU time", md)
        self.assertIn("| dataload |", md)

    def test_no_gpu(self):
        path = os.path.join(self.tmp, "empty.sqlite")
        db = sqlite3.connect(path)
        db.executescript("create table StringIds (id integer primary key, value text);")
        db.close()
        p = nsys.parse(path)
        self.assertIn("no GPU activity", report.text([p]))
        self.assertEqual(report.hints(p), [])


class RanksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def ranks(self, compute, run="r1"):
        out = []
        for rank, ms in enumerate(compute):
            path = os.path.join(self.tmp, f"nsys.{run}.rank{rank}.sqlite")
            out.append(nsys.parse(make_profile(path, rank, ms)))
        return out

    def test_straggler(self):
        ps = self.ranks([20, 21, 20, 38])
        slow, ratio = report.straggler(ps)
        self.assertEqual(slow.rank, 3)
        self.assertGreater(ratio, 1.5)
        text = report.text(ps)
        self.assertIn("straggler: rank 3", text)
        self.assertIn("nsys.r1.rank3.sqlite", text)  # its detail
        md = report.markdown(ps)
        self.assertIn("xychart-beta", md)

    def test_balanced(self):
        self.assertIsNone(report.straggler(self.ranks([20, 20, 21]))[0])
        self.assertIn("balanced", report.text(self.ranks([20, 20, 21], "r2")))

    def test_resolve_and_compare(self):
        a, b = self.ranks([20, 20], "ra"), self.ranks([30, 30], "rb")
        self.assertEqual(len(nsys.resolve("ra", self.tmp)), 2)
        self.assertEqual(len(nsys.results(self.tmp + "/nsys.rb.*")), 2)
        self.assertEqual(len(nsys.resolve(self.tmp)), 2)  # the latest run
        text = report.compare(a, b, ("ra", "rb"))
        self.assertIn("+50.0%", text)  # kernels 20 -> 30ms
        self.assertIn("gemm", text)


def _gpu():
    if not (shutil.which("nvcc") and shutil.which("nsys")):
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


@unittest.skipUnless(_gpu(), "needs nvcc, nsys and a GPU")
class RealRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "bench_gpu")
        subprocess.run(
            ["nvcc", "-O2", "-lineinfo", "-o", cls.exe, str(HERE / "bench_gpu.cu")],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_profile(self):
        out = os.path.join(self.tmp, "out")
        res = subprocess.run(
            [sys.executable, "-m", "phlop.run.nsys", "-q", "-o", out, self.exe, "3"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("top kernel strided", res.stdout)
        self.assertIn("in NVTX 'idle'", res.stdout)
        self.assertIn("pageable", res.stdout)
        p = nsys.results(None, out)[0]
        self.assertEqual(
            {k.name.split("(")[0] for k in p.kernels}, {"strided", "saxpy"}
        )

    def test_exit_code(self):
        out = os.path.join(self.tmp, "rc")
        cmd = [
            sys.executable,
            "-m",
            "phlop.run.nsys",
            "-q",
            "-o",
            out,
            "sh",
            "-c",
            "exit 3",
        ]
        self.assertEqual(
            subprocess.run(cmd, capture_output=True, check=False).returncode, 3
        )


if __name__ == "__main__":
    unittest.main()
