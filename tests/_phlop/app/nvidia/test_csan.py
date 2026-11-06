# tests/_phlop/app/nvidia/test_csan.py
#
# the compute-sanitizer xml parser and reports on synthetic files, the MPI
# rank handshake, then real runs if nvcc, compute-sanitizer and a GPU are there

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from phlop.app.nvidia import csan
from phlop.app.nvidia import csan_report as report
from phlop.run import mpi_ranks

HERE = Path(__file__).parent

HOST = """<hostStack><saveLocation>x</saveLocation>
  <frame><func>cuMemAlloc_v2</func><module>/lib/libcuda.so.1</module></frame>
  <frame><func>cudaMalloc</func><module>./app</module></frame>
  <frame><func>{caller}</func><path>/src/app.cu</path><line>{line}</line><module>./app</module></frame>
  <frame><func>__libc_start_main</func><module>/lib/libc.so.6</module></frame>
</hostStack>"""


def _thread(t, b):
    return (
        f"<who><threadIdx><x>{t}</x><y>0</y><z>0</z></threadIdx>"
        f"<blockIdx><x>{b}</x><y>0</y><z>0</z></blockIdx></who>"
    )


def oob(t=1, b=3):
    return f"""<record><kind>Precise</kind><level>Error</level>{_thread(t, b)}
  <what><text>Invalid __global__ write of size 4 bytes</text><size>4</size></what>
  <where><func>kern</func><path>/src/app.cu</path><line>16</line></where>
  {HOST.format(caller="main", line=40)}</record>"""


API = f"""<record><kind>Api</kind><what><text>Program hit ...</text>
  <api>cudaDeviceSynchronize</api><error>cudaErrorLaunchFailure</error>
  <message>unspecified launch failure</message></what>
  {HOST.format(caller="main", line=61)}</record>"""
STICKY = """<record><kind>Sanitizer</kind><level>Error</level>
  <what><text>CUDA API Error: Sticky error detected</text></what></record>"""


def leak(n, line):
    return f"""<record><kind>Leak</kind><what><text>Leaked {n} bytes</text>
  <size>{n}</size><type>generic</type></what>
  {HOST.format(caller="setup(int)", line=line)}</record>"""


RACE = """<record><kind>Analysis</kind><level>Error</level><what><text>Race condition</text>
  <source><direction>Write</direction><where><func>race</func><path>/src/app.cu</path><line>26</line></where></source>
  <destination><direction>Read</direction><where><func>race</func><path>/src/app.cu</path><line>28</line></where>
  <count>1024</count></destination></what></record>"""


def xml(*records, closed=True):
    s = '<?xml version="1.0" encoding="utf-8"?>\n<ComputeSanitizerOutput>\n'
    s += "\n".join(records)
    return s + ("\n</ComputeSanitizerOutput>\n" if closed else "")


class CsanParseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write(self, name, content):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_memcheck_grouped(self):
        recs = [oob(t) for t in range(5)] + [
            API,
            STICKY,
            leak(4096, 50),
            leak(4096, 50),
            leak(10, 7),
        ]
        r = csan.parse(self.write("csan-memcheck.r1.p12.xml", xml(*recs)))
        self.assertEqual(
            (r.tool, r.rank, r.records, r.complete), ("memcheck", None, 10, True)
        )
        kinds = [i.kind for i in r.issues]
        self.assertEqual(kinds, ["error", "api", "leak", "leak", "note"])
        err = r.issues[0]
        self.assertEqual((err.count, err.where.where()), (5, "app.cu:16"))
        self.assertEqual(err.example, "block 3 thread 0")
        api = r.issues[1]
        self.assertEqual(
            str(api.caller), "main (app.cu:61)"
        )  # CUDA runtime frames skipped
        self.assertEqual((r.leaks[0].bytes, r.leaks[0].count), (8192, 2))

        text = report.text([r])
        self.assertIn(
            "6 errors in 2 places, 3 leaks, 8.0KiB", text
        )  # the API error counts
        self.assertIn("at kern (app.cu:16), eg block 3 thread 0", text)
        self.assertIn("cudaDeviceSynchronize: cudaErrorLaunchFailure", text)
        self.assertIn("allocated from setup(int) (app.cu:50)", text)
        self.assertIn("a kernel faulted: CUDA errors are sticky", text)
        self.assertIn("| memory errors | 5 |", report.markdown([r]))

    def test_race_and_clean(self):
        r = csan.parse(self.write("csan-racecheck.r2.p1.xml", xml(RACE)))
        self.assertEqual(r.issues[0].count, 1024)
        text = report.text([r])
        self.assertIn("race (app.cu:26) vs race (app.cu:28)", text)
        self.assertIn("__syncthreads()", text)
        clean = csan.parse(self.write("csan-memcheck.r3.p1.xml", xml()))
        self.assertIn("memcheck: clean", report.text([clean]))

    def test_cut_short(self):  # the process died mid record: keep the closed ones
        content = xml(oob(), oob(2), closed=False) + "<record><kind>Prec"
        r = csan.parse(self.write("csan-memcheck.r4.p1.xml", content))
        self.assertFalse(r.complete)
        self.assertEqual(r.issues[0].count, 2)
        self.assertIn("cut short", report.text([r]))

    def test_print_limit(self):
        r = csan.parse(
            self.write("csan-memcheck.r5.p1.xml", xml(*[oob()] * csan.PRINT_LIMIT))
        )
        self.assertTrue(r.limited)
        self.assertIn(f"{csan.PRINT_LIMIT}+ errors", report.text([r]))

    def test_ranks_merged_and_resolve(self):
        self.write("csan-memcheck.mpi-1.rank0.xml", xml(oob()))
        self.write("csan-memcheck.mpi-1.rank1.xml", xml(oob(), RACE))
        self.write("csan-memcheck.mpi-1.rank2.xml", xml())
        self.write("other.xml", "<x/>")
        rs = csan.results("mpi-1", self.tmp)
        self.assertEqual([r.rank for r in rs], [0, 1, 2])
        self.assertEqual(len(csan.resolve(None, self.tmp)), 3)  # the latest run
        text = report.text(rs)
        self.assertIn("3 processes, issues on ranks 0-1", text)
        self.assertIn("[ranks 0-1]", text)
        self.assertIn("| ranks |", report.markdown(rs))
        self.assertEqual(report._ranks({0, 1, 2, 5, 7, 8}), "0-2,5,7-8")


class RanksHandshakeTest(unittest.TestCase):
    def test_rank0_reports_after_all_and_exits_worst(self):
        tmp = tempfile.mkdtemp()
        try:
            out, reported = {}, []

            def rank(r, rc, delay):
                threading.Event().wait(delay)
                out[r] = mpi_ranks.finish(
                    tmp,
                    "mpi-9",
                    str(r),
                    3,
                    rc,
                    lambda: reported.append(len(out)),
                    timeout=30,
                )

            threads = [
                threading.Thread(target=rank, args=a)
                for a in ((0, 0, 0), (1, 99, 0.6), (2, 0, 0.2))
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(60)
            self.assertEqual(out, {0: 99, 1: 99, 2: 0})  # rank 0: the worst
            self.assertEqual(reported, [0])  # once, before any rank exited
        finally:
            shutil.rmtree(tmp)

    def test_slurm_steps_differ(self):
        env = dict(os.environ)
        try:
            for k in mpi_ranks.JOB_ENVS:
                os.environ.pop(k, None)
            os.environ.update(SLURM_JOB_ID="77", SLURM_STEP_ID="3")
            self.assertEqual(mpi_ranks.run_id(), "mpi-77-3")
        finally:
            os.environ.clear()
            os.environ.update(env)


def _gpu():
    if not (shutil.which("nvcc") and shutil.which("compute-sanitizer")):
        return False
    return (
        subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, check=False
        ).returncode
        == 0
    )


@unittest.skipUnless(_gpu(), "needs nvcc, compute-sanitizer and a GPU")
class CsanRealRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "bench_csan")
        cmd = [
            "nvcc",
            "-O0",
            "-g",
            "-lineinfo",
            "-o",
            cls.exe,
            str(HERE / "bench_csan.cu"),
        ]
        subprocess.run(cmd, check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def run_csan(self, tool, what):
        out = os.path.join(self.tmp, tool + what)
        cmd = [
            sys.executable,
            "-m",
            "phlop.run.csan",
            "-q",
            "-t",
            tool,
            "-o",
            out,
            self.exe,
            what,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def test_each_tool_finds_its_bug(self):
        for tool, what, expect in (
            ("memcheck", "oob", "at oob (bench_csan.cu:16)"),
            ("initcheck", "uninit", "at sum (bench_csan.cu:21)"),
            ("racecheck", "race", "race (bench_csan.cu:26) vs race (bench_csan.cu:28)"),
            ("synccheck", "sync", "at divergent (bench_csan.cu:35)"),
        ):
            with self.subTest(tool=tool):
                res = self.run_csan(tool, what)
                self.assertEqual(
                    res.returncode, csan.ERROR_EXITCODE, res.stdout + res.stderr
                )
                self.assertIn(expect, res.stdout)

    def test_clean(self):
        res = self.run_csan("memcheck", "clean")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("memcheck: clean", res.stdout)


if __name__ == "__main__":
    unittest.main()
