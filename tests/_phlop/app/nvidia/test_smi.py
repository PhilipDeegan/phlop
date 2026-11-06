# tests/_phlop/app/nvidia/test_smi.py
#
# nvidia-smi parsing, CUDA_VISIBLE_DEVICES, and stats_man's GPU columns and
# summary on fake samples (no GPU needed), then a real query and a real
# stats_man run when nvidia-smi is there: counters only, no GPU load

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

from phlop.app import stats_man as sm
from phlop.app.nvidia import smi

U0, U1 = "GPU-aaaa-0", "GPU-bbbb-1"
NA = "[N/A]"


def gpu_row(index, uuid, util="50", mem="1000", power="100.5", clock="2000",
            temp="60", slow=()):  # fmt: skip
    flags = ["Active" if s in slow else "Not Active" for s in smi.SLOWDOWNS]
    return [str(index), uuid, util, mem, power, clock, temp, *flags]


class SmiParseTest(unittest.TestCase):
    def test_parse(self):
        gpus = [gpu_row(0, U0, slow=("hw_thermal_slowdown",)),
                gpu_row(1, U1, power=NA), ["2", "short row"]]  # fmt: skip
        apps = [["10", U0, "300"], ["11", U0, "200"], ["99", U0, "5000"],
                ["10", U1, "7"], ["bad"]]  # fmt: skip
        s0, s1 = smi.parse(gpus, apps, {0, 1}, {10, 11})
        self.assertEqual(
            (s0.index, s0.util, s0.mem_mb, s0.power_w), (0, 50, 1000, 100.5)
        )
        self.assertEqual((s0.sm_mhz, s0.temp_c), (2000, 60))
        self.assertEqual(s0.slowdowns, ["hw_thermal_slowdown"])
        self.assertEqual(s0.proc_mem_mb, 500)  # pid 99 isn't ours
        self.assertEqual((s1.power_w, s1.proc_mem_mb, s1.slowdowns), (None, 7, []))

    def test_only_asked_indices(self):
        rows = [gpu_row(0, U0), gpu_row(1, U1)]
        self.assertEqual([s.index for s in smi.parse(rows, [], {1}, set())], [1])

    def test_visible(self):
        gpus = [smi.Gpu(0, U0), smi.Gpu(1, U1)]
        idx = lambda env: [g.index for g in smi.visible(gpus, env)]
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(idx(None), [0, 1])  # unset: all
        self.assertEqual(idx(""), [])  # set empty: none
        self.assertEqual(idx("1"), [1])
        self.assertEqual(idx("GPU-bbbb,0"), [1, 0])  # uuid prefix, order kept
        self.assertEqual(idx("5"), [])

    def test_no_nvidia_smi(self):
        with mock.patch.object(smi.shutil, "which", return_value=None):
            self.assertFalse(smi.available())
            self.assertEqual(smi.gpus(), [])
            self.assertEqual(sm.gpus_to_sample(None), [])


def samples(index, n, util=80, slow_every=0):
    return [
        smi.Sample(index, util=util, mem_mb=1000, proc_mem_mb=250, power_w=100 + i,
                   sm_mhz=2000 - i, temp_c=70,
                   slowdowns=["sw_power_cap"] if slow_every and i % slow_every == 0 else [])
        for i in range(n)
    ]  # fmt: skip


class StatsManGpuTest(unittest.TestCase):
    def test_columns(self):
        self.assertEqual(sm.columns(), sm.HEADERS)  # no GPU: as before
        cols = sm.columns([smi.Gpu(0, U0), smi.Gpu(3, U1)])
        self.assertEqual(cols[3:5], ["gpu0_util", "gpu0_mem_mb"])
        self.assertEqual(cols[-1], "gpu3_slowdown")
        self.assertEqual(len(cols), 3 + 2 * len(sm.GPU_HEADERS))

    def test_no_gpu_flag(self):
        with mock.patch.object(smi, "gpus") as gpus:
            self.assertEqual(sm.gpus_to_sample(False), [])
            gpus.assert_not_called()

    def test_yaml_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "s.yaml")
            stats = {"fds": 3, "mem_used_mb": 20, "cpu_usage": 1.5,
                     "gpu": [smi.Sample(0, 50, 1000, 300, None, 2000, 60,
                                        ["sw_power_cap", "hw_slowdown"])]}  # fmt: skip
            sm.append_yaml(f, stats)
            sm.append_yaml(f, {k: v for k, v in stats.items() if k != "gpu"})
            with open(f) as fh:
                lines = fh.read().splitlines()
        self.assertEqual(
            lines,
            ["- v: 3,20,1.5,50,1000,300,,2000,60,sw_power_cap+hw_slowdown",
             "- v: 3,20,1.5"],
        )  # fmt: skip

    def test_summary(self):
        data = sm.ProcessCaptureInfo()
        for a, b in zip(samples(0, 10, slow_every=3), samples(1, 10, util=20)):
            data.append({"fds": 1, "mem_used_mb": 1, "cpu_usage": 1, "gpu": [a, b]})
        text = "\n".join(sm.gpu_summary(data.gpu))
        self.assertIn("GPU 0: util avg 80% max 80%, process mem max 250MB", text)
        self.assertIn("GPU 0: power avg 104W max 109W, SM clock min 1991MHz", text)
        self.assertIn(
            "GPU 0: clocks slowed by sw_power_cap in 4 of 10 samples (40%)", text
        )
        self.assertIn("GPU 1: no kernel running 80% of the time", text)
        self.assertNotIn("GPU 0: no kernel running", text)
        self.assertNotIn("GPU 1: clocks slowed", text)

    def test_nvidia_smi_fails_warns_once(self):
        sampler = sm._Sampler(os.getpid(), [smi.Gpu(0, U0)])
        err = subprocess.TimeoutExpired("nvidia-smi", 10)
        with (
            mock.patch.object(smi, "sample", side_effect=err),
            self.assertLogs(sm.logger, "WARNING") as logs,
        ):
            first, second = sampler()["gpu"], sampler()["gpu"]
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(
            (first[0].index, first[0].util, second[0].mem_mb), (0, None, None)
        )


def _gpu():
    if not smi.available():
        return False
    return (
        subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, check=False
        ).returncode
        == 0
    )


@unittest.skipUnless(_gpu(), "needs nvidia-smi and a GPU")
class RealSmiTest(unittest.TestCase):
    def test_sample(self):
        gpus = smi.visible(smi.gpus())
        self.assertTrue(gpus)
        s = smi.sample([g.index for g in gpus], {os.getpid()})
        self.assertEqual([x.index for x in s], [g.index for g in gpus])
        self.assertIsNotNone(s[0].mem_mb)
        self.assertEqual(s[0].proc_mem_mb, 0)  # this process has no CUDA context

    def test_stats_man_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "s.yaml")
            cmd = [sys.executable, "-m", "phlop.run.stats_man", "-i", "1", "-y", f,
                   "sleep", "1.5"]  # fmt: skip
            subprocess.run(cmd, check=True, capture_output=True)
            with open(f) as fh:
                doc = yaml.safe_load(fh)
        index = smi.visible(smi.gpus())[0].index
        self.assertIn(f"gpu{index}_util", doc["headers"])
        self.assertEqual(doc["gpus"][0]["index"], index)
        row = doc["snapshots"][0]["v"].split(",")
        self.assertEqual(len(row), len(doc["headers"]))


if __name__ == "__main__":
    unittest.main()
