# tests/_phlop/app/test_thermal.py
#
# phlop.app.thermal on a fake sensor: the hold's hysteresis and environment, a
# command's whole process tree stopped while hot (a grandchild included), exit
# codes, no sensor: unguarded; the job processor starting nothing while hot

import itertools
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from phlop.app import thermal


def schedule(*phases):
    """a fake sensor: (seconds, temperature) phases from the first reading on,
    the last one's temperature after"""
    start = []

    def temperature():
        if not start:
            start.append(time.monotonic())
        elapsed = time.monotonic() - start[0]
        for seconds, t in phases:
            if elapsed < seconds:
                return t
            elapsed -= seconds
        return phases[-1][1]

    return temperature


class HoldTest(unittest.TestCase):
    def test_hysteresis(self):
        hold = thermal.Hold(high=88, low=78, temperature=lambda: 0)
        readings = [70, 87, 88, 85, 79, 78, 80, 90]
        expected = [False, False, True, True, True, False, False, True]
        self.assertEqual([hold.update(t) for t in readings], expected)

    def test_no_sensor_never_holds(self):
        hold = thermal.Hold(high=10, low=5, temperature=lambda: None)
        self.assertFalse(hold())

    def test_environment(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(thermal.Hold().enabled)
        with mock.patch.dict(os.environ, {"PHLOP_THERMAL_MAX": "85"}, clear=True):
            hold = thermal.Hold()
            self.assertEqual((hold.high, hold.low), (85.0, 75.0))
        env = {"PHLOP_THERMAL_MAX": "85", "PHLOP_THERMAL_RESUME": "60"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(thermal.Hold().low, 60.0)


class GovernorTest(unittest.TestCase):
    def governor(self, command, temperature):
        return thermal.Governor(
            command,
            high=88,
            low=78,
            every=0.1,
            temperature=temperature,
            log=lambda _: None,
        )

    def test_grandchild_paused(self):
        """a grandchild (its own process group, as an MPI rank) writes a time
        stamp every 0.1 s: none while hot, so a gap at least as long"""
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.skipTest("no psutil")
        with tempfile.TemporaryDirectory() as d:
            stamps = Path(d) / "stamps"
            work = (
                "import os, time\n"
                "os.setpgid(0, 0)\n"
                f"with open({str(stamps)!r}, 'w') as f:\n"
                "    for _ in range(15):\n"
                "        f.write(f'{time.monotonic()}\\n'); f.flush(); time.sleep(0.1)\n"
            )
            child = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {work!r}])"
            g = self.governor(
                [sys.executable, "-c", child], schedule((0.6, 40), (1.5, 95), (1, 40))
            )
            self.assertEqual(g.run(), 0)
            times = [float(t) for t in stamps.read_text().split()]
        self.assertEqual(len(times), 15)
        self.assertEqual(g.pauses, 1)
        self.assertGreater(max(b - a for a, b in itertools.pairwise(times)), 1.2)

    def test_exit_code(self):
        g = self.governor([sys.executable, "-c", "raise SystemExit(3)"], lambda: 40)
        self.assertEqual(g.run(), 3)

    def test_no_sensor_unguarded(self):
        g = self.governor([sys.executable, "-c", "raise SystemExit(5)"], lambda: None)
        self.assertEqual(g.run(), 5)
        self.assertEqual(g.pauses, 0)


class ProcessorTest(unittest.TestCase):
    def test_jobs_held_while_hot(self):
        """PHLOP_THERMAL_MAX: no job starts until the CPU cooled down"""
        from phlop.procs import parallel_processor as pp

        cmd = f"{sys.executable} -c pass"
        env = {"PHLOP_THERMAL_MAX": "88", "PHLOP_THERMAL_RESUME": "78"}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(thermal, "cpu_temperature", schedule((1.5, 95), (1, 40))),
        ):
            start = time.monotonic()
            results = pp.process(
                [pp.Job(cmd=cmd, logging=0) for _ in range(3)], n_cores=2
            )
            elapsed = time.monotonic() - start
        self.assertTrue(all(r.ok for r in results))
        self.assertGreater(elapsed, 1.4)


if __name__ == "__main__":
    unittest.main()
