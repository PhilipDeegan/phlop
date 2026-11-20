# tests/_phlop/app/test_stats_man.py
#
# phlop.app.stats_man's sampler over a process tree: a launcher's children
# (an MPI launcher's ranks) counted in its memory, fds and cpu

import subprocess
import sys
import time
import unittest

from phlop.app import stats_man as sm

MB = 1 << 20
CHILD = (
    "import sys, time; b = bytearray({mb} * (1 << 20)); b[::4096] = b'x' * len(b[::4096]); "
    "print('ready', flush=True); time.sleep(30)"
)
# a parent holding little that starts the child, waits for it, then stays
PARENT = (
    "import subprocess, sys, time; "
    "c = subprocess.Popen([sys.executable, '-c', sys.argv[1]], stdout=sys.stdout); c.wait(); "
    "time.sleep(30)"
)


class StatsManTreeTest(unittest.TestCase):
    def test_children_memory_counted(self):
        mb = 200
        parent = subprocess.Popen(
            [sys.executable, "-c", PARENT, CHILD.format(mb=mb)],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(parent.stdout.readline().strip(), "ready")
            sampler = sm._Sampler(parent.pid)
            stats = sampler()
            self.assertEqual(len(sampler.procs), 2)
            self.assertGreaterEqual(stats["mem_used_mb"], mb)
            own = sm.bytes_as_mb(sampler.proc.memory_info().rss)
            self.assertLess(own, mb)  # the parent alone holds little
            self.assertGreater(stats["fds"], 0)
        finally:
            for c in sm.psutil.Process(parent.pid).children(recursive=True):
                c.kill()
            parent.kill()
            parent.wait()

    def test_a_child_gone_is_not_counted(self):
        parent = subprocess.Popen(
            [sys.executable, "-c", PARENT, CHILD.format(mb=1)],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(parent.stdout.readline().strip(), "ready")
            sampler = sm._Sampler(parent.pid)
            sampler()
            for c in sampler.proc.children(recursive=True):
                c.kill()
                c.wait(5)
            time.sleep(0.1)
            sampler()  # no raise: the child is gone, the parent still there
        finally:
            parent.kill()
            parent.wait()


if __name__ == "__main__":
    unittest.main()
