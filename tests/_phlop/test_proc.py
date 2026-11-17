# tests/_phlop/test_proc.py
#
# run_mp: how many commands run at once, and no commands at all. usable_cores
# when the affinity query is refused

import concurrent.futures
import unittest
from unittest import mock

from phlop import proc


class RunMpTest(unittest.TestCase):
    def workers(self, cmds, n_cores=None, usable=4):
        """max_workers run_mp asks for, commands not run"""
        seen = []

        class Pool(concurrent.futures.ThreadPoolExecutor):
            def __init__(self, max_workers):
                seen.append(max_workers)
                super().__init__(max_workers=max_workers)

        with (
            mock.patch("concurrent.futures.ThreadPoolExecutor", Pool),
            mock.patch("phlop.os.usable_cores", return_value=usable),
            mock.patch.object(proc, "run", return_value=0),
        ):
            results = proc.run_mp(cmds, n_cores)
        self.assertEqual(len(results), len(cmds))
        return seen[0]

    def test_capped_at_usable_cores(self):
        self.assertEqual(self.workers(["true"] * 500), 4)

    def test_fewer_cmds_than_cores(self):
        self.assertEqual(self.workers(["true"] * 2), 2)

    def test_explicit(self):
        self.assertEqual(self.workers(["true"] * 500, n_cores=16), 16)

    def test_no_cmds(self):
        self.assertEqual(self.workers([]), 1)  # was max_workers=0: ValueError

    def test_runs(self):
        results = proc.run_mp(["exit 0", "exit 3"])
        self.assertEqual(sorted(r.exitcode for r in results), [0, 3])


class UsableCoresTest(unittest.TestCase):
    def test_affinity_refused(self):
        """python < 3.13 without the affinity syscall (seccomp): os.cpu_count"""
        import types

        import phlop.os

        def refused(pid):
            raise PermissionError(1, "Operation not permitted")

        fake_os = types.SimpleNamespace(sched_getaffinity=refused, cpu_count=lambda: 3)
        with (
            mock.patch.object(phlop.os, "os", fake_os),
            mock.patch.object(phlop.os, "cgroup_cpu_limit", return_value=None),
        ):
            self.assertEqual(phlop.os.usable_cores(), 3)


if __name__ == "__main__":
    unittest.main()
