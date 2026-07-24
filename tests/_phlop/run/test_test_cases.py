# tests/_phlop/run/test_test_cases.py

import os
import shlex
import sys
import tempfile
import unittest

from phlop.procs.runtimer import RunTimer
from phlop.testing.test_cases import _python_invocation_index, _python_test_target


class PhlopGitTestCasesTest(unittest.TestCase):
    def test_fails(self):
        """Force some test failures conditionally to see exit codes in bash etc"""
        force_failure = int(os.environ.get("PHLOP_FORCE_TEST_CASE_FAILURE", "0"))

        if force_failure == 1:
            raise RuntimeError("Fail")
        if force_failure == 2:
            sys.exit(1)
        if force_failure == 3:
            self.assertEqual(1, 2)
        if force_failure == 4:
            self.fail("force_failure")


class PythonTestTargetTest(unittest.TestCase):
    def _target(self, cmd):
        bits = shlex.split(cmd)
        idx = _python_invocation_index(bits, cmd)
        return _python_test_target(bits, idx, cmd)

    def test_script_file_after_flags(self):
        cmd = (
            "/usr/bin/mpirun -n 2 --bind-to none python3 -u "
            "test_alfven2d_amr.py AlfvenConvergenceTest.test_spatial_convergence_1_2"
        )
        self.assertEqual(self._target(cmd), "test_alfven2d_amr.py")

    def test_value_consuming_flag_is_skipped(self):
        self.assertEqual(self._target("python3 -W default test.py"), "test.py")

    def test_dash_m_direct_module(self):
        cmd = "python3 -um mypkg.mymodule ClassName.test_id"
        self.assertEqual(self._target(cmd), "mypkg.mymodule")

    def test_dash_m_unittest_wrapper(self):
        cmd = "python3 -m unittest package.test"
        self.assertEqual(self._target(cmd), "package.test")

    def test_no_python3_invocation_raises(self):
        cmd = "mpirun -n 2 script.sh"
        with self.assertRaises(ValueError):
            _python_invocation_index(shlex.split(cmd), cmd)

    def test_dash_m_unittest_without_test_id_raises(self):
        with self.assertRaises(ValueError):
            self._target("python3 -m unittest")

    def test_no_target_raises(self):
        with self.assertRaises(ValueError):
            self._target('python3 -c "print(1)"')


class RunTimerLogFileHandleTest(unittest.TestCase):
    def _open_fd_count(self):
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))

    def test_run_path_closes_log_file_handles(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "job")
            before = self._open_fd_count()
            for _ in range(10):
                RunTimer(
                    "echo hi",
                    shell=True,
                    capture_output=False,
                    log_file_path=log_path,
                    popen=False,
                )
            self.assertEqual(self._open_fd_count(), before)

    def test_popen_path_closes_log_file_handles(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "job")
            before = self._open_fd_count()
            for _ in range(10):
                RunTimer(
                    "echo hi",
                    shell=True,
                    capture_output=False,
                    log_file_path=log_path,
                    popen=True,
                )
            self.assertEqual(self._open_fd_count(), before)

    def test_capture_disabled_without_log_file_uses_devnull(self):
        import subprocess

        rt = RunTimer("echo hi", shell=True, capture_output=False, popen=True)
        self.assertEqual(rt.stdout, None)
        self.assertEqual(rt.stderr, None)
        self.assertEqual(rt.exitcode, 0)
        del subprocess


if __name__ == "__main__":
    unittest.main()
