# tests/_phlop/run/test_test_cases.py

import os
import shlex
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

from phlop.procs.parallel_processor import Job
from phlop.procs.runtimer import RunTimer
from phlop.reflection import classes_in_file
from phlop.run.test_cases import cli_args_parser
from phlop.sys import extend_sys_path
from phlop.testing.test_cases import (
    TestBatch,
    TestBatchesList,
    _log_subpath,
    _python_invocation_index,
    _python_test_target,
    _variants_for,
    extract_load,
    load_config_tests,
    load_py_test_cases_from_cmake,
    resolve_cores,
)


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


class ExecConfigDemoTest(unittest.TestCase):
    """Sample target for tests/_phlop/run/.phlop.exec.yaml: intentionally a no-op
    regardless of the env it's given, so its declared variants are harmless to run
    for real as part of the default test suite scan."""

    def test_scaled(self):
        os.environ.get("DEMO_GRID_SIZE", "1")

    def test_plain(self):
        """Not declared in .phlop.exec.yaml: always runs once with defaults."""


class PythonTestTargetTest(unittest.TestCase):
    def _target(self, cmd):
        bits = shlex.split(cmd)
        idx = _python_invocation_index(bits, cmd)
        target, _ = _python_test_target(bits, idx, cmd)
        return target

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

    def test_next_idx_points_past_target(self):
        bits = shlex.split("python3 -u test.py TestClass.test_id")
        idx = _python_invocation_index(bits, "")
        target, next_idx = _python_test_target(bits, idx, "")
        self.assertEqual(target, "test.py")
        self.assertEqual(bits[next_idx:], ["TestClass.test_id"])

    def test_next_idx_at_end_when_no_trailing_args(self):
        bits = shlex.split("python3 -u test.py")
        idx = _python_invocation_index(bits, "")
        target, next_idx = _python_test_target(bits, idx, "")
        self.assertEqual(target, "test.py")
        self.assertEqual(bits[next_idx:], [])


class LoadPyTestCasesFromCmakeTest(unittest.TestCase):
    """Covers the ctest-registration path (as opposed to load_config_tests,
    which scans files off disk directly): the same .py file can be registered
    by cmake more than once with different trailing args, e.g. one entry per
    `Class.test_id`. Re-scanning the file for each such entry would both
    duplicate every test in the file per entry and discard the args that
    distinguish them, so a trailing arg means "run this exact command",
    not "expand this file"."""

    def test_trailing_test_id_is_treated_as_a_single_test(self):
        job = Job(
            cmd="python3 -u nonexistent_test_file.py SomeTest.test_it",
            working_dir=".",
        )
        self.assertIsNone(load_py_test_cases_from_cmake(job))

    def test_no_trailing_args_expands_the_file(self):
        job = Job(
            cmd="python3 -u tests/_phlop/run/test_test_cases.py",
            working_dir=".",
        )
        jobs = load_py_test_cases_from_cmake(job)
        self.assertTrue(any("PhlopGitTestCasesTest.test_fails" in j.cmd for j in jobs))


class DuplicateTestNameTest(unittest.TestCase):
    """https://github.com/PhilipDeegan/phlop/issues/59

    ctest entries like `python3 -um test_duplicate` in different working dirs share
    a bare module name - each must load its own file (and its own sibling
    imports), not whichever was imported first. Fixtures are in
    tests/_phlop/duplicate_test_name/{one,two}, which the default `-i tests` scan
    also collects and runs."""

    ROOT = Path("tests/_phlop/duplicate_test_name")
    CLASSES: ClassVar = {
        "one": "DuplicateTestNameOneTest",
        "two": "DuplicateTestNameTwoTest",
    }

    def _load(self, name):
        job = Job(cmd="python3 -um test_duplicate", working_dir=str(self.ROOT / name))
        return load_py_test_cases_from_cmake(job)

    def test_each_dir_gets_its_own_classes_and_logfile(self):
        jobs = {name: self._load(name) for name in self.CLASSES}

        for name, clazz in self.CLASSES.items():
            self.assertEqual(len(jobs[name]), 1)
            job = jobs[name][0]
            self.assertIn(f"test_duplicate {clazz}.test_run", job.cmd)
            self.assertIn(
                str(_log_subpath(self.ROOT / name) / "test_duplicate" / clazz),
                job.log_file_path,
            )

    def test_sibling_imports_are_not_shared(self):
        for name in self.CLASSES:
            with extend_sys_path([str(self.ROOT / name)]):
                (clazz,) = classes_in_file("test_duplicate.py", unittest.TestCase)
            self.assertEqual(clazz.test_run.__globals__["WHO"], name)

    def test_directory_scan(self):
        jobs = [j for b in load_config_tests(str(self.ROOT)) for j in b.tests]
        for name, clazz in self.CLASSES.items():
            (job,) = [j for j in jobs if clazz in j.cmd]
            self.assertIn(f"{name}.test_duplicate {clazz}.test_run", job.cmd)
            self.assertIn(
                str(_log_subpath(self.ROOT / name) / "test_duplicate" / clazz),
                job.log_file_path,
            )

    def test_modules_are_not_kept(self):
        self._load("one")
        self.assertNotIn("test_duplicate", sys.modules)
        self.assertNotIn("helper", sys.modules)


class ImportedTestCaseTest(unittest.TestCase):
    """A TestCase imported into a scanned file is collected only from the file
    that defines it, not again from every file that imports it."""

    def test_imported_classes_are_collected_once(self):
        root = "tests/_phlop/imported_test_case"
        cmds = [j.cmd for b in load_config_tests(root) for j in b.tests]
        self.assertEqual(
            sorted(c.split(" ", 2)[2] for c in cmds),
            [
                "tests._phlop.imported_test_case.base ImportedBaseTest.test_shared",
                "tests._phlop.imported_test_case.test_a ImportedATest.test_own",
                "tests._phlop.imported_test_case.test_a ImportedATest.test_shared",
                "tests._phlop.imported_test_case.test_b ImportedBTest.test_own",
                "tests._phlop.imported_test_case.test_b ImportedBTest.test_shared",
            ],
        )


class CliArgsTest(unittest.TestCase):
    def test_dump_requires_a_filepath(self):
        parser = cli_args_parser()
        self.assertEqual(parser.parse_args(["--dump", "out"]).dump, "out")
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["--dump"])
        self.assertEqual(ctx.exception.code, 2)


class TestBatchCoresTest(unittest.TestCase):
    """A batch's jobs all run with the batch's core count - anything else is a bug"""

    def test_mismatched_job_raises(self):
        with self.assertRaisesRegex(ValueError, "'b' \\(cores=4\\)"):
            TestBatch([Job(cmd="a", cores=2), Job(cmd="b", cores=4)], 2)

    def _load(self, batches):
        import codecs

        import dill

        with tempfile.TemporaryDirectory() as d:
            # same encoding as phlop.run.test_cases.dump_batches
            Path(d, "tests.dump").write_text(
                codecs.encode(
                    dill.dumps(TestBatchesList(batch_list=batches)), "hex"
                ).decode()
            )
            loaded = extract_load(d, "tests.dump")
        return {b.cores: sorted(j.cmd for j in b.tests) for b in loaded}

    def test_load_round_trip(self):
        batches = [
            TestBatch([Job(cmd="a"), Job(cmd="b")], 1),
            TestBatch([Job(cmd="c", cores=4)], 4),
            TestBatch([Job(cmd="d", cores=4)], 4),
        ]
        self.assertEqual(self._load(batches), {1: ["a", "b"], 4: ["c", "d"]})

    def test_load_rejects_mismatched_batch(self):
        batch = TestBatch([Job(cmd="a", cores=4)], 4)
        batch.cores = 2  # e.g. a dump written by something other than phlop
        with self.assertRaisesRegex(ValueError, "'a' \\(cores=4\\)"):
            self._load([batch])


class ResolveCoresTest(unittest.TestCase):
    def test_explicit_wins_over_mpirun(self):
        self.assertEqual(resolve_cores("mpirun -n 8 foo", explicit=2), 2)

    def test_mpirun_dash_n(self):
        self.assertEqual(resolve_cores("mpirun -n 4 foo"), 4)

    def test_mpirun_dash_np(self):
        self.assertEqual(resolve_cores("mpirun -np 6 foo"), 6)

    def test_mpirun_double_dash_np(self):
        self.assertEqual(resolve_cores("mpirun --np 3 foo"), 3)

    def test_omp_num_threads_env(self):
        self.assertEqual(resolve_cores("foo", {"OMP_NUM_THREADS": "5"}), 5)

    def test_mpirun_and_omp_num_threads_combine(self):
        # 2 mpi ranks * 4 threads/rank = 8 total cores
        self.assertEqual(resolve_cores("mpirun -n 2 foo", {"OMP_NUM_THREADS": "4"}), 8)

    def test_explicit_wins_over_combined_mpirun_and_omp(self):
        cores = resolve_cores("mpirun -n 2 foo", {"OMP_NUM_THREADS": "4"}, explicit=3)
        self.assertEqual(cores, 3)

    def test_threads_alone_sets_cores(self):
        # no mpirun/OMP signal: threads (e.g. a std::thread_pool size) is the
        # total core count on its own
        self.assertEqual(resolve_cores("foo", threads=4), 4)

    def test_threads_multiplies_with_mpirun(self):
        # 2 mpi ranks * 4 threads/rank = 8 total cores
        self.assertEqual(resolve_cores("mpirun -n 2 foo", threads=4), 8)

    def test_explicit_wins_over_threads(self):
        cores = resolve_cores("mpirun -n 2 foo", threads=4, explicit=3)
        self.assertEqual(cores, 3)

    def test_defaults_to_one_when_uninferrable(self):
        self.assertEqual(resolve_cores("foo"), 1)


class VariantsForTest(unittest.TestCase):
    def test_qualified_key_takes_precedence(self):
        dir_config = {
            "test_foo.py:MyTestCase": [{"tags": ["class"]}],
            "test_foo.py:MyTestCase.test_a": [{"tags": ["method"]}],
        }
        self.assertEqual(
            _variants_for(dir_config, "test_foo.py", "MyTestCase", "test_a"),
            [{"tags": ["method"]}],
        )

    def test_bare_class_key_applies_to_other_methods(self):
        dir_config = {"test_foo.py:MyTestCase": [{"tags": ["class"]}]}
        self.assertEqual(
            _variants_for(dir_config, "test_foo.py", "MyTestCase", "test_b"),
            [{"tags": ["class"]}],
        )

    def test_file_default_key_applies_to_every_class_and_method(self):
        dir_config = {"test_foo.py": [{"tags": ["file"]}]}
        self.assertEqual(
            _variants_for(dir_config, "test_foo.py", "MyTestCase", "test_a"),
            [{"tags": ["file"]}],
        )
        self.assertEqual(
            _variants_for(dir_config, "test_foo.py", "OtherTestCase", "test_z"),
            [{"tags": ["file"]}],
        )

    def test_bare_class_key_takes_precedence_over_file_default(self):
        dir_config = {
            "test_foo.py": [{"tags": ["file"]}],
            "test_foo.py:MyTestCase": [{"tags": ["class"]}],
        }
        self.assertEqual(
            _variants_for(dir_config, "test_foo.py", "MyTestCase", "test_a"),
            [{"tags": ["class"]}],
        )
        self.assertEqual(
            _variants_for(dir_config, "test_foo.py", "OtherTestCase", "test_a"),
            [{"tags": ["file"]}],
        )

    def test_empty_list_means_default_run(self):
        dir_config = {"test_foo.py:MyTestCase.test_a": []}
        self.assertIsNone(
            _variants_for(dir_config, "test_foo.py", "MyTestCase", "test_a")
        )

    def test_no_entry_means_default_run(self):
        self.assertIsNone(_variants_for({}, "test_foo.py", "MyTestCase", "test_a"))

    def test_same_class_name_in_different_file_is_not_confused(self):
        dir_config = {"test_foo.py:MyTestCase.test_a": [{"tags": ["foo"]}]}
        self.assertIsNone(
            _variants_for(dir_config, "test_bar.py", "MyTestCase", "test_a")
        )


class LoadConfigTestsTest(unittest.TestCase):
    """Exercises the sample tests/_phlop/run/.phlop.exec.yaml fixture end-to-end."""

    def _jobs(self, tags=None):
        # classes_in_file derives a module name from this path, so it must be
        # relative to CWD (as the rest of this tooling always assumes) rather
        # than absolute.
        batches = load_config_tests("tests/_phlop/run", tags=tags)
        return [job for batch in batches for job in batch.tests]

    def _by_cmd_substr(self, jobs, substr):
        return [j for j in jobs if substr in j.cmd]

    def test_undeclared_test_runs_once_with_defaults(self):
        jobs = self._by_cmd_substr(self._jobs(), "ExecConfigDemoTest.test_plain")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].cores, 1)
        self.assertEqual(jobs[0].env, {})

    def test_tagged_variants_are_opt_in_by_default(self):
        # Neither variant is untagged, and no --tags was requested, so this
        # mpirun-dependent test stays off unless explicitly opted into - it
        # must not run as part of a plain, untagged scan (e.g. CI without
        # mpirun installed).
        jobs = self._by_cmd_substr(self._jobs(), "ExecConfigDemoTest.test_scaled")
        self.assertEqual(jobs, [])

        # undeclared tests are unaffected and always run
        plain = self._by_cmd_substr(self._jobs(), "ExecConfigDemoTest.test_plain")
        self.assertEqual(len(plain), 1)

    def test_tags_opts_into_matching_variant_only(self):
        jobs = self._by_cmd_substr(self._jobs(tags=["small"]), "ExecConfigDemoTest")
        scaled = [j for j in jobs if "test_scaled" in j.cmd]
        plain = [j for j in jobs if "test_plain" in j.cmd]

        self.assertEqual(len(scaled), 1)
        self.assertEqual(scaled[0].env.get("DEMO_GRID_SIZE"), "64")
        self.assertEqual(scaled[0].cores, 1)
        self.assertIsNone(scaled[0].working_dir)
        self.assertTrue(scaled[0].log_file_path.endswith("_small"))
        # undeclared/untagged tests are unaffected by --tags
        self.assertEqual(len(plain), 1)

    def test_tags_opts_into_mpirun_and_threads_variant(self):
        jobs = self._by_cmd_substr(self._jobs(tags=["high_load"]), "test_scaled")
        self.assertEqual(len(jobs), 1)

        large = jobs[0]
        self.assertEqual(large.env.get("DEMO_GRID_SIZE"), "256")
        # 2 mpi ranks (from prefix) * 4 threads/rank (declared) = 8 total cores
        self.assertEqual(large.cores, 8)
        self.assertIn("mpirun -n 2", large.cmd)
        self.assertEqual(large.working_dir, "tests/_phlop/run")
        self.assertTrue(large.log_file_path.endswith("_large-high_load"))

    def test_tags_opts_into_thread_pool_variant(self):
        jobs = self._by_cmd_substr(self._jobs(tags=["thread_pool"]), "test_scaled")
        self.assertEqual(len(jobs), 1)

        pool = jobs[0]
        self.assertEqual(pool.env.get("DEMO_GRID_SIZE"), "128")
        # no mpirun in this variant: threads is the total core count directly
        self.assertEqual(pool.cores, 4)
        self.assertNotIn("mpirun", pool.cmd)
        self.assertIsNone(pool.working_dir)
        self.assertTrue(pool.log_file_path.endswith("_thread_pool"))

    def test_unmatched_tag_yields_nothing(self):
        jobs = self._by_cmd_substr(self._jobs(tags=["nonexistent"]), "test_scaled")
        self.assertEqual(jobs, [])


class LoadConfigTestsValidationTest(unittest.TestCase):
    """A directory's .phlop.exec.yaml keys must match something real found
    while scanning it - a stale/mistyped key (like the "eq" project's, which
    used a "." instead of a ":" and named a class no longer present) should
    fail loudly rather than silently doing nothing.

    classes_in_file needs a path relative to CWD, so this uses a scratch
    directory under the repo (not tempfile.TemporaryDirectory, which is
    absolute). It lives under tests/_phlop/ rather than tests/_phlop/run/,
    since LoadConfigTestsTest globs tests/_phlop/run recursively at runtime -
    nesting there would let this class's deliberately-invalid config get
    picked up mid-run by a concurrently-executing LoadConfigTestsTest method
    and fail it with an unrelated ValueError. Each test method also gets its
    own scratch directory (rather than sharing one at the class level), since
    phlop's own runner turns each method into a separate process that can run
    concurrently with its siblings - a shared directory races between them."""

    def setUp(self):
        self._scratch = Path(
            f"tests/_phlop/_scratch_config_validation_{self._testMethodName}"
        )
        self._module = (
            "tests._phlop."
            f"_scratch_config_validation_{self._testMethodName}.test_scratch"
        )
        self._scratch.mkdir(exist_ok=True)
        (self._scratch / "__init__.py").write_text("")
        (self._scratch / "test_scratch.py").write_text(
            "import unittest\n\n\n"
            "class ScratchTest(unittest.TestCase):\n"
            "    def test_it(self):\n"
            "        pass\n"
        )

    def tearDown(self):
        shutil.rmtree(self._scratch, ignore_errors=True)
        sys.modules.pop(self._module, None)

    def _write_config(self, text):
        (self._scratch / ".phlop.exec.yaml").write_text(text)

    def test_raises_on_stale_config_key(self):
        self._write_config("test_scratch.py:NoSuchClass.test_missing:\n  - tags: [x]\n")
        with self.assertRaises(ValueError):
            load_config_tests(str(self._scratch))

    def test_no_error_when_all_keys_match(self):
        self._write_config("test_scratch.py:ScratchTest.test_it:\n  - tags: [x]\n")
        load_config_tests(str(self._scratch))  # should not raise


class LogSubpathTest(unittest.TestCase):
    def test_path_inside_log_dir_is_relative(self):
        with mock.patch("phlop.testing.test_cases._LOG_DIR", Path("/a/b")):
            self.assertEqual(_log_subpath("/a/b/c/d"), Path("c/d"))

    def test_path_outside_log_dir_keeps_full_path(self):
        with mock.patch("phlop.testing.test_cases._LOG_DIR", Path("/a/b")):
            self.assertEqual(_log_subpath("/x/y/z"), Path("x/y/z"))


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
        rt = RunTimer("echo hi", shell=True, capture_output=False, popen=True)
        self.assertIsNone(rt.stdout)
        self.assertIsNone(rt.stderr)
        self.assertEqual(rt.exitcode, 0)


class RunTimerStallTest(unittest.TestCase):
    def _run(self, script, stall_timeout, stall_busy_timeout=None):
        with tempfile.TemporaryDirectory() as tmp:
            return RunTimer(
                [sys.executable, "-uc", script],
                capture_output=False,
                log_file_path=os.path.join(tmp, "job"),
                stall_timeout=stall_timeout,
                stall_busy_timeout=stall_busy_timeout,
            )

    def test_silent_process_without_psutil_is_killed(self):
        with mock.patch.dict(sys.modules, {"psutil": None}):
            rt = self._run("import time; time.sleep(60)", stall_timeout=1)
        self.assertEqual(rt.stalled, "no output for 1s")

    def test_silent_busy_process_gets_busy_timeout(self):
        if not _has_psutil():
            self.skipTest("psutil unavailable")
        busy = "import time\nend = time.time() + {}\nwhile time.time() < end: pass"
        rt = self._run(busy.format(3), stall_timeout=1, stall_busy_timeout=10)
        self.assertFalse(rt.stalled)
        self.assertEqual(rt.exitcode, 0)

        rt = self._run(busy.format(60), stall_timeout=1, stall_busy_timeout=3)
        self.assertEqual(rt.stalled, "no output for 3s, busy")

    def test_silent_process_is_killed(self):
        rt = self._run("import time; time.sleep(60)", stall_timeout=1)
        self.assertTrue(rt.stalled)
        self.assertNotEqual(rt.exitcode, 0)
        self.assertLess(rt.run_time, 30)

    def test_process_with_output_is_not_killed(self):
        script = "import time\nfor i in range(15):\n print(i)\n time.sleep(0.2)"
        rt = self._run(script, stall_timeout=1)
        self.assertFalse(rt.stalled)
        self.assertEqual(rt.exitcode, 0)

    def _run_no_log(self, script, stall_timeout, stall_busy_timeout=None):
        return RunTimer(
            [sys.executable, "-c", script],
            capture_output=False,
            stall_timeout=stall_timeout,
            stall_busy_timeout=stall_busy_timeout,
        )

    def test_no_log_file_without_psutil_is_not_monitored(self):
        with mock.patch.dict(sys.modules, {"psutil": None}):
            rt = self._run_no_log("import time; time.sleep(2)", stall_timeout=1)
        self.assertFalse(rt.stalled)
        self.assertEqual(rt.exitcode, 0)

    def test_no_log_file_idle_process_is_killed(self):
        if not _has_psutil():
            self.skipTest("psutil unavailable")
        rt = self._run_no_log("import time; time.sleep(60)", stall_timeout=1)
        self.assertIn("cpu", rt.stalled)
        self.assertLess(rt.run_time, 30)

    def test_no_log_file_busy_process_gets_busy_timeout(self):
        if not _has_psutil():
            self.skipTest("psutil unavailable")
        busy = "import time\nend = time.time() + {}\nwhile time.time() < end: pass"
        rt = self._run_no_log(busy.format(3), stall_timeout=1, stall_busy_timeout=10)
        self.assertFalse(rt.stalled)
        self.assertEqual(rt.exitcode, 0)

        rt = self._run_no_log(busy.format(60), stall_timeout=1, stall_busy_timeout=3)
        self.assertIn("for 3s, busy", rt.stalled)
        self.assertLess(rt.run_time, 30)

    def test_descendant_ignoring_sigterm_is_killed(self):
        if not _has_psutil():
            self.skipTest("psutil unavailable")
        import psutil

        child = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        script = (
            "import subprocess, sys, time\n"
            f"c = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
            "print(c.pid, flush=True)\n"
            "time.sleep(60)"
        )
        # captured output: the surviving child would hold the pipes open
        with mock.patch("phlop.procs.runtimer.STALL_KILL_GRACE", 2):
            rt = RunTimer(
                [sys.executable, "-c", script],
                capture_output=True,
                stall_timeout=1,
            )
        self.assertTrue(rt.stalled)
        self.assertLess(rt.run_time, 30)
        child_pid = int(rt.stdout.split()[0])
        # SIGKILL is async, the pipes can close before the child is fully dead
        deadline = time.monotonic() + 1
        while True:
            try:
                child_alive = psutil.Process(child_pid).status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                child_alive = False
            if not child_alive or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        self.assertFalse(child_alive)


def _has_psutil():
    try:
        import psutil  # noqa: F401

        return True
    except ImportError:
        return False


if __name__ == "__main__":
    unittest.main()
