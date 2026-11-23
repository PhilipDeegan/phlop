# tests/_phlop/run/test_exec_scope.py
#
# .phlop.exec.yaml scopes, chaining and phases: phlop/testing/exec_scope.py and
#  load_config_tests in phlop/testing/test_cases.py

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phlop.procs.parallel_processor import Job, ProcessorFailure
from phlop.testing import parallel_processor as pp
from phlop.testing import test_cases as tc
from phlop.testing.exec_scope import nest, phase_fixtures, phase_label


class NestTest(unittest.TestCase):
    def test_inner_replaces(self):
        outer = {"prefix": "timeout 60", "cores": 2, "tags": ["a"]}
        self.assertEqual(
            nest(outer, {"prefix": "mpirun -n 2", "cores": 4, "tags": ["b"]}),
            {"prefix": "mpirun -n 2", "cores": 4, "tags": ["b"]},
        )

    def test_outer_value_by_name(self):
        outer = {"prefix": "timeout 60", "tags": ["a"]}
        inner = {"prefix": "${prefix} mpirun -n 2", "tags": ["${tags}", "b"]}
        self.assertEqual(
            nest(outer, inner),
            {"prefix": "timeout 60 mpirun -n 2", "tags": ["a", "b"]},
        )
        self.assertEqual(nest({}, {"prefix": "${prefix} x"})["prefix"], " x")

    def test_env_per_variable(self):
        outer = {"env": {"A": "1", "B": "2"}}
        env = nest(outer, {"env": {"A": "${A}:3", "C": "4"}})["env"]
        self.assertEqual(env, {"A": "1:3", "B": "2", "C": "4"})

    def test_env_falls_back_to_process(self):
        with mock.patch.dict(os.environ, {"PHLOP_NEST_TEST": "p"}):
            env = nest({}, {"env": {"PHLOP_NEST_TEST": "x:${PHLOP_NEST_TEST}"}})
        self.assertEqual(env["env"], {"PHLOP_NEST_TEST": "x:p"})

    def test_other_keys_kept(self):
        self.assertEqual(nest({}, {"requires": ["a"]}), {"requires": ["a"]})


class BracesTest(unittest.TestCase):
    def test_expanded(self):
        self.assertEqual(
            tc._braces("b/{x,y}/t{1,2}"), ["b/x/t1", "b/x/t2", "b/y/t1", "b/y/t2"]
        )
        self.assertEqual(tc._braces("a{b,{c,d}}e"), ["abe", "ace", "ade"])
        self.assertEqual(tc._braces("{,x}y"), ["y", "xy"])

    def test_no_alternatives_kept(self):
        self.assertEqual(tc._braces("a{b}c"), ["a{b}c"])
        self.assertEqual(tc._braces("a{b"), ["a{b"])


class LogSubpathTest(unittest.TestCase):
    def test_outside_log_dir_relative_to_cwd(self):
        with mock.patch("phlop.testing.test_cases._LOG_DIR", Path("/nowhere")):
            self.assertEqual(tc._log_subpath("bin/x"), Path("bin/x"))


class ActiveLogTest(unittest.TestCase):
    def test_running_job_tail_from_its_temp_log(self):
        import contextlib
        import io

        from phlop.procs.parallel_processor import _print_jobs

        with tempfile.TemporaryDirectory() as tmp:
            job = Job(cmd="x", log_file_path=os.path.join(tmp, "final"))
            job.active_log_path = os.path.join(tmp, "active")
            Path(f"{job.active_log_path}.stdout").write_text("live line\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                _print_jobs([job], verbose=True)
            self.assertIn("live line", out.getvalue())


class StallEnvTest(unittest.TestCase):
    def test_job_env_first(self):
        job = Job(cmd="x", env={"PHLOP_STALL_TIMEOUT": "60"})
        self.assertEqual(job._stall("PHLOP_STALL_TIMEOUT", 3600), 60)
        self.assertEqual(job._stall("PHLOP_STALL_BUSY_TIMEOUT", 7), 7)

    def test_job_killed_by_its_env_timeout(self):
        job = Job(cmd="sleep 30", env={"PHLOP_STALL_TIMEOUT": "1"})
        with tempfile.TemporaryDirectory() as tmp:
            job.log_file_path = os.path.join(tmp, "job")
            with self.assertRaisesRegex(ProcessorFailure, "killed"):
                job()  # idle: killed with or without psutil


class MpirunCoresTest(unittest.TestCase):
    def test_options_before_n(self):
        for cmd, n in (
            ("mpirun --oversubscribe -n 8 ./x", 8),
            ("mpirun --bind-to none -np 4 ./x", 4),
            ("mpirun --np=3 ./x", 3),
            ("mpirun -n 2 ./x -n 5", 2),  # the program's own -n is not mpirun's
            ("./x -n 5", 1),
            ("mpiexec -n 6 ./x", 6),
            ("mpiexec --bind-to core -np 4 ./x", 4),
            ("srun -n 3 ./x", 3),
            ("srun --exclusive --ntasks=5 ./x", 5),
            ("srun -n7 ./x", 7),
        ):
            self.assertEqual(tc.resolve_cores(cmd), n, cmd)
        self.assertEqual(tc._launch_dir("mpirun --oversubscribe -n 8 x"), "mpirun_n_8")
        self.assertEqual(tc._launch_dir("mpiexec -n 2 x"), "mpiexec_n_2")
        self.assertEqual(tc._launch_dir("srun --ntasks 4 x"), "srun_n_4")


class EnvStringsTest(unittest.TestCase):
    def test_yaml_scalars(self):
        env = Job(cmd="x", env={"A": 60, "B": 1.5, "C": True, "D": "s"}).env
        self.assertEqual(env, {"A": "60", "B": "1.5", "C": "true", "D": "s"})

    def test_other_values_name_the_key(self):
        with self.assertRaisesRegex(TypeError, "BAD"):
            Job(cmd="x", env={"BAD": [1]})


def _job(cmd, phase):
    job = Job(cmd=cmd)
    job.phase = phase
    return job


class PhaseFixturesTest(unittest.TestCase):
    def test_waits_for_previous_strict(self):
        a = _job("a", [("f", 0, None, False)])
        b = _job("b", [("f", 1, "one", False)])
        c = _job("c", [("f", 3, "three", False)])  # phase 2 empty
        phase_fixtures([a, b, c])
        self.assertEqual(b.requires, {"phase f#0"})
        self.assertEqual(c.requires, {"phase f#1"})
        self.assertEqual(a.provides, {"phase f#0"})
        self.assertEqual(phase_label(c.phase), "three")

    def test_ignore_errors_cleans(self):
        a = _job("a", [("f", 1, "one", False)])
        b = _job("b", [("f", 2, "two", True)])
        c = _job("c", [("f", 3, "three", False)])
        phase_fixtures([a, b, c])
        self.assertEqual(c.cleans, {"phase f#2"})  # waits, whatever b's outcome
        self.assertEqual(c.requires, {"phase f#1"})  # a failed: c fails unrun

    def test_nested_lists(self):
        a = _job("a", [("f", 1, "one", False), ("g", 1, "x", False)])
        b = _job("b", [("f", 1, "one", False), ("g", 2, "y", False)])
        phase_fixtures([a, b])
        self.assertEqual(b.requires, {"phase g#1"})
        self.assertEqual(phase_label(b.phase), "one/y")


class PhaseRunTest(unittest.TestCase):
    """phases run in order through the processor"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_jobs(self, jobs):
        pp.process(jobs, n_cores=4, logging=0)

    def touch(self, name, phase, fail=False):
        path = Path(self.tmp.name) / name
        return _job(f"sh -c 'touch {path}; exit {int(fail)}'", phase)

    def check(self, name, needs, phase):
        need = Path(self.tmp.name) / needs
        path = Path(self.tmp.name) / name
        return _job(f"sh -c 'test -e {need} && touch {path}'", phase)

    def test_next_phase_after_previous(self):
        self.run_jobs(
            [
                self.check("b", "a", [("f", 2, "two", False)]),
                _job("sleep 0.2", [("f", 1, "one", False)]),
                self.touch("a", [("f", 1, "one", False)]),
            ]
        )
        self.assertTrue((Path(self.tmp.name) / "b").exists())

    def test_failing_phase_stops(self):
        with self.assertRaises(ProcessorFailure):
            self.run_jobs(
                [
                    self.touch("a", [("f", 1, "one", False)], fail=True),
                    self.touch("b", [("f", 2, "two", False)]),
                ]
            )
        self.assertFalse((Path(self.tmp.name) / "b").exists())

    def test_ignore_errors_goes_on(self):
        with self.assertRaises(ProcessorFailure):  # still failed overall
            self.run_jobs(
                [
                    self.touch("a", [("f", 1, "one", True)], fail=True),
                    self.check("b", "a", [("f", 2, "two", False)]),
                ]
            )
        self.assertTrue((Path(self.tmp.name) / "b").exists())


class ConfigChainTest(unittest.TestCase):
    """configs in a scratch package under tests/_phlop, see
    LoadConfigTestsValidationTest (relative imports, concurrent methods)"""

    def setUp(self):
        name = f"_scratch_scope_{self._testMethodName}"
        self.dir = Path(f"tests/_phlop/{name}")
        self.package = f"tests._phlop.{name}"
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        for sub in ("", "sub", "sub/deep", "other"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
            (self.dir / sub / "__init__.py").write_text("")
        for sub, module in (
            ("", "test_top"),
            ("sub", "test_sub"),
            ("sub/deep", "test_deep"),
            ("other", "test_other"),
        ):
            clazz = module.title().replace("_", "")
            (self.dir / sub / f"{module}.py").write_text(
                f"import unittest\n\n\nclass {clazz}(unittest.TestCase):\n"
                "    def test_a(self):\n        pass\n"
            )
            dotted = ".".join(p for p in (self.package, *sub.split("/"), module) if p)
            self.addCleanup(sys.modules.pop, dotted, None)
        (self.dir / "bin").mkdir()
        for exe in ("one.sh", "two.sh"):
            (self.dir / "bin" / exe).write_text("#!/bin/sh\n")
            (self.dir / "bin" / exe).chmod(0o755)
        (self.dir / "bin" / "not_exe.sh").write_text("#!/bin/sh\n")
        importlib.invalidate_caches()

    def config(self, text, sub=""):
        (self.dir / sub / ".phlop.exec.yaml").write_text(text)

    def jobs(self, **kw):
        batches = tc.load_config_tests(str(self.dir), **kw)
        return [j for b in batches for j in b.tests]

    def by_name(self, **kw):
        """{test or binary name: [jobs]}"""
        found = {}
        for job in self.jobs(**kw):
            key = job.test_names[0] if job.test_names else Path(job.cmd).name
            found.setdefault(key, []).append(job)
        return found

    def test_phases_and_scopes(self):
        self.config(
            "env: {A: '1'}\n"
            "phases:\n"
            "  - name: bins\n"
            "    bin/*.sh:\n"
            "      - {}\n"
            "      - {prefix: 'mpirun -n 2'}\n"
            "  - name: subs\n"
            "    env: {A: '${A}2'}\n"
            "    path: sub\n"
        )
        jobs = self.by_name()
        self.assertEqual(
            sorted(jobs),
            [
                "TestDeep.test_a",
                "TestOther.test_a",
                "TestSub.test_a",
                "TestTop.test_a",
                "one.sh",
                "two.sh",
            ],
        )
        root = str(self.dir / ".phlop.exec.yaml")
        (top,) = jobs["TestTop.test_a"]
        self.assertEqual(top.phase, [(root, 0, None, False)])
        self.assertEqual(top.env, {"A": "1"})
        one = jobs["one.sh"]
        self.assertEqual([j.cores for j in one], [1, 2])
        self.assertEqual({phase_label(j.phase) for j in one}, {"bins"})
        self.assertNotEqual(one[0].log_file_path, one[1].log_file_path)
        (sub,) = jobs["TestSub.test_a"]  # the path, not again as a subdirectory
        self.assertEqual(phase_label(sub.phase), "subs")
        self.assertEqual(sub.env, {"A": "12"})
        (deep,) = jobs["TestDeep.test_a"]  # in sub's scope
        self.assertEqual(phase_label(deep.phase), "subs")
        (other,) = jobs["TestOther.test_a"]  # unphased
        self.assertEqual(other.phase, [(root, 0, None, False)])

    def test_select_phases(self):
        self.config("phases:\n  - name: bins\n    bin/one.sh: [{}]\n  - path: sub\n")
        self.assertEqual(sorted(self.by_name(phases=["bins"])), ["one.sh"])
        self.assertEqual(
            sorted(self.by_name(phases=["2"])), ["TestDeep.test_a", "TestSub.test_a"]
        )
        with self.assertRaisesRegex(ValueError, "nope"):
            self.jobs(phases=["nope"])

    def test_no_phases_to_select(self):
        self.config("bin/one.sh: [{}]\n")
        with self.assertRaisesRegex(ValueError, "has none"):
            self.jobs(phases=["1"])

    def test_nested_scope_extends(self):
        self.config("prefix: timeout 60\ntags: [slow]\n")
        self.config("prefix: ${prefix} nice\ntags: ['${tags}', deep]\n", "sub/deep")
        self.assertNotIn("TestDeep.test_a", self.by_name())  # tagged: opt in
        (deep,) = self.by_name(tags=["deep"])["TestDeep.test_a"]
        self.assertTrue(deep.cmd.startswith("timeout 60 nice python3"), deep.cmd)
        self.assertEqual(deep.meta, ["slow", "deep"])

    def test_python_key_in_phase(self):
        self.config(
            "phases:\n  - name: late\n    test_top.py:TestTop.test_a:\n"
            "      - {env: {B: '1'}}\n"
        )
        (top,) = self.by_name()["TestTop.test_a"]
        self.assertEqual((phase_label(top.phase), top.env), ("late", {"B": "1"}))

    def test_python_file_path(self):
        self.config("phases:\n  - name: one\n    path: other/test_other.py\n")
        self.config("test_other.py:\n  - {env: {C: '1'}}\n", "other")
        (other,) = self.by_name()["TestOther.test_a"]
        self.assertEqual((phase_label(other.phase), other.env), ("one", {"C": "1"}))

    def test_config_file_path(self):
        (self.dir / "sub" / "alt.yaml").write_text("env: {D: '1'}\n")
        self.config("path: sub/alt.yaml\n")
        (sub,) = self.by_name()["TestSub.test_a"]
        self.assertEqual(sub.env, {"D": "1"})

    def test_cycle(self):
        self.config("path: sub\n")
        self.config("path: ..\n", "sub")
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.jobs()

    def test_reached_twice_runs_once(self):
        self.config("paths: [sub, sub]\n")
        self.assertEqual(len(self.by_name()["TestSub.test_a"]), 1)

    def test_stale_executable(self):
        self.config("bin/not_exe.sh: [{}]\n")
        with self.assertRaisesRegex(ValueError, "not_exe"):
            self.jobs()

    def test_stale_in_chained_file(self):
        self.config("path: sub\n")
        self.config("test_sub.py:Nope: [{}]\n", "sub")
        with self.assertRaisesRegex(ValueError, "Nope"):
            self.jobs()

    def test_executable_paths(self):
        self.config(
            "sub/../bin/{one,two}.sh:\n"
            "  - {tags: [mpi], prefix: mpirun -n 2}\n"
            "  - {tags: [mpi], prefix: mpirun -n 8}\n"
            "  - {}\n"
            "  - {}\n"
        )
        jobs = self.by_name(tags=["mpi"])
        one = jobs["one.sh"]
        self.assertEqual(len(jobs["two.sh"]), 4)
        self.assertEqual(one[2].cmd, str(self.dir / "bin" / "one.sh"))  # no sub/..
        logs = [Path(j.log_file_path).name for j in one]
        self.assertEqual(
            logs,
            [
                "one.sh_mpi-mpirun_n_2",
                "one.sh_mpi-mpirun_n_8",
                "one.sh_2",
                "one.sh_3",
            ],
        )

    def test_own_python_file_in_paths(self):
        self.config("phases:\n  - name: late\n    path: test_top.py\n")
        (top,) = self.by_name()["TestTop.test_a"]  # no cycle, runs once
        self.assertEqual(phase_label(top.phase), "late")

    def test_braces_before_normalising(self):
        self.config("'{sub/../bin,bin}/one.sh': [{}]\n")
        self.assertEqual(len(self.by_name()["one.sh"]), 1)

    def test_bare_executable_not_from_path(self):
        (self.dir / "bin" / ".phlop.exec.yaml").write_text("one.sh: [{}]\n")
        here = os.getcwd()
        os.chdir(self.dir / "bin")
        self.addCleanup(os.chdir, here)
        (job,) = [j for b in tc.load_config_tests(".") for j in b.tests]
        self.assertEqual(job.cmd, "./one.sh")

    def test_ranks_from_launcher_only(self):
        self.config(
            "bin/one.sh:\n"
            "  - {prefix: mpirun --oversubscribe, postfix: -n 5}\n"
            "  - {prefix: mpirun --oversubscribe -n 2, postfix: -n 5}\n"
        )
        cores = [j.cores for j in self.jobs() if "one.sh -n 5" in j.cmd]
        self.assertEqual(cores, [1, 2])

    def test_missing_path(self):
        self.config("path: nowhere\n")
        with self.assertRaisesRegex(ValueError, "nowhere"):
            self.jobs()


if __name__ == "__main__":
    unittest.main()
