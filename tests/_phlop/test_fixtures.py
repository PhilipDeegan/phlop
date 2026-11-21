# tests/_phlop/test_fixtures.py
#
# ctest fixtures (phlop.procs.fixtures): a job requiring a fixture runs after its
#  providers, fails unrun if one failed, a missing provider is ignored, cycles are
#  errors, providers start first, cleanups wait for every user; parsed from ctest's
#  json, carried onto every job split from an entry, merged across --load dumps,
#  kept with what --regex keeps

import codecs
import importlib
import os
import shlex
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phlop.app.cmake import CTest_test
from phlop.procs.fixtures import FixtureCycleError, FixtureGraph
from phlop.procs.parallel_processor import Job, ProcessorFailure, process
from phlop.run.test_cases import cli_args_parser, filter_out_regex_fails
from phlop.testing import test_cases as tc


def py(code):
    return shlex.join([sys.executable, "-c", code])


def run(jobs, n_cores=4):
    """(ids finished in order, ids failed with their error)"""
    finished, failed = [], {}
    try:
        process(
            jobs,
            n_cores=n_cores,
            fail_fast=False,
            on_success=lambda r: finished.append(r.job.id),
            on_failure=lambda r: failed.update({r.job.id: str(r.error)}),
        )
    except ProcessorFailure:
        pass
    return finished, failed


class SchedulingTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.data = Path(self.dir.name) / "data"

    def tearDown(self):
        self.dir.cleanup()

    def writer(self, **kw):
        code = f"import time; time.sleep(0.5); open({str(self.data)!r}, 'w').close()"
        return Job(cmd=py(code), id="writer", **kw)

    def reader(self, id="reader", **kw):
        code = f"import os, sys; sys.exit(not os.path.exists({str(self.data)!r}))"
        return Job(cmd=py(code), id=id, **kw)

    def test_dependent_runs_after_provider(self):
        jobs = [self.reader(requires={"data"}), self.writer(provides={"data"})]
        finished, failed = run(jobs)
        self.assertEqual(failed, {})
        self.assertEqual(finished, ["writer", "reader"])

    def test_without_fixtures_reader_runs_alongside(self):  # the bug fixed
        _, failed = run([self.reader(), self.writer()])
        self.assertIn("reader", failed)

    def test_failed_provider_fails_dependents(self):
        jobs = [
            Job(cmd=py("import sys; sys.exit(1)"), id="gen", provides={"a"}),
            Job(cmd="true", id="mid", requires={"a"}, provides={"b"}),
            Job(cmd="true", id="leaf", requires={"b"}),
            Job(cmd="true", id="other"),
            Job(cmd="true", id="cleanup", cleans={"a"}),
        ]
        finished, failed = run(jobs)
        self.assertEqual(sorted(finished), ["cleanup", "other"])
        self.assertEqual(failed["mid"], "fixture a failed (gen)")
        self.assertEqual(
            failed["leaf"], "fixture b failed (mid: fixture a failed (gen))"
        )
        self.assertIn("exited 1", failed["gen"])

    def test_cleanup_waits_for_every_user(self):
        code = f"import os, sys; os.remove({str(self.data)!r})"
        jobs = [
            Job(cmd=py(code), id="cleanup", cleans={"data"}),
            self.reader(requires={"data"}),
            self.reader(id="reader2", requires={"data"}),
            self.writer(provides={"data"}),
        ]
        finished, failed = run(jobs)
        self.assertEqual(failed, {})
        self.assertEqual(finished[0], "writer")
        self.assertEqual(finished[-1], "cleanup")

    def test_missing_provider_is_ignored(self):
        finished, failed = run([Job(cmd="true", id="j", requires={"nobody"})])
        self.assertEqual((finished, failed), (["j"], {}))

    def test_cycle_is_an_error(self):
        jobs = [
            Job(cmd="true", id="a", provides={"x"}, requires={"y"}),
            Job(cmd="true", id="b", provides={"y"}, requires={"x"}),
        ]
        with self.assertRaisesRegex(FixtureCycleError, "a -> b -> a|b -> a -> b"):
            process(jobs, n_cores=2)

    def test_self_reference_is_not_a_cycle(self):
        jobs = [Job(cmd="true", id="a", provides={"x"}, requires={"x"})]
        self.assertEqual(run(jobs), (["a"], {}))

    def test_providers_start_before_wider_jobs(self):
        jobs = [
            Job(cmd=py("import time; time.sleep(0.5)"), id="wide", cores=2),
            self.reader(requires={"data"}),
            self.writer(provides={"data"}),
        ]
        finished, failed = run(jobs, n_cores=2)
        self.assertEqual(failed, {})
        self.assertEqual(finished[0], "writer")

    def test_priority_counts_transitive_dependents(self):
        jobs = [
            Job(cmd="a", provides={"a"}),
            Job(cmd="b", requires={"a"}, provides={"b"}),
            Job(cmd="c", requires={"b"}),
            Job(cmd="d", requires={"b"}),
        ]
        self.assertEqual(FixtureGraph(jobs).priority, [3, 2, 0, 0])


class FailedProviderLogTest(unittest.TestCase):
    def test_dependent_names_the_providers_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = str(Path(d) / "gen")
            jobs = [
                Job(cmd="false", id="gen", provides={"a"}, log_file_path=log),
                Job(cmd="true", id="use", requires={"a"}),
            ]
            _, failed = run(jobs)
            self.assertTrue(Path(f"{log}.stderr").exists())
        self.assertEqual(
            failed["use"], f"fixture a failed (gen, see {log}.stdout/.stderr)"
        )


class NotEnoughCoresTest(unittest.TestCase):
    def test_cli_prints_the_minimum(self):
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".phlop.exec.yaml").write_text(
                "test_x.py:\n  - prefix: mpirun -n 4\n"
            )
            (Path(d) / "test_x.py").write_text(
                "import unittest\n\n\nclass X(unittest.TestCase):\n"
                "    def test_x(self):\n        pass\n"
            )
            p = subprocess.run(
                [sys.executable, "-m", "phlop.run.test_cases", "-i", ".", "-c", "2"],
                capture_output=True,
                text=True,
                cwd=d,
                env={**os.environ, "PYTHONPATH": os.getcwd()},
                check=False,
            )
        self.assertEqual(p.returncode, 1)
        self.assertIn("need at least 4 cores, have 2", p.stderr)
        self.assertIn("run with -c 4 or more", p.stderr)


class LogHeaderTest(unittest.TestCase):
    def test_stdout_starts_with_the_command(self):
        with tempfile.TemporaryDirectory() as d:
            log = str(Path(d) / "job")
            cmd = py("print('out'); import sys; print('err', file=sys.stderr)")
            Job(cmd=cmd, log_file_path=log)()
            self.assertEqual(Path(f"{log}.stdout").read_text(), f"{cmd}\nout\n")
            self.assertEqual(Path(f"{log}.stderr").read_text(), "err\n")


class FixtureLogTest(unittest.TestCase):
    """a setup or cleanup job keeps its log only if it failed (or --logging 2),
    a test always"""

    def logs(self, logging=1, **kw):
        with tempfile.TemporaryDirectory() as d:
            jobs = {
                name: Job(
                    cmd=cmd,
                    id=name,
                    log_file_path=str(Path(d) / "sub" / name),
                    logging=logging,
                    **fixtures,
                )
                for name, cmd, fixtures in (
                    ("gen_ok", "true", {"provides": {"a"}}),
                    ("gen_bad", "false", {"provides": {"b"}}),
                    ("rm", "true", {"cleans": {"a"}}),
                    ("test", "true", {}),
                )
            }
            (Path(d) / "sub").mkdir()
            run(list(jobs.values()))
            return sorted(p.name for p in Path(d).glob("sub/*"))

    def test_only_failed_fixtures_keep_logs(self):
        self.assertEqual(
            self.logs(),
            ["gen_bad.stderr", "gen_bad.stdout", "test.stderr", "test.stdout"],
        )

    def test_logging_always_keeps_them(self):
        self.assertEqual(len(self.logs(logging=2)), 8)


class CTestTest(unittest.TestCase):
    def ctest(self, **properties):
        return CTest_test(
            backtrace=0,
            command=["sh", "-c", "true"],
            name="t",
            properties=[{"name": k, "value": v} for k, v in properties.items()],
        )

    def test_properties_parsed(self):
        t = self.ctest(
            FIXTURES_SETUP=["a", "b"],
            FIXTURES_REQUIRED="c;d",
            FIXTURES_CLEANUP=["e"],
            WORKING_DIRECTORY="/tmp",
        )
        self.assertEqual(t.fixtures_setup, {"a", "b"})
        self.assertEqual(t.fixtures_required, {"c", "d"})
        self.assertEqual(t.fixtures_cleanup, {"e"})
        self.assertEqual(t.working_dir, "/tmp")

    def test_every_job_of_an_entry_carries_its_fixtures(self):
        here = os.getcwd()
        tests = [
            CTest_test(
                backtrace=0,
                # split per Class.method
                command=["python3", "-u", "tests/_phlop/run/test_test_cases.py"],
                name="py",
                properties=[
                    {"name": "WORKING_DIRECTORY", "value": here},
                    {"name": "FIXTURES_REQUIRED", "value": ["data"]},
                ],
            ),
            self.ctest(FIXTURES_SETUP=["data"], WORKING_DIRECTORY=here),  # a plain job
        ]
        with mock.patch.object(tc, "get_cmake_tests", return_value=tests):
            jobs = [j for b in tc.load_cmake_tests("build") for j in b.tests]
        py_jobs = [j for j in jobs if "python3" in j.cmd]
        self.assertGreater(len(py_jobs), 1)
        self.assertTrue(all(j.requires == {"data"} for j in py_jobs))
        plain = [j for j in jobs if "sh -c" in j.cmd]
        self.assertEqual([j.provides for j in plain], [{"data"}])

    def test_plain_jobs_log_under_their_test_name(self):
        here = os.getcwd()

        def ctest(name, *command):
            return CTest_test(
                backtrace=0,
                command=list(command),
                name=name,
                properties=[{"name": "WORKING_DIRECTORY", "value": f"{here}/build/x"}],
            )

        gen = ("python3", "-u", "amr/gen.py", "--root", "/data")
        tests = [
            ctest("sample-data-1d-h5", *gen, "1d", "h5"),
            ctest("sample-data-2d/vtk", *gen, "2d", "vtk"),
            # same names under another launch: another directory
            ctest("sample-data-mpi3-h5", "mpirun", "-n", "3", *gen, "--mpi3", "h5"),
            ctest("sample-data-mpi3-vtk", "mpirun", "-n", "3", *gen, "--mpi3", "vtk"),
            ctest("other-1d", "python3", "-u", "other.py", "1d"),  # alone: as named
            ctest("sample-data-1d", "./bin", "1d"),  # another program
        ]
        with mock.patch.object(tc, "get_cmake_tests", return_value=tests):
            jobs = [j for b in tc.load_cmake_tests("build") for j in b.tests]
        logs = sorted(str(Path(j.log_file_path).relative_to(here)) for j in jobs)
        x = ".phlop/build/x"
        self.assertEqual(
            logs,
            [
                f"{x}/mpirun_-n_3/sample-data-mpi3/h5",
                f"{x}/mpirun_-n_3/sample-data-mpi3/vtk",
                f"{x}/other-1d",
                f"{x}/sample-data-1d",
                f"{x}/sample-data/1d-h5",
                f"{x}/sample-data/2d_vtk",
            ],
        )

    def test_split_jobs_log_under_their_launch(self):
        here = os.getcwd()
        script = "tests/_phlop/run/test_test_cases.py"
        cmds = (
            f"python3 -u {script}",
            f"mpirun -n 2 --bind-to none python3 -u {script}",
        )
        logs = []
        for cmd in cmds:
            job = Job(cmd=cmd, working_dir=here)
            split = tc.load_py_test_cases_from_cmake(job)
            one = next(
                j
                for j in split
                if j.test_names == ["ResolveCoresTest.test_mpirun_dash_n"]
            )
            logs.append(str(Path(one.log_file_path).relative_to(here)))
        method = "test_test_cases/ResolveCoresTest/test_mpirun_dash_n"
        self.assertEqual(
            logs,
            [
                f".phlop/tests/_phlop/run/{method}",
                f".phlop/tests/_phlop/run/mpirun_-n_2/{method}",
            ],
        )


class PrintOnlyTest(unittest.TestCase):
    def test_fixtures_section(self):
        import contextlib
        import io

        from phlop.testing import parallel_processor as tpp

        jobs = [
            Job(cmd="gen", provides={"data"}),
            Job(cmd="mpirun -n 2 read", cores=2, requires={"data", "nobody"}),
            Job(cmd="rm", cleans={"data"}),
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tpp.process(jobs, print_only=True)
        self.assertEqual(
            out.getvalue().splitlines(),
            [
                "fixtures:",
                "data  # required by 1",
                "  setup: gen",
                "  cleanup: rm",
                "nobody  # required by 1, no setup loaded: not waited for",
                "",
                "tests:",
                "mpirun -n 2 read  # cores=2",
            ],
        )


class LoadAndRegexTest(unittest.TestCase):
    def test_duplicates_merged_across_dumps(self):
        import dill

        def gen(**kw):
            return Job(cmd="gen.py", env={"A": "1"}, working_dir="w", **kw)

        mpi = "mpirun -n {} test.py"
        dumps = [
            [
                tc.TestBatch([gen(provides={"x"})], 1),
                tc.TestBatch([Job(cmd=mpi.format(2), cores=2)], 2),
            ],
            [
                tc.TestBatch([gen(requires={"y"})], 1),
                tc.TestBatch([Job(cmd=mpi.format(4), cores=4)], 4),
            ],
        ]
        with tempfile.TemporaryDirectory() as d:
            for i, batches in enumerate(dumps):
                (Path(d) / f"ctests.{i}.dill").write_text(
                    codecs.encode(
                        dill.dumps(tc.TestBatchesList(batch_list=batches)), "hex"
                    ).decode()
                )
            jobs = [j for b in tc.extract_load(d, "ctests.*.dill") for j in b.tests]
        gens = [j for j in jobs if j.cmd == "gen.py"]
        self.assertEqual(len(gens), 1)
        self.assertEqual((gens[0].provides, gens[0].requires), ({"x"}, {"y"}))
        # an mpirun -n count is part of what runs: both kept
        self.assertEqual(
            sorted(j.cmd for j in jobs if "mpirun" in j.cmd),
            [mpi.format(2), mpi.format(4)],
        )

    def test_regex_keeps_providers_and_cleanups(self):
        batches = [
            tc.TestBatch(
                [
                    Job(cmd="gen_base", provides={"base"}),
                    Job(cmd="gen_data", provides={"data"}, requires={"base"}),
                    Job(cmd="test_reader", requires={"data"}),
                    Job(cmd="rm_data", cleans={"data"}),
                    Job(cmd="gen_other", provides={"other"}),
                    Job(cmd="test_unrelated"),
                ],
                1,
            )
        ]
        args = cli_args_parser().parse_args(["-r", "test_reader"])
        kept = [j.cmd for b in filter_out_regex_fails(args, batches) for j in b.tests]
        self.assertEqual(kept, ["gen_base", "gen_data", "test_reader", "rm_data"])


class ExecYamlTest(unittest.TestCase):
    """fixtures in .phlop.exec.yaml: dict entries on file / class / method keys
    (every level adds), fixtures on variants, the same yaml under --cmake.
    A scratch package per method under tests/_phlop, as
    LoadConfigTestsValidationTest (relative imports, concurrent methods)"""

    def setUp(self):
        name = f"_scratch_fixtures_{self._testMethodName}"
        self.dir = Path(f"tests/_phlop/{name}")
        self.package = f"tests._phlop.{name}"
        self.dir.mkdir(exist_ok=True)
        (self.dir / "__init__.py").write_text("")
        for module, clazz in (("test_gen", "GenTest"), ("test_read", "ReadTest")):
            (self.dir / f"{module}.py").write_text(
                "import unittest\n\n\n"
                f"class {clazz}(unittest.TestCase):\n"
                "    def test_a(self):\n        pass\n\n"
                "    def test_b(self):\n        pass\n"
            )
        importlib.invalidate_caches()  # a package made after tests._phlop's scan

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        for module in ("test_gen", "test_read"):
            sys.modules.pop(f"{self.package}.{module}", None)

    def config(self, text):
        (self.dir / ".phlop.exec.yaml").write_text(text)

    def jobs(self, **kw):
        batches = tc.load_config_tests(str(self.dir), **kw)
        return {
            (
                j.test_names[0] if len(j.test_names) == 1 else j.cmd,
                j.meta and j.meta[0],
            ): j
            for b in batches
            for j in b.tests
        }

    def test_dict_entries_add_up(self):
        self.config(
            "test_gen.py:\n  provides: [data]\n"
            "test_read.py:\n  requires: [data]\n"
            "test_read.py:ReadTest.test_b:\n  requires: [more]\n  cleans: [data]\n"
        )
        jobs = self.jobs()
        self.assertEqual(jobs["GenTest.test_a", None].provides, {"data"})
        self.assertEqual(jobs["ReadTest.test_a", None].requires, {"data"})
        b = jobs["ReadTest.test_b", None]
        self.assertEqual((b.requires, b.cleans), ({"data", "more"}, {"data"}))

    def test_variant_fixtures(self):
        self.config(
            "test_read.py:\n  requires: [data]\n"
            "test_read.py:ReadTest.test_a:\n"
            "  - tags: [mpi]\n    prefix: mpirun -n 2\n    requires: [mpi_data]\n"
        )
        job = self.jobs(tags=["mpi"])["ReadTest.test_a", "mpi"]
        self.assertEqual(job.requires, {"data", "mpi_data"})
        self.assertEqual(job.cores, 2)

    def test_grouped_tests_split_on_fixtures(self):
        self.config("test_read.py:ReadTest.test_b:\n  requires: [data]\n")
        jobs = self.jobs(granularity="class")
        read = [j for j in jobs.values() if "ReadTest" in j.cmd]
        self.assertEqual(sorted(sorted(j.requires) for j in read), [[], ["data"]])

    def test_unknown_key_in_dict_entry(self):
        self.config("test_gen.py:\n  provide: [data]\n")
        with self.assertRaisesRegex(ValueError, "provide"):
            self.jobs()

    def test_cmake_jobs_get_yaml_fixtures(self):
        self.config(
            "test_gen.py:\n  provides: [data]\n"
            "test_read.py:ReadTest.test_b:\n  requires: [data]\n"
        )
        here = os.getcwd()
        plain = Job(cmd="python3 -u test_gen.py 2d h5", working_dir=str(self.dir))
        split = Job(
            cmd=f"python3 -um {self.package}.test_read ReadTest.test_b",
            working_dir=here,
            test_names=["ReadTest.test_b"],
            requires={"ctest"},
        )
        other = Job(cmd=f"python3 -um {self.package}.test_read ReadTest.test_a")
        tc.yaml_fixtures(plain)
        tc.yaml_fixtures(split)
        tc.yaml_fixtures(other)
        self.assertEqual(plain.provides, {"data"})
        self.assertEqual(split.requires, {"ctest", "data"})
        self.assertEqual(other.requires, set())


if __name__ == "__main__":
    unittest.main()
