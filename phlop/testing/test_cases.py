# phlop/testing/test_cases.py

import os
import shlex
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

from phlop.app.cmake import list_tests as get_cmake_tests
from phlop.os import env_sep
from phlop.procs.parallel_processor import Job
from phlop.reflection import classes_in_file
from phlop.sys import extend_sys_path

_LOG_DIR = Path(os.environ.get("PHLOP_LOG_DIR", os.getcwd()))
CMD_PREFIX = ""
CMD_POSTFIX = ""
PYTHON_FLAGS = "-um"  # -u always; omit -O so assertions run in tests


@dataclass
class TestBatch:
    tests: list
    cores: int


class DefaultTestCaseExtractor:
    def __call__(self, ctest_test):
        return [
            Job(
                cmd=ctest_test.cmd,
                env=ctest_test.env,
                working_dir=ctest_test.working_dir,
                log_file_path=_LOG_DIR
                / ".phlop"
                / f"{Path(ctest_test.working_dir).relative_to(_LOG_DIR)}",
            )
        ]


class GoogleTestCaseExtractor:
    def __call__(self, ctest_test):
        # not configured, assumed fast per file
        # print("GoogleTestCaseExtractor")
        # exec binary with `--gtest_list_tests` and see if it doesn't fail
        # p = run(
        #     ctest_test.cmd + " --gtest_list_tests --gtest_output=json:gtest.json",
        #     working_dir=ctest_test.working_dir,
        # )
        # print(p.stdout)

        return


class PythonUnitTestCaseExtractor:
    def __call__(self, ctest_test):
        if "python3" in ctest_test.cmd:  # hacky
            return load_py_test_cases_from_cmake(ctest_test)
        return None


EXTRACTORS = [
    PythonUnitTestCaseExtractor(),
    GoogleTestCaseExtractor(),
    DefaultTestCaseExtractor(),
]


def python3_default_test_cmd(clazz, test_id, python_flags=None):
    flags = python_flags if python_flags is not None else PYTHON_FLAGS
    return f"python3 {flags} {clazz.__module__} {clazz.__name__}.{test_id}"


def logfile(log_file_path, test_class, suite):
    if not log_file_path:
        return None
    pyfile = Path(sys.modules[test_class.__module__].__file__)
    logfile = str(
        log_file_path
        / pyfile.parent.relative_to(_LOG_DIR)
        / pyfile.stem
        / test_class.__name__
        / suite._testMethodName
    )
    return logfile


def load_test_cases_in(
    classes,
    test_cmd_pre="",
    test_cmd_post="",
    test_cmd_fn=None,
    python_flags=None,
    **kwargs,
):
    if test_cmd_fn is None:

        def test_cmd_fn(c, t):
            return python3_default_test_cmd(c, t, python_flags)

    tests, loader = [], unittest.TestLoader()
    for test_class in classes:
        for suite in loader.loadTestsFromTestCase(test_class):
            cmd = test_cmd_fn(type(suite), suite._testMethodName)

            tests += [
                Job(
                    cmd=f"{test_cmd_pre} {cmd} {test_cmd_post}".strip(),
                    log_file_path=logfile(_LOG_DIR / ".phlop", test_class, suite),
                    **kwargs,
                )
            ]
    return tests


_PY_VALUE_FLAGS = "cmWX"  # trailing char of a python3 option cluster that takes a value


def _python_invocation_index(bits, cmd):
    for i, tok in enumerate(bits):
        if "python3" in tok:
            return i
    raise ValueError(f"no python3 invocation found in command: {cmd!r}")


def _python_test_target(bits, idx, cmd):
    i = idx + 1
    while i < len(bits):
        tok = bits[i]
        if tok.startswith("-") and len(tok) > 1 and tok[-1] in _PY_VALUE_FLAGS:
            if i + 1 >= len(bits):
                raise ValueError(f"'{tok}' given without a value in command: {cmd!r}")
            value = bits[i + 1]
            if tok[-1] != "m":
                i += 2
                continue
            if value != "unittest":
                return value
            # `-m unittest <test-id>`: unittest is the runner, not the target
            if i + 2 >= len(bits):
                raise ValueError(
                    f"'-m unittest' given without a test id in command: {cmd!r}"
                )
            return bits[i + 2]
        if tok.startswith("-"):
            i += 1
            continue
        return tok
    raise ValueError(f"no python test target found in command: {cmd!r}")


def load_py_test_cases_from_cmake(ctest_test):
    ppath = ctest_test.env.get("PYTHONPATH", "")
    bits = shlex.split(ctest_test.cmd)
    idx = _python_invocation_index(bits, ctest_test.cmd)
    prefix = " ".join(bits[:idx])
    with extend_sys_path([ctest_test.working_dir] + ppath.split(env_sep())):
        target = _python_test_target(bits, idx, ctest_test.cmd)
        pyfile = (
            target if target.endswith(".py") else target.replace(".", os.sep) + ".py"
        )

        return load_test_cases_in(
            classes_in_file(pyfile, unittest.TestCase, fail_on_import_error=True),
            env=ctest_test.env,
            working_dir=ctest_test.working_dir,
            test_cmd_pre=CMD_PREFIX + prefix + CMD_POSTFIX,
        )


def determine_cores_for_test_case(test_case):
    try:
        if "mpirun -n" in test_case.cmd:
            bits = test_case.cmd.split(" ")
            idx = next(i for i, x in enumerate(bits) if "mpirun" in x)
            test_case.cores = int(bits[idx + 2])
    except Exception as e:  # noqa: BLE001 - best-effort, never fatal
        print("EXXXX", e)

    return test_case


def binless(test_case):
    if test_case.cmd.startswith("/usr/"):
        bits = test_case.cmd.split(" ")
        test_case.cmd = " ".join([bits[0].split("/")[-1]] + bits[1:])
    # print("test_case.cmd ", test_case.cmd)
    return test_case


MUTATORS = [determine_cores_for_test_case, binless]


def load_cmake_tests(cmake_dir, cores=1, test_cmd_pre="", test_cmd_post=""):
    cmake_tests = get_cmake_tests(cmake_dir)
    tests = []
    for cmake_test in cmake_tests:
        cmd = f"{test_cmd_pre} " + " ".join(cmake_test.command) + f" {test_cmd_post}"
        tests += [Job(cmd=cmd, env=cmake_test.env, working_dir=cmake_test.working_dir)]

    test_batches = {}

    def _add(test_cases):
        for test_case in test_cases:
            for mutator in MUTATORS:
                test_case = mutator(test_case)
            if test_case.cores not in test_batches:
                test_batches[test_case.cores] = []
            test_batches[test_case.cores].append(test_case)

    for test in tests:
        for extractor in EXTRACTORS:
            res = extractor(test)
            if res:
                _add(res)
                break

    return [TestBatch(v, k) for k, v in test_batches.items()]


@dataclass
class TestBatchesList:
    batch_list: list[TestBatch]


def deserialize(s):
    import codecs

    import dill

    return dill.loads(codecs.decode(s, "hex"))


def extract_load(directory, globbing):
    test_batches = {}

    file_paths = list(Path(directory).glob(globbing))

    if not file_paths:
        raise ValueError("No load files found")

    for path in file_paths:
        with open(path, "r") as file:
            batch_list = deserialize(file.read()).batch_list
            for batch in batch_list:
                for test_case in batch.tests:
                    if batch.cores not in test_batches:
                        test_batches[test_case.cores] = []
                    test_batches[test_case.cores].append(test_case)

    return [TestBatch(v, k) for k, v in test_batches.items()]
