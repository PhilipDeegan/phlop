#
#
#
#
#


import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from phlop.app.cmake import list_tests as get_cmake_tests
from phlop.os import env_sep
from phlop.proc import run
from phlop.reflection import classes_in_file
from phlop.sys import extend_sys_path

_LOG_DIR = Path(os.environ.get("PHLOP_LOG_DIR", os.getcwd()))

CMD_PREFIX = ""
CMD_POSTFIX = ""


@dataclass
class TestCase:
    cmd: str
    env: dict = field(default_factory=lambda: {})  # dict[str, str] # eventually
    working_dir: str = field(default_factory=lambda: None)
    log_file_path: str = field(default_factory=lambda: None)
    cores: int = field(default_factory=lambda: 1)

    def __post_init__(self):
        self.cmd = self.cmd.strip()


@dataclass
class TestBatch:
    tests: list
    cores: int


class DefaultTestCaseExtractor:
    def __call__(self, ctest_test):
        return [
            TestCase(
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
        ...
        # not configured, assumed fast per file
        # print("GoogleTestCaseExtractor")
        # exec binary with `--gtest_list_tests` and see if it doesn't fail
        # p = run(
        #     ctest_test.cmd + " --gtest_list_tests --gtest_output=json:gtest.json",
        #     working_dir=ctest_test.working_dir,
        # )
        # print(p.stdout)

        return None


class PythonUnitTestCaseExtractor:
    def __call__(self, ctest_test):
        if "python3 " in ctest_test.cmd:  # hacky
            return load_py_test_cases_from_cmake(ctest_test)
        return None


EXTRACTORS = [
    PythonUnitTestCaseExtractor(),
    GoogleTestCaseExtractor(),
    DefaultTestCaseExtractor(),
]


def python3_default_test_cmd(clazz, test_id):
    return f"python3 -Oum {clazz.__module__} {clazz.__name__}.{test_id}"


def load_test_cases_in(
    classes, test_cmd_pre="", test_cmd_post="", test_cmd_fn=None, **kwargs
):
    test_cmd_fn = test_cmd_fn if test_cmd_fn else python3_default_test_cmd
    log_file_path = kwargs.pop("log_file_path", None)
    tests, loader = [], unittest.TestLoader()
    for test_class in classes:
        for suite in loader.loadTestsFromTestCase(test_class):
            cmd = test_cmd_fn(type(suite), suite._testMethodName)
            tests += [
                TestCase(
                    cmd=f"{test_cmd_pre} {cmd} {test_cmd_post}".strip(),
                    log_file_path=None
                    if not log_file_path
                    else f"{log_file_path}/{suite._testMethodName}",
                    **kwargs,
                )
            ]
    return tests


def load_py_test_cases_from_cmake(ctest_test):
    ppath = ctest_test.env.get("PYTHONPATH", "")
    bits = ctest_test.cmd.split(" ")
    idx = [i for i, x in enumerate(bits) if "python3" in x][0]
    prefix = " ".join(bits[:idx])
    with extend_sys_path([ctest_test.working_dir] + ppath.split(env_sep())):
        pyfile = bits[-1]
        return load_test_cases_in(
            classes_in_file(pyfile, unittest.TestCase, fail_on_import_error=True),
            env=ctest_test.env,
            working_dir=ctest_test.working_dir,
            log_file_path=_LOG_DIR
            / ".phlop"
            / f"{Path(ctest_test.working_dir).relative_to(_LOG_DIR)}",
            test_cmd_pre=CMD_PREFIX + prefix + CMD_POSTFIX,
        )


def determine_cores_for_test_case(test_case):
    try:
        if "mpirun -n" in test_case.cmd:
            bits = test_case.cmd.split(" ")
            idx = [i for i, x in enumerate(bits) if "mpirun" in x][0]
            test_case.cores = int(bits[idx + 2])
    except Exception as e:
        print("EXXXX", e)

    return test_case


def binless(test_case):
    if test_case.cmd.startswith("/usr/bin/"):
        test_case.cmd = test_case.cmd[9:]
    return test_case


MUTATORS = [determine_cores_for_test_case, binless]


def load_cmake_tests(cmake_dir, cores=1, test_cmd_pre="", test_cmd_post=""):
    cmake_tests = get_cmake_tests(cmake_dir)
    tests = []
    for cmake_test in cmake_tests:
        tests += [
            TestCase(
                cmd=test_cmd_pre + " ".join(cmake_test.command) + " " + test_cmd_post,
                env=cmake_test.env,
                working_dir=cmake_test.working_dir,
            )
        ]

    test_batches = {}

    def _add(test_cases):
        for test_case in test_cases:
            for mutator in MUTATORS:
                test_case = mutator(test_case)
            if test_case.cores not in test_batches:
                test_batches[test_case.cores] = []
            test_batches[test_case.cores].append(test_case)

    test_cases = []
    for test in tests:
        for extractor in EXTRACTORS:
            res = extractor(test)
            if res:
                _add(res)
                break

    return [TestBatch(v, k) for k, v in test_batches.items()]


@dataclass
class TestBatchesList:
    batch_list: List[TestBatch]


def deserialize(s):
    import codecs

    import dill

    return dill.loads(codecs.decode(s, "hex"))


def extract_load(cli_args):
    test_batches = {}

    file_paths = list(Path(cli_args.dir).glob(cli_args.load))

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
