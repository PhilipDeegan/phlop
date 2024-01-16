#
#
#
#
#


import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from phlop.app.cmake import list_tests as get_cmake_tests
from phlop.os import env_sep
from phlop.proc import run
from phlop.reflection import classes_in_file
from phlop.sys import extend_sys_path

_LOG_DIR = Path(os.environ.get("PHLOP_LOG_DIR", os.getcwd()))


@dataclass
class TestCase:
    cmd: str
    env: dict = field(default_factory=lambda: {})  # dict[str, str] # eventually
    working_dir: str = field(default_factory=lambda: None)
    log_file_path: str = field(default_factory=lambda: None)

    def __post_init__(self):
        self.cmd = self.cmd.strip()


class TestBatch:
    def __init__(self, tests, cores=1):
        self.tests = tests
        self.cores = cores


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
        if "python3" in ctest_test.cmd:  # hacky
            return load_test_cases_from_cmake(ctest_test)
        return None


EXTRACTORS = [
    PythonUnitTestCaseExtractor(),
    GoogleTestCaseExtractor(),
    DefaultTestCaseExtractor(),
]


def python3_default_test_cmd(clazz, test_id):
    return f"python3 -m {clazz.__module__} {clazz.__name__}.{test_id}"


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


def load_test_cases_from_cmake(ctest_test):
    ppath = ctest_test.env.get("PYTHONPATH", "")
    with extend_sys_path([ctest_test.working_dir] + ppath.split(env_sep())):
        pyfile = ctest_test.cmd.split(" ")[-1]
        return load_test_cases_in(
            classes_in_file(pyfile, unittest.TestCase, fail_on_import_error=True),
            env=ctest_test.env,
            working_dir=ctest_test.working_dir,
            log_file_path=_LOG_DIR
            / ".phlop"
            / f"{Path(ctest_test.working_dir).relative_to(_LOG_DIR)}",
        )


# probably return a list of TestBatch if we do some core count detection per test
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

    test_cases = []
    for test in tests:
        for extractor in EXTRACTORS:
            res = extractor(test)
            if res:
                test_cases += res
                break

    return TestBatch(test_cases, cores)
