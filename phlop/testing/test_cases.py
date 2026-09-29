# phlop/testing/test_cases.py

import os
import re
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
CONFIG_FILENAME = ".phlop.exec.yaml"


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
    """Return `(target, next_idx)`: the script/module the command runs, and
    the index right after it. Anything remaining at `bits[next_idx:]` is an
    explicit argument (e.g. a specific `Class.test_id`) the ctest entry
    already scoped itself to."""
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
                return value, i + 2
            # `-m unittest <test-id>`: unittest is the runner, not the target
            if i + 2 >= len(bits):
                raise ValueError(
                    f"'-m unittest' given without a test id in command: {cmd!r}"
                )
            return bits[i + 2], i + 3
        if tok.startswith("-"):
            i += 1
            continue
        return tok, i + 1
    raise ValueError(f"no python test target found in command: {cmd!r}")


def load_py_test_cases_from_cmake(ctest_test):
    ppath = ctest_test.env.get("PYTHONPATH", "")
    bits = shlex.split(ctest_test.cmd)
    idx = _python_invocation_index(bits, ctest_test.cmd)
    prefix = " ".join(bits[:idx])
    with extend_sys_path([ctest_test.working_dir] + ppath.split(env_sep())):
        target, next_idx = _python_test_target(bits, idx, ctest_test.cmd)
        if bits[next_idx:]:
            # ctest already scoped this entry to a specific invocation (e.g.
            # a trailing `Class.test_id`) - run it as-is rather than
            # expanding every unittest.TestCase method in the file, which
            # would otherwise duplicate work across every such entry for
            # the same file and discard the args that distinguish them.
            return None
        pyfile = (
            target if target.endswith(".py") else target.replace(".", os.sep) + ".py"
        )

        return load_test_cases_in(
            classes_in_file(pyfile, unittest.TestCase, fail_on_import_error=True),
            env=ctest_test.env,
            working_dir=ctest_test.working_dir,
            test_cmd_pre=CMD_PREFIX + prefix + CMD_POSTFIX,
        )


_MPIRUN_CORES_RE = re.compile(r"mpirun\s+(?:-n|-np|--np)\s+(\d+)")


def mpirun_cores(cmd, env):
    m = _MPIRUN_CORES_RE.search(cmd)
    return int(m.group(1)) if m else None


def omp_num_threads_cores(cmd, env):
    value = (env or {}).get("OMP_NUM_THREADS")
    return int(value) if value else None


CORES_RESOLVERS = [mpirun_cores, omp_num_threads_cores]


def resolve_cores(cmd, env=None, explicit=None, threads=None):
    """Resolve a total core count for `cmd`/`env`: an explicit value always
    wins outright. Otherwise `threads` (a declared threads-per-rank count for
    parallelism phlop can't introspect, e.g. std::thread/thread_pool) and
    every CORES_RESOLVERS match are multiplied together - e.g. `mpirun -n 2`
    with `threads=4` is 2*4=8 total cores - defaulting to 1 if none apply."""
    if explicit is not None:
        return int(explicit)
    total = int(threads) if threads is not None else None
    for resolver in CORES_RESOLVERS:
        cores = resolver(cmd, env or {})
        if cores is not None:
            total = (total or 1) * cores
    return total or 1


def determine_cores_for_test_case(test_case):
    try:
        test_case.cores = resolve_cores(test_case.cmd, test_case.env)
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


def _load_dir_config(directory):
    config_path = Path(directory) / CONFIG_FILENAME
    if not config_path.exists():
        return {}
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _variants_for(dir_config, filename, class_name, method_name, used_keys=None):
    """Look up declared execution variants for a test, keyed by
    `filename:Class.method` (or bare `filename:Class` to apply to every
    method in that class, or bare `filename` with no class - written in yaml
    as `filename:` - to apply to every test in that file). More specific
    keys take precedence: `Class.method` > `Class` > file-default. The
    filename is always required, since a directory can hold multiple files
    that each define a class of the same name. None means: run once, as
    normal.

    Note: the yaml key `filename:` parses to the plain string `filename`
    (the trailing colon is yaml's key/value delimiter, not part of the
    key), so the file-default lookup below is the bare filename.

    `used_keys`, if given, collects whichever key actually matched - so
    callers can tell, after scanning every file, which declared keys never
    matched anything."""
    qualified = f"{filename}:{class_name}.{method_name}"
    if qualified in dir_config:
        if used_keys is not None:
            used_keys.add(qualified)
        return dir_config[qualified] or None
    bare = f"{filename}:{class_name}"
    if bare in dir_config:
        if used_keys is not None:
            used_keys.add(bare)
        return dir_config[bare] or None
    if filename in dir_config:
        if used_keys is not None:
            used_keys.add(filename)
        return dir_config[filename] or None
    return None


def _variant_job(
    base_cmd, test_class, suite, variant, index, test_cmd_pre, test_cmd_post
):
    prefix = variant.get("prefix", "")
    env = dict(variant.get("env") or {})
    working_dir = variant.get("working_dir")
    cmd = " ".join(
        p for p in (test_cmd_pre, prefix, base_cmd, test_cmd_post) if p
    ).strip()
    tags = variant.get("tags") or []

    log_file_path = logfile(_LOG_DIR / ".phlop", test_class, suite)
    if log_file_path:
        log_file_path = f"{log_file_path}_{'-'.join(tags) if tags else index}"

    return Job(
        cmd=cmd,
        env=env,
        working_dir=working_dir,
        log_file_path=log_file_path,
        cores=resolve_cores(
            cmd, env, explicit=variant.get("cores"), threads=variant.get("threads")
        ),
        meta=tags or None,
    )


def load_config_tests(
    input_path, test_cmd_pre="", test_cmd_post="", python_flags=None, tags=None
):
    """Scan a file or directory for tests, honouring per-directory
    `.phlop.exec.yaml` configs that fan a test out into tagged execution
    variants with their own prefix/env/working_dir/cores/threads. Tests/files
    not declared in a config (or with no config present at all) run once, as
    normal.

    Tagged variants are opt-in: they only run when `tags` is given and
    intersects with the variant's own tags. Untagged variants and undeclared
    (default) tests are unaffected and always run, regardless of `tags` - this
    keeps anything that needs e.g. mpirun or other unavailable/heavy resources
    from running unless explicitly requested.

    Every key declared in a directory's `.phlop.exec.yaml` must match a real
    `file:Class(.method)` found while scanning that directory - a stale or
    mistyped entry (wrong filename, missing `:`, renamed class/method) raises
    a ValueError rather than silently doing nothing.
    """
    tags = set(tags) if tags else None
    path = Path(input_path)
    py_files = [path] if path.is_file() else sorted(path.glob("**/*.py"))

    dir_configs = {}
    used_keys = {}

    def dir_config_for(directory):
        directory = str(directory)
        if directory not in dir_configs:
            dir_configs[directory] = _load_dir_config(directory)
            used_keys[directory] = set()
        return dir_configs[directory]

    loader = unittest.TestLoader()
    test_batches = {}

    def _add(job):
        test_batches.setdefault(job.cores, []).append(job)

    for py_file in py_files:
        directory = str(py_file.parent)
        dir_config = dir_config_for(directory)
        for test_class in classes_in_file(py_file, unittest.TestCase):
            for suite in loader.loadTestsFromTestCase(test_class):
                method = suite._testMethodName
                base_cmd = python3_default_test_cmd(test_class, method, python_flags)
                variants = _variants_for(
                    dir_config,
                    py_file.name,
                    test_class.__name__,
                    method,
                    used_keys[directory],
                )

                if not variants:
                    cmd = f"{test_cmd_pre} {base_cmd} {test_cmd_post}".strip()
                    _add(
                        Job(
                            cmd=cmd,
                            log_file_path=logfile(
                                _LOG_DIR / ".phlop", test_class, suite
                            ),
                            cores=resolve_cores(cmd),
                        )
                    )
                    continue

                for index, variant in enumerate(variants):
                    variant_tags = set(variant.get("tags") or [])
                    if variant_tags and not (tags and (variant_tags & tags)):
                        continue
                    _add(
                        _variant_job(
                            base_cmd,
                            test_class,
                            suite,
                            variant,
                            index,
                            test_cmd_pre,
                            test_cmd_post,
                        )
                    )

    stale = {
        directory: sorted(set(config) - used_keys[directory])
        for directory, config in dir_configs.items()
        if set(config) - used_keys[directory]
    }
    if stale:
        details = "; ".join(f"{d}: {keys}" for d, keys in stale.items())
        raise ValueError(
            f"{CONFIG_FILENAME} references tests that were never found "
            f"while scanning - {details}"
        )

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
