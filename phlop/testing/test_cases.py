# phlop/testing/test_cases.py

import os
import re
import shlex
import unittest
from dataclasses import dataclass
from pathlib import Path

from phlop.app.cmake import list_tests as get_cmake_tests
from phlop.os import env_sep
from phlop.procs.fixtures import fixtures_of
from phlop.procs.parallel_processor import Job
from phlop.reflection import class_file, classes_in_file
from phlop.sys import extend_sys_path

_LOG_DIR = Path(os.environ.get("PHLOP_LOG_DIR", os.getcwd()))
CMD_PREFIX = ""
CMD_POSTFIX = ""
PYTHON_FLAGS = "-um"  # -u always; omit -O so assertions run in tests
CONFIG_FILENAME = ".phlop.exec.yaml"
# how many test methods one process runs:
#   method: one process per test method - most isolated, pays interpreter
#           startup and module imports for every method
#   class:  one process per TestCase class
#   file:   one process per test file
GRANULARITIES = ("method", "class", "file")
GRANULARITY = "method"


def _log_subpath(path):
    """`path` relative to _LOG_DIR, or the full path without its root if outside"""
    path = Path(path).resolve()
    try:
        return path.relative_to(_LOG_DIR.resolve())
    except ValueError:
        return path.relative_to(path.anchor)


@dataclass
class TestBatch:
    tests: list
    cores: int

    def __post_init__(self):
        self.check_cores()

    def check_cores(self):
        """Every job in a batch runs with the batch's core count"""
        bad = [t for t in self.tests if getattr(t, "cores", self.cores) != self.cores]
        if bad:
            raise ValueError(
                f"TestBatch(cores={self.cores}) given jobs with other core counts: "
                + ", ".join(f"{t.cmd!r} (cores={t.cores})" for t in bad)
            )


def _fixture_kwargs(ctest_test):
    """the fixtures of a ctest entry, for every job made from it"""
    provides, requires, cleans = fixtures_of(ctest_test)
    return {"provides": provides, "requires": requires, "cleans": cleans}


class DefaultTestCaseExtractor:
    def __call__(self, ctest_test):
        return [
            Job(
                cmd=ctest_test.cmd,
                env=ctest_test.env,
                working_dir=ctest_test.working_dir,
                # set by load_cmake_tests, named after the ctest test
                log_file_path=ctest_test.log_file_path
                or _LOG_DIR / ".phlop" / _log_subpath(ctest_test.working_dir),
                **_fixture_kwargs(ctest_test),
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


def python3_test_cmd(module, test_names, python_flags=None):
    """`test_names` are unittest names (`Class` or `Class.method`), passed to
    the module's own `unittest.main()`"""
    flags = python_flags if python_flags is not None else PYTHON_FLAGS
    return f"python3 {flags} {module} {' '.join(test_names)}"


def python3_default_test_cmd(clazz, test_id, python_flags=None):
    return python3_test_cmd(
        clazz.__module__, [f"{clazz.__name__}.{test_id}"], python_flags
    )


def _check_granularity(granularity):
    if granularity is None:
        granularity = GRANULARITY
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity {granularity!r} not one of {GRANULARITIES}")
    return granularity


def _test_methods(loader, test_class):
    return [suite._testMethodName for suite in loader.loadTestsFromTestCase(test_class)]


def _group_tests(entries, granularity):
    """Group `(test_class, method, key)` entries into one group per process to
    start, in first-seen order. Only entries with equal `key` (e.g. the same
    .phlop.exec.yaml variants) are grouped. Returns `[(key, {class: [methods]})]`"""
    groups = {}
    for test_class, method, key in entries:
        if granularity == "method":
            group = (test_class, method, key)
        elif granularity == "class":
            group = (test_class, key)
        else:
            group = (class_file(test_class), key)
        groups.setdefault(group, (key, {}))[1].setdefault(test_class, []).append(method)
    return list(groups.values())


def _test_names(by_class, all_methods):
    """unittest names for a group: the bare class if all its methods are in it"""
    names = []
    for test_class, methods in by_class.items():
        if len(methods) > 1 and methods == all_methods[test_class]:
            names.append(test_class.__name__)
        else:
            names += [f"{test_class.__name__}.{m}" for m in methods]
    return names


def _qualified_names(by_class):
    """every `Class.method` in a group, so --regex can match a method in a
    grouped job whose cmd names only the class (or file)"""
    return [f"{c.__name__}.{m}" for c, methods in by_class.items() for m in methods]


def _unique_path(path, used):
    """groups in one class/file with different variant lists would share a log
    path, suffix all but the first"""
    n = used.get(path, 0)
    used[path] = n + 1
    return f"{path}_g{n}" if n else path


def _launch_dir(cmd):
    """a log path part for how a command is launched, `mpirun_-n_2`, None if
    plainly: the same test run with different rank counts logs apart"""
    m = _MPIRUN_CORES_RE.search(cmd)
    return f"mpirun_-n_{m.group(1)}" if m else None


def _group_logfile(log_file_path, by_class, granularity, launch=None):
    """as logfile(), shortened to the class or file for coarser granularity,
    under `launch` (see _launch_dir) if given"""
    test_class = next(iter(by_class))
    pyfile = Path(class_file(test_class))
    path = log_file_path / _log_subpath(pyfile.parent)
    if launch:
        path = path / launch
    path = path / pyfile.stem
    if granularity != "file":
        path = path / test_class.__name__
    if granularity == "method":
        path = path / by_class[test_class][0]
    return str(path)


def logfile(log_file_path, test_class, suite):
    if not log_file_path:
        return None
    pyfile = Path(class_file(test_class))
    logfile = str(
        log_file_path
        / _log_subpath(pyfile.parent)
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
    granularity=None,
    **kwargs,
):
    granularity = _check_granularity(granularity)
    if test_cmd_fn is not None and granularity != "method":
        raise ValueError("test_cmd_fn requires granularity 'method'")

    loader = unittest.TestLoader()
    classes = list(dict.fromkeys(classes))
    all_methods = {c: _test_methods(loader, c) for c in classes}
    entries = ((c, m, None) for c in classes for m in all_methods[c])

    tests = []
    for _, by_class in _group_tests(entries, granularity):
        test_class = next(iter(by_class))
        if test_cmd_fn is not None:
            cmd = test_cmd_fn(test_class, by_class[test_class][0])
        else:
            names = _test_names(by_class, all_methods)
            cmd = python3_test_cmd(test_class.__module__, names, python_flags)
        cmd = f"{test_cmd_pre} {cmd} {test_cmd_post}".strip()
        tests += [
            Job(
                cmd=cmd,
                log_file_path=_group_logfile(
                    _LOG_DIR / ".phlop", by_class, granularity, _launch_dir(cmd)
                ),
                test_names=_qualified_names(by_class),
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
            **_fixture_kwargs(ctest_test),
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


def _python_file_of(job):
    """(the .py file a python3 job runs, its trailing args), None if not one"""
    if "python3" not in job.cmd:
        return None
    try:
        bits = shlex.split(job.cmd)
        target, next_idx = _python_test_target(
            bits, _python_invocation_index(bits, job.cmd), job.cmd
        )
    except ValueError:
        return None
    rel = target if target.endswith(".py") else target.replace(".", os.sep) + ".py"
    ppath = (job.env or {}).get("PYTHONPATH", "")
    for root in [job.working_dir or "."] + [p for p in ppath.split(env_sep()) if p]:
        path = Path(root) / rel
        if path.is_file():
            return path, bits[next_idx:]
    return None


def yaml_fixtures(job, dir_configs=None):
    """with --cmake: fixtures a .phlop.exec.yaml beside the python file a job
    runs declares for it (its file, or the Class.method it was split into),
    added to ctest's FIXTURES_* properties. Variants are not applied, and keys
    are not checked: the yaml can name tests ctest does not register"""
    found = _python_file_of(job)
    if found is None:
        return job
    path, args = found
    dir_configs = {} if dir_configs is None else dir_configs
    directory = str(path.parent.resolve())
    if directory not in dir_configs:
        dir_configs[directory] = _load_dir_config(directory)
    config = dir_configs[directory]
    if not config:
        return job
    names = getattr(job, "test_names", None) or [a for a in args if "-" not in a[:1]]
    for name in names or [""]:
        class_name, _, method = name.partition(".")
        _with_fixtures(job, _fixtures_for(config, path.name, class_name, method))
    return job


MUTATORS = [determine_cores_for_test_case, binless]


def _ctest_script(cmake_test):
    """what a ctest entry runs: its python file or module, else its program"""
    cmd = " ".join(cmake_test.command)
    if "python3" in cmd:
        try:
            bits = shlex.split(cmd)
            return _python_test_target(bits, _python_invocation_index(bits, cmd), cmd)[
                0
            ]
        except ValueError:
            pass
    return cmake_test.command[0] if cmake_test.command else ""


def _ctest_log_paths(cmake_tests):
    """log paths for ctest entries run as they are registered: under their
    working directory and launch (see _launch_dir), named after the test (the
    working directory is shared by every entry in it). Entries running the same
    script there share a directory named for the common start of their names,
    each logging as the rest: `sample-data-1d-h5` and `sample-data-2d-h5` log
    to `sample-data/1d-h5` and `sample-data/2d-h5`"""
    groups = {}
    for i, t in enumerate(cmake_tests):
        log_dir = _LOG_DIR / ".phlop" / _log_subpath(t.working_dir or ".")
        if launch := _launch_dir(" ".join(t.command)):
            log_dir = log_dir / launch
        groups.setdefault((log_dir, _ctest_script(t)), []).append(i)

    paths = [None] * len(cmake_tests)
    for (log_dir, _), idxs in groups.items():
        names = [re.sub(r"[^\w.+-]", "_", cmake_tests[i].name) for i in idxs]
        words = [n.split("-") for n in names]
        common = 0
        if len(names) > 1:
            shortest = min(len(w) for w in words)
            # every name keeps at least one word of its own
            while common < shortest - 1 and len({w[common] for w in words}) == 1:
                common += 1
        for i, w in zip(idxs, words):
            if common:
                paths[i] = str(log_dir / "-".join(w[:common]) / "-".join(w[common:]))
            else:
                paths[i] = str(log_dir / "-".join(w))
    return paths


def load_cmake_tests(cmake_dir, cores=1, test_cmd_pre="", test_cmd_post=""):
    cmake_tests = get_cmake_tests(cmake_dir)
    tests = []
    for cmake_test, log_file_path in zip(cmake_tests, _ctest_log_paths(cmake_tests)):
        cmd = f"{test_cmd_pre} " + " ".join(cmake_test.command) + f" {test_cmd_post}"
        tests += [
            Job(
                cmd=cmd,
                env=cmake_test.env,
                working_dir=cmake_test.working_dir,
                log_file_path=log_file_path,
                provides=cmake_test.fixtures_setup,
                requires=cmake_test.fixtures_required,
                cleans=cmake_test.fixtures_cleanup,
            )
        ]

    test_batches = {}
    dir_configs = {}

    def _add(test_cases):
        for test_case in test_cases:
            yaml_fixtures(test_case, dir_configs)
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
    keys = [
        f"{filename}:{class_name}.{method_name}",
        f"{filename}:{class_name}",
        filename,
    ]
    for key in keys:
        # a dict entry only declares fixtures, see _fixtures_for
        if key in dir_config and not isinstance(dir_config[key], dict):
            if used_keys is not None:
                used_keys.add(key)
            return dir_config[key] or None
    return None


FIXTURE_KEYS = ("provides", "requires", "cleans")


def _fixture_names(value, where):
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset([value])
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return frozenset(value)
    raise ValueError(f"{CONFIG_FILENAME} {where}: fixtures must be a list of names")


def _fixtures_in(entry, where):
    """(provides, requires, cleans) declared in a dict entry or a variant"""
    return tuple(_fixture_names(entry.get(k), where) for k in FIXTURE_KEYS)


def _fixtures_for(dir_config, filename, class_name, method_name, used_keys=None):
    """fixtures declared for a test: a dict value (`provides`, `requires`,
    `cleans` lists) on its file, class or method key, every level that matches
    adds to the others (unlike variants, where the most specific key wins).
    Empty `class_name`/`method_name`: only the file or class keys are looked at"""
    keys = [filename]
    if class_name:
        keys.append(f"{filename}:{class_name}")
        if method_name:
            keys.append(f"{filename}:{class_name}.{method_name}")
    found = (frozenset(), frozenset(), frozenset())
    for key in keys:
        entry = dir_config.get(key)
        if not isinstance(entry, dict):
            continue
        unknown = sorted(set(entry) - set(FIXTURE_KEYS))
        if unknown:
            raise ValueError(
                f"{CONFIG_FILENAME} {key}: unknown keys {unknown}, a dict entry "
                f"declares fixtures only: {', '.join(FIXTURE_KEYS)}"
            )
        if used_keys is not None:
            used_keys.add(key)
        found = tuple(a | b for a, b in zip(found, _fixtures_in(entry, key)))
    return found


def _with_fixtures(job, fixtures):
    """job with `fixtures` (provides, requires, cleans) added to its own"""
    sets = zip(fixtures_of(job), fixtures)
    job.provides, job.requires, job.cleans = (set(a | b) for a, b in sets)
    return job


def _variant_job(
    base_cmd,
    log_file_path,
    variant,
    index,
    test_cmd_pre,
    test_cmd_post,
    test_names,
    fixtures=(),
):
    prefix = variant.get("prefix", "")
    env = dict(variant.get("env") or {})
    working_dir = variant.get("working_dir")
    cmd = " ".join(
        p for p in (test_cmd_pre, prefix, base_cmd, test_cmd_post) if p
    ).strip()
    tags = variant.get("tags") or []

    if log_file_path:
        log_file_path = f"{log_file_path}_{'-'.join(tags) if tags else index}"

    job = Job(
        cmd=cmd,
        env=env,
        working_dir=working_dir,
        log_file_path=log_file_path,
        cores=resolve_cores(
            cmd, env, explicit=variant.get("cores"), threads=variant.get("threads")
        ),
        meta=tags or None,
        test_names=test_names,
    )
    _with_fixtures(job, fixtures)
    return _with_fixtures(job, _fixtures_in(variant, cmd))


def load_config_tests(
    input_path,
    test_cmd_pre="",
    test_cmd_post="",
    python_flags=None,
    tags=None,
    granularity=None,
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

    With a coarser `granularity` than "method", tests sharing a class/file
    and the same declared variants run in one process per variant.
    """
    granularity = _check_granularity(granularity)
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
        classes = classes_in_file(py_file, unittest.TestCase)
        all_methods = {c: _test_methods(loader, c) for c in classes}

        variant_lists = {}  # id -> variants list from dir_config
        entries = []
        for test_class in classes:
            for method in all_methods[test_class]:
                variants = _variants_for(
                    dir_config,
                    py_file.name,
                    test_class.__name__,
                    method,
                    used_keys[directory],
                )
                vkey = None
                if variants:
                    vkey = id(variants)
                    variant_lists[vkey] = variants
                fixtures = _fixtures_for(
                    dir_config,
                    py_file.name,
                    test_class.__name__,
                    method,
                    used_keys[directory],
                )
                # tests grouped into one process share their fixtures
                entries.append((test_class, method, (vkey, fixtures)))

        used_paths = {}
        for key, by_class in _group_tests(entries, granularity):
            test_class = next(iter(by_class))
            names = _test_names(by_class, all_methods)
            test_names = _qualified_names(by_class)
            base_cmd = python3_test_cmd(test_class.__module__, names, python_flags)
            log_file_path = _unique_path(
                _group_logfile(_LOG_DIR / ".phlop", by_class, granularity), used_paths
            )
            vkey, fixtures = key
            variants = variant_lists.get(vkey)

            if not variants:
                cmd = f"{test_cmd_pre} {base_cmd} {test_cmd_post}".strip()
                job = Job(
                    cmd=cmd,
                    log_file_path=log_file_path,
                    cores=resolve_cores(cmd),
                    test_names=test_names,
                )
                _add(_with_fixtures(job, fixtures))
                continue

            for index, variant in enumerate(variants):
                variant_tags = set(variant.get("tags") or [])
                if variant_tags and not (tags and (variant_tags & tags)):
                    continue
                _add(
                    _variant_job(
                        base_cmd,
                        log_file_path,
                        variant,
                        index,
                        test_cmd_pre,
                        test_cmd_post,
                        test_names,
                        fixtures,
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


def _job_key(job):
    """jobs equal in what runs, and where: an `mpirun -n` count is in the cmd"""
    env = getattr(job, "env", None) or {}
    return job.cmd, tuple(sorted(env.items())), getattr(job, "working_dir", None)


def merge_duplicates(test_batches):
    """one job per cmd, env and working_dir, the first seen, with the union of
    the fixtures: dumps loaded together can each hold the same test (e.g. a
    serial and an MPI build both registering the same data generator), which
    would otherwise run twice, concurrently, into the same directory"""
    first = {}
    merged = {}
    for batch in test_batches:
        for job in batch.tests:
            key = _job_key(job)
            if key not in first:
                first[key] = job
                merged.setdefault(batch.cores, []).append(job)
                continue
            kept = first[key]
            sets = zip(fixtures_of(kept), fixtures_of(job))
            kept.provides, kept.requires, kept.cleans = (a | b for a, b in sets)
    return [TestBatch(v, k) for k, v in merged.items()]


def extract_load(directory, globbing):
    test_batches = {}

    file_paths = sorted(Path(directory).glob(globbing))

    if not file_paths:
        raise ValueError("No load files found")

    for path in file_paths:
        with open(path, "r") as file:
            batch_list = deserialize(file.read()).batch_list
            for batch in batch_list:
                batch.check_cores()  # unpickling skips __post_init__
                test_batches.setdefault(batch.cores, []).extend(batch.tests)

    return merge_duplicates([TestBatch(v, k) for k, v in test_batches.items()])
