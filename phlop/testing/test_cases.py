# phlop/testing/test_cases.py

import glob as globmod
import itertools
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
from phlop.testing.exec_scope import RESERVED_KEYS, nest, scope_of

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
    """`path` relative to _LOG_DIR, else to the current directory, else the
    full path without its root"""
    path = Path(path).resolve()
    for base in (_LOG_DIR, Path.cwd()):
        try:
            return path.relative_to(base.resolve())
        except ValueError:
            pass
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
    """a log path part for how a command is launched, `mpirun_n_2` (or
    `mpiexec_n_2`, `srun_n_2`), None if
    plainly: the same test run with different rank counts logs apart"""
    m = _MPIRUN_CORES_RE.search(cmd)
    return f"{m.group(1)}_n_{m.group(2)}" if m else None


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


# the rank count of an mpirun, mpiexec or srun: its first -n/-np/--np/--ntasks
#  (`-n 8`, `-n8`, `--ntasks=8`), other options before it (`mpirun
#  --oversubscribe --bind-to none -n 8`): (launcher, ranks)
_MPIRUN_CORES_RE = re.compile(
    r"\b(mpirun|mpiexec|srun)(?:\s+\S+)*?\s+(?:-n|-np|--np|--ntasks)(?:\s+|=)?(\d+)\b"
)


def mpirun_cores(cmd, env):
    m = _MPIRUN_CORES_RE.search(cmd)
    return int(m.group(2)) if m else None


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


def _braces(pattern):
    """`pattern` with its `{a,b}` groups expanded (nested too), as a shell
    does: python's glob does not"""
    depth, start = 0, None
    for i, c in enumerate(pattern):
        if c == "{":
            if depth == 0:
                start, commas = i, []
            depth += 1
        elif c == "," and depth == 1:
            commas.append(i)
        elif c == "}" and depth:
            depth -= 1
            if depth == 0 and commas:
                cuts = [start, *commas, i]
                rest = _braces(pattern[i + 1 :])
                return [
                    pattern[:start] + alt + tail
                    for a, b in itertools.pairwise(cuts)
                    for alt in _braces(pattern[a + 1 : b])
                    for tail in rest
                ]
            if depth == 0:  # `{x}`: no alternatives, kept as is
                return [pattern[: i + 1] + tail for tail in _braces(pattern[i + 1 :])]
    return [pattern]


def _glob(directory, pattern):
    """paths matching `pattern` (relative to `directory`), `{a,b}` expanded,
    normalised (`tst/../bin` -> `bin`), sorted, each once"""
    if not os.path.isabs(pattern):
        pattern = os.path.join(directory, pattern)
    found = set()
    for each in _braces(os.path.normpath(pattern)):
        if re.search(r"[*?[]", each):
            found.update(globmod.glob(each))
        elif os.path.exists(each):
            found.add(each)
    return sorted(os.path.normpath(p) for p in found)


def _is_py_key(key):
    """`file.py[:Class[.method]]`, else an executable glob"""
    return key.partition(":")[0].endswith(".py")


def _read_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    import yaml

    with open(config_path) as f:
        body = yaml.safe_load(f) or {}
    if not isinstance(body, dict):
        raise ValueError(f"{config_path}: not a mapping")  # noqa: TRY004
    return body


def _config_paths(body, where):
    """the `paths:` (or `path:`) of a scope, as a list"""
    found = []
    for key in ("path", "paths"):
        value = body.get(key)
        if value is None:
            continue
        value = [value] if isinstance(value, str) else value
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{where}: {key} must be a path or a list of paths")
        found += value
    return found


def _phases_of(body, where):
    phases = body.get("phases") or []
    if not isinstance(phases, list) or not all(isinstance(p, dict) for p in phases):
        raise ValueError(f"{where}: phases must be a list of mappings")
    return phases


def _phase_name(phase, index):
    return str(phase.get("name") or index)


def _entry(value, where):
    """(variants, fixtures) of a key's value: a list of variants, or a dict
    declaring fixtures only, or nothing (run once)"""
    if value is None:
        return None, (frozenset(), frozenset(), frozenset())
    if isinstance(value, dict):
        unknown = sorted(set(value) - set(FIXTURE_KEYS))
        if unknown:
            raise ValueError(
                f"{CONFIG_FILENAME} {where}: unknown keys {unknown}, a dict entry "
                f"declares fixtures only: {', '.join(FIXTURE_KEYS)}"
            )
        return None, _fixtures_in(value, where)
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"{CONFIG_FILENAME} {where}: must be a list of variants")
    return value or None, (frozenset(), frozenset(), frozenset())


def _variant_suffix(variant, scope, base_cmd, index):
    """a variant's log name: its own tags and how it is launched
    (`mpi-mpirun_n_8`), its index if neither"""
    own = list(scope.get("tags") or []) if "tags" in variant else []
    launch = _launch_dir(f"{scope.get('prefix') or ''} {base_cmd}")
    parts = [str(t) for t in own] + ([launch] if launch else [])
    return "-".join(parts) if parts else index


def _scoped_job(base_cmd, log_file_path, scope, test_cmd_pre, test_cmd_post, phase):
    """a job running `base_cmd` in `scope` (a scope with a variant nested in)"""
    parts = (test_cmd_pre, scope.get("prefix"), base_cmd, scope.get("postfix"))
    cmd = " ".join(p for p in (*parts, test_cmd_post) if p).strip()
    env = dict(scope.get("env") or {})
    tags = list(scope.get("tags") or [])
    job = Job(
        cmd=cmd,
        env=env,
        working_dir=scope.get("working_dir"),
        log_file_path=log_file_path,
        cores=resolve_cores(
            cmd, env, explicit=scope.get("cores"), threads=scope.get("threads")
        ),
        meta=tags or None,
    )
    if phase:
        job.phase = list(phase)  # see phlop.testing.exec_scope
    return job


class _Keys:
    """the test keys of one scope of a config: its python and executable keys,
    the scope they run in, which python keys matched a test"""

    def __init__(self, directory, label, keys, scope, phase):
        self.directory = directory
        self.label = label
        self.keys = keys
        self.scope = scope
        self.phase = phase
        self.used = set()

    def match(self, filename, class_name, method):
        """None if no key names the test, else its (variants, fixtures): the
        most specific variants key, the fixtures of every key"""
        names = [f"{filename}:{class_name}.{method}", f"{filename}:{class_name}"]
        names.append(filename)
        claimed, variants = False, None
        for key in names:
            if key in self.keys and not isinstance(self.keys[key], dict):
                self.used.add(key)
                variants, _ = _entry(self.keys[key], key)
                claimed = True
                break
        fixtures = _fixtures_for(self.keys, filename, class_name, method, self.used)
        claimed |= any(isinstance(self.keys.get(key), dict) for key in names)
        return (variants, fixtures) if claimed else None


class _ConfigWalk:
    """the jobs of directories and the .phlop.exec.yaml files chained from them,
    see load_config_tests"""

    def __init__(self, tags, granularity, test_cmd_pre, test_cmd_post, python_flags):
        self.tags = tags
        self.granularity = granularity
        self.test_cmd_pre = test_cmd_pre
        self.test_cmd_post = test_cmd_post
        self.python_flags = python_flags
        self.loader = unittest.TestLoader()
        self.jobs = []
        self.keys = []  # every _Keys, for the stale check
        self.exe_keys = []  # (label, key, matched anything)
        self.scanned = {}  # directory -> names of the python files scanned
        self.seen = set()  # (directory, config) walked
        self.seen_files = set()  # python files scanned
        self.loading = []  # configs being read, for cycles
        self.used_logs = {}

    def directory(self, directory, scope, phase, config_path=None, py_files=None):
        """the tests of `directory` (only `py_files` of it if given, and then
        none of its executables, chained paths or subdirectories) with its
        config (or `config_path`) nested in `scope`"""
        directory = Path(directory)
        config_path = Path(config_path or directory / CONFIG_FILENAME)
        whole = py_files is None
        resolved = config_path.resolve()
        if resolved in self.loading:
            chain = [*self.loading[self.loading.index(resolved) :], resolved]
            raise ValueError(
                f"{CONFIG_FILENAME} cycle: " + " -> ".join(str(c) for c in chain)
            )
        if whole:
            if (directory.resolve(), resolved) in self.seen:
                return  # reached before, runs once
            self.seen.add((directory.resolve(), resolved))

        body = _read_config(config_path)
        self.loading.append(resolved)
        try:
            keys = []
            top = nest(scope, scope_of(body))
            self._scope(directory, body, top, phase, str(config_path), keys, whole)
            files = sorted(directory.glob("*.py")) if whole else py_files
            self._python(directory, files, keys)
            if whole:
                for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
                    self.directory(sub, keys[0].scope, keys[0].phase)
        finally:
            self.loading.pop()

    def _scope(self, directory, body, scope, phase, label, keys, follow):
        """a scope (`body` already nested into `scope`) and its phases"""
        phases = _phases_of(body, label)
        outer = phase
        if phases:  # its unphased entries run first
            phase = (*phase, (label, 0, None, bool(scope.get("ignore_errors"))))
        own = {str(k): v for k, v in body.items() if k not in RESERVED_KEYS}
        py_keys = {k: v for k, v in own.items() if _is_py_key(k)}
        keys.append(_Keys(directory, label, py_keys, scope, phase))
        self.keys.append(keys[-1])
        if follow:
            for key, value in own.items():
                if not _is_py_key(key):
                    self._executables(directory, key, value, scope, phase, label)
            for entry in _config_paths(body, label):
                self._path(directory, entry, scope, phase, label)
        for i, sub in enumerate(phases, 1):
            sub_scope = nest(scope, scope_of(sub))
            ignore = bool(sub_scope.get("ignore_errors"))
            sub_phase = (*outer, (label, i, _phase_name(sub, i), ignore))
            sub_label = f"{label}[{i}]"
            self._scope(directory, sub, sub_scope, sub_phase, sub_label, keys, follow)

    def _path(self, directory, entry, scope, phase, label):
        """a `paths:` entry: directories, config files, python test files"""
        matches = _glob(directory, entry)
        if not matches:
            raise ValueError(f"{label}: paths entry {entry!r} matches nothing")
        py_files = {}
        for match in map(Path, matches):
            if match.is_dir():
                self.directory(match, scope, phase)
            elif match.suffix in (".yaml", ".yml"):
                self.directory(match.parent, scope, phase, config_path=match)
            elif match.suffix == ".py":
                py_files.setdefault(match.parent, []).append(match)
            else:
                raise ValueError(f"{label}: paths entry {entry!r} matches {match}")
        for parent, files in py_files.items():
            self.directory(parent, scope, phase, py_files=files)

    def _executables(self, directory, key, value, scope, phase, label):
        exes = [
            p
            for p in _glob(directory, key)
            if os.path.isfile(p) and os.access(p, os.X_OK)
        ]
        self.exe_keys.append((label, key, bool(exes)))
        variants, fixtures = _entry(value, key)
        for exe in exes:

            def base(scope, exe=exe):
                # elsewhere than here, a relative path would not be found
                return shlex.quote(
                    os.path.abspath(exe) if scope.get("working_dir") else exe
                )

            log = _LOG_DIR / ".phlop" / _log_subpath(exe)
            log = _unique_path(str(log), self.used_logs)
            self._jobs(base, log, variants, scope, phase, [], fixtures)

    def _python(self, directory, files, keys):
        top, phased = keys[0], keys[1:]
        scanned = self.scanned.setdefault(directory.resolve(), set())
        for py_file in files:
            if py_file.resolve() in self.seen_files:
                continue  # reached before, runs once
            self.seen_files.add(py_file.resolve())
            scanned.add(py_file.name)
            classes = classes_in_file(py_file, unittest.TestCase)
            all_methods = {c: _test_methods(self.loader, c) for c in classes}

            variant_lists = {}  # id -> variants list from a config
            entries = []
            for test_class in classes:
                for method in all_methods[test_class]:
                    names = py_file.name, test_class.__name__, method
                    # named in a phase: runs there, else unphased
                    claims = [(i, k.match(*names)) for i, k in enumerate(phased, 1)]
                    claims = [(i, found) for i, found in claims if found]
                    if not claims:
                        found = top.match(*names)
                        claims = [(0, found or (None, _entry(None, "")[1]))]
                    for i, (variants, fixtures) in claims:
                        vkey = None
                        if variants:
                            vkey = id(variants)
                            variant_lists[vkey] = variants
                        # tests grouped into one process share their fixtures
                        entries.append((test_class, method, (i, vkey, fixtures)))

            for key, by_class in _group_tests(entries, self.granularity):
                test_class = next(iter(by_class))
                names = _test_names(by_class, all_methods)
                base_cmd = python3_test_cmd(
                    test_class.__module__, names, self.python_flags
                )
                log = _unique_path(
                    _group_logfile(_LOG_DIR / ".phlop", by_class, self.granularity),
                    self.used_logs,
                )
                i, vkey, fixtures = key
                self._jobs(
                    lambda _, cmd=base_cmd: cmd,
                    log,
                    variant_lists.get(vkey),
                    keys[i].scope,
                    keys[i].phase,
                    _qualified_names(by_class),
                    fixtures,
                )

    def _jobs(self, base, log, variants, scope, phase, test_names, fixtures):
        """a job per variant nested in `scope` (one if no variants), tagged
        ones only if asked for with `tags`"""
        runs = [(None, {})] if not variants else list(enumerate(variants))
        for index, variant in runs:
            run_scope = nest(scope, variant)
            run_tags = set(run_scope.get("tags") or [])
            if run_tags and not (self.tags and (run_tags & self.tags)):
                continue
            path = log
            if log and index is not None:
                suffix = _variant_suffix(variant, run_scope, base(run_scope), index)
                path = _unique_path(f"{log}_{suffix}", self.used_logs)
            job = _scoped_job(
                base(run_scope),
                path,
                run_scope,
                self.test_cmd_pre,
                self.test_cmd_post,
                phase,
            )
            job.test_names = test_names
            _with_fixtures(job, fixtures)
            self.jobs.append(_with_fixtures(job, _fixtures_in(variant, job.cmd)))

    def check_stale(self):
        """every declared key matched something: a python key whose file was
        scanned (or does not exist), an executable key"""
        used = {(k.label, key) for k in self.keys for key in k.used}
        stale = []
        for k in self.keys:
            scanned = self.scanned.get(k.directory.resolve(), set())
            for key in k.keys:
                filename = key.partition(":")[0]
                if (k.label, key) in used:
                    continue
                if filename in scanned or not (k.directory / filename).exists():
                    stale.append(f"{k.label}: {key}")
        stale += [f"{label}: {key}" for label, key, ok in self.exe_keys if not ok]
        if stale:
            raise ValueError(
                f"{CONFIG_FILENAME} references tests that were never found "
                f"while scanning - {'; '.join(sorted(set(stale)))}"
            )


def _select_phases(jobs, config_path, phases):
    """the jobs in `phases` (names, or 1-based indices) of `config_path`"""
    listed = _phases_of(_read_config(config_path), str(config_path))
    if not listed:
        raise ValueError(f"phases {phases} asked for, {config_path} has none")
    names = {_phase_name(p, i): i for i, p in enumerate(listed, 1)}
    wanted = set()
    for token in phases:
        token = str(token).strip()
        if token in names:
            wanted.add(names[token])
        elif token.isdigit() and 1 <= int(token) <= len(listed):
            wanted.add(int(token))
        else:
            raise ValueError(
                f"phase {token!r} not in {config_path}: {', '.join(names)}"
            )

    def selected(job):
        phase = getattr(job, "phase", None)
        return phase and phase[0][0] == str(config_path) and phase[0][1] in wanted

    return [job for job in jobs if selected(job)]


def load_config_tests(
    input_path,
    test_cmd_pre="",
    test_cmd_post="",
    python_flags=None,
    tags=None,
    granularity=None,
    phases=None,
):
    """Scan a file or directory for tests, honouring `.phlop.exec.yaml` configs.

    A config is a scope: `env prefix postfix tags cores threads working_dir
    ignore_errors`, inherited by the scopes nested in it (see
    phlop.testing.exec_scope for how), holding any mix of:
      - python test keys `file.py[:Class[.method]]` of its own directory, and
        executable globs (any other key), each with a list of variants (the
        innermost scope, each a run) or a dict declaring fixtures
      - `paths:` (`path:`): directories, config files, python test files
        (globs too), relative to it, run as nested scopes
      - `phases:`: scopes run one after the other, a failing one stopping the
        rest unless `ignore_errors`; its other entries run before them
    A directory's subdirectories are scanned in its scope, except those reached
    through `paths:` (a directory or file reached twice runs once). A config
    reached from itself through `paths:` is an error.

    Tests/files not declared in a config (or with no config present at all)
    run once, as normal. Tagged runs are opt-in: they only run when `tags` is
    given and intersects with their tags. Untagged ones always run.

    Every key must match: a python key one of the tests scanned (or name a file
    not scanned, as when one file is given), an executable key an executable
    file. A stale or mistyped entry raises a ValueError rather than silently
    doing nothing.

    `phases` (names, or 1-based indices) of the input directory's config: only
    those run, not its unphased entries.

    With a coarser `granularity` than "method", tests sharing a class/file
    and the same declared variants run in one process per variant.
    """
    granularity = _check_granularity(granularity)
    tags = set(tags) if tags else None
    path = Path(input_path)
    walk = _ConfigWalk(tags, granularity, test_cmd_pre, test_cmd_post, python_flags)
    if path.is_file():
        walk.directory(path.parent, {}, (), py_files=[path])
    else:
        walk.directory(path, {}, ())
    walk.check_stale()

    jobs = walk.jobs
    if phases:
        root = path.parent if path.is_file() else path
        jobs = _select_phases(jobs, root / CONFIG_FILENAME, phases)

    test_batches = {}
    for job in jobs:
        test_batches.setdefault(job.cores, []).append(job)
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
