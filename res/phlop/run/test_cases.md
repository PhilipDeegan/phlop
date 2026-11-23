# `phlop.run.test_cases`

Flexible parallel test runner. Discovers `unittest.TestCase` tests (from
plain directory scans, cmake/ctest registration, or a dump/load cache),
optionally fans individual tests out into declared execution variants via
`.phlop.exec.yaml`, and runs them in parallel.

Invoked as `python3 -m phlop.run.test_cases [options]`.

## CLI arguments

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | `.` | Input file or directory to scan for tests. |
| `--cmake` | `False` | Discover tests via `ctest --show-only=json-v1` in `--input` instead of scanning `.py` files directly (see [cmake/ctest discovery](#cmakectest-discovery) below). |
| `-c`, `--cores` | `1` | Parallelism core/thread count. Accepts `a`/`all` for `multiprocessing.cpu_count()`. |
| `-p`, `--print_only` | `False` | Print discovered tests and exit, no execution. Fixtures print first, each with its setup and cleanup jobs and how many jobs require it, then the remaining tests (see [Fixtures](#fixtures)). |
| `-v`, `--verbose` | `False` | With `--print_only`, print full per-test info (cores, tags, env, working_dir, log_file_path) instead of just the execution string. |
| `--prefix` | `""` | Prepend this string to every execution command. |
| `--postfix` | `""` | Append this string to every execution command. |
| `--dump` | `None` | Dump discovered tests as a dill/hex blob to this filepath, no execution. Mutually exclusive with `--load`. |
| `--load` | `None` | Globbing filepath for files previously written by `--dump`; loads and runs those instead of scanning. Jobs with equal `cmd`, `env` and `working_dir` across the loaded files run once, with their fixtures merged (an `mpirun -n` count is part of `cmd`, so `-n 2` and `-n 4` stay separate). |
| `-r`, `--regex` | `None` | Filter out tests whose execution string doesn't match this regex (falls back to plain substring search if the pattern is invalid). The setup jobs of fixtures the kept tests require are kept too (transitively), as are the cleanups of the fixtures they use, like `ctest -R`. |
| `-R`, `--reverse` | `False` | Reverse batch order - higher core-count batches preferred/run first. |
| `--rerun` | `1` | Number of times to re-execute each discovered test (duplicates the job, suffixing `log_file_path` with `_<i>`). |
| `--logging` | `1` | `0`=off, `1`=on non-zero exit code, `2`=always. Logs go to `.phlop/<dir>/[<mpirun\|mpiexec\|srun>_n_<N>/]<file>/<Class>/<method>.stdout/.stderr` (a ctest entry run as-is: `.phlop/<working_dir>/[<mpirun\|mpiexec\|srun>_n_<N>/]<ctest name>`, entries running the same script there grouped as `<common start of their names>/<rest>`, e.g. `sample-data/1d-h5`); each `.stdout` starts with the exact command run. With `1` a job's logs are written to a temporary directory while it runs (the stall check watches them) and moved under `.phlop/` only if it fails: nothing is written there for a job that passes. |
| `--no-phlop-exec-yaml` | `False` | Ignore any `.phlop.exec.yaml` configs found while scanning; just run discovered tests once each, with defaults. Has no effect together with `--cmake` (cmake discovery reads only fixtures from `.phlop.exec.yaml` - see below). |
| `--tags` | `None` | Comma-separated tags to opt into running tagged `.phlop.exec.yaml` variants (tagged variants are otherwise skipped). Untagged/undeclared tests are unaffected and always run. |
| `--phases` | `None` | Comma-separated phases (names, or 1-based indices) of the input directory's `.phlop.exec.yaml` to run; its unphased entries are skipped (see [Scopes, chaining and phases](#scopes-chaining-and-phases)). |

## Discovery modes

`get_test_cases` (`phlop/run/test_cases.py`) picks one of three discovery
paths, in this order:

1. **`--cmake`** → `tc.load_cmake_tests(input)` - reads ctest's registered
   tests and expands Python ones into their individual `unittest` methods
   (see [cmake/ctest discovery](#cmakectest-discovery)). Only the fixtures
   of a `.phlop.exec.yaml` are used in this mode, not its variants.
2. **default** (no `--cmake`, no `--no-phlop-exec-yaml`) →
   `tc.load_config_tests(input, tags=...)` - scans `.py` files under `input`
   directly and honours `.phlop.exec.yaml` (see below).
3. **`--no-phlop-exec-yaml`** → scans `.py` files under `input` directly via
   `classes_in_file`/`classes_in_directory`, ignoring any `.phlop.exec.yaml`.
   Every test runs exactly once, with defaults.

## `.phlop.exec.yaml`

Dropped into any directory scanned by `load_config_tests` (the default
discovery mode). Declares execution **variants** for specific tests - e.g. an
mpirun-wrapped run, a scaled-up env var, an explicit core count - without
touching the test file itself. A test with no matching entry (or with no
`.phlop.exec.yaml` present at all) just runs once, with `cores=1` and no
extra env/prefix/working_dir.

### Key format

```
<filename>:<ClassName>:
  - <variant>
  - <variant>

<filename>:<ClassName>.<method_name>:
  - <variant>
```

- The **filename** (e.g. `test_foo.py`, matching `Path.name` - extension
  included) is always required. A directory can hold multiple files that
  each define a class of the same name, so the bare class name alone would
  be ambiguous.
- `filename:ClassName.method_name` (qualified) takes precedence over a bare
  `filename:ClassName` entry for that method.
- A bare `filename:ClassName` entry applies to every method on that class
  that doesn't have its own qualified entry.
- The value is always a **list** of variant dicts - a test can have several
  variants (e.g. `small`/`large` sizings), each becoming its own job when
  its tags are opted into (or unconditionally, if untagged).
- An empty list (`[]`) is equivalent to no entry: the test runs once, as
  normal.
- Any key not naming a `.py` file is a glob of **executables** relative to
  the yaml's directory (e.g. a gtest binary; `{a,b}` alternatives expanded),
  with the same variants: one job per executable file matched per variant,
  logged under `.phlop/<path of the binary>_<variant>`, `<variant>` being its
  own tags and launch (`mpi_mpirun_n_8`), joined with `_`, else its index.

**Every key must reference something that actually exists.** After
scanning a directory, `load_config_tests` raises `ValueError` if any
declared key never matched a real `Class`/`Class.method` found in that
directory (or an executable key matched no executable file; keys of every
chained file are checked) - this catches typos, wrong separators (`.` instead of `:`),
missing `.py` extensions, and stale entries left over after a rename, rather
than silently doing nothing.

### Variant fields

| Field | Type | Effect |
|---|---|---|
| `tags` | list of str | Opts this variant in via `--tags` on the CLI. A **tagged** variant only runs when `--tags` is passed and intersects with it. An **untagged** variant (no `tags`, or empty) always runs regardless of `--tags` - this keeps anything relying on e.g. `mpirun` or other unavailable/heavy resources off by default. |
| `prefix` | str | Prepended to the test's invocation, e.g. an `mpirun -n 2` wrapper. |
| `env` | dict | Extra environment variables for this run (merged into the job's env). |
| `working_dir` | str | Working directory for this run. |
| `cores` | int | Explicit total core count. Always wins outright over everything below. |
| `threads` | int | Threads-per-rank for parallelism phlop can't introspect from the command itself (e.g. a `std::thread`/thread-pool size passed via `env`/args). Multiplies with any `mpirun`/`mpiexec`/`srun` rank count detected from `prefix` (`mpirun -n 2` + `threads: 4` ⇒ 8 total cores); on its own (no mpirun) it's just the total core count directly. |

Core-count resolution order (`resolve_cores`, `phlop/testing/test_cases.py`):
`cores` (explicit) wins outright → otherwise `threads` × any detected
`mpirun`/`mpiexec`/`srun` rank count (`-n`/`-np`/`--np`/`--ntasks`, other options before it) → otherwise an `OMP_NUM_THREADS` in
`env` → otherwise defaults to `1`.

### Scopes, chaining and phases

A file's top level is a **scope**: `env prefix postfix tags cores threads
working_dir ignore_errors` there apply to everything in it. Scopes nest - a
subdirectory's file, a file reached through `paths:`, a phase, and finally a
variant - each inheriting the one above, mkn style: an inner key replaces the
outer one, `${key}` in it is the outer value.

```yaml
prefix: timeout 600
env: {PHLOP_STALL_BUSY_TIMEOUT: 300}
tags: [gpu]
# ... in a nested scope (a phase, a subdirectory's yaml, a variant):
prefix: ${prefix} mpirun -n 2   # timeout 600 mpirun -n 2
env: {PATH: /opt/x/bin:${PATH}} # per variable; the process env if no outer one
tags: ["${tags}", mpi]          # [gpu, mpi]
```

A scope also holds:

- `paths:` (`path:` for one): directories, other yaml files, python test files
  (globs and `{a,b}` too), relative to the yaml, run as scopes nested in this one. A
  directory reached this way is not scanned again as a subdirectory; a
  directory or file reached twice runs once; a file chaining back to itself is
  an error.
- `phases:` a list of scopes (optionally `name:`d) run one after the other:
  each starts once the previous has finished. A failing phase stops the later
  ones (their jobs fail unrun, like a failed fixture's), unless it has
  `ignore_errors: true`. The scope's other entries (its own tests, its
  directory's other python files and subdirectories, its executables) run
  before the first phase. A python key inside a phase moves those tests into it.

```yaml
# tst/.phlop.exec.yaml
env: {PHLOP_STALL_BUSY_TIMEOUT: 300}
phases:
  - name: core
    "../bin/test_core/*": [{}]
  - name: mpi
    tags: [mpi]
    "../bin/test_mpi/*":
      - {prefix: "mpirun --oversubscribe -n 2"}
      - {prefix: "mpirun --oversubscribe -n 8"}
  - name: hdf5
    ignore_errors: true
    path: hdf5
```

`--phases mpi,hdf5` runs just those (in order); `-p` shows each job's phase.

### Example

```yaml
# tests/_phlop/run/.phlop.exec.yaml
test_test_cases.py:ExecConfigDemoTest.test_scaled:
  - tags: [small]
    env: {DEMO_GRID_SIZE: "64"}
  - tags: [thread_pool]
    env: {DEMO_GRID_SIZE: "128"}
    threads: 4
  - tags: [large, high_load]
    prefix: "mpirun -n 2"
    env: {DEMO_GRID_SIZE: "256"}
    working_dir: tests/_phlop/run
    threads: 4
```

- `ExecConfigDemoTest.test_scaled` gets 3 variants; none run by default
  (all are tagged). `--tags small` opts in just the first; `--tags
  thread_pool,high_load` would opt in the second and third.
- Any other test in `test_test_cases.py`, or any test in a different file in
  the same directory, is unaffected and runs once, as normal.

## Stalled jobs, logs and CI

- `PHLOP_STALL_TIMEOUT` (default 3600, 0=off): seconds without a job's log
  files changing before it is killed as stalled; with psutil only while it is
  also idle. `PHLOP_STALL_BUSY_TIMEOUT` (default 14400, 0=off): with psutil, a
  busy job without output is killed after this anyway. Both are read from a
  job's own env first, so a `.phlop.exec.yaml` `env:` sets them per scope or
  variant, else from phlop's environment.
- psutil is optional (`pip install phlop[psutil]`), and a CI running MPI jobs
  wants it: without it a stalled `mpirun`'s ranks are not killed with it, and
  the stall check cannot tell a busy job from a hung one.
- Logs go under `$PHLOP_LOG_DIR/.phlop/` (default the current directory), at
  each file's path relative to `PHLOP_LOG_DIR`, else to the current directory,
  else its absolute path. Upload that directory from CI, e.g.
  `actions/upload-artifact` with `if: always()`.

## cmake/ctest discovery

With `--cmake`, `load_cmake_tests` (`phlop/testing/test_cases.py`) reads
`ctest --show-only=json-v1` (`phlop/app/cmake.py:list_tests`) and routes each
registered test through `EXTRACTORS` in order:
`PythonUnitTestCaseExtractor` → `GoogleTestCaseExtractor` (currently a
no-op) → `DefaultTestCaseExtractor` (runs the ctest command verbatim,
unexpanded).

For a Python ctest entry (its command contains `python3`),
`load_py_test_cases_from_cmake` finds the target script/module in the
command, then:

- **If the command has no trailing arguments after the target** (e.g.
  `python3 -u test_foo.py`), it imports that file and expands it into one
  job per `unittest.TestCase` method found - each running
  `python3 -um <module> <Class>.<method>`.
- **If the command has trailing arguments** (e.g. `python3 -u test_foo.py
  SomeTest.test_id`, or any other trailing arg), it is treated as already
  scoped to a specific invocation and run as-is, unexpanded. This matters
  because cmake can register the *same* `.py` file more than once with
  different trailing args (e.g. one ctest entry per test id) - re-expanding
  the file for each such entry would both duplicate every test in the file
  per entry and discard the args that distinguish them.

Variants and tags in `.phlop.exec.yaml` are a `load_config_tests`-only
feature; its [fixtures](#fixtures) apply in cmake mode too.

## Fixtures

Jobs can depend on other jobs' output, as ctest fixtures (CMake >= 3.7):

- a job **requiring** a fixture starts only once every job **providing** it
  has finished; if one of them failed it fails without running (`fixture X
  failed`), and so do the jobs requiring a fixture *it* provides;
- a job **cleaning** a fixture starts once every job providing or requiring
  it is done, and runs whatever their outcome;
- a required fixture that no loaded job provides is ignored;
- a dependency cycle is an error before anything runs.

Jobs are scheduled as each one finishes, not in waves: a test waits only for
its own fixtures. Jobs others wait on (by how many, transitively) start
first, then the widest.

Fixtures come from:

- **ctest**, with `--cmake`: the `FIXTURES_SETUP`, `FIXTURES_REQUIRED` and
  `FIXTURES_CLEANUP` properties of an entry, given to every job made from it
  (each `Class.method` a Python file is split into);
- **`.phlop.exec.yaml`**, in both modes: a dict value on a file, class or
  method key declares the fixtures of the tests it matches (every matching
  level adds to the others, unlike variants), and a variant can declare its
  own. With `--cmake` the yaml beside the Python file a job runs adds to the
  ctest properties; its keys are not checked there, since the yaml may name
  tests ctest doesn't register.

```yaml
gen_data.py:
  provides: [amr_data]
test_run_amr.py:
  requires: [amr_data]
test_run_amr.py:RunTest.test_cleanup:
  cleans: [amr_data]
test_mhd.py:MHDTest.test_2d:
  - tags: [mpi]
    prefix: mpirun -n 2
    requires: [amr_data]
```

In Python, `phlop.procs.parallel_processor.Job` carries the fixture names as
the sets `provides`, `requires` and `cleans`.
