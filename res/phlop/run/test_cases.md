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
| `-p`, `--print_only` | `False` | Print discovered tests and exit, no execution. |
| `-v`, `--verbose` | `False` | With `--print_only`, print full per-test info (cores, tags, env, working_dir, log_file_path) instead of just the execution string. |
| `--prefix` | `""` | Prepend this string to every execution command. |
| `--postfix` | `""` | Append this string to every execution command. |
| `--dump` | `None` | Dump discovered tests as a dill/hex blob to this filepath, no execution. Mutually exclusive with `--load`. |
| `--load` | `None` | Globbing filepath for files previously written by `--dump`; loads and runs those instead of scanning. |
| `-r`, `--regex` | `None` | Filter out tests whose execution string doesn't match this regex (falls back to plain substring search if the pattern is invalid). |
| `-R`, `--reverse` | `False` | Reverse batch order - higher core-count batches preferred/run first. |
| `--rerun` | `1` | Number of times to re-execute each discovered test (duplicates the job, suffixing `log_file_path` with `_<i>`). |
| `--logging` | `1` | `0`=off, `1`=on non-zero exit code, `2`=always. |
| `--no-phlop-exec-yaml` | `False` | Ignore any `.phlop.exec.yaml` configs found while scanning; just run discovered tests once each, with defaults. Has no effect together with `--cmake` (cmake discovery never reads `.phlop.exec.yaml` - see below). |
| `--tags` | `None` | Comma-separated tags to opt into running tagged `.phlop.exec.yaml` variants (tagged variants are otherwise skipped). Untagged/undeclared tests are unaffected and always run. |

## Discovery modes

`get_test_cases` (`phlop/run/test_cases.py`) picks one of three discovery
paths, in this order:

1. **`--cmake`** → `tc.load_cmake_tests(input)` - reads ctest's registered
   tests and expands Python ones into their individual `unittest` methods
   (see [cmake/ctest discovery](#cmakectest-discovery)). `.phlop.exec.yaml`
   is never consulted in this mode.
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

**Every key must reference something that actually exists.** After
scanning a directory, `load_config_tests` raises `ValueError` if any
declared key never matched a real `Class`/`Class.method` found in that
directory - this catches typos, wrong separators (`.` instead of `:`),
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
| `threads` | int | Threads-per-rank for parallelism phlop can't introspect from the command itself (e.g. a `std::thread`/thread-pool size passed via `env`/args). Multiplies with any `mpirun` rank count detected from `prefix` (`mpirun -n 2` + `threads: 4` ⇒ 8 total cores); on its own (no mpirun) it's just the total core count directly. |

Core-count resolution order (`resolve_cores`, `phlop/testing/test_cases.py`):
`cores` (explicit) wins outright → otherwise `threads` × any detected
`mpirun -n`/`-np`/`--np` rank count → otherwise an `OMP_NUM_THREADS` in
`env` → otherwise defaults to `1`.

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

`.phlop.exec.yaml` is **not** consulted in cmake mode - variants/tags are a
`load_config_tests`-only feature.
