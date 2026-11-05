# phlop module summary

phlop wraps tools (test running, perf, valgrind, process stats, scope timers)
behind sane defaults. This is a map of the code, with the non-obvious
behaviour and the performance notes behind past changes.

## Process running

### `phlop/proc.py`
- `run()` → `RunTimer`, the one place subprocesses are started.
- `run_raw()` → bare `Popen`, used by stats_man.
- `run_mp()` runs commands on a thread pool, by default
  `min(len(cmds), usable_cores())` threads, at least 1.

### `phlop/procs/runtimer.py` - `RunTimer`
- Starts a command with `Popen` (default) or `subprocess.run`, timed, with
  optional stdout/stderr capture and per-job `.stdout`/`.stderr` log files.
- `working_dir` is passed as Popen `cwd=`, never `os.chdir`: the cwd is
  process wide and jobs run on threads. A relative `log_file_path` resolves
  against `working_dir`.
- Stall detection (`_StallMonitor`): a job is killed after `stall_timeout`
  seconds without its log files changing. With psutil, only if the process
  tree is also under 50% of a cpu, or after `stall_busy_timeout` regardless.
- `_kill_tree` terminates, then kills, the process and its descendants
  (psutil), e.g. mpi ranks.
- `on_popen` hands the started `Popen` to the caller, used for cancellation.

### `phlop/procs/parallel_processor.py` - the scheduler
- `Job` is a command plus env, working_dir, log file and `cores`. Calling it
  runs it via `RunTimer`. `normalize()` accepts Jobs, TestBatches,
  `{cores: [items]}` or strings.
- `process()` runs jobs on threads, one thread per running job, waiting on
  that job's subprocess. There is no Python worker process per job. Each
  thread always puts exactly one result on the queue.
- Scheduling: pending jobs are sorted widest (`cores`) first, stably. Jobs
  start whenever their cores fit in what's free. Results come back in input
  order.
- Fail-fast and Ctrl-C (any exception out of the loop) terminate, then kill,
  every running job's process tree.
- Env: `PHLOP_STALL_TIMEOUT` (3600), `PHLOP_STALL_BUSY_TIMEOUT` (14400),
  `PHLOP_FAIL_FAST`.

## Testing

### `phlop/testing/test_cases.py` - discovery
- `load_config_tests()` is the default CLI path. It scans `*.py`, imports
  each file (`phlop.reflection.classes_in_file`) to find `unittest.TestCase`
  classes, and applies per-directory `.phlop.exec.yaml` variants
  (prefix/env/working_dir/cores/threads/tags). Stale yaml keys raise.
- Granularity (`GRANULARITY`, CLI `-g`): `method` (default) runs one process
  per test method. `class`/`file` group methods that share the same yaml
  variants into one `python3 -um module Class ClassB.test_x` process. Each
  process pays interpreter startup and imports, so grouping helps most with
  heavy imports, and least when tests are mostly sleeping.
- `resolve_cores()` infers a job's cores from `mpirun -n N`,
  `OMP_NUM_THREADS` and yaml `threads`/`cores`.
- `load_cmake_tests()` uses `ctest --show-only=json-v1`. Python ctest
  entries are expanded per method, others run as-is.
- Log files go to `$PHLOP_LOG_DIR/.phlop/<path>/<file>/<Class>/<method>`,
  shortened for coarser granularity.

### `phlop/run/test_cases.py` - CLI (`python3 -m phlop.run.test_cases`)
- `-c` defaults to usable cores (`phlop.os.usable_cores`, affinity aware).
- `-g method|class|file`, `-r` regex on commands (with class/file, a match
  runs the whole group), `--tags`, `--rerun`, `--logging 0|1|2`,
  `--dump/--load` (hex dill pickle of discovered batches), `-p` print only.

### `phlop/testing/parallel_processor.py`
Test flavoured wrapper of `procs.process`: print-only mode, and log tails
printed on failure.

### `phlop/reflection.py`, `phlop/sys.py`
`classes_in_file` imports a file in a scoped way: modules first imported from
its directory are dropped afterwards (`scoped_modules`). Same-named helpers in
different directories therefore don't collide, at the cost of re-importing
shared helpers for each file.

## Tool wrappers (`phlop/app`, CLIs in `phlop/run`)
- `perf.py`: `perf stat`/`record` command builders. Needs
  `kernel.perf_event_paranoid` low enough. `sh/test.sh` tolerates failure.
- `valgrind.py`: memcheck / massif.
- `stats_man.py`: runs a program and samples its fds, rss and cpu.
  - Keeps one `psutil.Process` and calls `cpu_percent(None)`, so each sample
    covers the whole interval (the first covers 0.1s).
  - A thread samples while the main thread waits on (reaps) the child.
    Polling `os.kill(pid, 0)` instead never ends: zombies still answer it.
  - yaml headers: `fds, mem_used_mb, cpu_usage`.
  - `AttachableRuntimeStatsManager` samples the current process from a
    daemon subprocess (not held up by our GIL). SIGINT (`kill()`) or SIGTERM
    (exit) stop it, and it writes `end:`.
- `cmake.py` (ctest listing), `git.py`, `nvidia/` (ncu, compute-sanitizer),
  `pfm/` (libpfm event lookup, needs `sh/setup_pfm.sh`).

## Scope timers

### C++ (`inc/phlop/timing`, `src/phlop/timing`)
- `PHLOP_SCOPE_TIMER("key")` records start and duration per scope into a
  tree. Pick one variant per binary: `scope_timer.hpp` (single thread),
  `threaded_scope_timer.hpp` (per-thread storage, merged at shutdown) or
  `mpi_scope_timer.hpp` (MPI_Wtime clock, per-rank file names).
- Hot path: one `make_unique` snapshot per scope exit, and child vectors
  grow geometrically. Don't `reserve(size()+1)`: that reallocates for every
  child, quadratic for loops.
- `-DPHLOP_SCOPE_TIMER_SELF_STATS` (single-thread timer only, off by
  default) records the timer's own constructor/destructor cost:
  `print_self_stats()` gives count, mean and max ns. It costs about as much
  as the timer itself, so keep it out of normal builds.
- Output (`BinaryTimerFile::write`, text despite the name): a `id key` table,
  a blank line, then `depth id start:time` lines, depth first. `'\n'`, not
  `std::endl` (a flush per line). `fn_id` is `uint32_t`, depth `uint16_t`.
- Built and tested with mkn: CI runs `mkn clean build test run -p
  scope_timer,threaded_scope_timer -Oa "-fPIC -std=c++20" -W 9` before
  `sh/test.sh`, which builds and tests each profile again (`mkn clean build
  test`): `mkn test` alone does nothing on a fresh checkout.

### Python (`phlop/timing/scope_timer.py`)
- `file_parser`/`lines_parser` read the format above, keeping the path of
  open nodes, so there's no depth limit and O(1) per line.
- `print_scope_timings(path)` prints one file with % of root and "loss" (time
  not covered by children). `print_variance_across(glob)` compares many
  files, e.g. MPI ranks, and is the only function that globs.

## Repo
- `sh/test.sh`: full test run (CLI modes, concurrency, timers, valgrind,
  perf, pfm, forced failure exit codes). `sh/lint.sh` rewrites files (black,
  autoflake); with `CI` set (GitHub sets it) it only checks, failing on
  changes: `CI=true ./sh/lint.sh` for checks only.
- Version lives in both `pyproject.toml` and `setup.py`. Publishing to PyPI
  runs when a GitHub release is published.
- `todo.md`: known follow-ups.
