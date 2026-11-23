# phlop/procs/parallel_processor.py

from __future__ import annotations

import copy
import inspect
import json
import os
import queue as queue_module
import shlex
import shutil
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any

from phlop.os import usable_cores
from phlop.procs.fixtures import FixtureGraph

# seconds without output before a job is killed as stalled, 0=off; with psutil
# only if it's also idle (below runtimer.BUSY_CPU_FRACTION of a cpu), see
# phlop.procs.runtimer._StallMonitor
STALL_TIMEOUT = float(os.environ.get("PHLOP_STALL_TIMEOUT", "3600"))
# with psutil, seconds without output before a busy job is killed anyway, 0=off
STALL_BUSY_TIMEOUT = float(os.environ.get("PHLOP_STALL_BUSY_TIMEOUT", "14400"))
FAIL_FAST = bool(json.loads(os.environ.get("PHLOP_FAIL_FAST", "false")))
# seconds between temperature checks while jobs are held (PHLOP_THERMAL_MAX,
#  see phlop.app.thermal.Hold)
THERMAL_POLL = 1.0
# seconds cancelled jobs get to exit on SIGTERM before SIGKILL
CANCEL_GRACE = 5


class ProcessorFailure(Exception): ...


class NotEnoughCores(ProcessorFailure):
    """a job needs more cores than the processor may use, before anything runs"""

    def __init__(self, msg, needed):
        super().__init__(msg)
        self.needed = needed


class LoggingMode(Enum):
    OFF = 0
    ON_NON_ZERO_EXIT_CODE = 1
    ALWAYS = 2


def _env_strings(env, cmd):
    """env values as strings, yaml's `60` or `true` (as `true`) too: a process
    environment holds only strings"""
    out = {}
    for name, value in (env or {}).items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (int, float)):
            value = str(value)
        elif not isinstance(value, str):
            raise TypeError(
                f"env {name}={value!r} of {cmd!r}: values must be strings or numbers"
            )
        out[str(name)] = value
    return out


@dataclass
class Job:
    cmd: str = ""
    env: dict = field(default_factory=dict)
    working_dir: str | None = None
    log_file_path: str | None = None
    cores: int = 1
    logging: int = 1
    id: str = ""  # display label; defaults to cmd
    meta: Any = field(default=None)
    test_names: list = field(default_factory=list)  # Class.method, for --regex
    # fixture names, see phlop.procs.fixtures: jobs requiring one wait for its
    #  providers, jobs cleaning one for its providers and requirers
    provides: set = field(default_factory=set)
    requires: set = field(default_factory=set)
    cleans: set = field(default_factory=set)

    def __post_init__(self):
        self.env = _env_strings(self.env, self.cmd)
        self.provides, self.requires = set(self.provides), set(self.requires)
        self.cleans = set(self.cleans)
        self.cmd = self.cmd.strip()
        if not self.id:
            self.id = self.cmd
        if not self.cmd:
            raise ValueError("Job requires cmd")
        if self.cores < 1:
            raise ValueError(f"Job.cores must be >= 1, got {self.cores}")

    def _stall(self, name, default):
        """a stall timeout from the job's own env (a .phlop.exec.yaml `env:`),
        else phlop's"""
        value = (self.env or {}).get(name)
        return default if value is None else float(value)

    def _final_log_path(self):
        """where the logs belong: a relative path is under working_dir, as
        RunTimer has it"""
        path = self.log_file_path
        if path and self.working_dir and not os.path.isabs(path):
            return os.path.join(self.working_dir, path)
        return path

    def __call__(self, on_popen=None):
        if not self.log_file_path or self.logging >= LoggingMode.ALWAYS.value:
            return self._run(self.log_file_path, on_popen)
        # logs only on failure: written while running (the stall check watches
        #  them) in a directory of its own, moved to log_file_path if it fails
        tmp = tempfile.mkdtemp(prefix="phlop-log-")
        self.active_log_path = os.path.join(tmp, "job")  # for _print_jobs
        passed = False
        try:
            self._run(self.active_log_path, on_popen)
            passed = True
            return self
        finally:
            if not passed:  # failed, stalled, cancelled
                _move_logs(self.active_log_path, self._final_log_path())
            self.active_log_path = None
            shutil.rmtree(tmp, ignore_errors=True)

    def _run(self, log_file_path, on_popen):
        from phlop.proc import run

        self.run = run(
            shlex.split(self.cmd),
            shell=False,
            capture_output=False,
            check=False,
            print_cmd=False,
            env=self.env,
            working_dir=self.working_dir,
            log_file_path=log_file_path,
            logging=self.logging,
            stall_timeout=self._stall("PHLOP_STALL_TIMEOUT", STALL_TIMEOUT) or None,
            stall_busy_timeout=self._stall(
                "PHLOP_STALL_BUSY_TIMEOUT", STALL_BUSY_TIMEOUT
            )
            or float("inf"),
            on_popen=on_popen,
            # the exact command: its launcher's args (mpirun --bind-to ...) are
            #  not in the log path
            log_header=self.cmd,
        )
        if self.run.stalled:
            raise ProcessorFailure(f"{self.run.stalled}, killed")
        if self.run.exitcode != 0:
            raise ProcessorFailure(f"exited {self.run.exitcode}")
        return self


def _move_logs(src, dst):
    """`src`.stdout/.stderr to `dst`.stdout/.stderr, its directory made"""
    for suffix in (".stdout", ".stderr"):
        if os.path.exists(src + suffix):
            os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
            shutil.move(src + suffix, dst + suffix)


@dataclass
class JobResult:
    job: Job
    value: Any = None
    error: Exception = None

    @property
    def ok(self):
        return self.error is None


def _to_seconds(val: timedelta | float | None) -> float | None:
    if val is None:
        return None
    if isinstance(val, timedelta):
        return val.total_seconds()
    return float(val)


@dataclass
class ProcessorOptions:
    print_errors_on_exit: bool = True
    # Print IDs of running jobs when no queue output for this long (seconds or timedelta).
    poll_no_output: timedelta | float | None = None
    # Print running jobs + last log lines when a job has been alive this long.
    poll_long_lived: timedelta | float | None = None


def _print_jobs(jobs, verbose):
    if not jobs:
        return

    if not verbose:
        print("running jobs:", ", ".join(job.id for job in jobs))
        return

    from phlop.os import read_last_lines_of

    print("pending jobs start")
    for job in jobs:
        print("cmd:", job.id)
        # while running, logging 1 writes elsewhere, see Job.__call__
        log = getattr(job, "active_log_path", None) or job.log_file_path
        if log:
            for suffix in (".stdout", ".stderr"):
                lines = read_last_lines_of(f"{log}{suffix}")
                if lines:
                    print(f"  {suffix[1:]}:", "\n".join(lines))
    print("pending jobs end")


class _Cancelled(Exception): ...


def _accepts_on_popen(job):
    try:
        params = inspect.signature(job).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.name == "on_popen" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params
    )


def _runner(job, job_idx, queue, popens, cancelled):
    """Runs on a thread: the job's subprocess is the only process started, a
    thread waiting on it costs next to nothing. Always reports a result."""

    def on_popen(p):
        popens[job_idx] = p
        if cancelled.is_set():  # cancelled between launch and Popen
            # cancel_running's snapshot missed this child, so it would never
            # get the SIGKILL that follows SIGTERM, kill it now: it only just
            # started, there is nothing for it to clean up
            p.kill()

    try:
        if cancelled.is_set():
            raise _Cancelled("cancelled")
        if _accepts_on_popen(job):
            result = (job_idx, job(on_popen=on_popen), None)
        else:  # subclass overriding __call__(self), not cancellable mid-run
            result = (job_idx, job(), None)
    except BaseException as e:  # noqa: BLE001 - report any job failure to the parent
        result = (job_idx, None, e)
    popens.pop(job_idx, None)
    queue.put(result)


# --- Adapters ---


def _from_test_case(item) -> Job:
    return Job(
        cmd=item.cmd,
        env=getattr(item, "env", {}),
        working_dir=getattr(item, "working_dir", None),
        log_file_path=getattr(item, "log_file_path", None),
        cores=getattr(item, "cores", 1),
    )


def _adapt_dict(d: dict) -> Job:
    # API TBD
    raise NotImplementedError("dict Job format is not yet defined")


def adapt(item) -> Job:
    if isinstance(item, Job):
        return item
    if isinstance(item, str):
        return Job(cmd=item)
    if isinstance(item, dict):
        return _adapt_dict(item)
    if hasattr(item, "test_case"):
        return _from_test_case(item.test_case)
    if hasattr(item, "cmd") and hasattr(item, "cores"):
        return _from_test_case(item)
    raise TypeError(f"Cannot adapt {type(item).__name__} to Job")


def _jobs_from_batch(batch) -> list:
    jobs = []
    for item in batch.tests:
        job = copy.copy(adapt(item))
        job.cores = batch.cores
        jobs.append(job)
    return jobs


def normalize(items) -> list:
    """Flatten any supported input format into a flat list of Job objects.

    Accepted formats:
      - list[Job]                flat job list (pass-through)
      - list[TestBatch]          batch.cores is applied to every job in the batch
      - dict[int, list]          keys are core counts, values are lists of adaptable items
      - any single adaptable item (str, TestCase, …)
    """
    if isinstance(items, dict):
        jobs = []
        for cores, batch_items in items.items():
            for item in batch_items:
                job = copy.copy(adapt(item))
                job.cores = cores
                jobs.append(job)
        return jobs

    if not isinstance(items, list):
        if hasattr(items, "tests") and hasattr(items, "cores"):
            return _jobs_from_batch(items)
        return [adapt(items)]

    if not items:
        return []

    if hasattr(items[0], "tests") and hasattr(items[0], "cores"):
        jobs = []
        for batch in items:
            jobs.extend(_jobs_from_batch(batch))
        return jobs

    return [adapt(item) for item in items]


# --- Scheduler ---


def process(
    items,
    n_cores: int | None = None,
    fail_fast: bool | None = None,
    on_success: Callable[[JobResult], None] | None = None,
    on_failure: Callable[[JobResult], None] | None = None,
    options: ProcessorOptions | None = None,
) -> list:
    """Run items as Jobs on threads, at most n_cores cores in use at once,
    widest jobs first, a job waiting for the fixtures it requires (see
    phlop.procs.fixtures). Returns results in input order. Any failure raises
    ProcessorFailure: at once with fail_fast (running jobs are cancelled),
    else after every job has finished."""
    jobs = normalize(items)
    if not jobs:
        return []

    opts = options or ProcessorOptions()
    poll_no_output_secs = _to_seconds(opts.poll_no_output)
    poll_long_lived_secs = _to_seconds(opts.poll_long_lived)

    # None blocks until a result: every job thread always reports one
    queue_timeout = min(
        (s for s in [poll_no_output_secs, poll_long_lived_secs] if s is not None),
        default=None,
    )

    # PHLOP_THERMAL_MAX: no new job while the CPU is too hot (running ones go
    #  on), the queue polled meanwhile: nothing running may otherwise be waited on
    from phlop.app.thermal import Hold

    hot = Hold()
    if hot.enabled:
        queue_timeout = min(queue_timeout or THERMAL_POLL, THERMAL_POLL)
    held_reported = False

    fail_fast = fail_fast if fail_fast is not None else FAIL_FAST
    n_cores = n_cores or usable_cores()
    max_cost = max(j.cores for j in jobs)
    if n_cores < max_cost:
        wide = sorted({j.id for j in jobs if j.cores > n_cores})
        shown = ", ".join(wide[:3]) + (
            f" (+{len(wide) - 3} more)" if len(wide) > 3 else ""
        )
        raise NotEnoughCores(
            f"need at least {max_cost} cores, have {n_cores}: {len(wide)} job(s) "
            f"wider, e.g. {shown}",
            max_cost,
        )

    graph = FixtureGraph(jobs)  # FixtureCycleError before anything runs

    cores_avail = n_cores
    # jobs others wait for first (most waiting, transitively), then widest
    # first: a wide job left until the narrow ones are done would only find
    # free cores at the end, and then run alone. Stable, so input order is
    # kept between equal jobs. Results keep input order.
    pending = deque(
        sorted(enumerate(jobs), key=lambda ij: (-graph.priority[ij[0]], -ij[1].cores))
    )
    running = {}  # idx -> (Thread, Job)
    popens = {}  # idx -> Popen, written by job threads
    cancelled = threading.Event()
    results = [None] * len(jobs)
    failures = []
    queue = queue_module.Queue()

    job_start_times: dict = {}
    long_lived_warned: dict = {}  # idx -> last warn time
    last_output_time = time.time()
    last_no_output_print = 0.0

    def finish(idx, job, value, error):
        graph.done(idx, error is None)
        r = JobResult(job=job, value=value, error=error)
        if error is not None:
            msg = f"{job.id} FAILED: {error}"
            failures.append(msg)
            if on_failure:
                on_failure(r)
            else:
                print(msg)
            if fail_fast:
                raise ProcessorFailure(msg)  # cancels the rest, see below
        else:
            results[idx] = r
            if on_success:
                on_success(r)
            else:
                run_time = getattr(getattr(r.value, "run", None), "run_time", None)
                timing = f" in {run_time:.2f}s" if run_time is not None else ""
                print(f"{job.id} finished{timing}")

    def fail_doomed():
        """pending jobs requiring a fixture whose provider failed fail unrun,
        which can doom the jobs requiring a fixture they provide"""
        while graph.doomed:
            doomed = [ij for ij in pending if ij[0] in graph.doomed]
            if not doomed:
                return
            keep = [ij for ij in pending if ij[0] not in graph.doomed]
            pending.clear()
            pending.extend(keep)
            for idx, job in doomed:
                error = ProcessorFailure(graph.doomed[idx])
                finish(idx, job, None, error)

    def launch():
        nonlocal cores_avail, held_reported
        fail_doomed()
        if pending and hot():
            if not held_reported:
                print(f"phlop: CPU at or above {hot.high} C, {len(pending)} jobs held")
                held_reported = True
            return
        held_reported = False
        skipped = deque()
        while pending:
            idx, job = pending.popleft()
            if graph.ready(idx) and job.cores <= cores_avail:
                cores_avail -= job.cores
                t = threading.Thread(
                    target=_runner,
                    args=(job, idx, queue, popens, cancelled),
                    name=f"phlop-job-{idx}",
                    daemon=True,
                )
                t.start()
                running[idx] = (t, job)
                job_start_times[idx] = time.time()
            else:
                skipped.append((idx, job))
        pending.extend(skipped)
        if pending and not running:  # an acyclic graph never gets here
            raise ProcessorFailure(
                "no job can start: " + ", ".join(job.id for _, job in pending)
            )

    def cancel_running():
        from phlop.procs.runtimer import _descendants, _signal_all

        pending.clear()
        cancelled.set()
        procs = []
        for p in list(popens.values()):
            procs += [p, *_descendants(p)]
        _signal_all(procs, "terminate")
        deadline = time.monotonic() + CANCEL_GRACE
        for t, _ in running.values():
            t.join(timeout=max(0, deadline - time.monotonic()))
        _signal_all(procs, "kill")
        # killed: their threads finish (a failed job's logs moved in place),
        #  each bounded, a descendant outside the tree may hold its pipes
        deadline = time.monotonic() + CANCEL_GRACE
        for t, _ in running.values():
            t.join(timeout=max(0, deadline - time.monotonic()))

    def run():
        nonlocal cores_avail, last_output_time, last_no_output_print
        launch()

        while running or pending:
            try:
                job_idx, value, error = queue.get(timeout=queue_timeout)
            except queue_module.Empty:
                now = time.time()

                if (
                    poll_no_output_secs is not None
                    and now - last_output_time >= poll_no_output_secs
                    and now - last_no_output_print >= poll_no_output_secs
                ):
                    alive = [job for p, job in running.values() if p.is_alive()]
                    _print_jobs(alive, verbose=False)
                    last_no_output_print = now

                if poll_long_lived_secs is not None:
                    for idx, (p, job) in list(running.items()):
                        if not p.is_alive():
                            continue
                        age = now - job_start_times.get(idx, now)
                        last_warned = long_lived_warned.get(idx, 0.0)
                        if (
                            age >= poll_long_lived_secs
                            and now - last_warned >= poll_long_lived_secs
                        ):
                            _print_jobs([job], verbose=True)
                            long_lived_warned[idx] = now

                launch()
                continue

            last_output_time = time.time()
            _, job = running.pop(job_idx)
            cores_avail += job.cores
            finish(job_idx, job, value, error)

            launch()

        if failures:
            summary = f"{len(failures)} job(s) failed"
            if opts.print_errors_on_exit:
                print(f"\n{summary}:")
                for msg in failures:
                    print(f"  {msg}")
                raise ProcessorFailure(summary)
            raise ProcessorFailure(f"{summary}:\n" + "\n".join(failures))

        return results

    try:
        return run()
    except BaseException:  # failures, KeyboardInterrupt: leave nothing running
        if running:
            cancel_running()
        raise
