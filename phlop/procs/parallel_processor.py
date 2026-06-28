#
#
#
#
#

import copy
import json
import os
import shlex
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process, Queue, cpu_count
from typing import Any, Callable, Optional

from phlop.logger import getLogger

TIMEOUT = 60 * 60  # seconds
FAIL_FAST = bool(json.loads(os.environ.get("PHLOP_FAIL_FAST", "false")))

logger = getLogger(__name__)


class ProcessorFailure(Exception): ...


class LoggingMode(Enum):
    OFF = 0
    ON_NON_ZERO_EXIT_CODE = 1
    ALWAYS = 2


@dataclass
class Job:
    cmd: str = ""
    env: dict = field(default_factory=dict)
    working_dir: Optional[str] = None
    log_file_path: Optional[str] = None
    cores: int = 1
    logging: int = 1
    id: str = ""  # display label; defaults to cmd
    meta: Any = field(default=None)

    def __post_init__(self):
        self.cmd = self.cmd.strip()
        if not self.id:
            self.id = self.cmd
        if not self.cmd:
            raise ValueError("Job requires cmd")

    def __call__(self):
        from phlop.proc import run

        self.run = run(
            shlex.split(self.cmd),
            shell=False,
            capture_output=False,
            check=False,
            print_cmd=False,
            env=self.env,
            working_dir=self.working_dir,
            log_file_path=self.log_file_path,
            logging=self.logging,
        )
        if self.run.exitcode != 0:
            raise ProcessorFailure(f"exited {self.run.exitcode}")
        return self


@dataclass
class JobResult:
    job: Job
    value: Any = None
    error: Exception = None

    @property
    def ok(self):
        return self.error is None


def _print_running(running):
    from phlop.os import read_last_lines_of

    print("pending jobs start")
    for _, (p, job) in running.items():
        if not p.is_alive():
            continue
        print("cmd:", job.id)
        if job.log_file_path:
            for suffix in (".stdout", ".stderr"):
                lines = read_last_lines_of(f"{job.log_file_path}{suffix}")
                if lines:
                    print(f"  {suffix[1:]}:", " ".join(lines))
    print("pending jobs end")


def _runner(job, job_idx, queue):
    try:
        queue.put((job_idx, job(), None))
    except Exception as e:
        queue.put((job_idx, None, e))


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
    n_cores: Optional[int] = None,
    fail_fast: Optional[bool] = None,
    on_success: Optional[Callable[[JobResult], None]] = None,
    on_failure: Optional[Callable[[JobResult], None]] = None,
) -> list:
    jobs = normalize(items)
    if not jobs:
        return []

    fail_fast = fail_fast if fail_fast is not None else FAIL_FAST
    n_cores = n_cores or cpu_count()
    max_cost = max(j.cores for j in jobs)
    assert n_cores >= max_cost, f"need {max_cost} cores, have {n_cores}"

    cores_avail = n_cores
    pending = deque(enumerate(jobs))
    running = {}  # idx -> (Process, Job)
    results = [None] * len(jobs)
    failures = []
    queue = Queue()

    def launch():
        nonlocal cores_avail
        skipped = deque()
        while pending:
            idx, job = pending.popleft()
            if job.cores <= cores_avail:
                cores_avail -= job.cores
                p = Process(target=_runner, args=(job, idx, queue), daemon=True)
                p.start()
                running[idx] = (p, job)
            else:
                skipped.append((idx, job))
        pending.extend(skipped)

    def reap_dead():
        nonlocal cores_avail
        for idx, (p, job) in list(running.items()):
            if not p.is_alive():
                del running[idx]
                cores_avail += job.cores
                err = RuntimeError("worker process died without reporting result")
                r = JobResult(job=job, error=err)
                failures.append(f"{job.id}: {err}")
                if on_failure:
                    on_failure(r)
                else:
                    print(f"{job.id} FAILED: {err}")
                logger.warning("job %r worker died silently", job.id)

    launch()

    while running or pending:
        try:
            job_idx, value, error = queue.get(timeout=TIMEOUT)
        except Exception:
            logger.info("queue timeout — checking for dead workers")
            _print_running(running)
            reap_dead()
            launch()
            if fail_fast and failures:
                raise ProcessorFailure(failures[-1])
            continue

        _, job = running.pop(job_idx)
        cores_avail += job.cores
        r = JobResult(job=job, value=value, error=error)

        if error is not None:
            msg = f"{job.id} FAILED: {error}"
            failures.append(msg)
            if on_failure:
                on_failure(r)
            else:
                print(msg)
            if fail_fast:
                raise ProcessorFailure(msg)
        else:
            results[job_idx] = r
            if on_success:
                on_success(r)
            else:
                print(f"{job.id} finished")

        launch()

    if failures:
        raise ProcessorFailure(
            f"{len(failures)} job(s) failed:\n" + "\n".join(failures)
        )

    return results
