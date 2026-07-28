# phlop/testing/parallel_processor.py

from phlop.os import read_last_lines_of
from phlop.procs.parallel_processor import (
    LoggingMode,  # noqa: F401
    ProcessorFailure,
    ProcessorOptions,  # noqa: F401
    normalize,
)
from phlop.procs.parallel_processor import process as _process

TestCaseFailure = ProcessorFailure

FAIL_FAST = False


def _on_failure(result):
    job = result.job
    print(f"{job.id} FAILED: {result.error}")
    if job.log_file_path:
        for suffix in (".stdout", ".stderr"):
            lines = read_last_lines_of(f"{job.log_file_path}{suffix}")
            if lines:
                print(f"  {suffix[1:]}:", "\n".join(lines))


def _print_job(job, verbose):
    if not verbose:
        info = []
        if job.cores != 1:
            info.append(f"cores={job.cores}")
        if job.meta:
            info.append(f"tags={','.join(job.meta)}")
        print(f"{job.cmd}  # {', '.join(info)}" if info else job.cmd)
        return

    print(job.cmd)
    if job.cores != 1:
        print(f"  cores: {job.cores}")
    if job.meta:
        print(f"  tags: {','.join(job.meta)}")
    if job.env:
        print(f"  env: {job.env}")
    if job.working_dir:
        print(f"  working_dir: {job.working_dir}")


def process(
    batches,
    n_cores=None,
    print_only=False,
    fail_fast=None,
    logging=1,
    options=None,
    verbose=False,
):
    jobs = normalize(batches)
    if not jobs:
        return

    if print_only:
        for job in jobs:
            _print_job(job, verbose)
        return

    for job in jobs:
        job.logging = logging

    _process(
        jobs,
        n_cores=n_cores,
        fail_fast=fail_fast,
        on_failure=_on_failure,
        options=options,
    )
