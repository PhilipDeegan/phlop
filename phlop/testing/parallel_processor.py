#
#
#
#
#

from phlop.os import read_last_lines_of
from phlop.procs.parallel_processor import LoggingMode  # noqa: F401
from phlop.procs.parallel_processor import ProcessorFailure
from phlop.procs.parallel_processor import normalize
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


def process(batches, n_cores=None, print_only=False, fail_fast=None, logging=1):
    jobs = normalize(batches)
    if not jobs:
        return

    if print_only:
        for job in jobs:
            print(job.cmd)
        return

    for job in jobs:
        job.logging = logging

    _process(jobs, n_cores=n_cores, fail_fast=fail_fast, on_failure=_on_failure)
