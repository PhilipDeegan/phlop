# phlop/testing/parallel_processor.py

from phlop.os import read_last_lines_of
from phlop.procs.fixtures import fixtures_of
from phlop.procs.parallel_processor import (
    LoggingMode,  # noqa: F401
    NotEnoughCores,  # noqa: F401
    ProcessorFailure,
    ProcessorOptions,  # noqa: F401
    normalize,
)
from phlop.procs.parallel_processor import process as _process
from phlop.testing.exec_scope import phase_fixtures, phase_label

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
        phase = phase_label(getattr(job, "phase", None))
        if phase:
            info.append(f"phase={phase}")
        print(f"{job.cmd}  # {', '.join(info)}" if info else job.cmd)
        return

    print(job.cmd)
    if job.cores != 1:
        print(f"  cores: {job.cores}")
    if job.meta:
        print(f"  tags: {','.join(job.meta)}")
    phase = phase_label(getattr(job, "phase", None))
    if phase:
        print(f"  phase: {phase}")
    if job.env:
        print(f"  env: {job.env}")
    if job.working_dir:
        print(f"  working_dir: {job.working_dir}")
    for name, fixtures in zip(("provides", "requires", "cleans"), fixtures_of(job)):
        if fixtures:
            print(f"  {name}: {','.join(sorted(fixtures))}")


def _print_fixtures(jobs, verbose):
    """every fixture, its setup and cleanup jobs printed as tests are (they run
    like them), and how many jobs wait on it. Returns the ids of jobs printed"""
    fixtures = {}  # name -> (setups, n requiring, cleanups)
    for job in jobs:
        provides, requires, cleans = fixtures_of(job)
        for kind, names in ((0, provides), (1, requires), (2, cleans)):
            for name in names:
                fixture = fixtures.setdefault(name, ([], [], []))
                fixture[kind].append(job)
    shown = set()
    if not fixtures:
        return shown

    print("fixtures:")
    for name, (setups, requiring, cleanups) in sorted(fixtures.items()):
        ignored = "" if setups else ", no setup loaded: not waited for"
        print(f"{name}  # required by {len(requiring)}{ignored}")
        for kind, kind_jobs in (("setup", setups), ("cleanup", cleanups)):
            for job in kind_jobs:
                print(f"  {kind}: ", end="")
                _print_job(job, verbose)
                shown.add(id(job))
    return shown


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
        # fixtures first: they run before the tests needing them
        shown = _print_fixtures(jobs, verbose)
        if shown:
            print("\ntests:")
        for job in jobs:
            if id(job) not in shown:
                _print_job(job, verbose)
        return

    for job in jobs:
        job.logging = logging
    phase_fixtures(jobs)  # after any filtering: an earlier phase is not pulled in

    _process(
        jobs,
        n_cores=n_cores,
        fail_fast=fail_fast,
        on_failure=_on_failure,
        options=options,
    )
