# phlop/run/mpi_ranks.py
#
# for the mpi_* runners (valgrind, nsys, compute-sanitizer): which rank this
# is, and a file based handshake so rank 0 can report on every rank.
#
# mpirun kills the whole job when one rank exits non zero, so a rank whose
# tool found errors (exit 99) would kill rank 0 before it reports. Instead each
# rank writes its exit code to <outdir>/<run id>.rank<N>.rc, rank 0 waits for
# them all, reports, writes <run id>.done and exits with the worst code; the
# others exit, with their own codes, once they see .done
#
# --outdir must be on storage every rank sees, eg not node local /tmp: rank 0
# reads the other ranks' files. Waits time out after PHLOP_MPI_TIMEOUT seconds,
# default 600, for the other ranks to finish after rank 0. The ranks agree on a
# run id from the launcher's job id, see run_id, PHLOP_MPI_RUN_ID overrides it

import glob
import os
import re
import sys
import time
from pathlib import Path

RANK_ENVS = (
    "OMPI_COMM_WORLD_RANK",
    "PMIX_RANK",
    "PMI_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
)
SIZE_ENVS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS")
JOB_ENVS = ("PMIX_NAMESPACE", "OMPI_MCA_ess_base_jobid", "PMI_KVS_NAME", "SLURM_JOB_ID")


def env(keys):
    return next((os.environ[k] for k in keys if k in os.environ), None)


def rank_size():
    """(rank, size) as strings/int, rank None outside an MPI launch"""
    return env(RANK_ENVS), int(env(SIZE_ENVS) or 1)


_warned_ppid = False


def run_id():
    """the same on every rank: PHLOP_MPI_RUN_ID, else the launcher's job id, else
    the parent pid. srun steps of one slurm job share SLURM_JOB_ID, the step id
    separates them. The parent pid is only shared by ranks under one launcher
    process, eg one node with mpich's hydra: ranks on other nodes would wait on
    a different id, so that fallback warns once"""
    global _warned_ppid
    if rid := os.environ.get("PHLOP_MPI_RUN_ID"):
        return f"mpi-{rid}"
    job = env(JOB_ENVS)  # eg prterun-host-234099@1: keep 234099-1
    if not job and env(RANK_ENVS) is not None and not _warned_ppid:
        _warned_ppid = True
        print(
            "phlop: no launcher job id in the env, run id from the parent pid:"
            " ranks on other nodes would not match. Set PHLOP_MPI_RUN_ID on every"
            " rank, eg mpirun -genv PHLOP_MPI_RUN_ID=$RANDOM",
            file=sys.stderr,
        )
    if job and job == os.environ.get("SLURM_JOB_ID") and "SLURM_STEP_ID" in os.environ:
        job += f"-{os.environ['SLURM_STEP_ID']}"
    digits = re.findall(r"\d+", job or "")
    return f"mpi-{'-'.join(digits) if digits else os.getppid()}"


def timeout():
    return float(os.environ.get("PHLOP_MPI_TIMEOUT", "600"))


def show_saved(cli_args, rank, main):
    """--show/--compare: saved results, no MPI run. Rank 0, or a plain launch,
    runs the non MPI runner's main (the same arguments), the others exit"""
    if cli_args.show is None and not getattr(cli_args, "compare", None):
        return
    if rank in (None, "0"):
        main()
    sys.exit(0)


def begin(outdir, rid, rank):
    """before the run: files of an earlier invocation with the same run id, eg
    a second wrapper call in one srun step, must not count for this one"""
    (Path(outdir) / f"{rid}.rank{rank}.rc").unlink(missing_ok=True)
    if rank == "0":
        (Path(outdir) / f"{rid}.done").unlink(missing_ok=True)


def _rc_files(outdir, rid):
    return glob.glob(str(Path(outdir) / f"{rid}.rank*.rc"))


def finished(outdir, rid, rank, rc):
    """this rank's tool is done, its result files are written"""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    tmp = Path(outdir) / f".{rid}.rank{rank}.rc"
    tmp.write_text(str(rc))
    tmp.rename(Path(outdir) / f"{rid}.rank{rank}.rc")  # atomic: never half written


def gather(outdir, rid, size, wait=600):
    """rank 0: the worst exit code once every rank finished, None on timeout"""
    deadline = time.time() + wait
    while time.time() < deadline:
        files = _rc_files(outdir, rid)
        if len(files) >= size:
            codes = [int(Path(f).read_text() or 0) for f in files]
            return max(codes, key=lambda c: (c != 0, c))
        time.sleep(0.5)
    return None


def release(outdir, rid):
    """rank 0: reported, the others can exit"""
    (Path(outdir) / f"{rid}.done").touch()


def wait_released(outdir, rid, wait=900):
    done = Path(outdir) / f"{rid}.done"
    deadline = time.time() + wait
    while time.time() < deadline and not done.exists():
        time.sleep(0.5)


def finish(outdir, rid, rank, size, rc, report, wait=None):
    """the end of every mpi_* runner: rank 0 reports once all ranks are done,
    returns the exit code to use. report(worst exit code over the ranks)"""
    wait = timeout() if wait is None else wait
    finished(outdir, rid, rank, rc)
    if rank != "0":
        wait_released(outdir, rid, wait + 300)
        return rc
    worst = gather(outdir, rid, size, wait)
    if worst is None:
        print(
            f"phlop mpi: timed out waiting for the other ranks, {wait:g}s"
            " (PHLOP_MPI_TIMEOUT): is --outdir on storage every rank sees?",
            flush=True,
        )
        worst = rc or 1
    try:
        report(worst)
    finally:
        release(outdir, rid)
    return worst
