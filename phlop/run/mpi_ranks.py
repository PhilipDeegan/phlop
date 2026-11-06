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

import glob
import os
import re
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


def run_id():
    """the same on every rank: from the launcher's job id, else its pid.
    srun steps of one slurm job share SLURM_JOB_ID, the step id separates them"""
    job = env(JOB_ENVS)  # eg prterun-host-234099@1: keep 234099-1
    if job and job == os.environ.get("SLURM_JOB_ID") and "SLURM_STEP_ID" in os.environ:
        job += f"-{os.environ['SLURM_STEP_ID']}"
    digits = re.findall(r"\d+", job or "")
    return f"mpi-{'-'.join(digits) if digits else os.getppid()}"


def _rc_files(outdir, rid):
    return glob.glob(str(Path(outdir) / f"{rid}.rank*.rc"))


def finished(outdir, rid, rank, rc):
    """this rank's tool is done, its result files are written"""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    tmp = Path(outdir) / f".{rid}.rank{rank}.rc"
    tmp.write_text(str(rc))
    tmp.rename(Path(outdir) / f"{rid}.rank{rank}.rc")  # atomic: never half written


def gather(outdir, rid, size, timeout=600):
    """rank 0: the worst exit code once every rank finished, None on timeout"""
    deadline = time.time() + timeout
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


def wait_released(outdir, rid, timeout=900):
    done = Path(outdir) / f"{rid}.done"
    deadline = time.time() + timeout
    while time.time() < deadline and not done.exists():
        time.sleep(0.5)


def finish(outdir, rid, rank, size, rc, report, timeout=600):
    """the end of every mpi_* runner: rank 0 reports once all ranks are done,
    returns the exit code to use"""
    finished(outdir, rid, rank, rc)
    if rank != "0":
        wait_released(outdir, rid, timeout + 300)
        return rc
    worst = gather(outdir, rid, size, timeout)
    if worst is None:
        print("phlop mpi: timed out waiting for the other ranks", flush=True)
        worst = rc or 1
    try:
        report()
    finally:
        release(outdir, rid)
    return worst
