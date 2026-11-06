# phlop/run/mpi_valgrind.py
#
# each MPI rank under valgrind, one report for all of them:
#   mpirun -n 4 python3 -m phlop.run.mpi_valgrind -t massif ./exe args
# files are labelled by rank, rank 0 waits for the others' files and reports:
# per rank peaks (sorted), the worst rank's detail. Options as phlop.run.valgrind

import glob
import os
import re
import sys
import time
from pathlib import Path

from phlop.app import valgrind as vg
from phlop.run import valgrind as rv

RANK_ENVS = ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK")
SIZE_ENVS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE")
JOB_ENVS = ("PMIX_NAMESPACE", "OMPI_MCA_ess_base_jobid", "PMI_KVS_NAME", "SLURM_JOB_ID")


def _env(keys):
    return next((os.environ[k] for k in keys if k in os.environ), None)


def run_id():
    """the same on every rank: from the launcher's job id, else its pid"""
    job = _env(JOB_ENVS)  # eg prterun-host-234099@1: keep 234099-1
    digits = re.findall(r"\d+", job or "")
    return f"mpi-{'-'.join(digits) if digits else os.getppid()}"


def _complete(path):
    if path.endswith(".xml"):
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 64))
            return b"</valgrindoutput>" in f.read()
    return os.path.getsize(path) > 0


def wait_for(outdir, tool, rid, size, timeout=300):
    """rank 0: until every rank's result file is written and stops growing"""
    ext = ".xml" if tool == "memcheck" else ".out"
    pattern = str(Path(outdir) / f"{tool}.{rid}.rank*{ext}")
    sizes, deadline = None, time.time() + timeout
    while time.time() < deadline:
        files = sorted(glob.glob(pattern))
        now = [os.path.getsize(f) for f in files]
        if len(files) >= size and now == sizes and all(_complete(f) for f in files):
            return True
        sizes = now
        time.sleep(0.5)
    return False


def main():
    parser = rv.cli_args_parser("each MPI rank under valgrind, one report")
    cli_args = parser.parse_args()
    rank, size = _env(RANK_ENVS), _env(SIZE_ENVS)
    if rank is None:
        print(
            "phlop mpi_valgrind: no MPI rank in the environment, use mpirun",
            file=sys.stderr,
        )
        sys.exit(1)
    if not cli_args.remaining:
        parser.print_help()
        sys.exit(1)

    rid = run_id()
    rc, _ = vg.run(
        cli_args.tool,
        cli_args.remaining,
        rv.shlex.split(cli_args.extra),
        cli_args.outdir,
        cli_args.quiet,
        run_id=rid,
        label=f"rank{rank}",
    )
    if rank == "0" and cli_args.tool in ("memcheck", "massif"):
        if not wait_for(cli_args.outdir, cli_args.tool, rid, int(size or 1)):
            print(
                "phlop mpi_valgrind: timed out waiting for the other ranks",
                file=sys.stderr,
            )
        rv.report_run(cli_args, rid, rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
