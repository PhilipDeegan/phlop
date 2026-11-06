# phlop/run/mpi_nsys.py
#
# each MPI rank under nsys, one report for all of them:
#   mpirun -n 4 python3 -m phlop.run.mpi_nsys ./exe args
#   srun python3 -m phlop.run.mpi_nsys ./exe args
# files are labelled by rank, MPI calls are traced. Rank 0 waits for the
# others to finish and reports: per rank GPU time, MPI and NCCL time, the
# straggler the others wait for, and the slowest rank's detail. It exits with
# the worst rank's code. Options as phlop.run.nsys

import sys

from phlop.app.nvidia import nsys
from phlop.run import mpi_ranks
from phlop.run import nsys as rn


def main():
    parser = rn.cli_args_parser("each MPI rank under nsys, one report")
    cli_args = parser.parse_args()
    rank, size = mpi_ranks.rank_size()
    if rank is None:
        print(
            "phlop mpi_nsys: no MPI rank in the environment, use mpirun",
            file=sys.stderr,
        )
        sys.exit(1)
    if not cli_args.remaining:
        parser.print_help()
        sys.exit(1)
    trace = cli_args.trace
    if "mpi" not in trace.split(","):
        trace += ",mpi"

    rid = mpi_ranks.run_id()
    rc, _ = nsys.run(
        cli_args.remaining,
        rn.shlex.split(cli_args.extra),
        cli_args.outdir,
        cli_args.quiet,
        run_id=rid,
        label=f"rank{rank}",
        trace=trace,
    )
    report = lambda: rn.report_run(cli_args, rid, rc)
    sys.exit(mpi_ranks.finish(cli_args.outdir, rid, rank, size, rc, report))


if __name__ == "__main__":
    main()
