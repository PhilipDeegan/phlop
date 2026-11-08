# phlop/run/mpi_ncu.py
#
# each MPI rank's kernels under ncu, one report for all of them:
#   mpirun -n 4 python3 -m phlop.run.mpi_ncu ./exe args
#   srun python3 -m phlop.run.mpi_ncu ./exe args
# files are labelled by rank. Rank 0 waits for the others to finish and
# reports: per rank profiled kernel time, kernels slower on one rank than the
# median rank, and the slowest rank's detail. It exits with the worst rank's
# code. Each profiled kernel is replayed ~30 times on every rank: pick few (-k).
# Kernels that wait on other ranks (NCCL) need the peers launched in step:
# -e="--lockstep-kernel-launch" (see ncu --help). Options as phlop.run.ncu

import sys

from phlop.app.nvidia import ncu
from phlop.run import mpi_ranks
from phlop.run import ncu as rn


def main():
    parser = rn.cli_args_parser("each MPI rank's kernels under ncu, one report")
    cli_args = parser.parse_args()
    rank, size = mpi_ranks.rank_size()
    mpi_ranks.show_saved(cli_args, rank, rn.main)
    if rank is None:
        print(
            "phlop mpi_ncu: no MPI rank in the environment, use mpirun",
            file=sys.stderr,
        )
        sys.exit(1)
    if not cli_args.remaining:
        parser.print_help()
        sys.exit(1)

    rid = mpi_ranks.run_id()
    mpi_ranks.begin(cli_args.outdir, rid, rank)
    rc, _ = ncu.run(
        cli_args.remaining,
        rn.extra_args(cli_args),
        cli_args.outdir,
        cli_args.quiet,
        run_id=rid,
        label=f"rank{rank}",
        count=cli_args.count,
    )
    report = lambda worst: rn.report_run(cli_args, rid, worst)
    sys.exit(mpi_ranks.finish(cli_args.outdir, rid, rank, size, rc, report))


if __name__ == "__main__":
    main()
