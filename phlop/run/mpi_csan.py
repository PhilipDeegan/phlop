# phlop/run/mpi_csan.py
#
# each MPI rank under compute-sanitizer, one report for all of them:
#   mpirun -n 4 python3 -m phlop.run.mpi_csan -t memcheck ./exe args
# files are labelled by rank, rank 0 waits for the others to finish and
# reports each problem once, with the ranks that hit it. It exits with the
# worst rank's code. Options as phlop.run.csan

import sys

from phlop.app.nvidia import csan
from phlop.run import csan as rc_
from phlop.run import mpi_ranks


def main():
    parser = rc_.cli_args_parser("each MPI rank under compute-sanitizer, one report")
    cli_args = parser.parse_args()
    rank, size = mpi_ranks.rank_size()
    mpi_ranks.show_saved(cli_args, rank, rc_.main)
    if rank is None:
        print(
            "phlop mpi_csan: no MPI rank in the environment, use mpirun",
            file=sys.stderr,
        )
        sys.exit(1)
    if not cli_args.remaining:
        parser.print_help()
        sys.exit(1)

    rid = mpi_ranks.run_id()
    mpi_ranks.begin(cli_args.outdir, rid, rank)
    rc, _ = csan.run(
        cli_args.tool,
        cli_args.remaining,
        rc_.shlex.split(cli_args.extra),
        cli_args.outdir,
        cli_args.quiet,
        run_id=rid,
        label=f"rank{rank}",
    )
    report = lambda worst: rc_.report_run(cli_args, rid, worst)
    sys.exit(mpi_ranks.finish(cli_args.outdir, rid, rank, size, rc, report))


if __name__ == "__main__":
    main()
