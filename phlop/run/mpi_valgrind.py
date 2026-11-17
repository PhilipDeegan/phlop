# phlop/run/mpi_valgrind.py
#
# each MPI rank under valgrind, one report for all of them:
#   mpirun -n 4 python3 -m phlop.run.mpi_valgrind -t massif ./exe args
# files are labelled by rank, rank 0 waits for the others to finish and
# reports: per rank peaks (sorted), the worst rank's detail. It exits with the
# worst rank's code. Options as phlop.run.valgrind

import sys

from phlop.app import valgrind as vg
from phlop.run import mpi_ranks
from phlop.run import valgrind as rv


def main():
    parser = rv.cli_args_parser("each MPI rank under valgrind, one report")
    cli_args = parser.parse_args()
    rank, size = mpi_ranks.rank_size()
    mpi_ranks.show_saved(cli_args, rank, rv.main)
    if rank is None:
        print(
            "phlop mpi_valgrind: no MPI rank in the environment, use mpirun",
            file=sys.stderr,
        )
        sys.exit(1)
    if not cli_args.remaining:
        parser.print_help()
        sys.exit(1)

    rid = mpi_ranks.run_id()
    mpi_ranks.begin(cli_args.outdir, rid, rank)
    rc, _ = vg.run(
        cli_args.tool,
        cli_args.remaining,
        rv.shlex.split(cli_args.extra),
        cli_args.outdir,
        cli_args.quiet,
        run_id=rid,
        label=f"rank{rank}",
    )

    def report(worst):
        if cli_args.tool in ("memcheck", "massif"):
            rv.report_run(cli_args, rid, worst)

    sys.exit(mpi_ranks.finish(cli_args.outdir, rid, rank, size, rc, report))


if __name__ == "__main__":
    main()
