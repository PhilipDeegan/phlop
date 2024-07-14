#
#

import logging
import os
import sys

from phlop.app import stats_man as sman

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MPI_RANK = os.environ.get("OMPI_COMM_WORLD_RANK")
USAGE = """MPI Stats Manager - see CPU/RAM/FD usage"""


def verify_cli_args(cli_args):
    try:
        cli_args.interval = int(cli_args.interval)
    except ValueError:
        raise ValueError("Interval must be an integer")
    if cli_args.yaml:
        cli_args.yaml = f"{cli_args.yaml}.{MPI_RANK}.yaml"
        cli_args.summary = False
    sys.argv = [sys.argv[0]]  # drop everything!
    return cli_args


def main():
    parser = sman.cli_args_parser(USAGE)
    cli_args = verify_cli_args(parser.parse_args())
    try:
        info = dict(exe=cli_args.remaining, rank=MPI_RANK)
        statsman = sman.RuntimeStatsManager(cli_args, info).join()
        if cli_args.summary:
            sman.print_summary(statsman)
    except (Exception, SystemExit) as e:
        logger.exception(e)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
