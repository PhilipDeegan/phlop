#
#

import logging
import os
import sys

from phlop.app import perf as p

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MPI_RANK = os.environ.get("OMPI_COMM_WORLD_RANK")


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
    parser = p.cli_args_parser()
    # cli_args = verify_cli_args(parser.parse_args())
    try:
        ...
    except (Exception, SystemExit) as e:
        logger.exception(e)
        parser.print_help()


if __name__ == "__main__":
    main()
