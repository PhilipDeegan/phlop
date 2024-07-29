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
    cli_args = p.verify_cli_args(cli_args)
    try:
        cli_args.interval = int(cli_args.interval)
    except ValueError:
        raise ValueError("Interval must be an integer") from None
    if cli_args.outfile:
        cli_args.outfile = f"{cli_args.outfile}.{MPI_RANK}."

    sys.argv = [sys.argv[0]]  # drop everything!
    return cli_args


def main():
    parser = p.cli_args_parser()
    cli_args = verify_cli_args(parser.parse_args())
    try:
        if cli_args.tool == "stat":
            ...
        elif cli_args.tool == "record":
            ...
        else:
            raise RuntimeError("PHLOP ERROR: Perf tool not recognized")
    except (Exception, SystemExit) as e:
        logger.exception(e)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
