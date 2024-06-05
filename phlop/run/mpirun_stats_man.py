#
#

import logging
import os
import sys

import numpy as np
import psutil
import yaml
from phlop.app import stats_man as sman
from phlop.dict import ValDict
from phlop.proc import run_raw

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


MPI_RANK = os.environ.get("OMPI_COMM_WORLD_RANK")


def verify_cli_args(cli_args):
    try:
        cli_args.interval = int(cli_args.interval)
        if cli_args.yaml:
            cli_args.yaml = f"{cli_args.yaml}.{MPI_RANK}.yaml"
            cli_args.summary = False
    except ValueError:
        raise ValueError("Interval must be an integer")
    sys.argv = [sys.argv[0]]  # drop everything!
    return cli_args


def main():
    parser = sman.cli_args_parser()
    cli_args = verify_cli_args(parser.parse_args())
    try:
        info = dict(exe=cli_args.remaining, rank=MPI_RANK)
        statsman = sman.RuntimeStatsManager(cli_args, info).join()
        if cli_args.summary:
            sman.print_summary(statsman)
    except (Exception, SystemExit) as e:
        logger.exception(e)
        parser.print_help()
    except:
        e = sys.exc_info()[0]
        print(f"Error: Unknown Error {e}")


if __name__ == "__main__":
    main()
