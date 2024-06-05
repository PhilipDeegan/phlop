#
#

import logging
import sys

import numpy as np
import psutil
import yaml
from phlop.app import stats_man as sman
from phlop.dict import ValDict
from phlop.proc import run_raw

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    parser = sman.cli_args_parser()
    cli_args = sman.verify_cli_args(parser.parse_args())
    try:
        info = dict(exe=cli_args.remaining)
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
