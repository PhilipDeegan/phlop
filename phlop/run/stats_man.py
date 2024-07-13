#
#

import sys

from phlop.app import stats_man as sman
from phlop.logger import getLogger

logger = getLogger(__name__)


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
        sys.exit(1)


if __name__ == "__main__":
    main()
