#
#

import sys

from phlop.app import valgrind as vg
from phlop.dict import ValDict
from phlop.logger import getLogger

logger = getLogger(__name__)


def cli_args_parser():
    import argparse

    _help = ValDict(
        quiet="Redirect output to /dev/null",
        tool="which valgrind tool to pick (massif/etc)",
        outfile="outfile path base if available",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument(
        "-q", "--quiet", action="store_true", default=False, help=_help.quiet
    )
    parser.add_argument("-t", "--tool", default="", help=_help.tool)
    parser.add_argument("-o", "--outfile", default="", help=_help.outfile)
    return parser


def verify_cli_args(cli_args):
    return cli_args


def main():
    parser = cli_args_parser()
    cli_args = verify_cli_args(parser.parse_args())
    try:
        vg.run_valgrind(cli_args)

    except (Exception, SystemExit) as e:
        logger.exception(e)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
