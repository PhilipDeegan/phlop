#
#
#
#
#


import logging
import multiprocessing
import sys
import unittest
from pathlib import Path

from phlop.dict import ValDict
from phlop.reflection import classes_in_directory
from phlop.testing import parallel_processor as pp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def cli_args_parser():
    import argparse

    _help = ValDict(
        dir="Working directory",
        cmake="Enable cmake build config tests extraction",
        cores="Parallism core/thread count",
        print_only="Print only, no execution",
        prefix="Prepend string to execution string",
        postfix="Append string to execution string",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", action="store_true", default=False, help=_help.cmake)
    parser.add_argument("-d", "--dir", default=".", help=_help.dir)
    parser.add_argument("-r", "--retries", type=int, default=0, help="")
    parser.add_argument("-c", "--cores", type=int, default=1, help=_help.cores)
    parser.add_argument("-f", "--filter", type=str, default="", help="")
    parser.add_argument(
        "-p", "--print_only", action="store_true", default=False, help=_help.print_only
    )
    parser.add_argument("--prefix", default="", help=_help.prefix)
    parser.add_argument("--postfix", default="", help=_help.postfix)
    return parser


def verify_cli_args(cli_args):
    if cli_args.cores == "a" or cli_args.cores == "all":
        cli_args.cores = multiprocessing.cpu_count()
    cli_args.cores = int(cli_args.cores)
    if not Path(cli_args.dir).exists():
        raise RuntimeError(
            "phlop.run.test_cases error: directory provided does not exist"
        )
    return cli_args


def get_test_cases(cli_args):
    if cli_args.cmake:
        return pp.load_cmake_tests(
            cli_args.dir, test_cmd_pre=cli_args.prefix, test_cmd_post=cli_args.postfix
        )
    return pp.TestBatch(
        pp.load_test_cases_in(
            classes_in_directory(cli_args.dir, unittest.TestCase),
            test_cmd_pre=cli_args.prefix,
            test_cmd_post=cli_args.postfix,
        )
    )


def main():
    parser = cli_args_parser()
    cli_args = verify_cli_args(parser.parse_args())
    try:
        pp.process(
            get_test_cases(cli_args),
            n_cores=cli_args.cores,
            print_only=cli_args.print_only,
        )
    except (Exception, SystemExit) as e:
        logger.exception(e)
        parser.print_help()
    except:
        e = sys.exc_info()[0]
        print(f"Error: Unknown Error {e}")


if __name__ == "__main__":
    main()
