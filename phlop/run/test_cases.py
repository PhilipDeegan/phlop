import multiprocessing
import os
import unittest
from pathlib import Path

from phlop.reflection import classes_in_directory
from phlop.testing import parallel_processor as pp


def parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", help="", action="store_true", default=False)
    parser.add_argument("-d", "--dir", help="", default=None)
    parser.add_argument("-r", "--retries", type=int, default=0, help="")
    parser.add_argument("-c", "--cores", type=int, default=1, help="")
    parser.add_argument("-f", "--filter", type=str, default=None, help="")
    parser.add_argument(
        "-p", "--print_only", action="store_true", default=False, help=""
    )

    return parser.parse_args()


def verify_cli_args(cli_args):
    if cli_args.cores == "a" or cli_args.cores == "all":
        cli_args.cores = multiprocessing.cpu_count()
    cli_args.cores = int(cli_args.cores)

    if cli_args.dir is None:
        cli_args.dir = os.getcwd()
    if not Path(cli_args.dir).exists():
        raise RuntimeError(
            "phlop.run.test_cases error: directory provided does not exist"
        )
    return cli_args


def get_test_cases(cli_args):
    if cli_args.cmake:
        return pp.load_cmake_tests(cli_args.dir)
    return pp.TestBatch(
        pp.load_test_cases_in(classes_in_directory(cli_args.dir, unittest.TestCase))
    )


def main():
    cli_args = verify_cli_args(parse_cli_args())

    pp.process(
        get_test_cases(cli_args), n_cores=cli_args.cores, print_only=cli_args.print_only
    )


if __name__ == "__main__":
    main()
