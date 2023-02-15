import unittest
from distutils.util import strtobool

from phlop.reflection import classes_in_directory
from phlop.testing.parallel_processor import load_test_cases_in, process


def parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="", required=True)
    parser.add_argument("-r", "--retries", type=int, default=0, help="")
    parser.add_argument("-c", "--cores", type=int, default=1, help="")
    parser.add_argument("-p", "--print_only", type=strtobool, default=False, help="")

    return parser.parse_args()


def verify_cli_args(cli_args):
    if cli_args.cores == "a" or cli_args.cores == "all":
        cli_args.cores = cpu_count()
    cli_args.cores = int(cli_args.cores)

    return cli_args


def get_test_classes(cli_args):
    return load_test_cases_in(classes_in_directory(cli_args.dir, unittest.TestCase))


def main():
    cli_args = verify_cli_args(parse_cli_args())

    process(
        get_test_classes(cli_args),
        n_cores=cli_args.cores,
        print_only=cli_args.print_only,
    )


if __name__ == "__main__":
    main()
