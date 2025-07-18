#
#

import re
import os
import sys
import unittest
import multiprocessing
from pathlib import Path

from phlop.dict import ValDict
from phlop.logger import getLogger
from phlop.testing import parallel_processor as pp
from phlop.testing import test_cases as tc

from phlop import reflection as refl

logger = getLogger(__name__)

USAGE = """Flexible parallel test runner"""


def cli_args_parser():
    import argparse

    _help = ValDict(
        # dir="Working directory",
        input="Input file or directory.",
        cmake="Enable cmake build config tests extraction",
        cores="Parallism core/thread count",
        print_only="Print only, no execution",
        prefix="Prepend string to execution string",
        postfix="Append string to execution string",
        dump="Dump discovered tests as YAML to filepath, no execution",
        load="globbing filepath for files exported from dump",
        regex="Filter out non-matching execution strings",
        reverse="reverse order - higher core count tests preferred",
        logging="0=off, 1=on non zero exit code, 2=always",
        rerun="number of times to re-execute discovered tests",
    )

    parser = argparse.ArgumentParser(
        description=USAGE, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cmake", action="store_true", default=False, help=_help.cmake)
    parser.add_argument("-c", "--cores", type=int, default=1, help=_help.cores)
    # parser.add_argument("-d", "--dir", default=".", help=_help.dir)
    parser.add_argument("-i", "--input", default=".", help=_help.input)
    parser.add_argument(
        "-p", "--print_only", action="store_true", default=False, help=_help.print_only
    )
    parser.add_argument("--prefix", default="", help=_help.prefix)
    parser.add_argument("--postfix", default="", help=_help.postfix)

    parser.add_argument(
        "--dump", default=None, action="store", nargs="?", help=_help.dump
    )
    parser.add_argument("--load", default=None, help=_help.load)
    parser.add_argument("-r", "--regex", default=None, help=_help.regex)
    parser.add_argument(
        "-R", "--reverse", action="store_true", default=False, help=_help.reverse
    )
    parser.add_argument("--rerun", type=int, default=1, help=_help.rerun)
    parser.add_argument("--logging", type=int, default=1, help=_help.logging)

    return parser


def verify_cli_args(cli_args):
    if cli_args.cores == "a" or cli_args.cores == "all":
        cli_args.cores = multiprocessing.cpu_count()
    cli_args.cores = int(cli_args.cores)
    if not Path(cli_args.input).exists():
        raise RuntimeError("phlop.run.test_cases error: input provided does not exist")
    pp.LoggingMode(cli_args.logging)  # check convertible
    sys.argv = [sys.argv[0]]  # drop everything!
    return cli_args


def get_test_cases(cli_args):
    if cli_args.cmake:
        return tc.load_cmake_tests(
            cli_args.input, test_cmd_pre=cli_args.prefix, test_cmd_post=cli_args.postfix
        )
    if os.path.isfile(cli_args.input):
        return [
            tc.TestBatch(
                tc.load_test_cases_in(
                    refl.classes_in_file(cli_args.input, unittest.TestCase),
                    test_cmd_pre=cli_args.prefix,
                    test_cmd_post=cli_args.postfix,
                ),
                1,
            )
        ]
    return [
        tc.TestBatch(
            tc.load_test_cases_in(
                refl.classes_in_directory(cli_args.input, unittest.TestCase),
                test_cmd_pre=cli_args.prefix,
                test_cmd_post=cli_args.postfix,
            ),
            1,
        )
    ]


def dump_batches(cli_args):
    import codecs

    import dill

    with open(cli_args.dump, "w") as f:
        f.write(
            codecs.encode(
                dill.dumps(tc.TestBatchesList(batch_list=get_test_cases(cli_args))),
                "hex",
            ).decode("utf8")
        )


def filter_out_regex_fails(cli_args, test_batches):
    if not cli_args.regex:
        return test_batches
    try:
        pattern = re.compile(cli_args.regex)

        def op(x):
            return pattern.search(x)

    except re.error:
        print("regex invalid, resorting to 'str in str' approach")

        def op(x):
            return cli_args.regex in x

    filtered = {tb.cores: [] for tb in test_batches}
    for tb in test_batches:
        for test in tb.tests:
            if op(test.cmd):
                filtered[tb.cores].append(test)
    return [tc.TestBatch(v, k) for k, v in filtered.items() if v]


def noLog(test_batches):
    for tb in test_batches:
        for test in tb.tests:
            test.log_file_path = None
    return test_batches


def log(test_batches):
    # synchronously make all directories before executions
    for tb in test_batches:
        for test in tb.tests:
            if test.log_file_path:
                Path(test.log_file_path).parent.mkdir(parents=True, exist_ok=True)
    return test_batches


def duplicate_reruns(cli_args, test_batches):
    if cli_args.rerun == 1:
        return test_batches

    import copy

    duped = {tb.cores: [] for tb in test_batches}

    for tb in test_batches:
        for test in tb.tests:
            for i in range(cli_args.rerun):
                duped[tb.cores].append(copy.copy(test))
                if duped[tb.cores][-1].log_file_path:
                    duped[tb.cores][-1].log_file_path += f"_{i}"

    return [tc.TestBatch(v, k) for k, v in duped.items() if v]


def main():
    parser = cli_args_parser()
    cli_args = verify_cli_args(parser.parse_args())
    try:
        if cli_args.dump and cli_args.load:
            raise ValueError("Cannot use 'dump' and 'load' options simultaneously.")

        if cli_args.dump:
            dump_batches(cli_args)
            return

        test_batches = (
            tc.extract_load(cli_args.input, cli_args.load)
            if cli_args.load
            else get_test_cases(cli_args)
        )
        test_batches = filter_out_regex_fails(cli_args, test_batches)

        if cli_args.logging == 0:
            test_batches = noLog(test_batches)
        else:
            test_batches = log(test_batches)

        test_batches = duplicate_reruns(cli_args, test_batches)

        if cli_args.reverse:
            test_batches = list(reversed(test_batches))

        pp.process(
            test_batches,
            n_cores=cli_args.cores,
            print_only=cli_args.print_only,
            logging=cli_args.logging,
        )

    except pp.TestCaseFailure:
        sys.exit(1)
    except (Exception, SystemExit) as e:
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
