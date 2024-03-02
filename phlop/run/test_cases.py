#
#
#
#
#


import logging
import multiprocessing
import sys
import re
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
        dump="Dump discovered tests as YAML, no execution",
        load="Run tests exported from dump",
        regex="Filter out non-matching execution strings",
        nolog="Do not write test stdout/stderr to disk",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", action="store_true", default=False, help=_help.cmake)
    parser.add_argument("-c", "--cores", type=int, default=1, help=_help.cores)
    parser.add_argument("-d", "--dir", default=".", help=_help.dir)
    parser.add_argument("-f", "--filter", type=str, default="", help="")
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
    parser.add_argument("--nolog", action="store_true", default=False, help=_help.nolog)

    return parser


def verify_cli_args(cli_args):
    if cli_args.cores == "a" or cli_args.cores == "all":
        cli_args.cores = multiprocessing.cpu_count()
    cli_args.cores = int(cli_args.cores)
    if not Path(cli_args.dir).exists():
        raise RuntimeError(
            "phlop.run.test_cases error: directory provided does not exist"
        )
    sys.argv = [sys.argv[0]]  # drop everything!
    return cli_args


def get_test_cases(cli_args):
    if cli_args.cmake:
        return pp.load_cmake_tests(
            cli_args.dir, test_cmd_pre=cli_args.prefix, test_cmd_post=cli_args.postfix
        )
    return [
        pp.TestBatch(
            pp.load_test_cases_in(
                classes_in_directory(cli_args.dir, unittest.TestCase),
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
                dill.dumps(pp.TestBatchesList(batch_list=get_test_cases(cli_args))),
                "hex",
            ).decode("utf8")
        )


def filter_out_regex_fails(cli_args, test_batches):
    if not cli_args.regex:
        return test_batches
    try:
        pattern = re.compile(cli_args.regex)
        is_valid = True
        op = lambda x: pattern.search(x)
    except re.error:
        print("regex invalid, resorting to 'str in str' approach")
        is_valid = False
        op = lambda x: regex in x
    filtered = {tb.cores: [] for tb in test_batches}
    for tb in test_batches:
        for test in tb.tests:
            if op(test.cmd):
                filtered[tb.cores].append(test)
    return [pp.TestBatch(v, k) for k, v in filtered.items() if v]


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
            pp.extract_load(cli_args) if cli_args.load else get_test_cases(cli_args)
        )
        test_batches = filter_out_regex_fails(cli_args, test_batches)

        pp.process(
            noLog(test_batches) if cli_args.nolog else log(test_batches),
            n_cores=cli_args.cores,
            print_only=cli_args.print_only,
        )

    except pp.TestCaseFailure as e:
        sys.exit(1)
    except (Exception, SystemExit) as e:
        logger.exception(e)
        parser.print_help()
    except:
        e = sys.exc_info()[0]
        print(f"Error: Unknown Error {e}")


if __name__ == "__main__":
    main()
