#
#

import sys
from pathlib import Path

from phlop.app import perf as p
from phlop.logger import getLogger
from phlop.testing import parallel_processor as pp
from phlop.testing import test_cases as tc

logger = getLogger(__name__)
logpath = Path(".phlop") / "perf"

"""
if you get an error about js-d3-flamegraph: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=996839
mkdir /usr/share/d3-flame-graph
wget -O /usr/share/d3-flame-graph/d3-flamegraph-base.html https://cdn.jsdelivr.net/npm/d3-flame-graph@4/dist/templates/d3-flamegraph-base.html
"""


def perf_stat_cmd(cli_args, path, line, options):
    file = Path(line.split(" ")[-1]).stem
    outpath = logpath / path.stem
    outpath.mkdir(parents=True, exist_ok=True)
    return p.stat_cmd(line, p.stat_events, outpath / f"{file}.json", options)


def get_from_files(cli_args):
    test_batches = {}
    file_paths = list(Path(cli_args.dir).glob(cli_args.infiles))

    if not file_paths:
        raise ValueError("No load files found")

    for path in file_paths:
        with open(path, "r") as file:
            while line := file.readline():
                test_case = tc.determine_cores_for_test_case(
                    tc.TestCase(cmd=perf_stat_cmd(cli_args, path, line.rstrip()))
                )
                if test_case.cores not in test_batches:
                    test_batches[test_case.cores] = []
                test_batches[test_case.cores].append(test_case)

    return [tc.TestBatch(v, k) for k, v in test_batches.items()]


def get_remaining(cli_args):
    test_batches = {}
    path = Path(cli_args.remaining[-1]).parent
    test_case = tc.determine_cores_for_test_case(
        tc.TestCase(
            cmd=perf_stat_cmd(
                cli_args, path, " ".join(cli_args.remaining), cli_args.extra
            )
        )
    )
    test_batches[test_case.cores] = [test_case]
    return [tc.TestBatch(v, k) for k, v in test_batches.items()]


def main():
    logpath.mkdir(parents=True, exist_ok=True)

    parser = p.cli_args_parser()
    cli_args = p.verify_cli_args(parser.parse_args())
    try:
        test_batches = []
        if cli_args.infiles:
            test_batches = get_from_files(cli_args)
        else:
            test_batches = get_remaining(cli_args)

        pp.process(
            test_batches,
            n_cores=cli_args.cores,
            print_only=cli_args.print_only,
            logging=cli_args.logging,
        )

    except (Exception, SystemExit) as e:
        logger.exception(e)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
