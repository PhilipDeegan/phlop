#
#
#

import os

from phlop.dict import ValDict
from phlop.proc import run, run_mp

# can be modified
perf_events = [
    "duration_time",
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
]
# "perf stat" can support more events than "perf record"
stat_events = perf_events + ["bus-cycles"]


def version():
    # validated on perf version: 5.19
    proc = run("perf -v", shell=True, capture_output=True).out()
    if " " not in proc or "." not in proc:
        raise ValueError("Unparsable result from 'perf -v'")
    return [int(digit) for digit in proc.split(" ")[-1].split(".")]


def check(force_kernel_space=False):
    """perf can require some system config / read the error if thrown"""
    kernel_space_opt = "a" if force_kernel_space else ""
    run(
        f"perf stat -{kernel_space_opt}d sleep 1",
        shell=True,
        capture_output=True,
        check=True,
    )
    record("ls", [], "/tmp/perf_record_check.dat")


def parse_key(key, force_kernel_space):
    user_space_postfix = ":u"
    if key.endswith(user_space_postfix):
        if force_kernel_space:
            raise RuntimeError(f"Userspace event found {key}")
        return key[: -len(user_space_postfix)]
    return key


def parse_stat_csv(file, force_kernel_space=False):
    import csv

    comments_lines = 2  # validate
    row_val_idx, row_id_idx = 0, 2
    with open(file, newline="") as csvfile:
        [next(csvfile) for i in range(comments_lines)]  # skip headers
        return {
            parse_key(row[row_id_idx], force_kernel_space): row[row_val_idx]
            for row in csv.reader(csvfile, delimiter=",")
        }


def parse_stat_json(file):
    import json

    with open(file, newline="") as f:
        return json.loads(f)


# https://perf.wiki.kernel.org/index.php/Tutorial
# http://www.brendangregg.com/perf.html
# or run "perf list"
def events_str(events):
    if len(events) == 0:
        return ""
    return f"-e {events if isinstance(events, str) else ','.join(events)}"


def out_str(output_file):
    return "" if output_file is None else f"-o {os.path.relpath(output_file)}"


def stat_cmd(exe, events, output_file, options=""):
    return f"perf stat -j {options} {out_str(output_file)} {events_str(events)} {exe}"
    return f"perf stat -x , {options} {out_str(output_file)} {events_str(events)} {exe}"


def stat(exe, events=stat_events, output_file=None):
    return run(stat_cmd(exe, events, output_file), check=True)


def record_cmd(exe, events, output_file, options=""):
    return f"perf record {options} {out_str(output_file)} {events_str(events)} {exe}"


def record(exe, events, output_file=None):
    return run(record_cmd(exe, events, output_file), check=True)


def stat_mp(exe, events, output_files):
    return run_mp([stat_cmd(exe, events, out) for out in output_files])


def cli_args_parser():
    import argparse

    _help = ValDict(
        dir="working directory",
        quiet="Redirect output to /dev/null",
        cores="Parallism core/thread count",
        infiles="infiles",
        print_only="Print only, no execution",
        regex="Filter out non-matching execution strings",
        logging="0=off, 1=on non zero exit code, 2=always",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument("-d", "--dir", default=".", help=_help.dir)
    parser.add_argument("-c", "--cores", type=int, default=1, help=_help.cores)
    parser.add_argument(
        "-p", "--print_only", action="store_true", default=False, help=_help.print_only
    )
    parser.add_argument("-i", "--infiles", default=None, help=_help.infiles)
    parser.add_argument("-r", "--regex", default=None, help=_help.regex)
    parser.add_argument("--logging", type=int, default=1, help=_help.logging)
    return parser


def verify_cli_args(cli_args):
    return cli_args
