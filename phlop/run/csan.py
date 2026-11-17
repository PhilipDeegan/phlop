# phlop/run/csan.py
#
# run under compute-sanitizer and get a short answer:
#   python3 -m phlop.run.csan ./exe args                memcheck: errors, leaks
#   python3 -m phlop.run.csan -t racecheck ./exe        shared memory races
#   python3 -m phlop.run.csan -t initcheck ./exe        uninitialised device reads
#   python3 -m phlop.run.csan -t synccheck ./exe        barrier misuse
#   python3 -m phlop.run.csan --show                    the latest run again
# exit code 99 on errors, else the program's. --md FILE writes markdown (- for stdout)
# build with -lineinfo (device lines) and -g (host lines)
# MPI: mpirun -n 4 python3 -m phlop.run.mpi_csan ./exe

import shlex
import sys

from phlop.app.nvidia import csan
from phlop.app.nvidia import csan_report as report
from phlop.dict import ValDict


def cli_args_parser(description="compute-sanitizer with short reports"):
    import argparse

    _help = ValDict(
        tool=f"sanitizer tool: {', '.join(csan.TOOLS)}",
        extra='more compute-sanitizer options, eg -e="--padding=64"',
        outdir="where compute-sanitizer's files go",
        quiet="hide the program's output",
        top="rows per report section",
        md="also write a markdown report, - for stdout",
        show="report existing results: a run id, file, directory or glob,"
        " nothing for the latest run",
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument(
        "-t", "--tool", default="memcheck", choices=csan.TOOLS, help=_help.tool
    )
    parser.add_argument("-e", "--extra", default="", help=_help.extra)
    parser.add_argument("-o", "--outdir", default=str(csan.OUTDIR), help=_help.outdir)
    parser.add_argument("-q", "--quiet", action="store_true", help=_help.quiet)
    parser.add_argument("--top", type=int, default=10, help=_help.top)
    parser.add_argument("--md", default="", help=_help.md)
    parser.add_argument("--show", nargs="*", default=None, help=_help.show)
    return parser


def outputs(cli_args, res):
    """the text report, and the markdown when asked. --md - replaces the text"""
    if cli_args.md != "-":
        print(report.text(res, cli_args.top))
    if cli_args.md:
        md = report.markdown(res, cli_args.top)
        if cli_args.md == "-":
            print(md)
        else:
            with open(cli_args.md, "w") as f:
                f.write(md + "\n")
            print(f"markdown: {cli_args.md}")


def run(cli_args):
    """returns the exit code"""
    if cli_args.show is not None:
        specs = cli_args.show or [None]
        files = [f for s in specs for f in csan.resolve(s, cli_args.outdir)]
        outputs(cli_args, [csan.parse(f) for f in files])
        return 0
    if not cli_args.remaining:
        raise ValueError("nothing to run")
    rc, run_id = csan.run(
        cli_args.tool,
        cli_args.remaining,
        shlex.split(cli_args.extra),
        cli_args.outdir,
        cli_args.quiet,
    )
    report_run(cli_args, run_id, rc)
    return rc


def report_run(cli_args, run_id, rc):
    try:
        files = csan.resolve(run_id, cli_args.outdir)
    except FileNotFoundError:
        files = []
    print(f"\n== phlop compute-sanitizer {cli_args.tool}, exit code {rc} ==")
    if files:
        outputs(cli_args, [csan.parse(f) for f in files])
    else:
        print(_why_no_results(cli_args.outdir, run_id))
    print(report.files_line(cli_args.outdir, cli_args.tool, run_id))


def _why_no_results(outdir, run_id):
    import glob

    for log in glob.glob(f"{outdir}/csan-*.{run_id}.*.log"):
        with open(log) as f:
            if "before first instrumented API call" in f.read():
                return "the program exited before any CUDA call, nothing to check"
    return f"no results written, see {outdir}/csan-*.{run_id}.*.log"


def main():
    parser = cli_args_parser()
    cli_args = parser.parse_args()
    try:
        sys.exit(run(cli_args))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"phlop csan: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
