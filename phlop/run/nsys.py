# phlop/run/nsys.py
#
# profile under Nsight Systems and get a short answer:
#   python3 -m phlop.run.nsys ./exe args             where GPU time goes, idle gaps, hints
#   python3 -m phlop.run.nsys --show                 the latest run again
#   python3 -m phlop.run.nsys --show run.nsys-rep    a profile from elsewhere, eg a cluster
#   python3 -m phlop.run.nsys --compare ID1 ID2      eg master vs a branch
# --md FILE writes markdown with mermaid charts (- for stdout)
# MPI: mpirun -n 4 python3 -m phlop.run.mpi_nsys ./exe   per rank, finds stragglers

import shlex
import sys

from phlop.app.nvidia import nsys
from phlop.app.nvidia import nsys_report as report
from phlop.dict import ValDict


def cli_args_parser(description="nsys with short reports"):
    import argparse

    _help = ValDict(
        trace="nsys --trace, eg cuda,nvtx,osrt,cublas,mpi",
        extra='more nsys options, eg -e="--capture-range=cudaProfilerApi"',
        outdir="where nsys's files go",
        quiet="hide the program's output",
        top="rows per report section",
        all="details for every process, not only the slowest",
        md="also write a markdown report, with mermaid charts, - for stdout",
        show="report existing results: a run id, file (.sqlite, .nsys-rep),"
        " directory or glob, nothing for the latest run",
        compare="compare two results (run ids, files, directories or globs)",
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument("-t", "--trace", default=nsys.TRACE, help=_help.trace)
    parser.add_argument("-e", "--extra", default="", help=_help.extra)
    parser.add_argument("-o", "--outdir", default=str(nsys.OUTDIR), help=_help.outdir)
    parser.add_argument("-q", "--quiet", action="store_true", help=_help.quiet)
    parser.add_argument("--top", type=int, default=10, help=_help.top)
    parser.add_argument("--all", action="store_true", help=_help.all)
    parser.add_argument("--md", default="", help=_help.md)
    parser.add_argument("--show", nargs="*", default=None, help=_help.show)
    parser.add_argument("--compare", nargs=2, default=None, help=_help.compare)
    return parser


def _name(spec, default):
    return spec if spec and len(spec) <= 24 else default


def outputs(cli_args, res, compare=None, names=("A", "B")):
    """the text report, and the markdown when asked. --md - replaces the text,
    so the markdown can be piped, eg into gh pr comment -F -"""
    if cli_args.md != "-":
        if compare is not None:
            print(report.compare(res, compare, names, cli_args.top))
        else:
            print(report.text(res, cli_args.top, cli_args.all))
    if cli_args.md:
        md = report.markdown(res, cli_args.top, compare, names)
        if cli_args.md == "-":
            print(md)
        else:
            with open(cli_args.md, "w") as f:
                f.write(md + "\n")
            print(f"markdown: {cli_args.md}")


def run(cli_args):
    """returns the exit code"""
    if cli_args.compare:
        a, b = cli_args.compare
        names = (_name(a, "A"), _name(b, "B"))
        ra, rb = nsys.results(a, cli_args.outdir), nsys.results(b, cli_args.outdir)
        outputs(cli_args, ra, rb, names)
        return 0

    if cli_args.show is not None:
        specs = cli_args.show or [None]
        files = [f for s in specs for f in nsys.resolve(s, cli_args.outdir)]
        outputs(cli_args, [nsys.load(f) for f in files])
        return 0

    if not cli_args.remaining:
        raise ValueError("nothing to run")
    rc, run_id = nsys.run(
        cli_args.remaining,
        shlex.split(cli_args.extra),
        cli_args.outdir,
        cli_args.quiet,
        trace=cli_args.trace,
    )
    report_run(cli_args, run_id, rc)
    return rc


def report_run(cli_args, run_id, rc):
    try:
        files = nsys.resolve(run_id, cli_args.outdir)
    except FileNotFoundError:
        files = []
    print(f"\n== phlop nsys, exit code {rc} ==")
    if files:
        outputs(cli_args, [nsys.load(f) for f in files])
    else:
        print(f"no profile written, see {cli_args.outdir}/nsys.{run_id}.*.log")
    print(f"files: {cli_args.outdir}/nsys.{run_id}.*  (--show {run_id})")


def main():
    parser = cli_args_parser()
    cli_args = parser.parse_args()
    try:
        sys.exit(run(cli_args))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"phlop nsys: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
