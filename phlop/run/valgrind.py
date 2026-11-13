# phlop/run/valgrind.py
#
# run under valgrind and get a short answer:
#   python3 -m phlop.run.valgrind ./exe args            memcheck: errors, leaks
#   python3 -m phlop.run.valgrind -t massif ./exe       heap peak and who holds it
#   python3 -m phlop.run.valgrind --show                the latest run again
#   python3 -m phlop.run.valgrind --compare ID1 ID2     eg master vs a branch
# --md FILE writes markdown with mermaid charts (- for stdout), --plot FILE a png
# MPI: mpirun -n 4 python3 -m phlop.run.mpi_valgrind -t massif ./exe

import shlex
import sys

from phlop.app import valgrind as vg
from phlop.app.valgrind import report
from phlop.dict import ValDict
from phlop.logger import getLogger

logger = getLogger(__name__)


def cli_args_parser(description="valgrind with short reports"):
    import argparse

    _help = ValDict(
        tool="valgrind tool: memcheck, massif, or any other (dhat, callgrind...)",
        extra='more valgrind options, eg -e="--threshold=0.1 --num-callers=30"',
        outdir="where valgrind's files go",
        quiet="hide the program's output",
        top="rows per report section",
        all="details for every process, not only the worst",
        md="also write a markdown report, with mermaid charts, - for stdout",
        plot="also plot massif results to this png (needs matplotlib)",
        show="report existing results: a run id, file, directory or glob, "
        "nothing for the latest run",
        compare="compare two results (run ids, files, directories or globs)",
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument("-t", "--tool", default="memcheck", help=_help.tool)
    parser.add_argument("-e", "--extra", default="", help=_help.extra)
    parser.add_argument("-o", "--outdir", default=str(vg.OUTDIR), help=_help.outdir)
    parser.add_argument("-q", "--quiet", action="store_true", help=_help.quiet)
    parser.add_argument("--top", type=int, default=None, help=_help.top)
    parser.add_argument("--all", action="store_true", help=_help.all)
    parser.add_argument("--md", default="", help=_help.md)
    parser.add_argument("--plot", default="", help=_help.plot)
    parser.add_argument("--show", nargs="*", default=None, help=_help.show)
    parser.add_argument("--compare", nargs=2, default=None, help=_help.compare)
    return parser


def _name(spec, default):
    return spec if spec and len(spec) <= 24 else default


def outputs(cli_args, res, compare=None, names=("A", "B")):
    """the text report, and the markdown/plot when asked. --md - replaces
    the text, so the markdown can be piped, eg into gh pr comment -F -"""
    if cli_args.md != "-":
        print(report.text(res, cli_args.top, cli_args.all, compare, names))
    if cli_args.md:
        md = report.markdown(res, cli_args.top, compare, names)
        if cli_args.md == "-":
            print(md)
        else:
            with open(cli_args.md, "w") as f:
                f.write(md + "\n")
            print(f"markdown: {cli_args.md}")
    if cli_args.plot:
        print(f"plot: {report.plot(cli_args.plot, res, cli_args.top, compare, names)}")


def run(cli_args):
    """returns the exit code"""
    if cli_args.compare:
        a, b = cli_args.compare
        names = (_name(a, "A"), _name(b, "B"))
        ra, rb = vg.results(a, cli_args.outdir), vg.results(b, cli_args.outdir)
        outputs(cli_args, ra, rb, names)  # says which are incomplete
        return 1 if any(not getattr(r, "complete", True) for r in ra + rb) else 0

    if cli_args.show is not None:
        specs = cli_args.show or [None]
        files = [f for s in specs for f in vg.resolve(s, cli_args.outdir)]
        res = [vg.load(f) for f in files]
        outputs(cli_args, res)
        return 1 if any(not getattr(r, "complete", True) for r in res) else 0

    if not cli_args.remaining:
        raise ValueError("nothing to run")
    extra = shlex.split(cli_args.extra)
    rc, run_id = vg.run(
        cli_args.tool, cli_args.remaining, extra, cli_args.outdir, cli_args.quiet
    )
    report_run(cli_args, run_id, rc)
    return rc


def report_run(cli_args, run_id, rc):
    files = (
        vg.resolve(run_id, cli_args.outdir) if _has_results(cli_args, run_id) else []
    )
    status = sys.stderr if cli_args.md == "-" else sys.stdout  # stdout: markdown only
    print(f"\n== phlop valgrind {cli_args.tool}, exit code {rc} ==", file=status)
    if files:
        outputs(cli_args, [vg.load(f) for f in files])
    print(
        f"files: {cli_args.outdir}/{cli_args.tool}.{run_id}.*  (--show {run_id})",
        file=status,
    )


def _has_results(cli_args, run_id):
    try:
        return bool(vg.resolve(run_id, cli_args.outdir))
    except FileNotFoundError:
        return False


def main():
    parser = cli_args_parser()
    cli_args = parser.parse_args()
    try:
        sys.exit(run(cli_args))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"phlop valgrind: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
