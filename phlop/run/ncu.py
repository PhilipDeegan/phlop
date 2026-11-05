# phlop/run/ncu.py
#
# profile kernels under Nsight Compute and get a short answer:
#   python3 -m phlop.run.ncu ./exe args             per kernel: what bounds it, hints
#   python3 -m phlop.run.ncu -k 'gemm|axpy' ./exe   only kernels matching a regex
#   python3 -m phlop.run.ncu --show                 the latest run again
#   python3 -m phlop.run.ncu --show run.ncu-rep     a report from elsewhere, eg a cluster
#   python3 -m phlop.run.ncu --compare ID1 ID2      eg master vs a branch
# each distinct kernel launch is profiled --count times (default once)
# --md FILE writes markdown with mermaid charts (- for stdout)
# MPI: mpirun -n 4 python3 -m phlop.run.mpi_ncu ./exe   per rank, slow kernels by rank

import shlex
import sys

from phlop.app.nvidia import ncu
from phlop.app.nvidia import ncu_report as report
from phlop.dict import ValDict


def cli_args_parser(description="ncu with short reports"):
    import argparse

    _help = ValDict(
        kernel="only kernels whose name matches this regex",
        count="profiled launches per distinct kernel launch (name, grid, block, smem)",
        extra='more ncu options, eg -e="--set full" or -e="--nvtx-include compute/"',
        outdir="where ncu's files go",
        quiet="hide the program's output",
        top="kernels per report",
        all="details for every process, not only the slowest",
        md="also write a markdown report, with mermaid charts, - for stdout",
        show="report existing results: a run id, file (.ncu-rep, .ncu-repz),"
        " directory or glob, nothing for the latest run",
        compare="compare two results (run ids, files, directories or globs)",
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument("-k", "--kernel", default="", help=_help.kernel)
    parser.add_argument("-c", "--count", type=int, default=1, help=_help.count)
    parser.add_argument("-e", "--extra", default="", help=_help.extra)
    parser.add_argument("-o", "--outdir", default=str(ncu.OUTDIR), help=_help.outdir)
    parser.add_argument("-q", "--quiet", action="store_true", help=_help.quiet)
    parser.add_argument("--top", type=int, default=10, help=_help.top)
    parser.add_argument("--all", action="store_true", help=_help.all)
    parser.add_argument("--md", default="", help=_help.md)
    parser.add_argument("--show", nargs="*", default=None, help=_help.show)
    parser.add_argument("--compare", nargs=2, default=None, help=_help.compare)
    return parser


def extra_args(cli_args):
    extra = shlex.split(cli_args.extra)
    if cli_args.kernel:
        extra = [f"--kernel-name=regex:{cli_args.kernel}", *extra]
    return extra


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
        ra, rb = ncu.results(a, cli_args.outdir), ncu.results(b, cli_args.outdir)
        outputs(cli_args, ra, rb, names)
        return 0

    if cli_args.show is not None:
        specs = cli_args.show or [None]
        files = [f for s in specs for f in ncu.resolve(s, cli_args.outdir)]
        outputs(cli_args, [ncu.load(f) for f in files])
        return 0

    if not cli_args.remaining:
        raise ValueError("nothing to run")
    rc, run_id = ncu.run(
        cli_args.remaining,
        extra_args(cli_args),
        cli_args.outdir,
        cli_args.quiet,
        count=cli_args.count,
    )
    report_run(cli_args, run_id, rc)
    return rc


def report_run(cli_args, run_id, rc):
    try:
        files = ncu.resolve(run_id, cli_args.outdir)
    except FileNotFoundError:
        files = []
    print(f"\n== phlop ncu, exit code {rc} ==")
    if files:
        outputs(cli_args, [ncu.load(f) for f in files])
    else:
        why = ncu.log_problem(cli_args.outdir, run_id) or "see the log"
        print(f"no report written: {why}")
    print(f"files: {cli_args.outdir}/ncu.{run_id}.*  (--show {run_id})")


def main():
    parser = cli_args_parser()
    cli_args = parser.parse_args()
    try:
        sys.exit(run(cli_args))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"phlop ncu: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
