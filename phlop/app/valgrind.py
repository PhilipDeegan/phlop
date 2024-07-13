# valgrind.py
#

from phlop.proc import run


def run_massif(cli_args):
    outfile = cli_args.outfile if cli_args.outfile else "massif.phlop"
    cmd = f"valgrind --tool=massif --massif-out-file={outfile} " + " ".join(
        cli_args.remaining
    )
    run(cmd, capture_output=True)


def run_memcheck(cli_args):
    cmd = "valgrind " + " ".join(cli_args.remaining)
    run(cmd, capture_output=True)


def run_valgrind(cli_args):
    tool = cli_args.tool
    if tool == "massif":
        run_massif(cli_args)
    elif tool == "memcheck" or not tool:
        run_memcheck(cli_args)
    elif tool:
        raise RuntimeError(f"Phlop ERROR: Unhandled valgrind tool: {tool}")
