# valgrind.py
#

from phlop.proc import run


def run_massif(cli_args):
    run(
        " ".join(
            [
                "valgrind",
                "--tool=massif",
                "--massif-out-file=massif.phlop",
                *cli_args.remaining,
            ]
        ),
        capture_output=True,
    )


def run_memcheck(cli_args):
    run(
        " ".join(["valgrind", *cli_args.remaining]),
        capture_output=True,
    )


def run_valgrind(cli_args):
    tool = cli_args.tool
    if tool == "massif":
        run_massif(cli_args)
    elif tool == "memcheck" or not tool:
        run_memcheck(cli_args)
    elif tool:
        raise RuntimeError(f"Phlop ERROR: Unhandled valgrind tool: {tool}")
