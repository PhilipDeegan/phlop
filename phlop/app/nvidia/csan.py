# compute sanitizer frontend

# https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html

## samples
#  compute-sanitizer --tool memcheck [sanitizer_options] app_name [app_options]
#  compute-sanitizer --tool racecheck [sanitizer_options] app_name [app_options]
#
#
#


from phlop.dict import ValDict
from phlop.proc import run

metrics = [
    "all",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",  # read
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",  # wrte
]


def build_command(cli_args):
    cmd_parts = [
        "compute-sanitizer",
        f"--tool {cli_args.tool}",
        cli_args.extra if cli_args.extra else "",
        " ".join(cli_args.remaining) if cli_args.remaining else "",
    ]
    return " ".join(filter(None, cmd_parts))


def exec(cli_args):
    return run(build_command(cli_args), check=True, cwd=cli_args.dir)


def cli_args_parser(description="compute-sanitizer tool"):
    import argparse

    _help = ValDict(
        dir="working directory",
        quiet="Redirect output to /dev/null",
        logging="0=off, 1=on non zero exit code, 2=always",
        outfile="path for saved file if active",
        tool="Sanitizer tool to use (memcheck, racecheck, initcheck, synccheck)",
        extra="forward string to csan command",
    )

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument("-d", "--dir", default=".", help=_help.dir)
    parser.add_argument("-o", "--outfile", default=None, help=_help.outfile)
    parser.add_argument("-t", "--tool", default="memcheck", help=_help.tool)
    parser.add_argument("--logging", type=int, default=1, help=_help.logging)
    parser.add_argument("-e", "--extra", type=str, default="", help=_help.extra)

    return parser


def verify_cli_args(cli_args):
    return cli_args
