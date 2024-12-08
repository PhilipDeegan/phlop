# Nsight Compute CLI

# https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

## samples
#  ncu --help
#  ncu --metrics all
#  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
#  ncu --target-processes all -o <report-name> mpirun [mpi arguments] <app> [app arguments]
#


from phlop.dict import ValDict
from phlop.proc import run

metrics = [
    "all",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",  # read
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",  # wrte
]


def build_command(cli_args):
    return f"ncu {cli_args.remaining}"


def exec(cli_args):
    return run(build_command(cli_args), check=True)


def cli_args_parser(description="ncu tool"):
    import argparse

    _help = ValDict(
        dir="working directory",
        quiet="Redirect output to /dev/null",
        logging="0=off, 1=on non zero exit code, 2=always",
        outfile="path for saved file if active",
        tool="",
        extra="forward string to csan command",
    )

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument("-d", "--dir", default=".", help=_help.dir)
    parser.add_argument("-i", "--infiles", default=None, help=_help.infiles)
    parser.add_argument("-o", "--outfile", default=None, help=_help.outfile)
    parser.add_argument("-t", "--tool", default="stat", help=_help.tool)
    parser.add_argument("--logging", type=int, default=1, help=_help.logging)
    parser.add_argument("-e", "--extra", type=str, default="", help=_help.extra)

    return parser


def verify_cli_args(cli_args):
    return cli_args
