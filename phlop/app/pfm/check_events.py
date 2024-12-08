#
#
#
#
#


import logging
from pathlib import Path

from phlop.os import pushd
from phlop.proc import run
from phlop.string import decode_bytes

FILE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)
check_events_start = "Total events:"


def parse_check_events_output(lines):
    return lines[-1].split(":")[1].strip().replace("0x", "r")


def run_check_events(code):
    with pushd(FILE_DIR.parent.parent.parent):
        return decode_bytes(
            run(f"./tpp/pfm/examples/check_events {code}").stdout
        ).splitlines()


def get_evt_perf_code(code):
    return parse_check_events_output(run_check_events(code))


if __name__ == "__main__":
    from phlop.app.pfm.showevtinfo import get_evt_info

    key, code = "[MULT_FLOPS]", ""
    for info in get_evt_info():
        if key in info.umask:
            code = f"{info.name}:{info.umask[key].code}"
            break

    assert code != ""

    # print("get_evt_perf_code", get_evt_perf_code(code))
    print(run(f"perf stat -e {get_evt_perf_code(code)} sleep 5"))
