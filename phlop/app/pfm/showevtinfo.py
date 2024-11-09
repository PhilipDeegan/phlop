#
#
#
#
#


import logging
from dataclasses import dataclass, field
from pathlib import Path

from phlop.os import pushd
from phlop.proc import run
from phlop.string import decode_bytes

FILE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)
EVTINFO_delimiter = "#-----------------------------"


@dataclass
class EVTUMask:
    id: str
    desc: str
    code: str


@dataclass
class EVTInfo:
    idx: str
    pmu: str
    name: str
    umask: dict = field(default_factory=lambda: {})
    ect: dict = field(default_factory=lambda: {})


def _parse_evtinfo(bits_list):
    assert len(bits_list) >= 7
    info = EVTInfo(
        idx=bits_list[0][1].strip(),
        pmu=bits_list[1][1].strip(),
        name=bits_list[2][1].strip(),
    )
    for bits in bits_list[7:]:
        if bits[0].strip().startswith("Umask"):
            info.umask[bits[3].strip()] = EVTUMask(
                id=bits[3].strip(), desc=bits[5].strip(), code=bits[1].strip()
            )
    return info


def parse_evtinfo_output(lines):
    start_idx = 0
    for line in lines:
        start_idx += 1
        if line.strip() == EVTINFO_delimiter:
            break

    bits_list, results = [], []
    for line in lines[start_idx:]:
        if line == EVTINFO_delimiter:
            results.append(_parse_evtinfo(bits_list))
            bits_list = []
            continue
        bits_list.append(line.strip().split(":"))

    return results


def run_evtinfo():
    with pushd(FILE_DIR.parent.parent.parent):
        return decode_bytes(run("./tpp/pfm/examples/showevtinfo").stdout).splitlines()


def get_evt_info():
    return parse_evtinfo_output(run_evtinfo())


if __name__ == "__main__":
    print(get_evt_info())
