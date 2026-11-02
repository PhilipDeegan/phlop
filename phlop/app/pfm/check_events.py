# phlop/app/pfm/check_events.py

import logging
import os
from functools import lru_cache
from pathlib import Path

from phlop.proc import run
from phlop.string import decode_bytes

FILE_DIR = Path(__file__).resolve().parent
PERF_PMU_FORMAT_DIR = Path("/sys/bus/event_source/devices/cpu/format")

logger = logging.getLogger(__name__)
check_events_start = "Total events:"


def pfm_examples_dir():
    """libpfm4 example binaries, from ./sh/setup_pfm.sh or $PHLOP_PFM_DIR"""
    root = os.environ.get("PHLOP_PFM_DIR", FILE_DIR.parent.parent.parent / "tpp/pfm")
    return Path(root) / "examples"


def parse_check_events_output(lines):
    return lines[-1].split(":")[1].strip().replace("0x", "r")


def run_check_events(code):
    exe = pfm_examples_dir() / "check_events"
    return decode_bytes(run(f"{exe} {code}", print_cmd=False).stdout).splitlines()


def get_evt_perf_code(code):
    """raw perf event string for code, eg "r1005301a0"
    note: includes the usr/os/int/en bits, newer perf kernels reject those,
    prefer `encode`/`perf_event_string`"""
    return parse_check_events_output(run_check_events(code))


def _parse_encodings(lines):
    """{requested event: config int} from check_events output"""
    results, requested = {}, None
    for line in lines:
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "Requested Event":
            requested = val
        elif key == "Codes" and requested is not None:
            results[requested] = int(val.split()[0], 16)
            requested = None
    return results


def _parse_detected_pmus(lines):
    """["amd64_fam1ah_zen5", ...] from the "Detected PMU models:" block"""
    pmus, active = [], False
    for line in lines:
        if line.startswith("Detected PMU models:"):
            active = True
            continue
        if active:
            if not line.startswith("\t["):
                break
            pmus.append(line.split(",")[1].strip())
    return pmus


@lru_cache(maxsize=1)
def detected_pmus():
    """PMU models libpfm detects on this machine, excluding the generic perf ones"""
    pmus = _parse_detected_pmus(run_check_events(""))
    return [p for p in pmus if p not in ("perf", "perf_raw")]


@lru_cache(maxsize=1)
def perf_config_mask():
    """union of the config bits the kernel `cpu` PMU declares in its format dir,
    eg AMD: event 0-7,32-35 umask 8-15 edge 18 inv 23 cmask 24-31
    libpfm also sets usr/os/int/en bits which perf rejects as unsupported"""
    mask = 0
    if not PERF_PMU_FORMAT_DIR.exists():
        return ~0
    for f in PERF_PMU_FORMAT_DIR.iterdir():
        field, _, ranges = f.read_text().strip().partition(":")
        if field != "config":
            continue
        for rng in ranges.split(","):
            lo, _, hi = rng.partition("-")
            lo, hi = int(lo), int(hi or lo)
            mask |= ((1 << (hi - lo + 1)) - 1) << lo
    return mask


_cache = {}


def encode(events):
    """{event: config} for libpfm event strings, eg "RETIRED_OPS",
    "DISPATCH_STALLS_1:FE_NO_OPS:c=8". Raises ValueError naming any
    event libpfm cannot encode on this machine"""
    todo = [e for e in dict.fromkeys(events) if e not in _cache]
    if todo:
        _cache.update(_parse_encodings(run_check_events(" ".join(todo))))
        # check_events stops at the first bad event, find which
        bad = []
        for e in todo:
            if e in _cache:
                continue
            got = _parse_encodings(run_check_events(e))
            if e in got:
                _cache.update(got)
            else:
                bad.append(e)
        if bad:
            raise ValueError(f"libpfm cannot encode: {bad} (PMUs: {detected_pmus()})")
    return {e: _cache[e] for e in events}


def perf_event_string(name, config, modifiers=""):
    """perf -e string for a raw config, named so `perf stat` reports it as name"""
    return f"cpu/config={config & perf_config_mask():#x},name={name}/{modifiers}"


if __name__ == "__main__":
    from phlop.app.pfm.showevtinfo import get_evt_info

    key, code = "[MULT_FLOPS]", ""
    for info in get_evt_info():
        if key in info.umask:
            code = f"{info.name}:{key[1:-1]}"
            break

    assert code != ""

    config = encode([code])[code]
    print(code, hex(config), perf_event_string("mult_flops", config, "u"))
    print(
        decode_bytes(
            run(
                f"perf stat -e {perf_event_string('mult_flops', config, 'u')} sleep 1"
            ).stderr
        )
    )
