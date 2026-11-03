# phlop/app/pfm/check_events.py

import logging
import os
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path

from phlop.app.pfm import host
from phlop.proc import run
from phlop.string import decode_bytes

FILE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)
check_events_start = "Total events:"
FORCE_PMU_ENV = "LIBPFM_FORCE_PMU"  # libpfm encodes for this model, not the host's


def pfm_examples_dir():
    """libpfm4 example binaries, from ./sh/setup_pfm.sh or $PHLOP_PFM_DIR"""
    root = os.environ.get("PHLOP_PFM_DIR", FILE_DIR.parent.parent.parent / "tpp/pfm")
    return Path(root) / "examples"


def parse_check_events_output(lines):
    return lines[-1].split(":")[1].strip().replace("0x", "r")


def run_check_events(code):
    exe = pfm_examples_dir() / "check_events"
    cmd = f"{shlex.quote(str(exe))} {code}"  # checkouts with spaces
    return decode_bytes(run(cmd, print_cmd=False).stdout).splitlines()


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


def _cpu_pmus(pmus):
    return [p for p in pmus if p not in ("perf", "perf_raw")]


def forced_pmu():
    """the LIBPFM_FORCE_PMU model, None if not forced"""
    return os.environ.get(FORCE_PMU_ENV) or None


@lru_cache(maxsize=8)
def _probe(forced):
    """libpfm's PMU models with LIBPFM_FORCE_PMU=forced, or unset if None.
    None if check_events failed: missing, or no "Detected PMU models" block"""
    env = {k: v for k, v in os.environ.items() if k != FORCE_PMU_ENV}
    if forced:
        env[FORCE_PMU_ENV] = forced
    try:
        out = subprocess.run(
            [str(pfm_examples_dir() / "check_events")],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
    except OSError:
        return None
    if not any(line.startswith("Detected PMU models:") for line in out):
        return None
    return tuple(_cpu_pmus(_parse_detected_pmus(out)))


def detected_pmus():
    """PMU models libpfm detects on this machine, excluding the generic perf
    ones. With LIBPFM_FORCE_PMU set: only the forced model, if libpfm knows it"""
    return list(_probe(forced_pmu()) or [])


def host_pmus():
    """PMU models libpfm detects for this CPU, ignoring LIBPFM_FORCE_PMU.
    None if that check failed, unlike [] for a CPU libpfm has no model for"""
    pmus = _probe(None)
    return None if pmus is None else list(pmus)


def pmu_warnings(pmu):
    """sets.pmu_warnings for the sets picked for pmu on this host"""
    from phlop.app.pfm.sets import pmu_warnings as warnings_for

    forced = forced_pmu()
    return warnings_for(pmu, detected_pmus(), forced, host_pmus() if forced else [])


def forced_encodings(pmu):
    """True if pmu's sets are encoded for a forced model that is not this CPU.
    False if this CPU's model could not be checked: pmu_warnings says so"""
    forced, host = forced_pmu(), host_pmus()
    return pmu != "generic" and bool(forced) and host is not None and forced not in host


@lru_cache(maxsize=1)
def perf_config_mask():
    """union of the config bits the kernel core PMU declares in its format dir,
    eg AMD: event 0-7,32-35 umask 8-15 edge 18 inv 23 cmask 24-31, arm64:
    event 0-15. libpfm also sets usr/os/int/en bits which perf rejects"""
    mask = 0
    pmu = host.core_pmu()
    if pmu is None:
        return ~0
    for f in (pmu / "format").iterdir():
        field, _, ranges = f.read_text().strip().partition(":")
        if field != "config":
            continue
        for rng in ranges.split(","):
            lo, _, hi = rng.partition("-")
            lo, hi = int(lo), int(hi or lo)
            mask |= ((1 << (hi - lo + 1)) - 1) << lo
    return mask


def _format_fields():
    """{"event": [(lo, hi), ...]} for the core PMU's config format, ranges
    hold a value's bits low first, eg AMD event: [(0, 7), (32, 35)]"""
    fields = {}
    pmu = host.core_pmu()
    if pmu is None:
        return fields
    for f in (pmu / "format").iterdir():
        target, _, ranges = f.read_text().strip().partition(":")
        if target != "config":
            continue
        fields[f.name] = []
        for rng in ranges.split(","):
            lo, _, hi = rng.partition("-")
            fields[f.name].append((int(lo), int(hi or lo)))
    return fields


def pmu_terms_config(pmu_string):
    """raw config for a perf PMU event string, eg "cpu/event=0x165,umask=0xc8/"
    on AMD -> 0x10000c865, laying each term's bits into its format ranges"""
    terms = pmu_string.split("/")[1]
    fields, config = _format_fields(), 0
    for term in filter(None, terms.split(",")):
        name, _, value = term.partition("=")
        if name not in fields:
            raise ValueError(f"{pmu_string}: {name} not a config format field")
        value = int(value or "1", 0)
        for lo, hi in fields[name]:
            width = hi - lo + 1
            config |= (value & ((1 << width) - 1)) << lo
            value >>= width
        if value:
            raise ValueError(f"{pmu_string}: {term} does not fit its format bits")
    return config


_cache = {}  # forced pmu (None: host) -> {event: config}


def encode(events):
    """{event: config} for libpfm event strings, eg "RETIRED_OPS",
    "DISPATCH_STALLS_1:FE_NO_OPS:c=8". Raises ValueError naming any
    event libpfm cannot encode on this machine"""
    cache = _cache.setdefault(forced_pmu(), {})  # codes differ per forced model
    todo = [e for e in dict.fromkeys(events) if e not in cache]
    if todo:
        cache.update(_parse_encodings(run_check_events(" ".join(todo))))
        # check_events stops at the first bad event, find which
        bad = []
        for e in todo:
            if e in cache:
                continue
            got = _parse_encodings(run_check_events(e))
            if e in got:
                cache.update(got)
            else:
                bad.append(e)
        if bad:
            raise ValueError(f"libpfm cannot encode: {bad} (PMUs: {detected_pmus()})")
    return {e: cache[e] for e in events}


def perf_event_string(name, config, modifiers=""):
    """perf -e string for a raw config, named so `perf stat` reports it as name"""
    pmu = host.core_pmu_name()
    return f"{pmu}/config={config & perf_config_mask():#x},name={name}/{modifiers}"


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
