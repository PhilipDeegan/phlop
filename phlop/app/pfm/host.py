# phlop/app/pfm/host.py
#
# What the kernel and CPU expose for hardware counting: the perf core PMU
# (named "cpu" on x86, eg "armv8_pmuv3_0" on arm64), its capabilities, and
# /proc/cpuinfo details the event set formulas depend on.

from functools import cache
from pathlib import Path

EVENT_SOURCE_DIR = Path("/sys/bus/event_source/devices")


def _is_core_pmu(dev):
    """perf's rule: "cpu", or a PMU with a `cpus` file (arm64 core PMUs, intel
    hybrid cpu_core/cpu_atom). Uncore PMUs carry a `cpumask` file instead"""
    return dev.name == "cpu" or (dev / "cpus").is_file()


def core_pmus(root=EVENT_SOURCE_DIR):
    """perf core PMU sysfs dirs, "cpu" first. Several on big.LITTLE / hybrid"""
    try:
        devs = sorted(Path(root).iterdir(), key=lambda d: (d.name != "cpu", d.name))
    except OSError:
        return []
    return [d for d in devs if (d / "format").is_dir() and _is_core_pmu(d)]


@cache
def core_pmu():
    """the core PMU perf events go to, None if perf_event has none"""
    pmus = core_pmus()
    return pmus[0] if pmus else None


def core_pmu_name():
    pmu = core_pmu()
    return pmu.name if pmu else "cpu"


def pmu_caps(pmu=None):
    """eg arm64 {"slots": "5", ...}, x86 {"max_precise": "0", ...}"""
    pmu = pmu or core_pmu()
    caps = {}
    if pmu and (pmu / "caps").is_dir():
        for f in (pmu / "caps").iterdir():
            try:
                caps[f.name] = f.read_text().strip()
            except OSError as e:
                caps[f.name] = f"<{e.strerror}>"
    return caps


# arm (implementer 0x41) part -> dispatch slots, for when caps/slots reads 0
# (eg hypervisors), values from the Neoverse software optimization guides
ARM_SLOTS = {0xD40: 8, 0xD49: 5, 0xD4F: 8}  # V1, N2, V2


def dispatch_slots_and_source():
    """(slots per cycle for topdown, where from), (None, reason) if unknown"""
    try:
        slots = int(pmu_caps().get("slots", ""), 0)
    except ValueError:
        slots = 0
    if slots:
        return slots, "caps/slots"
    midr = arm_midr()
    if midr is None:
        return None, "arm64 only, x86 sets use fixed widths"
    if midr[0] == 0x41 and midr[2] in ARM_SLOTS:
        return ARM_SLOTS[midr[2]], f"caps/slots is 0, known width of part {midr[2]:#x}"
    return None, "caps/slots missing or 0 and part unknown"


def dispatch_slots():
    return dispatch_slots_and_source()[0]


@cache
def cpuinfo():
    """first processor block of /proc/cpuinfo as a dict"""
    info = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if not line.strip():
                    if info:
                        break
                    continue
                key, _, val = line.partition(":")
                info[key.strip()] = val.strip()
    except OSError:
        pass
    return info


def cpu_flags():
    """x86 "flags", arm64 "Features" (eg sve, sve2, asimd)"""
    info = cpuinfo()
    return frozenset((info.get("flags") or info.get("Features") or "").split())


def arm_midr():
    """(implementer, variant, part, revision) from arm64 cpuinfo, else None"""
    info = cpuinfo()
    try:
        return tuple(
            int(info[k], 0)
            for k in ("CPU implementer", "CPU variant", "CPU part", "CPU revision")
        )
    except (KeyError, ValueError):
        return None


def stall_slot_counts_cycles():
    """Neoverse N2 r0p0/r0p3 STALL_SLOT* also count one slot per cycle, perf's
    neoverse-n2-v2 metrics subtract CPU_CYCLES for these (0x410fd490/3)"""
    midr = arm_midr()
    return (
        midr is not None
        and midr[0] == 0x41
        and midr[2] == 0xD49
        and ((midr[1], midr[3]) in ((0, 0), (0, 3)))
    )


def sve_vector_bits():
    """default SVE vector length in bits, None without SVE"""
    try:
        with open("/proc/sys/abi/sve_default_vector_length") as f:
            return int(f.read().strip()) * 8
    except (OSError, ValueError):
        return None


def context():
    """values metric formulas need besides counts, see sets.py"""
    return {
        "cpu_flags": cpu_flags(),
        "slots": dispatch_slots(),
        "stall_slot_counts_cycles": stall_slot_counts_cycles(),
        "sve_bits": sve_vector_bits(),
    }
