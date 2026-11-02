# phlop/app/pfm/diagnose.py
#
# Print everything that decides whether hardware counting works on this host,
# then try every event set on a small workload and report each event.
# Meant for CI logs on unfamiliar hardware (eg arm64 runners) as much as
# for local use:
#
#   python3 -m phlop.app.pfm.diagnose
#   python3 -m phlop.app.pfm.diagnose --strict  # exit 1 if nothing counted

import os
import platform
import shutil
import subprocess
import sys
import tempfile

from phlop.app import perf
from phlop.app.pfm import check_events as ce
from phlop.app.pfm import host
from phlop.app.pfm.sets import sets_for

# a busy loop long enough to count something, short enough for CI
WORKLOAD = [sys.executable, "-c", "s = 0\nfor i in range(2_000_000): s += i * i"]


def section(title):
    print(f"\n==== {title}")


def show(cmd, timeout=60):
    """run cmd, print its output, return (exitcode, stdout + stderr)"""
    print(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  failed: {e}")
        return None, ""
    out = (proc.stdout + proc.stderr).rstrip()
    for line in out.splitlines()[:60]:
        print(f"  {line}")
    if proc.returncode:
        print(f"  (exit code {proc.returncode})")
    return proc.returncode, out


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as e:
        return f"<{e.strerror}>"


def diagnose_platform():
    section("platform")
    print(
        f"  {platform.platform()} {platform.machine()} python {platform.python_version()}"
    )
    info = host.cpuinfo()
    for key in (
        "model name",
        "CPU implementer",
        "CPU architecture",
        "CPU variant",
        "CPU part",
        "CPU revision",
    ):
        if key in info:
            print(f"  {key}: {info[key]}")
    flags = sorted(host.cpu_flags())
    interesting = [
        f for f in flags if f.startswith(("avx", "sve", "asimd", "fp", "fma", "sme"))
    ]
    print(f"  {len(flags)} cpu flags/features, vector related: {' '.join(interesting)}")
    if host.arm_midr():
        imp, var, part, rev = host.arm_midr()
        print(f"  MIDR: implementer {imp:#x} part {part:#x} r{var}p{rev}")
        print(f"  STALL_SLOT counts cycles erratum: {host.stall_slot_counts_cycles()}")
    print(f"  SVE default vector bits: {host.sve_vector_bits()}")
    print(
        f"  virtualised: {'hypervisor' in host.cpu_flags() or _read('/sys/hypervisor/type')}"
    )


def diagnose_access():
    section("perf_event access")
    print(f"  perf_event_paranoid: {perf.paranoid_level()}")
    print(f"  privileged (root/CAP_PERFMON): {perf.privileged()}")
    print(f"  nmi_watchdog: {perf.nmi_watchdog()}")
    print(
        f"  perf_event_max_sample_rate: {_read('/proc/sys/kernel/perf_event_max_sample_rate')}"
    )
    can_count, warnings = perf.access_warnings()
    print(f"  can count: {can_count}")
    for w in warnings:
        print(f"  WARNING: {w}")
    return can_count


def diagnose_pmus():
    section("kernel PMUs (/sys/bus/event_source/devices)")
    try:
        devs = sorted(os.listdir(host.EVENT_SOURCE_DIR))
    except OSError as e:
        print(f"  cannot list: {e}")
        return False
    core = {p.name for p in host.core_pmus()}
    for d in devs:
        print(f"  {d}{'  <- core PMU' if d in core else ''}")
    if not core:
        print(
            "  ERROR: no core PMU, the kernel exposes no CPU hardware counters "
            "(VM without PMU passthrough? missing kernel driver?)"
        )
        return False
    for pmu in host.core_pmus():
        print(f"  {pmu.name}: type {_read(pmu / 'type')}, cpus {_read(pmu / 'cpus')}")
        for f in sorted((pmu / "format").iterdir()):
            print(f"    format {f.name}: {_read(f)}")
        for k, v in sorted(host.pmu_caps(pmu).items()):
            print(f"    caps {k}: {v}")
        events = pmu / "events"
        if events.is_dir():
            print(f"    {len(os.listdir(events))} named events in sysfs")
    print(f"  perf config mask: {ce.perf_config_mask():#x}")
    return True


def diagnose_perf_tool():
    section("perf tool")
    exe = shutil.which("perf")
    print(f"  perf: {exe}")
    if not exe:
        print("  ERROR: perf not on PATH (linux-tools-$(uname -r) / linux-perf)")
        return False
    show(["perf", "--version"])
    code, _ = show(["perf", "stat", "-e", "cycles:u,instructions:u", "--", *WORKLOAD])
    return code == 0


def diagnose_libpfm():
    section("libpfm")
    exe = ce.pfm_examples_dir() / "check_events"
    print(f"  check_events: {exe} exists={exe.exists()}")
    if not exe.exists():
        print("  ERROR: build libpfm first: ./sh/setup_pfm.sh (or set PHLOP_PFM_DIR)")
        return None
    pmus = ce.detected_pmus()
    print(f"  detected PMUs: {pmus}")
    pmu, sets = sets_for(pmus)
    from phlop.app.pfm.stat import pmu_counters

    print(f"  event sets for: {pmu}, counters: {pmu_counters(pmu)}")
    if pmu == "generic":
        print(f"  WARNING: no event sets for {pmus}, only perf generic events")
    return sets


def _try_set(eset):
    """encode + count one set on WORKLOAD, print per event.
    Returns {alias: value or None}, {} if encoding or perf failed outright"""
    from phlop.app.pfm.stat import collect, perf_events, perf_stat_cmd

    print(f"\n  -- {eset.name} ({eset.counters_needed()} counters): {eset.desc}")
    for alias, spec in eset.events.items():
        if spec.startswith(("raw:", "perf:")):
            continue
        try:
            config = ce.encode([spec])[spec]
            print(f"    encode {alias:<18} {spec} -> {config:#x}")
        except ValueError as e:
            print(f"    encode {alias:<18} {spec} FAILED: {e}")
            return {}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        outfile = f.name
    try:
        reported = perf_events(eset, "u")
        cmd = perf_stat_cmd(reported, outfile) + ["--", *WORKLOAD]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False
        )
        rows = perf.parse_stat_json(outfile) if os.path.getsize(outfile) else []
    finally:
        os.unlink(outfile)

    if proc.returncode or not rows:
        print(f"    perf failed (exit {proc.returncode}): {' '.join(cmd)}")
        for line in proc.stderr.strip().splitlines()[:20]:
            print(f"      {line}")
    values = {}
    for alias, ev in collect(eset, rows, reported).items():
        values[alias] = ev.value
        if ev.value is None:
            state = "NOT COUNTED"
        else:
            state = f"{ev.value:,.0f}"
            if ev.pcnt_running is not None and ev.pcnt_running < 100:
                state += f" (multiplexed {ev.pcnt_running:.0f}%)"
            if ev.value == 0:
                state += "  <- zero"
        print(f"    count  {alias:<18} {state}")
    return values


def main(argv=None):
    strict = "--strict" in (argv if argv is not None else sys.argv[1:])
    diagnose_platform()
    can_count = diagnose_access()
    has_pmu = diagnose_pmus()
    has_perf = diagnose_perf_tool()
    sets = diagnose_libpfm()

    counted = {}
    if sets and has_perf:
        section("event sets on a test workload")
        for name, eset in sets.items():
            counted[name] = _try_set(eset)

    section("summary")
    print(f"  can count: {can_count}, core PMU: {has_pmu}, perf works: {has_perf}")
    print(f"  dispatch slots: {host.dispatch_slots_and_source()}")
    print("  (zero can also mean the workload never does it, eg FP or RAM)")
    for name, values in counted.items():
        zero = [a for a, v in values.items() if not v]
        if not values or len(zero) == len(values):
            state = "NOTHING COUNTED"
        elif zero:
            state = f"PARTIAL, zero/uncounted: {', '.join(zero)}"
        else:
            state = "ok"
        print(f"  {name:<10} {state}")
    any_counted = any(v for values in counted.values() for v in values.values())
    if strict and not any_counted:
        print("  ERROR: no event set counted anything")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
