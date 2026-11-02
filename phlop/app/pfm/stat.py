# phlop/app/pfm/stat.py
#
# Run a program under `perf stat` once per named event set (see sets.py),
# events encoded by libpfm, and report derived metrics with optimization hints.
#
#   python3 -m phlop.app.pfm.stat --list
#   python3 -m phlop.app.pfm.stat -s topdown,vector,flops -- ./app args
#   python3 -m phlop.app.pfm.stat -s all -o stats.json -- ./app args
#
# Each set reruns the program, so it should be deterministic. Sets are sized
# to fit the hardware counters, avoiding multiplexing where possible.

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from functools import cache

from phlop.app import perf
from phlop.app.pfm import check_events as ce
from phlop.app.pfm import host
from phlop.app.pfm.sets import DEFAULT_SETS, EventSet, sets_for

NOT_COUNTED = ("<not counted>", "<not supported>")


@dataclass
class EventResult:
    spec: str
    value: float | None = None
    pcnt_running: float | None = None
    variance: float | None = None


@dataclass
class SetResult:
    name: str
    desc: str
    perf_cmd: list
    events: dict = field(default_factory=dict)  # alias -> EventResult
    metrics: dict = field(default_factory=dict)  # name -> value
    units: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)  # name -> shown instead of n/a
    hints: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    exitcode: int | None = None


@cache
def pmu_counters(pmu):
    """generic counters on pmu, from libpfm showevtinfo, None if unknown"""
    try:
        from phlop.app.pfm.showevtinfo import run_evtinfo

        for line in run_evtinfo():
            if line.startswith("Total events"):
                break
            if f", {pmu}," in line and " counters" in line:
                bits = [b.strip() for b in line.split(",")]
                return int(next(b for b in bits if b.endswith(" counters")).split()[0])
    except (OSError, ValueError, StopIteration) as e:
        print(f"WARNING: cannot read counter count for {pmu}: {e}")
    return None


def available_counters(pmu):
    n = pmu_counters(pmu)
    if n is not None and perf.nmi_watchdog() == 1:
        n -= 1
    return n


def perf_events(eset, modifiers):
    """[(alias, name perf reports, perf -e string)]"""
    pfm = [s for s in eset.events.values() if not s.startswith(("raw:", "perf:"))]
    enc = ce.encode(pfm) if pfm else {}
    out = []
    for alias, spec in eset.events.items():
        if spec.startswith("raw:"):
            pmu_str = spec[4:].rstrip("/")
            out.append((alias, alias, f"{pmu_str},name={alias}/{modifiers}"))
        elif spec.startswith("perf:"):
            sym = spec[5:]
            out.append((alias, sym, f"{sym}:{modifiers}" if modifiers else sym))
        else:
            out.append(
                (alias, alias, ce.perf_event_string(alias, enc[spec], modifiers))
            )
    return out


def perf_stat_cmd(events, outfile, repeat=1):
    cmd = ["perf", "stat", "-j", "-o", outfile]
    if repeat > 1:
        cmd += ["-r", str(repeat)]
    for _, _, ev in events:
        cmd += ["-e", ev]
    return cmd


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_perf_json(path):
    return perf.parse_stat_json(path)


def collect(eset, rows, reported_as):
    """fill EventResults from perf json rows, rows may carry ":u" suffixes"""
    by_name = {}
    for row in rows:
        name = row.get("event", "")
        by_name[name] = row
        by_name.setdefault(name.split(":")[0], row)

    out, reported = {}, {alias: name for alias, name, _ in reported_as}
    for alias, spec in eset.events.items():
        res = EventResult(spec=spec)
        row = by_name.get(reported.get(alias, alias))
        if row is not None and row.get("counter-value") not in NOT_COUNTED:
            res.value = _num(row.get("counter-value"))
            res.pcnt_running = _num(row.get("pcnt-running"))
            res.variance = _num(row.get("variance"))
        out[alias] = res
    return out


def evaluate(eset, result, ctx):
    """metrics see counts by alias, and ctx values as "@key", eg c["@slots"]"""
    counts = {k: v.value for k, v in result.events.items()}
    counts.update({f"@{k}": v for k, v in ctx.items()})
    for m in eset.metrics:
        try:
            val = m.fn(counts)
        except (TypeError, ZeroDivisionError, KeyError):
            val = None
        result.metrics[m.name] = val
        result.units[m.name] = m.unit
        if val is None and m.none_means is not None:
            try:
                if label := m.none_means(counts):
                    result.labels[m.name] = label
            except (TypeError, KeyError):
                pass
        if m.hint is not None:
            try:
                hint = m.hint(val, counts, ctx)
            except (TypeError, KeyError):
                hint = None
            if hint:
                result.hints.append(f"{m.name}: {hint}")


def run_set(eset, program, kernel=False, repeat=1, quiet=False, print_only=False):
    modifiers = "" if kernel else "u"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        outfile = f.name
    try:
        reported = perf_events(eset, modifiers)
        cmd = perf_stat_cmd(reported, outfile, repeat)
        result = SetResult(eset.name, eset.desc, cmd[:3] + cmd[5:] + ["--"] + program)
        if print_only:
            return result
        stream = subprocess.DEVNULL if quiet else None
        proc = subprocess.run(cmd + ["--"] + program, stdout=stream, check=False)
        result.exitcode = proc.returncode
        rows = parse_perf_json(outfile) if os.path.getsize(outfile) else []
    finally:
        os.unlink(outfile)

    if not rows:
        result.warnings.append(f"perf produced no counts (exit code {proc.returncode})")
        return result
    if proc.returncode != 0:
        result.warnings.append(f"program/perf exit code {proc.returncode}")

    result.events = collect(eset, rows, reported)
    for alias, ev in result.events.items():
        if ev.value is None:
            result.warnings.append(f"{alias} ({ev.spec}) not counted")
        elif ev.value == 0:
            result.warnings.append(
                f"{alias} ({ev.spec}) reads 0: not exercised by the program, or "
                "not supported here (VMs often filter events)"
            )
        elif ev.pcnt_running is not None and ev.pcnt_running < 100:
            result.warnings.append(
                f"{alias} multiplexed, counted {ev.pcnt_running:.0f}% of the time: "
                "value is a scaled estimate"
            )
    evaluate(eset, result, host.context())
    return result


def stat(
    program, set_names=None, kernel=False, repeat=1, quiet=False, print_only=False
):
    """run program once per event set, returns (pmu, [SetResult], [warnings])"""
    pmu, sets = sets_for(ce.detected_pmus())
    set_names = set_names or [s for s in DEFAULT_SETS if s in sets]
    if set_names == ["all"]:
        set_names = list(sets)
    unknown = [s for s in set_names if s not in sets]
    if unknown:
        raise ValueError(f"unknown sets {unknown} for {pmu}, have: {list(sets)}")

    can_count, warnings = perf.access_warnings(kernel)
    if shutil.which("perf") is None:
        warnings.append("perf not on PATH (linux-tools-$(uname -r) / linux-perf)")
        can_count = False
    if not can_count:
        print_only = True

    budget = available_counters(pmu)
    results = []
    for name in set_names:
        eset = sets[name]
        res = run_set(eset, program, kernel, repeat, quiet, print_only)
        if budget is not None and eset.counters_needed() > budget:
            res.warnings.append(
                f"needs {eset.counters_needed()} counters, {budget} available: "
                "expect multiplexing"
            )
        results.append(res)
    return pmu, results, warnings, can_count


def _fmt(val, unit):
    if val is None:
        return "n/a"
    if unit == "%":
        return f"{val * 100:.2f} %"
    if abs(val) >= 1e4:
        return f"{val:,.0f}"
    return f"{val:.3f}"


def report(pmu, results, warnings, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p(f"\nPMU: {pmu}")
    for w in warnings:
        p(f"WARNING: {w}")
    for r in results:
        p(f"\n== {r.name}: {r.desc}")
        if not r.events:
            p("   " + " ".join(r.perf_cmd))
        for alias, ev in r.events.items():
            extra = ""
            if ev.pcnt_running is not None and ev.pcnt_running < 100:
                extra = f"  ({ev.pcnt_running:.0f}% running)"
            if ev.variance:
                extra += f"  (+- {ev.variance:.2f}%)"
            p(f"   {alias:<22} {_fmt(ev.value, '')}{extra}")
        if r.events:
            p("   --")
        for name, val in r.metrics.items():
            shown = r.labels.get(name) or _fmt(val, r.units.get(name))
            p(f"   {name:<22} {shown}")
        for w in r.warnings:
            p(f"   warning: {w}")
    hints = [(r.name, h) for r in results for h in r.hints]
    if hints:
        p("\nHints:")
        for name, h in hints:
            p(f"  [{name}] {h}")


def list_sets(out=sys.stdout):
    pmu, sets = sets_for(ce.detected_pmus())
    budget = available_counters(pmu)
    print(f"PMU: {pmu}, counters available: {budget}", file=out)
    print(f"default sets: {','.join(DEFAULT_SETS)}", file=out)
    for s in sets.values():
        print(f"\n{s.name} ({s.counters_needed()} counters): {s.desc}", file=out)
        for alias, spec in s.events.items():
            print(f"    {alias:<18} {spec}", file=out)
        print(f"  metrics: {', '.join(m.name for m in s.metrics)}", file=out)


ENV_LEVELS = {"perf_light": "PHLOP_PERF_LIGHT", "perf_all": "PHLOP_PERF_ALL"}


def scope_env(set_name, sets=None, pmu=None):
    """PHLOP_SCOPE env value for an event set, see perf_counters.hpp:
    "<pmu>/<set>;<alias>=<raw config>,..." """
    if sets is None:
        pmu, sets = sets_for(ce.detected_pmus())
    eset = sets[set_name]
    pfm = [s for s in eset.events.values() if not s.startswith(("raw:", "perf:"))]
    enc = ce.encode(pfm) if pfm else {}
    items = []
    for alias, spec in eset.events.items():
        if spec.startswith("perf:"):
            raise ValueError(f"{set_name}: perf generic event {spec} has no raw config")
        if spec.startswith("raw:"):
            config = ce.pmu_terms_config(spec[4:])
        else:
            config = enc[spec] & ce.perf_config_mask()
        items.append(f"{alias}={config:#x}")
    return f"{pmu}/{set_name};{','.join(items)}"


def scope_env_exports(level_sets):
    """shell export lines for ["perf_light=light", "perf_all=topdown"]"""
    pmu, sets = sets_for(ce.detected_pmus())
    lines, union = [], {}  # config -> counters it costs
    for item in level_sets:
        level, _, set_name = item.partition("=")
        if level not in ENV_LEVELS or set_name not in sets:
            raise ValueError(
                f"--env {item}: want {'|'.join(ENV_LEVELS)}=<set>, sets: {list(sets)}"
            )
        value = scope_env(set_name, sets, pmu)
        configs = [c.split("=")[1] for c in value.split(";")[1].split(",")]
        for config, spec in zip(configs, sets[set_name].events.values()):
            union[config] = EventSet("", "", {"": spec}).counters_needed()
        lines.append(f"export {ENV_LEVELS[level]}='{value}'")
    from phlop.app.pfm import host

    ctx = {k: v for k, v in host.context().items() if k != "cpu_flags"}
    ctx = ",".join(f"{k}={int(v)}" for k, v in ctx.items() if v is not None)
    lines.append(f"export PHLOP_PERF_CTX='{ctx}'")  # written as the file's #ctx
    budget, needed = available_counters(pmu), sum(union.values())
    if budget is not None and needed > budget:
        print(
            f"WARNING: {len(union)} distinct events need {needed} counters, "
            f"{budget} available: the pinned group will not schedule, use smaller sets",
            file=sys.stderr,
        )
    return lines


def cli_args_parser(description="perf stat with libpfm event sets"):
    import argparse

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("remaining", nargs=argparse.REMAINDER, help="program and args")
    parser.add_argument(
        "-s",
        "--sets",
        default=None,
        help=f"comma separated, or all. default: {','.join(DEFAULT_SETS)}, "
        "those the detected PMU has",
    )
    parser.add_argument("-l", "--list", action="store_true", help="list event sets")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="LEVEL=SET",
        help="print PHLOP_SCOPE counter env exports, eg --env perf_light=light "
        '--env perf_all=topdown, use as: eval "$(... --env ...)"',
    )
    parser.add_argument("-r", "--repeat", type=int, default=1, help="perf stat -r")
    parser.add_argument("-o", "--outfile", default=None, help="write results json")
    parser.add_argument(
        "-k", "--kernel", action="store_true", help="count kernel space too"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="hide program stdout"
    )
    parser.add_argument(
        "-p", "--print_only", action="store_true", help="print perf commands only"
    )
    return parser


def main(argv=None):
    cli_args = cli_args_parser().parse_args(argv)
    if cli_args.list:
        list_sets()
        return 0
    if cli_args.env:
        print("\n".join(scope_env_exports(cli_args.env)))
        return 0
    program = cli_args.remaining
    if program and program[0] == "--":
        program = program[1:]
    if not program:
        cli_args_parser().print_usage()
        return 1

    pmu, results, warnings, can_count = stat(
        program,
        cli_args.sets.split(",") if cli_args.sets else None,
        cli_args.kernel,
        cli_args.repeat,
        cli_args.quiet,
        cli_args.print_only,
    )
    report(pmu, results, warnings)
    if cli_args.outfile:
        with open(cli_args.outfile, "w") as f:
            json.dump(
                {
                    "pmu": pmu,
                    "warnings": warnings,
                    "sets": [asdict(r) for r in results],
                },
                f,
                indent=2,
            )
    if not can_count and not cli_args.print_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
