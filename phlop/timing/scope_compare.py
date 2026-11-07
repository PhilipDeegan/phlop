# phlop/timing/scope_compare.py
#
# performance regression check between PHLOP_SCOPE timer files, a baseline
# and a candidate, eg the same test before and after a change:
#
#   python3 -m phlop.timing.scope_compare base.txt new.txt
#   python3 -m phlop.timing.scope_compare base.txt new.txt -t instructions=0.5 -t cycles=5
#   python3 -m phlop.timing.scope_compare base_runs/ new_runs.tar.gz -t time=5
#
# Each side is a timer file, a directory or a .tar.gz. Directories and tar
# files are searched recursively for files matching --pattern, and files in
# the same directory are one run, one file per rank, eg
#
#   base_runs/<run>/.phare/timings/rank.<N>.txt
#
# Every run of both sides must have the same number of ranks: counts are
# summed over ranks, so a rank count change is not a regression, an error.
# Ranks of a run are merged: counts summed, time the slowest rank's. Runs of
# a side are merged: counts the median run's, time the fastest run's. The
# spread of a value between a side's runs, (max - min) / merged value, is
# that side's noise: a value only regresses if it grows beyond its threshold
# plus both sides' noise. One run per side, no noise, the threshold alone.
#
# Values are time, the perf events' aliases and, for PHLOP_SCOPE("key", time,
# mem) scopes run with PHLOP_SCOPE_MEM=1 (or =peak), memory: mem_peak, the
# process high water mark at scope exit (with =peak the scope's own), mem_spike,
# the peak's rise during the scope, and mem_kept, resident growth exit - entry.
# Memory is the worst rank's and never "own": a peak is a level, not a sum.
# Gate them like events, eg -t mem_peak=2, --min-count is then bytes.
#
# Scopes match by path, root to scope, same named siblings merged (summed) as
# in print_scope_perf. Counts are the scope's own by default: children counted
# at its level subtracted, so a regression is reported where it happened, not
# in every ancestor. Exit 1 if any gated value grew beyond its threshold, 2 if
# the files can't be compared (different event sets or PMUs), else 0.
#
# Retired instructions are deterministic for the same binary and input, good
# for tight thresholds. Cycles and time move with the machine, keep those
# looser. Compare runs on the same CPU model and toolchain: libc picks memcpy
# and friends per CPU, compilers and flags change the counts. Spin waits
# (OpenMP barriers, MPI progress) retire instructions while waiting, gate on
# compute scopes or single threaded runs: counts that vary between runs of a
# side are reported as warnings.

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from phlop.timing.scope_timer import (
    MEM_VALUES,
    ScopeTimerFile,
    file_parser,
    format_bytes,
    group_scopes,
    lines_parser,
)

DEFAULT_THRESHOLDS = {"instructions": 1.0}  # percent
DEFAULT_MIN_COUNT = 100_000  # below this the scope's counter reads dominate
DEFAULT_PATTERN = "*.txt"


@dataclass
class Change:
    path: tuple
    name: str  # event alias, or "time" in ns
    base: float
    new: float
    threshold: float | None  # percent, None if only reported
    base_noise: float = 0.0  # percent, spread between the side's runs
    new_noise: float = 0.0

    @property
    def pct(self):
        return (self.new - self.base) / self.base * 100.0 if self.base else 0.0

    @property
    def noise(self):
        return self.base_noise + self.new_noise

    @property
    def limit(self):
        return None if self.threshold is None else self.threshold + self.noise

    @property
    def regressed(self):
        return self.threshold is not None and self.pct > self.limit

    @property
    def improved(self):
        return self.threshold is not None and self.pct < -self.limit


@dataclass
class Comparison:
    changes: list  # [Change], gated ones and, with all_events, the rest
    only_base: list  # paths
    only_new: list
    calls_changed: list  # (path, base calls, new calls)
    warnings: list
    runs: tuple = (1, 1)  # runs per side
    ranks: int = 1  # files per run, the same for every run

    @property
    def regressions(self):
        return [c for c in self.changes if c.regressed]

    @property
    def improvements(self):
        return [c for c in self.changes if c.improved]


class IncompatibleFiles(Exception):
    pass


@dataclass
class ScopeValues:
    """one scope path of one side: calls and perf level of its first run,
    {event alias or "time": [value per run]}"""

    n: int
    level: int
    counted: bool  # has perf counts, else values is only time
    values: dict = field(default_factory=dict)


def check_compatible(base, new):
    """the levels both files count with must be the same event sets on the
    same PMU, else counts mean different things. ctx differences only warn"""
    warnings = []
    for level in sorted(set(base.perf) & set(new.perf)):
        a, b = base.perf[level], new.perf[level]
        if a.label != b.label or a.names != b.names:
            raise IncompatibleFiles(
                f"perf level {level}: {a.label} {','.join(a.names)}"
                f" vs {b.label} {','.join(b.names)}"
            )
    if (base.ctx or {}) != (new.ctx or {}):
        warnings.append(f"#ctx differs: {base.ctx} vs {new.ctx}")
    for stf, which in ((base, "base"), (new, "new")):
        warnings += [f"{which}: {w}" for w in stf.perf_warnings()]
    return warnings


def flatten(groups, own, prefix=()):
    """{path: (ScopeGroup, own or inclusive time)}"""
    flat = {}
    for g in groups:
        path = prefix + (g.key,)
        t = g.t - sum(c.t for c in g.c) if own else g.t
        flat[path] = (g, t)
        flat.update(flatten(g.c, own, path))
    return flat


def as_runs(side):
    """[[ScopeTimerFile per rank] per run] from a ScopeTimerFile, a list of
    them (one run, a file per rank) or a list of those lists"""
    if isinstance(side, ScopeTimerFile):
        return [[side]]
    side = list(side)
    if side and isinstance(side[0], ScopeTimerFile):
        return [side]
    return [list(run) for run in side]


def merge_ranks(run, own, warnings):
    """{path: ScopeValues} of one run, a value per name: calls and counts
    summed over ranks, time and memory the worst rank's. Memory is never own,
    a peak is a level, not a sum over children"""
    merged = {}
    for stf in run:
        for path, (g, t) in flatten(group_scopes(stf, stf.roots, own), own).items():
            sv = merged.setdefault(path, ScopeValues(0, g.level, False, {"time": 0}))
            sv.n += g.n
            sv.values["time"] = max(sv.values["time"], t)
            for name, value in (g.mem or {}).items():
                sv.values[name] = max(sv.values.get(name, value), value)
            if not g.counts:
                continue
            if sv.counted and g.level != sv.level:
                warnings.append(
                    f"{'/'.join(path)}: perf level {sv.level} vs {g.level} across ranks"
                )
                continue
            sv.level, sv.counted = g.level, True
            for name, value in g.counts.items():
                sv.values[name] = sv.values.get(name, 0) + value
    return merged


def merge_runs(runs, own, which, warnings):
    """{path: ScopeValues} of one side, the paths seen in every run, values
    [one per run]"""
    per_run = [merge_ranks(run, own, warnings) for run in runs]
    common = [p for p in per_run[0] if all(p in r for r in per_run[1:])]
    if missing := len({p for r in per_run for p in r}) - len(common):
        warnings.append(f"{which}: {missing} scopes not in every run, skipped")
    side = {}
    for path in common:
        first = per_run[0][path]
        sv = side[path] = ScopeValues(first.n, first.level, first.counted)
        for r in per_run:
            if r[path].counted and (r[path].level != first.level or not first.counted):
                warnings.append(
                    f"{which}: {'/'.join(path)} perf level differs between runs"
                )
            for name, value in r[path].values.items():
                sv.values.setdefault(name, []).append(value)
        sv.values = {k: v for k, v in sv.values.items() if len(v) == len(per_run)}
    return side


def merged_value(name, values):
    """(value, noise percent): the fastest run's time, the median run's count"""
    value = float(min(values) if name == "time" else np.median(values))
    noise = (max(values) - min(values)) / value * 100.0 if value else 0.0
    return value, noise


def compare(
    base,
    new,
    thresholds=None,
    own=True,
    min_count=DEFAULT_MIN_COUNT,
    all_events=False,
):
    """Comparison of two sides, each a ScopeTimerFile or runs of them, see
    as_runs. thresholds: {event alias or "time": percent}, values growing
    more than that plus both sides' noise regress. Scopes whose baseline
    value is under min_count (ns for time) are skipped: too small to gate"""
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    base_runs, new_runs = as_runs(base), as_runs(new)
    if not base_runs or not new_runs or not all(base_runs + new_runs):
        raise ValueError("nothing to compare, a side or one of its runs has no files")
    if len(ranks := {len(run) for run in base_runs + new_runs}) > 1:
        raise IncompatibleFiles(
            "runs have different rank counts, base"
            f" {[len(r) for r in base_runs]} vs new {[len(r) for r in new_runs]}:"
            " counts are summed over ranks, compare runs of the same rank count"
        )

    ref = base_runs[0][0]
    warnings = check_compatible(ref, new_runs[0][0])
    for stf in [f for run in base_runs + new_runs for f in run][1:]:
        check_compatible(ref, stf)  # raises, warnings once from the first pair

    a = merge_runs(base_runs, own, "base", warnings)
    b = merge_runs(new_runs, own, "new", warnings)

    changes, calls_changed = [], []
    for path in (p for p in a if p in b):
        sa, sb = a[path], b[path]
        if sa.n != sb.n:
            calls_changed.append((path, sa.n, sb.n))
        values_a, values_b = sa.values, sb.values
        if sa.counted and sb.counted and sa.level != sb.level:
            warnings.append(f"{'/'.join(path)}: perf level {sa.level} vs {sb.level}")
            keep = ("time", *MEM_VALUES)
            values_a = {k: v for k, v in values_a.items() if k in keep}
            values_b = {k: v for k, v in values_b.items() if k in keep}
        for name in values_a:
            if name not in values_b:
                continue
            if name not in thresholds and not all_events:
                continue
            va, noise_a = merged_value(name, values_a[name])
            if va < min_count:
                continue
            vb, noise_b = merged_value(name, values_b[name])
            change = Change(path, name, va, vb, thresholds.get(name), noise_a, noise_b)
            if (
                name != "time"
                and change.threshold is not None
                and change.noise > change.threshold
            ):
                why = "" if name in MEM_VALUES else ": spin waits?"
                warnings.append(
                    f"{'/'.join(path)} {name} varies {change.noise:.2f}% between runs"
                    f" (threshold {change.threshold:g}%){why}"
                )
            changes.append(change)

    return Comparison(
        changes,
        [p for p in a if p not in b],
        [p for p in b if p not in a],
        calls_changed,
        warnings,
        (len(base_runs), len(new_runs)),
        ranks.pop(),
    )


def report(comparison, printer=print, top=None):
    """worst first, regressions then improvements then the rest"""
    if comparison.runs != (1, 1) or comparison.ranks != 1:
        (ra, rb), n = comparison.runs, comparison.ranks
        printer(f"runs: base {ra}, new {rb}, {n} ranks each")
    for w in comparison.warnings:
        printer(f"WARNING: {w}")

    def line(tag, c):
        fmt = "{:,.0f}".format
        if c.name == "time":
            fmt = lambda v: f"{v / 1e6:,.3f}ms"
        elif c.name in MEM_VALUES:
            fmt = lambda v: format_bytes(v, sign=c.name == "mem_kept")
        limit = ""
        if c.threshold is not None:
            noise = f" + {c.noise:.2f}% noise" if c.noise else ""
            limit = f" (limit {c.threshold:g}%{noise})"
        elif c.noise:
            limit = f" (noise {c.noise:.2f}%)"
        return (
            f"{tag:<11} {'/'.join(c.path)} {c.name} {c.pct:+.2f}%"
            f" {fmt(c.base)} -> {fmt(c.new)}{limit}"
        )

    regressions = sorted(comparison.regressions, key=lambda c: -c.pct)
    improvements = sorted(comparison.improvements, key=lambda c: c.pct)
    rest = [c for c in comparison.changes if not c.regressed and not c.improved]
    rest.sort(key=lambda c: -abs(c.pct))
    for tag, changes in (("REGRESSION", regressions), ("improved", improvements)):
        for c in changes[:top]:
            printer(line(tag, c))
    for c in rest[:top]:
        printer(line("", c))
    for path, n_a, n_b in comparison.calls_changed:
        printer(f"calls      {'/'.join(path)} x{n_a} -> x{n_b}")
    for path in comparison.only_base:
        printer(f"removed    {'/'.join(path)}")
    for path in comparison.only_new:
        printer(f"added      {'/'.join(path)}")
    printer(
        f"{len(regressions)} regressions, {len(improvements)} improvements,"
        f" {len(comparison.changes)} values compared"
    )


def load(path, pattern=DEFAULT_PATTERN):
    """[[ScopeTimerFile per rank] per run] from a timer file, or a directory
    or tar file searched recursively for pattern, a run per directory. Runs
    and ranks sorted by path"""
    path = Path(path)
    is_tar = path.name.endswith((".tar.gz", ".tgz", ".tar"))
    if path.is_file() and not is_tar:
        return [[file_parser(path)]]
    by_dir = {}  # run directory -> {file name: lines}
    if path.is_dir():
        for f in path.rglob(pattern):
            if f.is_file():
                with open(f) as file:
                    by_dir.setdefault(str(f.parent), {})[f.name] = file.readlines()
    elif path.is_file():
        import tarfile

        with tarfile.open(path) as tar:
            for m in tar:
                p = Path(m.name)
                if m.isfile() and p.match(pattern):
                    lines = tar.extractfile(m).read().decode().splitlines(keepends=True)
                    by_dir.setdefault(str(p.parent), {})[p.name] = lines
    else:
        raise ValueError(f"no such file or directory: {path}")
    if not by_dir:
        raise ValueError(f"no files matching {pattern!r} in {path}")

    def parse(run, name):
        try:
            return lines_parser(by_dir[run][name])
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"{run}/{name}: not a scope timer file ({e}), see --pattern"
            )

    return [
        [parse(run, name) for name in sorted(by_dir[run])] for run in sorted(by_dir)
    ]


def parse_thresholds(items):
    """["instructions=1", "cycles=5"] -> {"instructions": 1.0, "cycles": 5.0}"""
    thresholds = {}
    for item in items:
        name, eq, pct = item.partition("=")
        if not name or not eq:
            raise ValueError(f"threshold not name=percent: {item!r}")
        thresholds[name] = float(pct)
    return thresholds


def cli_args_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="PHLOP_SCOPE timer file regression check, base vs new",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "base", help="baseline timer file, or directory / tar file of runs"
    )
    parser.add_argument(
        "new", help="candidate timer file, or directory / tar file of runs"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        action="append",
        default=[],
        metavar="NAME=PCT",
        help="gate an event alias, time or mem_peak/spike/kept on growth over PCT percent plus"
        f" noise, repeatable. Default {DEFAULT_THRESHOLDS}",
    )
    parser.add_argument(
        "--inclusive",
        dest="own",
        action="store_false",
        help="compare counts including children, not each scope's own",
    )
    parser.add_argument(
        "--min-count",
        type=float,
        default=DEFAULT_MIN_COUNT,
        help="skip scopes whose baseline value is below this (ns for time)",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default=DEFAULT_PATTERN,
        help="timer file names searched for in directories and tar files",
    )
    parser.add_argument(
        "-a", "--all", action="store_true", help="also report ungated events"
    )
    parser.add_argument("--top", type=int, default=None, help="lines per section")
    return parser


def main(argv=None):
    cli_args = cli_args_parser().parse_args(argv)
    try:
        thresholds = parse_thresholds(cli_args.threshold) or DEFAULT_THRESHOLDS
        comparison = compare(
            load(cli_args.base, cli_args.pattern),
            load(cli_args.new, cli_args.pattern),
            thresholds,
            cli_args.own,
            cli_args.min_count,
            cli_args.all,
        )
    except (IncompatibleFiles, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    report(comparison, top=cli_args.top)
    return 1 if comparison.regressions else 0


if __name__ == "__main__":
    sys.exit(main())
