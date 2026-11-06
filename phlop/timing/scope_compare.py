# phlop/timing/scope_compare.py
#
# performance regression check between two PHLOP_SCOPE timer files, a
# baseline and a candidate, eg the same test before and after a change:
#
#   python3 -m phlop.timing.scope_compare base.txt new.txt
#   python3 -m phlop.timing.scope_compare base.txt new.txt -t instructions=0.5 -t cycles=5
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
# compute scopes or single threaded runs.

from __future__ import annotations

import sys
from dataclasses import dataclass

from phlop.timing.scope_timer import file_parser, group_scopes

DEFAULT_THRESHOLDS = {"instructions": 1.0}  # percent
DEFAULT_MIN_COUNT = 100_000  # below this the scope's counter reads dominate


@dataclass
class Change:
    path: tuple
    name: str  # event alias, or "time" in ns
    base: float
    new: float
    threshold: float | None  # percent, None if only reported

    @property
    def pct(self):
        return (self.new - self.base) / self.base * 100.0 if self.base else 0.0

    @property
    def regressed(self):
        return self.threshold is not None and self.pct > self.threshold

    @property
    def improved(self):
        return self.threshold is not None and self.pct < -self.threshold


@dataclass
class Comparison:
    changes: list  # [Change], gated ones and, with all_events, the rest
    only_base: list  # paths
    only_new: list
    calls_changed: list  # (path, base calls, new calls)
    warnings: list

    @property
    def regressions(self):
        return [c for c in self.changes if c.regressed]

    @property
    def improvements(self):
        return [c for c in self.changes if c.improved]


class IncompatibleFiles(Exception):
    pass


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


def compare(
    base,
    new,
    thresholds=None,
    own=True,
    min_count=DEFAULT_MIN_COUNT,
    all_events=False,
):
    """Comparison of two ScopeTimerFiles. thresholds: {event alias or "time":
    percent}, values growing more than that regress. Scopes whose baseline
    value is under min_count (ns for time) are skipped: too small to gate"""
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    warnings = check_compatible(base, new)
    a = flatten(group_scopes(base, base.roots, own), own)
    b = flatten(group_scopes(new, new.roots, own), own)

    changes, calls_changed = [], []
    for path in (p for p in a if p in b):
        (ga, ta), (gb, tb) = a[path], b[path]
        if ga.n != gb.n:
            calls_changed.append((path, ga.n, gb.n))
        values_a = {**(ga.counts or {}), "time": ta}
        values_b = {**(gb.counts or {}), "time": tb}
        if ga.counts and gb.counts and ga.level != gb.level:
            warnings.append(f"{'/'.join(path)}: perf level {ga.level} vs {gb.level}")
            values_a, values_b = {"time": ta}, {"time": tb}
        for name in values_a:
            if name not in values_b:
                continue
            if name not in thresholds and not all_events:
                continue
            if values_a[name] < min_count:
                continue
            changes.append(
                Change(path, name, values_a[name], values_b[name], thresholds.get(name))
            )

    return Comparison(
        changes,
        [p for p in a if p not in b],
        [p for p in b if p not in a],
        calls_changed,
        warnings,
    )


def report(comparison, printer=print, top=None):
    """worst first, regressions then improvements then the rest"""
    for w in comparison.warnings:
        printer(f"WARNING: {w}")

    def line(tag, c):
        fmt = (lambda v: f"{v / 1e6:,.3f}ms") if c.name == "time" else "{:,.0f}".format
        limit = f" (limit {c.threshold:g}%)" if c.threshold is not None else ""
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
    parser.add_argument("base", help="baseline scope timer file")
    parser.add_argument("new", help="candidate scope timer file")
    parser.add_argument(
        "-t",
        "--threshold",
        action="append",
        default=[],
        metavar="NAME=PCT",
        help="gate an event alias, or time, on growth over PCT percent,"
        f" repeatable. Default {DEFAULT_THRESHOLDS}",
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
        "-a", "--all", action="store_true", help="also report ungated events"
    )
    parser.add_argument("--top", type=int, default=None, help="lines per section")
    return parser


def main(argv=None):
    cli_args = cli_args_parser().parse_args(argv)
    try:
        thresholds = parse_thresholds(cli_args.threshold) or DEFAULT_THRESHOLDS
        comparison = compare(
            file_parser(cli_args.base),
            file_parser(cli_args.new),
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
