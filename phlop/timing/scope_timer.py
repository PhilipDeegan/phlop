# phlop/timing/scope_timer.py

# parsing PHARE scope funtion timers

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TERM_COLORS = bool(json.loads(os.environ.get("PHLOP_TERM_COLORS", "true")))


@dataclass
class RunTimerNode:
    k: int
    s: int
    t: int
    c: list = field(default_factory=list)
    level: int = 0  # PHLOP_SCOPE perf level: 1 perf_light, 2 perf_all, 0 none
    counts: list | None = None  # values in ScopeTimerFile.perf[level] order

    @staticmethod
    def from_scope_timer(line):
        ws, key, start_and_run_time, *perf = line.split(" ")
        start_time, run_time = start_and_run_time.split(":")
        node = RunTimerNode(*[int(e) for e in (key, start_time, run_time)])
        if perf:  # "<level>:<c0>,<c1>,..."
            level, _, counts = perf[0].partition(":")
            node.level, node.counts = int(level), [int(c) for c in counts.split(",")]
        return int(ws), node

    @property
    def key(self) -> int:
        return self.k

    @property
    def start_time(self) -> int:
        return self.s

    @property
    def run_time(self) -> int:
        return self.t


@dataclass
class PerfLevel:
    label: str  # "<pmu>/<event set>", eg "amd64_fam1ah_zen5/light"
    names: list  # event aliases, as in phlop.app.pfm.sets

    @property
    def pmu(self):
        return self.label.split("/")[0]

    @property
    def set_name(self):
        return self.label.split("/")[-1]


@dataclass
class ScopeTimerFile:
    id_keys: dict
    roots: list
    perf: dict = field(default_factory=dict)  # level -> PerfLevel
    ctx: dict | None = None  # measuring host's "#ctx", eg {"slots": 5}

    def fn_for(self, id):
        return self.id_keys[id]

    def __call__(self, id):
        return self.fn_for(id)

    def perf_counts(self, node):
        """{event alias: count} for a scope, inclusive of its children"""
        if not node.counts or node.level not in self.perf:
            return {}
        return dict(zip(self.perf[node.level].names, node.counts))

    def perf_self_counts(self, node):
        """perf_counts minus children counted at the same level, the scope's
        own work. Children at other levels or uncounted are not subtracted"""
        counts = self.perf_counts(node)
        for child in node.c:
            if child.level == node.level:
                for name, value in self.perf_counts(child).items():
                    counts[name] -= value
        return counts

    def perf_metrics(self, node, own=False, ctx=None, labels=None):
        """{metric: value} from the scope's event set formulas (phlop.app.pfm
        .sets). ctx defaults to this host's, eg arm topdown slots"""
        counts = self.perf_self_counts(node) if own else self.perf_counts(node)
        return self.metrics_for(node.level, counts, ctx, labels)

    def metrics_for(self, level, counts, ctx=None, labels=None):
        """{metric: value} for {event alias: count} counted at level. labels,
        if a dict, gets {metric: text} for values that are None for a known
        reason, eg "none (no RAM fetches)" """
        if not counts or level not in self.perf:
            return {}
        from phlop.app.pfm.sets import SETS, generic

        perf_level = self.perf[level]
        sets = {s.name: s for s in SETS.get(perf_level.pmu, generic)}
        if perf_level.set_name not in sets:
            return {}
        if ctx is None and self.ctx is not None:
            ctx = self.ctx  # the measuring host's, from the file
        if ctx is None:  # older files: this host's, may differ from the measuring one
            from phlop.app.pfm import host

            ctx = host.context()
        counts = {**counts, **{f"@{k}": v for k, v in ctx.items()}}
        metrics = {}
        for m in sets[perf_level.set_name].metrics:
            try:
                metrics[m.name] = m.fn(counts)
            except (TypeError, ZeroDivisionError, KeyError):
                metrics[m.name] = None
            if metrics[m.name] is None and m.none_means and labels is not None:
                try:
                    if label := m.none_means(counts):
                        labels[m.name] = label
                except (TypeError, KeyError):
                    pass
        return metrics


def lines_parser(lines):
    """scope timer text format: "<id> <key>" lines, a blank line, then one
    "<depth> <id> <start>:<time>" line per scope, depth is the nesting level
    under the preceding shallower line"""
    id_keys = {}
    roots = []
    perf = {}
    ctx = None

    idx = 0
    for idx, line in enumerate(lines):
        line = line.rstrip()
        if not line:
            break
        bits = line.split(" ")
        if bits[0] == "#perf":  # "#perf <level> <label> <name,...>"
            perf[int(bits[1])] = PerfLevel(bits[2], bits[3].split(","))
            continue
        if bits[0] == "#ctx":  # "#ctx key=value,..." measuring host context
            ctx = dict(kv.split("=", 1) for kv in bits[1].split(",") if "=" in kv)
            ctx = {k: int(v) for k, v in ctx.items()}
            continue
        id_keys[int(bits[0])] = bits[1]

    path = []  # path[d] is the latest node at depth d, no depth limit
    for line in lines[idx:]:
        stripped_line = line.strip()
        if not stripped_line:  # last line might be blank
            continue
        depth, node = RunTimerNode.from_scope_timer(stripped_line)
        if depth == 0:
            roots.append(node)
        elif depth <= len(path):
            path[depth - 1].c.append(node)
        else:
            raise ValueError(f"scope timer line nested without a parent: {line!r}")
        del path[depth:]
        path.append(node)
    return ScopeTimerFile(id_keys, roots, perf, ctx)


def file_parser(times_filepath):
    with open(times_filepath, "r") as file:
        return lines_parser(file.readlines())


def write_scope_timings(scope_timer_filepath, outfile, sort_worst_first=True):
    from contextlib import redirect_stdout

    with open(outfile, "w") as f, redirect_stdout(f):
        print_scope_timings(scope_timer_filepath, sort_worst_first)


@dataclass
class LossLine:
    padding: int
    tot: int
    key: str
    n: RunTimerNode
    par: RunTimerNode | None

    def __repr__(self):
        if self.par:
            pc = self.n.t / self.tot * 100.0
            return (
                self.padding
                + f"{pc:.2f}% loss({LossLine.loss(self.n)}) {self.key} {self.n.t}"
            )

        else:
            return f"100% loss({LossLine.loss(self.n)}) {self.key} {self.n.t}"

    def root_time(self):
        if self.par:
            return self.par.root_time()
        return self.n.t

    @staticmethod
    def loss(n):
        lss = n.t if n.c else 0
        for c in n.c:
            lss -= c.t
        return "-" if lss < 1e-2 else float(f"{lss / n.t * 100:.2f}")


def dedupe_leafs(stf, n):
    nodes = []
    leafs = {}
    keys = []

    for c in n.c:
        if c.c:
            return n.c  # skip

    for c in n.c:
        key = stf(c.k)
        if key not in leafs:
            leafs[key] = []
            keys.append(key)
        leafs[key].append(c)

    for key in keys:
        vals = leafs[key]
        base = leafs[key][0]
        for i in range(1, len(vals)):
            base.s += vals[i].s
            base.t += vals[i].t
        base.s /= len(vals)
        nodes.append(base)

    return nodes


def print_basic_scope_timings(
    scope_timer_filepath,
    sort_worst_first=True,
    root_id=None,
    dedupe=False,
    printer=print,
):
    """prints one timer file, see print_variance_across for many"""
    stf = file_parser(scope_timer_filepath)
    if sort_worst_first:
        stf.roots.sort(reverse=True, key=lambda x: x.t)

    def kinder(n, tabs, tot):
        if not n.c:
            return
        o = " " * tabs
        if dedupe:
            n.c = dedupe_leafs(stf, n)
        for c in n.c:
            printer(LossLine(o, tot, stf(c.k), c, n))
            kinder(c, tabs + 1, tot)

    for root in stf.roots:
        if root_id and not stf(root.k).startswith(root_id):
            continue
        printer(LossLine(0, root.t, stf(root.k), root, None))
        kinder(root, 1, root.t)


def print_scope_timings(
    scope_timer_filepath,
    sort_worst_first=False,
    root_id="",
    pretty_print=True,
    dedupe=True,
):
    if not pretty_print:
        print_basic_scope_timings(
            scope_timer_filepath, sort_worst_first, root_id, dedupe
        )
        return

    lines = []

    def printer(args):
        lines.append(args)

    print_basic_scope_timings(
        scope_timer_filepath, sort_worst_first, root_id, dedupe, printer
    )
    if not lines:
        return

    line_strs = [str(" ".join(str(line).split(" ")[:-1])) for line in lines]
    line_max = max([len(line_str) for line_str in line_strs])

    for i, line in enumerate(line_strs):
        padding = line_max - len(line)
        info = lines[i]
        print(f"{line}" + (" " * padding) + f"{info.n.t/1e6:>10,.2f}ms")


@dataclass
class ScopeGroup:
    """same named sibling scopes merged: summed time and counts, children
    merged the same way, see group_scopes"""

    key: str
    n: int
    t: int
    level: int = 0
    counts: dict | None = None  # summed, over the n_counted members with counts
    n_counted: int = 0
    c: list = field(default_factory=list)


def group_scopes(stf, nodes, own=False):
    """merge sibling nodes by name, first seen order, recursively"""
    groups, members = {}, {}
    for n in nodes:
        key = stf(n.k)
        if key not in groups:
            groups[key] = ScopeGroup(key, 0, 0)
            members[key] = []
        g = groups[key]
        g.n += 1
        g.t += n.t
        members[key].append(n)
        counts = stf.perf_self_counts(n) if own else stf.perf_counts(n)
        if counts and (g.counts is None or g.level == n.level):
            g.level = n.level
            g.counts = {k: (g.counts or {}).get(k, 0) + v for k, v in counts.items()}
            g.n_counted += 1
    for key, g in groups.items():
        g.c = group_scopes(stf, [c for m in members[key] for c in m.c], own)
    return list(groups.values())


def print_scope_perf(
    scope_timer_filepath, own=False, dedupe=True, verbose=False, printer=print
):
    """scope tree with time and, for counted scopes, their event set metrics.
    own: the scope's own work, children counted at its level subtracted.
    dedupe: same named siblings on one line, xN with total and mean time,
    metrics from their summed counts. verbose: also the raw event counts"""
    stf = file_parser(scope_timer_filepath)

    def fmt(v, label=None):
        if v is None:
            return label or "n/a"
        return f"{v:,.0f}" if abs(v) >= 1e4 else f"{v:.3g}"

    def perf_text(level, counts, depth):
        labels = {}
        metrics = stf.metrics_for(level, counts, labels=labels)
        text = f" [{stf.perf[level].set_name}] " + " ".join(
            f"{k}={fmt(v, labels.get(k))}" for k, v in metrics.items()
        )
        if verbose:
            text += f"\n{'  ' * depth}  counts: " + " ".join(
                f"{k}={v:,}" for k, v in counts.items()
            )
        return text

    def show_node(n, depth):
        line = f"{'  ' * depth}{stf(n.k)} {n.t / 1e6:,.3f}ms"
        counts = stf.perf_self_counts(n) if own else stf.perf_counts(n)
        if counts:
            line += perf_text(n.level, counts, depth)
        printer(line)
        for c in n.c:
            show_node(c, depth + 1)

    def show_group(g, depth):
        line = f"{'  ' * depth}{g.key}"
        if g.n > 1:
            line += f" x{g.n} {g.t / 1e6:,.3f}ms (mean {g.t / g.n / 1e6:,.3f}ms)"
        else:
            line += f" {g.t / 1e6:,.3f}ms"
        if g.counts:
            if g.n_counted != g.n:
                line += f" ({g.n_counted}/{g.n} counted)"
            line += perf_text(g.level, g.counts, depth)
        printer(line)
        for c in g.c:
            show_group(c, depth + 1)

    if dedupe:
        for g in group_scopes(stf, stf.roots, own):
            show_group(g, 0)
    else:
        for root in stf.roots:
            show_node(root, 0)


def write_variance_across(scope_timer_file_glob, outfile):
    from contextlib import redirect_stdout

    with open(outfile, "w") as f, redirect_stdout(f):
        print_variance_across(scope_timer_file_glob)


def are_consistent_trees(nodes):
    ref_node = nodes[0]
    for i in range(1, len(nodes)):
        cmp_node = nodes[i]
        if len(ref_node.c) != len(cmp_node.c):
            return 0
        for j in range(len(ref_node.c)):
            if ref_node.c[j].k != cmp_node.c[j].k:
                return 0
    return 1  # are consistent


def make_consistent(scope_timer_files):
    def traverse(nodes):
        if not are_consistent_trees(nodes):
            for node in nodes:
                node.c = []  # drop!
        for i in range(len(nodes[0].c)):
            traverse([node.c[i] for node in nodes])

    n_roots = len(scope_timer_files[0].roots)
    for i in range(1, len(scope_timer_files)):
        if n_roots != len(scope_timer_files[i].roots):
            raise RuntimeError(
                "Incompatible scope timer files, different root node count"
            )

    for i in range(n_roots):
        traverse([stf.roots[i] for stf in scope_timer_files])

    return scope_timer_files


def print_variance_across(scope_timer_file_glob, root_id=None, dedupe=True):
    scope_timer_files = [file_parser(f) for f in Path.cwd().glob(scope_timer_file_glob)]
    if not scope_timer_files:
        return

    scope_timer_files = make_consistent(scope_timer_files)

    stacks = [[] for _ in range(len(scope_timer_files))]

    def map_graph(stf, n, k, t=0):
        stacks[k].append((n.k, t, n.s, n.t))
        if not n.c:
            return
        if dedupe:
            n.c = dedupe_leafs(stf, n)
        for i in range(len(n.c)):
            map_graph(stf, n.c[i], k, t + 1)

    for i, stf in enumerate(scope_timer_files):
        for root in stf.roots:
            if root_id and not stf(root.k).startswith(root_id):
                continue
            map_graph(stf, root, i)

    if not stacks[0]:  # nothing to do
        return

    numerics = []
    for data in stacks[0]:
        numerics.append((data, [], []))

    for stack in stacks:
        for i, data in enumerate(stack):
            numerics[i][1].append(data[2])
            numerics[i][2].append(data[3])

    def metric(vals, stddev=False):
        if stddev:
            return int(np.std(vals))
        return max(vals) - min(vals)

    savg = 0
    ravg = 0
    for num in numerics:
        data, starts, times = num
        savg += metric(starts)
        ravg += metric(times)
    savg /= len(numerics)
    ravg /= len(numerics)

    colorize = TERM_COLORS and sys.stdout.isatty()
    red = "\033[91m" if colorize else ""
    white = "\033[0m" if colorize else ""
    orange = "\033[93m" if colorize else ""

    def color(v, avg):
        d = str(v / 1e6) + "ms"
        if v >= avg * 2:
            return red + d
        if v > avg:
            return orange + d
        return white + d

    stf = scope_timer_files[0]
    for num in numerics:
        data, starts, times = num
        key, tabs, _, __ = data
        o = " " * tabs
        start_sdev = color(metric(starts), savg)
        times_sdev = color(metric(times), ravg)
        print(
            white
            + o
            + f"{stf(key)} start: {start_sdev}"
            + white
            + f", runtime: {times_sdev}"
        )


def cli_args_parser(description="scope timer files"):
    import argparse

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("tool", choices=["perf", "times"], help="what to print")
    parser.add_argument("file", help="scope timer file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="perf: raw event counts too"
    )
    parser.add_argument(
        "--own",
        action="store_true",
        help="perf: scopes' own work, children counted at their level subtracted",
    )
    parser.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        help="one line per scope, not same named siblings merged",
    )
    return parser


def main(argv=None):
    """python3 -m phlop.timing.scope_timer perf .phlop_scope_times.txt -v"""
    cli_args = cli_args_parser().parse_args(argv)
    try:
        if cli_args.tool == "perf":
            print_scope_perf(
                cli_args.file, cli_args.own, cli_args.dedupe, cli_args.verbose
            )
        else:
            print_scope_timings(cli_args.file, dedupe=cli_args.dedupe)
    except BrokenPipeError:  # eg piped into head
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main())
