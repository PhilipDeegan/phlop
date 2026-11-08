# phlop/timing/scope_timer.py

# parsing PHARE scope funtion timers

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from phlop.timing.severity import OK, TERM_COLORS, WARN, Colors, level  # noqa: F401

# "io:" token values, the scope's exit - entry of /proc/thread-self/io
IO_FIELDS = (
    "rchar",  # bytes through read like syscalls, page cache included
    "wchar",  # bytes through write like syscalls
    "syscr",  # read like syscalls
    "syscw",  # write like syscalls
    "read_bytes",  # bytes fetched from the block layer, local filesystems
    "write_bytes",  # bytes sent to the block layer, local filesystems
    "cancelled_write_bytes",  # dirty page cache truncated before writeback
)


@dataclass
class RunTimerNode:
    k: int
    s: int
    t: int
    c: list = field(default_factory=list)
    level: int = 0  # PHLOP_SCOPE perf level: 1 perf_light, 2 perf_all, 0 none
    counts: list | None = None  # values in ScopeTimerFile.perf[level] order
    # memory, bytes, None unless the scope used the mem flag - PHLOP_SCOPE("key", time, mem)
    r: int | None = None  # resident set size at scope entry
    dr: int | None = None  # resident set size change, exit - entry
    p: int | None = None  # process peak resident set size at scope exit
    dp: int | None = None  # process peak growth during scope, exit - entry
    # the thread's I/O over the scope, None unless PHLOP_SCOPE("key", time, io):
    # {name: exit - entry} for IO_FIELDS, see phlop/os/io.hpp
    io: dict | None = None

    @staticmethod
    def from_scope_timer(line):
        ws, key, start_and_run_time, *extra = line.split(" ")
        start_time, run_time = start_and_run_time.split(":")
        node = RunTimerNode(*[int(e) for e in (key, start_time, run_time)])
        for token in extra:
            tag, _, values = token.partition(":")
            if tag == "io":  # "io:<rchar>,<wchar>,...", IO_FIELDS order
                node.io = dict(zip(IO_FIELDS, (int(v) for v in values.split(","))))
            elif tag == "m":  # "m:<rss_in>,<rss_out>,<peak_in>,<peak_out>"
                rss_in, rss_out, peak_in, peak_out = (int(v) for v in values.split(","))
                node.r, node.dr = rss_in, rss_out - rss_in
                node.p, node.dp = peak_out, peak_out - peak_in
            else:  # "<level>:<c0>,<c1>,..."
                node.level = int(tag)
                node.counts = [int(c) for c in values.split(",")]
        return int(ws), node

    @property
    def has_mem(self) -> bool:
        return self.dr is not None

    @property
    def has_io(self) -> bool:
        return self.io is not None

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
    def kernel(self):
        """counts include kernel space, --env ... --kernel"""
        return self.label.endswith("+kernel")

    @property
    def set_name(self):
        return self.label.split("/")[-1].removesuffix("+kernel")


@dataclass
class ScopeTimerFile:
    id_keys: dict
    roots: list
    perf: dict = field(default_factory=dict)  # level -> PerfLevel
    ctx: dict | None = None  # measuring host's "#ctx", eg {"slots": 5}
    # "#clock": clock values at points every process passed together, eg after an
    #  MPI barrier, phlop::mpi::sync_clock(), and whether the clock is global
    clock_syncs: list = field(default_factory=list)
    clock_global: bool | None = None

    def fn_for(self, id):
        return self.id_keys[id]

    def __call__(self, id):
        return self.fn_for(id)

    def perf_warnings(self):
        """why the counts may be inaccurate: generic events, forced PMU"""
        from phlop.app.pfm.sets import scope_file_warnings

        pmus = {level.pmu for level in self.perf.values()}
        return scope_file_warnings(pmus, bool((self.ctx or {}).get("pmu_forced")))

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

    def perf_metrics(self, node, own=False, ctx=None, labels=None, hints=None):
        """{metric: value} from the scope's event set formulas (phlop.app.pfm
        .sets). ctx defaults to this host's, eg arm topdown slots"""
        counts = self.perf_self_counts(node) if own else self.perf_counts(node)
        return self.metrics_for(node.level, counts, ctx, labels, hints)

    def metrics_for(self, level, counts, ctx=None, labels=None, hints=None):
        """{metric: value} for {event alias: count} counted at level. labels,
        if a dict, gets {metric: text} for values that are None for a known
        reason, eg "none (no RAM fetches)". hints, if a dict, gets {metric:
        text} for values the event set flags, eg "low IPC: see -s topdown" """
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
            if m.hint and hints is not None:
                try:  # hints see ctx without the "@", as in pfm.stat.evaluate
                    if hint := m.hint(metrics[m.name], counts, ctx):
                        hints[m.name] = hint
                except (TypeError, KeyError, ZeroDivisionError):
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
    clock_syncs, clock_global = [], None

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
            ctx = {}
            for kv in (bits[1] if len(bits) > 1 else "").split(","):
                key, _, value = kv.partition("=")
                try:  # key=integer only, others are skipped, not fatal
                    if key:
                        ctx[key] = int(value)
                except ValueError:
                    pass
            continue
        if bits[0] == "#clock":  # "#clock sync=<t>,... global=<0|1>"
            for kv in bits[1:]:
                key, _, value = kv.partition("=")
                if key == "sync" and value:
                    clock_syncs = [int(v) for v in value.split(",")]
                elif key == "global" and value in ("0", "1"):
                    clock_global = value == "1"
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
    return ScopeTimerFile(id_keys, roots, perf, ctx, clock_syncs, clock_global)


def file_parser(times_filepath):
    with open(times_filepath, "r") as file:
        return lines_parser(file.readlines())


def write_scope_timings(scope_timer_filepath, outfile, sort_worst_first=True):
    from contextlib import redirect_stdout

    with open(outfile, "w") as f, redirect_stdout(f):
        print_scope_timings(scope_timer_filepath, sort_worst_first)


def format_bytes(b, sign=False):
    v = float(abs(b))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if v < 1024 or unit == "GiB":
            break
        v /= 1024
    s = "-" if b < 0 else ("+" if sign else "")
    return f"{s}{v:.0f}{unit}" if unit == "B" else f"{s}{v:.2f}{unit}"


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

    def mem_str(self):
        if not self.n.has_mem:
            return ""
        return f"  rss {format_bytes(self.n.dr, sign=True)} peak {format_bytes(self.n.dp, sign=True)}"

    def io_str(self):
        return io_str(self.n.io) if self.n.has_io else ""

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


def _merge_mem(base, other):
    """fold other's memory into base: rss deltas sum, peaks max. Leaves
    without memory (eg memory off for that call) neither add nor erase it"""
    if not other.has_mem:
        return
    if not base.has_mem:
        base.r, base.dr, base.p, base.dp = other.r, other.dr, other.p, other.dp
        return
    base.dr += other.dr
    base.dp = max(base.dp, other.dp)
    base.p = max(base.p, other.p)


def _merge_counts(base, other):
    """fold other's perf counts into base, summed. Mixed levels or a member
    without counts leave none: partial sums would read as the whole"""
    if base.counts is None:
        return
    if other.counts is None or other.level != base.level:
        base.counts = None
        return
    base.counts = [a + b for a, b in zip(base.counts, other.counts)]


def io_str(io):
    """ "  write 8.00MiB/1 read 4.00KiB/2 disk w 8.00MiB" for an io dict"""
    text = f"  write {format_bytes(io['wchar'])}/{io['syscw']}"
    text += f" read {format_bytes(io['rchar'])}/{io['syscr']}"
    if io["write_bytes"] or io["read_bytes"]:
        text += f" disk w {format_bytes(io['write_bytes'])} r {format_bytes(io['read_bytes'])}"
    return text


def _merge_io(base, other):
    """fold other's io into base, summed. Members without io neither add nor
    erase it, as for memory"""
    if not other.has_io:
        return
    if not base.has_io:
        base.io = dict(other.io)
        return
    base.io = {k: v + other.io[k] for k, v in base.io.items()}


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
            _merge_mem(base, vals[i])
            _merge_counts(base, vals[i])
            _merge_io(base, vals[i])
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
        print(
            f"{line}"
            + (" " * padding)
            + f"{info.n.t/1e6:>10,.2f}ms"
            + info.mem_str()
            + info.io_str()
        )


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
    mem: dict | None = None  # MEM_VALUES over members with memory, see group_mem


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
        g.mem = group_mem(g.mem, n)
    for key, g in groups.items():
        g.c = group_scopes(stf, [c for m in members[key] for c in m.c], own)
    return list(groups.values())


# a scope's memory as values: peak, the process high water mark at exit (or
# the scope's own with PHLOP_SCOPE_MEM=peak), spike, the peak's rise during
# the scope, kept, resident growth exit - entry
MEM_VALUES = ("mem_peak", "mem_spike", "mem_kept")


def group_mem(mem, node):
    """mem with node's folded in, as _merge_mem: peaks max, kept summed"""
    if not node.has_mem:
        return mem
    if mem is None:
        return dict(zip(MEM_VALUES, (node.p, node.dp, node.dr)))
    return {
        "mem_peak": max(mem["mem_peak"], node.p),
        "mem_spike": max(mem["mem_spike"], node.dp),
        "mem_kept": mem["mem_kept"] + node.dr,
    }


def print_scope_perf(
    scope_timer_filepath, own=False, dedupe=True, verbose=False, printer=print
):
    """scope tree with time and, for counted scopes, their event set metrics.
    own: the scope's own work, children counted at its level subtracted.
    dedupe: same named siblings on one line, xN with total and mean time,
    metrics from their summed counts. Metrics the event set flags, eg low
    IPC, are colored. verbose: also the raw event counts and the flags' hints"""
    stf = file_parser(scope_timer_filepath)
    colors = Colors()
    flagged = [False]
    for w in stf.perf_warnings():
        printer(f"WARNING: {w}")

    def fmt(v, label=None):
        if v is None:
            return label or "n/a"
        return f"{v:,.0f}" if abs(v) >= 1e4 else f"{v:.3g}"

    def perf_text(level, counts, depth):
        labels, hints = {}, {}
        metrics = stf.metrics_for(level, counts, labels=labels, hints=hints)
        flagged[0] = flagged[0] or bool(hints)
        perf_level = stf.perf[level]
        kernel = " +kernel" if perf_level.kernel else ""
        text = f" [{perf_level.set_name}{kernel}] " + " ".join(
            colors(f"{k}={fmt(v, labels.get(k))}", WARN if k in hints else OK)
            for k, v in metrics.items()
        )
        if verbose:
            text += f"\n{'  ' * depth}  counts: " + " ".join(
                f"{k}={v:,}" for k, v in counts.items()
            )
            for k, hint in hints.items():
                text += f"\n{'  ' * depth}  {k}: {hint}"
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
    if flagged[0]:
        why = "" if verbose else ", -v for why"
        printer(colors("flagged", WARN) + f": metrics the event set flags{why}")


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


def _spread(vals):
    """(max - min) / median, relative so long and short scopes compare"""
    median = float(np.median(vals))
    return (max(vals) - min(vals)) / median if median else 0.0


def _load_across(scope_timer_file_glob, printer):
    """(file names, ScopeTimerFiles made consistent), None if under 2 files"""
    files = sorted(glob.glob(scope_timer_file_glob))
    if len(files) < 2:
        if files:
            printer(f"1 file matches {scope_timer_file_glob}, need 2 or more")
        return None
    return files, make_consistent([file_parser(f) for f in files])


def _aligned_scopes(stfs, root_id=None, dedupe=True):
    """[(key, depth, [node per file])], the consistent trees walked in step"""
    by_index = {}  # preorder position -> row

    def walk(stf, n, i, depth):
        by_index.setdefault(i[0], (n.k, depth, []))[2].append(n)
        i[0] += 1
        if dedupe and n.c:
            n.c = dedupe_leafs(stf, n)
        for c in n.c:
            walk(stf, c, i, depth + 1)

    for stf in stfs:
        i = [0]
        for root in stf.roots:
            if not root_id or stf(root.k).startswith(root_id):
                walk(stf, root, i, 0)
    rows = list(by_index.values())
    if any(len(nodes) != len(stfs) for _, _, nodes in rows):
        raise RuntimeError("Incompatible scope timer files, scope trees differ")
    return rows


def align_clocks(stfs):
    """Rewrites every start as time since the files' common clock sync point,
    from their "#clock sync=" values, phlop::mpi::sync_clock(). Returns False,
    changing nothing, if no file has any. All or nothing: sync_clock() is
    collective, so files with and without syncs, or with different numbers of
    them, aren't one consistent run and raise RuntimeError.

    A global clock (MPI_WTIME_IS_GLOBAL in every file) is shifted by one common
    origin, the earliest first sync. Otherwise each file's first sync is its
    own origin, and with two or more syncs its clock rate is scaled to the
    median sync span: a two point drift correction. Durations are unchanged,
    drift is ppm"""
    counts = [len(stf.clock_syncs) for stf in stfs]
    if not any(counts):
        return False
    if len(set(counts)) > 1:
        raise RuntimeError(
            f"Incompatible scope timer files, clock sync counts differ: {counts}"
        )
    spans = [stf.clock_syncs[-1] - stf.clock_syncs[0] for stf in stfs]
    common = all(stf.clock_global for stf in stfs)
    origin = min(stf.clock_syncs[0] for stf in stfs)
    ref = float(np.median(spans))
    for stf, span in zip(stfs, spans):
        t0, scale = (origin, 1.0) if common else (stf.clock_syncs[0], 1.0)
        if not common and span > 0 and ref > 0:
            scale = ref / span
        for n in _walk(stf.roots):
            n.s = round((n.s - t0) * scale)
    return True


def _pct(v):
    return f"{v * 100:.1f}%"


def _ms(v):
    return f"{v / 1e6:,.3f}ms"


def print_variance_across(
    scope_timer_file_glob,
    root_id=None,
    dedupe=True,
    warn=0.05,
    bad=0.2,
    min_ns=10_000,
    printer=print,
):
    """Per scope, how much runs or MPI ranks differ. Runtime spread is
    (max - min) / median across the files, colored at warn and bad. Scopes
    under min_ns median runtime are not colored, too short to time reliably.

    Starts are relative to each file's first root, as steady clock values
    differ between runs and hosts, or, if every file has "#clock" syncs
    (phlop::mpi::sync_clock()), to the common sync point, see align_clocks.
    Their spread is colored against the median run length.

    With the same perf counts in every file, the runtime spread splits in two:
    work, the spread of instructions, and speed, the spread of cycles per
    instruction. Ranks doing different work is load imbalance, the same work
    at different speed is the machine: memory contention, noisy neighbours,
    frequency. See print_mem_variance_across for memory"""
    loaded = _load_across(scope_timer_file_glob, printer)
    if not loaded:
        return
    files, stfs = loaded
    aligned = align_clocks(stfs)  # before dedupe averages starts
    rows = _aligned_scopes(stfs, root_id, dedupe)
    if not rows:
        return
    colors = Colors()
    with_perf = all(stf.perf == stfs[0].perf for stf in stfs)
    t0s = [0 if aligned else min(r.s for r in stf.roots) for stf in stfs]
    run_length = float(
        np.median(
            [max(r.s + r.t for r in stf.roots) - t0 for stf, t0 in zip(stfs, t0s)]
        )
    )

    printer(
        f"{len(files)} files, spread = (max - min) / median, starts from "
        + ("the clock sync. " if aligned else "each file's first scope. ")
        + colors.legend(f">= {_pct(warn)}", f">= {_pct(bad)}")
    )
    for key, depth, nodes in rows:
        times = [n.t for n in nodes]
        starts = [n.s - t0 for n, t0 in zip(nodes, t0s)]
        median = float(np.median(times))
        spread = _spread(times)
        timed = median >= min_ns
        line = f"{'  ' * depth}{stfs[0](key)} {_ms(median)} spread "
        line += colors(_pct(spread), level(spread, warn, bad) if timed else OK)
        start_spread = max(starts) - min(starts)
        line += f", start +{_ms(float(np.median(starts)))} spread "
        rel = start_spread / run_length if run_length else 0.0
        line += colors(_ms(start_spread), level(rel, warn, bad))
        counts = [
            stf.perf_counts(n) if with_perf else {} for stf, n in zip(stfs, nodes)
        ]
        if all(c.get("instructions") and c.get("cycles") for c in counts):
            work = _spread([c["instructions"] for c in counts])
            speed = _spread([c["cycles"] / c["instructions"] for c in counts])
            line += ", work " + colors(
                _pct(work), level(work, warn, bad) if timed else OK
            )
            line += ", speed " + colors(
                _pct(speed), level(speed, warn, bad) if timed else OK
            )
        printer(line)


def _per_scope_peaks(stf):
    """True if the file's mem scopes reset the peak on entry, PHLOP_SCOPE_MEM
    =peak: their peak on entry is then their rss on entry. Undeduped nodes,
    merging sums and maxes the fields"""
    nodes = [n for n in _walk(stf.roots) if n.has_mem]
    return bool(nodes) and all(n.p - n.dp == n.r for n in nodes)


def _walk(nodes):
    for n in nodes:
        yield n
        yield from _walk(n.c)


def print_mem_variance_across(
    scope_timer_file_glob,
    root_id=None,
    dedupe=True,
    warn=0.2,
    bad=0.5,
    printer=print,
):
    """Per mem scope, PHLOP_SCOPE("key", time, mem), how memory differs
    between runs or MPI ranks. The rank that runs out of memory is the
    highest one, not the average, so each value names the file with the max.

    peak: with PHLOP_SCOPE_MEM=peak the scope's own peak, else the process
    high water mark at scope exit. Its imbalance, max / median - 1, is
    colored at warn and bad.
    spike: the most the peak rose above the scope's entry, colored as a share
    of the median peak. Without PHLOP_SCOPE_MEM=peak only scopes setting a new
    process high show one, the header says which the files hold.
    kept: resident growth over the scope, exit - entry, median and max."""
    loaded = _load_across(scope_timer_file_glob, printer)
    if not loaded:
        return
    files, stfs = loaded
    per_scope = all(_per_scope_peaks(stf) for stf in stfs)  # before dedupe merges
    rows = _aligned_scopes(stfs, root_id, dedupe)
    colors = Colors()
    names = [os.path.basename(f) for f in files]

    def worst(values):
        i = int(np.argmax(values))
        return values[i], names[i]

    # mem scopes and their ancestors, preorder rows walked backwards: a scope
    # is shown if it has memory or a scope shown below it, before its sibling
    has_mem = [all(n.has_mem for n in nodes) for _, _, nodes in rows]
    show, below = [False] * len(rows), {}  # depth -> anything shown there
    for i in reversed(range(len(rows))):
        depth = rows[i][1]
        show[i] = has_mem[i] or any(v for d, v in below.items() if d > depth)
        below = {d: v for d, v in below.items() if d < depth}
        below[depth] = below.get(depth, False) or show[i]
    if not any(has_mem):
        printer('no mem scopes in these files, see PHLOP_SCOPE("key", time, mem)')
        return

    printer(
        f"{len(files)} files, peak imbalance = max / median - 1. "
        + colors.legend(f">= {_pct(warn)}", f">= {_pct(bad)}")
    )
    printer(
        "peaks per scope, PHLOP_SCOPE_MEM=peak"
        if per_scope
        else "peaks are the process high water mark: spikes only where a scope"
        " sets a new high, PHLOP_SCOPE_MEM=peak for every scope's"
    )
    for (key, depth, nodes), shown, mem in zip(rows, show, has_mem):
        line = f"{'  ' * depth}{stfs[0](key)}"
        if not shown:
            continue
        if not mem:
            printer(line)  # an ancestor of mem scopes, for the tree's shape
            continue
        peak = float(np.median([n.p for n in nodes]))
        top, top_file = worst([n.p for n in nodes])
        imbalance = top / peak - 1 if peak else 0.0
        line += f" peak {format_bytes(peak)} max {format_bytes(top)} "
        line += colors(f"+{_pct(imbalance)}", level(imbalance, warn, bad))
        line += f" {top_file}"
        spike, spike_file = worst([n.dp for n in nodes])
        if spike > 0:
            share = spike / peak if peak else 0.0
            line += ", spike " + colors(format_bytes(spike), level(share, warn, bad))
            line += f" {spike_file}"
        kept = [n.dr for n in nodes]
        line += f", kept {format_bytes(float(np.median(kept)), sign=True)}"
        line += f" max {format_bytes(max(kept), sign=True)}"
        printer(line)


def cli_args_parser(description="scope timer files"):
    import argparse

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "tool",
        choices=["perf", "times", "variance", "memvariance"],
        help="what to print. variance, memvariance: time and memory across files,"
        " eg runs or MPI ranks",
    )
    parser.add_argument(
        "file",
        help="scope timer file, for variance and memvariance a quoted glob,"
        " eg 'times.*.txt'",
    )
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
    parser.add_argument(
        "--warn",
        type=float,
        default=None,
        help="variance: spread percent colored as a warning, default 5."
        " memvariance: peak imbalance percent, default 20",
    )
    parser.add_argument(
        "--bad",
        type=float,
        default=None,
        help="variance: spread percent colored as bad, default 20."
        " memvariance: peak imbalance percent, default 50",
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
        elif cli_args.tool in ("variance", "memvariance"):
            mem = cli_args.tool == "memvariance"
            warn = cli_args.warn if cli_args.warn is not None else 20 if mem else 5
            bad = cli_args.bad if cli_args.bad is not None else 50 if mem else 20
            printer = print_mem_variance_across if mem else print_variance_across
            printer(
                cli_args.file, dedupe=cli_args.dedupe, warn=warn / 100, bad=bad / 100
            )
        else:
            print_scope_timings(cli_args.file, dedupe=cli_args.dedupe)
    except BrokenPipeError:  # eg piped into head
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main())
