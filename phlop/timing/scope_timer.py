#
# parsing PHARE scope funtion timers
#

import os
import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


TERM_COLORS = bool(json.loads(os.environ.get("PHLOP_TERM_COLORS", "true")))


@dataclass
class RunTimerNode:
    k: int
    s: int
    t: int
    c: list = field(default_factory=lambda: [])

    @staticmethod
    def from_scope_timer(line):
        ws, key, start_and_run_time = line.split(" ")
        start_time, run_time = start_and_run_time.split(":")
        return int(ws), RunTimerNode(*[int(e) for e in (key, start_time, run_time)])

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
class ScopeTimerFile:
    id_keys: dict
    roots: list

    def fn_for(self, id):
        return self.id_keys[id]

    def __call__(self, id):
        return self.fn_for(id)


def lines_parser(lines):
    id_keys = {}
    roots = []
    curr = None
    white_space = 0
    stack = {i: 0 for i in range(128)}
    stack_size = 0

    def _parent():
        assert stack_size >= 0
        n = roots[stack[0]]
        for s in range(1, stack_size):
            assert stack[s] >= 0
            n = n.c[stack[s]]
        return n

    for idx, line in enumerate(lines):
        line = line.rstrip()
        if not line:
            break
        bits = line.split(" ")
        id_keys[int(bits[0])] = bits[1]

    for i in range(idx, len(lines)):
        line = lines[i].rstrip()  # drop new line characters
        stripped_line = line.strip()
        if not stripped_line:  # last line might be blank
            continue
        idx, node = RunTimerNode.from_scope_timer(stripped_line)
        if idx == 0:  # is root node
            stack[0] = len(roots)
            roots.append(node)
            stack_size = 0
            white_space = 0
        else:  # is not root node
            if idx > white_space:
                parent = curr
                curr.c.append(node)
                stack_size += 1
            elif idx == white_space:
                parent = _parent()
                parent.c.append(node)
            elif idx < white_space:
                stack_size -= white_space - idx
                parent = _parent()
                parent.c.append(node)
            stack[stack_size] = len(parent.c) - 1
            white_space = idx
        curr = node
    return ScopeTimerFile(id_keys, roots)


def file_parser(times_filepath):
    with open(times_filepath, "r") as file:
        return lines_parser(file.readlines())


def write_scope_timings(scope_timer_file, outfile, sort_worst_first=True):
    from contextlib import redirect_stdout

    with open(outfile, "w") as f:
        with redirect_stdout(f):
            print_scope_timings(scope_timer_file, sort_worst_first)


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
    leafs = dict()
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
    scope_timer_file_glob,
    sort_worst_first=True,
    root_id=None,
    dedupe=False,
    printer=print,
):
    scope_timer_files = [file_parser(f) for f in Path.cwd().glob(scope_timer_file_glob)]
    if not scope_timer_files:
        return

    stf = scope_timer_files[0]
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
    scope_timer_file_glob,
    sort_worst_first=False,
    root_id="",
    pretty_print=True,
    dedupe=True,
):
    if not pretty_print:
        print_basic_scope_timings(
            scope_timer_file_glob, sort_worst_first, root_id, dedupe
        )
        return

    lines = []

    def printer(args):
        lines.append(args)

    print_basic_scope_timings(
        scope_timer_file_glob, sort_worst_first, root_id, dedupe, printer
    )
    if not lines:
        return

    line_strs = [str(" ".join(str(line).split(" ")[:-1])) for line in lines]
    line_max = max([len(line_str) for line_str in line_strs])

    for i, line in enumerate(line_strs):
        padding = line_max - len(line)
        info = lines[i]
        print(f"{line}" + (" " * padding) + f"{info.n.t/1e6:>10,.2f}ms")


def write_variance_across(scope_timer_file_glob, outfile):
    from contextlib import redirect_stdout

    with open(outfile, "w") as f:
        with redirect_stdout(f):
            print_variance_across(scope_timer_file_glob)


def print_variance_across(scope_timer_file_glob, root_id=None, dedupe=True):
    scope_timer_files = [file_parser(f) for f in Path.cwd().glob(scope_timer_file_glob)]
    if not scope_timer_files:
        return

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
