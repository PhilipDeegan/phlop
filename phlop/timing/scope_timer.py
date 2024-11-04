#
# parsing PHARE scope funtion timers
#

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RunTimerNode:
    k: int
    s: int
    t: int
    c: list = field(default_factory=lambda: [])

    @staticmethod
    def from_scope_timer(line):
        key, start_and_run_time = line.split(" ")
        start_time, run_time = start_and_run_time.split(":")
        return RunTimerNode(*[int(e) for e in (key, start_time, run_time)])

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


def file_parser(times_filepath):
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

    with open(times_filepath, "r") as file:
        while line := file.readline():
            line = line.rstrip()
            if not line:
                break
            bits = line.split(" ")
            id_keys[int(bits[0])] = bits[1]
        while line := file.readline():
            line = line.rstrip()  # drop new line characters
            stripped_line = line.strip()
            if not stripped_line:  # last line might be blank
                continue
            idx = len(line) - len(stripped_line)  # how many space indents from left
            node = RunTimerNode.from_scope_timer(stripped_line)
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


def write_scope_timings(scope_timer_file, outfile, sort_worst_first=True):
    from contextlib import redirect_stdout

    with open(outfile, "w") as f:
        with redirect_stdout(f):
            print_scope_timings(scope_timer_file, sort_worst_first)


def print_scope_timings(scope_timer_file, sort_worst_first=True):
    stf = scope_timer_file  # alias
    if sort_worst_first:
        stf.roots.sort(reverse=True, key=lambda x: x.t)

    def loss(n):
        lss = n.t if n.c else 0
        for c in n.c:
            lss -= c.t

        return "-" if lss < 1e-2 else float(f"{lss / n.t * 100:.2f}")

    def kinder(tot, n, tabs, rem):
        if not n.c:
            return
        o = " " * tabs
        for i in range(len(n.c)):
            c = n.c[i]
            pc = c.t / tot * 100.0
            print(o, f"{pc:.2f}% loss({loss(c)})", stf(c.k), c.t)
            kinder(tot, c, tabs + 1, c.t)

    for root in stf.roots:
        total = root.t
        print(f"100% loss({loss(root)})", stf(root.k), root.t)
        kinder(total, root, 0, total)


def write_variance_across(scope_timer_file_glob, outfile):
    from contextlib import redirect_stdout

    with open(outfile, "w") as f:
        with redirect_stdout(f):
            print_variance_across(scope_timer_file_glob)


def print_variance_across(scope_timer_file_glob, root_id=None):
    scope_timer_files = [file_parser(f) for f in Path.cwd().glob(scope_timer_file_glob)]
    if not scope_timer_files:
        return

    stacks = [[] for _ in range(len(scope_timer_files))]

    def map_graph(n, k, t=0):
        stacks[k].append((n.k, t, n.s, n.t))
        if not n.c:
            return
        for i in range(len(n.c)):
            map_graph(n.c[i], k, t + 1)

    for i, stf in enumerate(scope_timer_files):
        for root in stf.roots:
            if root_id and root_id != root.k:
                continue
            map_graph(root, i)

    numerics = []
    for data in stacks[0]:
        numerics.append((data, [], []))

    for stack in stacks:
        for i, data in enumerate(stack):
            numerics[i][1].append(data[2])
            numerics[i][2].append(data[3])

    stf = scope_timer_files[0]
    for num in numerics:
        data, starts, times = num
        key, tabs, _, __ = data
        o = " " * tabs
        print(o, stf(key), int(np.std(starts)), int(np.std(times)))
