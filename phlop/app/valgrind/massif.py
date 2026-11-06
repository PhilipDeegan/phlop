# phlop/app/valgrind/massif.py
#
# massif output files: snapshots of heap use over time, and the allocation
# trees (who allocated what is live) at the peak and some others

import re
from dataclasses import dataclass, field

from phlop.app.valgrind.common import Frame, is_std

_NODE = re.compile(r"^( *)n(\d+): (\d+) (.*)$")
_FRAME = re.compile(r"^0x[0-9A-Fa-f]+: (.*?)(?: \((?:in (.+)|(.+):(\d+))\))?$")
_BELOW = re.compile(
    r"in (\d+) places?, (?:all )?below massif's threshold \(([\d.]+)%\)"
)


@dataclass
class Node:
    bytes: int
    frame: Frame | None = None  # None: the root, or the below threshold entry
    below: int = 0  # places folded into this entry by massif's threshold
    children: list = field(default_factory=list)  # callers


@dataclass
class Snapshot:
    time: float
    heap: int
    extra: int  # allocator overhead (headers, alignment)
    stacks: int  # only with --stacks=yes
    kind: str = "empty"  # empty, detailed or peak
    tree: Node | None = None

    @property
    def total(self):
        return self.heap + self.extra + self.stacks


@dataclass
class Massif:
    path: str
    cmd: str = ""
    time_unit: str = "i"
    threshold: float = 1.0  # %, sites below it are folded
    snapshots: list = field(default_factory=list)

    @property
    def peak(self):
        """the snapshot with the most memory"""
        return max(self.snapshots, key=lambda s: s.total)

    @property
    def peak_tree(self):
        """the tree at massif's peak snapshot, else the largest detailed one"""
        trees = [s for s in self.snapshots if s.tree]
        peak = [s for s in trees if s.kind == "peak"]
        if peak:
            return peak[0]
        return max(trees, key=lambda s: s.total) if trees else None

    def frames(self):
        """every function in any tree, for comparisons"""
        out = set()

        def walk(n):
            if n.frame:
                out.add(n.frame.fn)
            for c in n.children:
                walk(c)

        for s in self.snapshots:
            if s.tree:
                walk(s.tree)
        return out


def _node(rest, nbytes):
    if rest.startswith("("):  # root: (heap allocation functions) malloc/new/...
        return Node(nbytes)
    if m := _BELOW.match(rest):
        return Node(nbytes, below=int(m.group(1)))
    if m := _FRAME.match(rest):
        fn, obj, file, line = m.groups()
        return Node(nbytes, Frame(fn, file or "", int(line or 0), obj or ""))
    return Node(nbytes, Frame(rest))


def parse(path):
    with open(path) as f:
        return parse_lines(f, path)


def parse_lines(lines, path=""):
    out = Massif(path)
    snap = None
    stack = []  # (depth, node) of the open tree
    for line in lines:
        line = line.rstrip("\n")
        if m := _NODE.match(line):
            depth, nbytes, rest = len(m.group(1)), int(m.group(3)), m.group(4)
            node = _node(rest, nbytes)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            if stack:
                stack[-1][1].children.append(node)
            else:
                snap.tree = node
            stack.append((depth, node))
            continue
        stack = []
        key, _, value = line.partition("=") if "=" in line else line.partition(": ")
        if line.startswith("desc: "):
            if m := re.search(r"--threshold=([\d.]+)", line):
                out.threshold = float(m.group(1))
        elif line.startswith("cmd: "):
            out.cmd = line[5:]
        elif line.startswith("time_unit: "):
            out.time_unit = line[11:]
        elif key == "snapshot":
            snap = Snapshot(0, 0, 0, 0)
            out.snapshots.append(snap)
        elif key == "time":
            snap.time = float(value)
        elif key == "mem_heap_B":
            snap.heap = int(value)
        elif key == "mem_heap_extra_B":
            snap.extra = int(value)
        elif key == "mem_stacks_B":
            snap.stacks = int(value)
        elif key == "heap_tree":
            snap.kind = value
    return out


@dataclass
class Site:
    """an allocation site: the first frame outside the C/C++ runtime, and
    what called it. None frame: massif's below threshold entries"""

    bytes: int
    frame: Frame | None
    callers: list = field(default_factory=list)  # Node
    below: int = 0


def sites(tree):
    """allocation sites in a tree, largest first, runtime frames collapsed:
    a vector's allocate/_M_allocate/... are skipped to the code using it"""
    found = {}
    below = Site(0, None)

    def walk(node):
        for c in node.children:
            if c.frame is None:
                below.bytes += c.bytes
                below.below += c.below
            elif is_std(c.frame) and c.children:
                walk(c)
            else:
                site = found.setdefault(c.frame, Site(0, c.frame))
                site.bytes += c.bytes
                site.callers += c.children

    if tree:
        walk(tree)
    out = sorted(found.values(), key=lambda s: -s.bytes)
    for s in out:  # merge callers reached through different runtime paths
        merged = {}
        for c in s.callers:
            if c.frame in merged:
                merged[c.frame].bytes += c.bytes
            else:
                merged[c.frame] = Node(c.bytes, c.frame, c.below, list(c.children))
        s.callers = sorted(merged.values(), key=lambda n: -n.bytes)
    if below.bytes:
        out.append(below)
    return out


def chain(node, depth=8):
    """the largest call path above a node: its caller, their caller..."""
    out = []
    while node.children and len(out) < depth:
        node = max(node.children, key=lambda n: n.bytes)
        if node.frame is None:
            break
        out.append(node.frame)
    return out
