#
# parsing PHARE scope funtion timers
#

from dataclasses import dataclass, field


@dataclass
class RunTimerNode:
    k: int
    t: int
    c: list = field(default_factory=lambda: [])


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
            bits = stripped_line.split(" ")
            node = RunTimerNode(*[int(b) for b in bits])
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


def print_scope_timings(scope_timer_file, percentages=True, sort_worst_first=True):
    """:|"""
    stf = scope_timer_file  # alias
    if sort_worst_first:
        stf.roots.sort(reverse=True, key=lambda x: x.t)

    def kinder(tot, n, tabs=0):
        o = " " * tabs
        for i in range(len(n.c)):
            c = n.c[i]
            pc = c.t / tot * 100.0
            print(o, f"{pc:.2f}%", stf(c.k), c.t)
            kinder(tot, c, tabs + 1)

    for root in stf.roots:
        total = root.t
        print("100%", stf(root.k), root.t)
        kinder(total, root)
