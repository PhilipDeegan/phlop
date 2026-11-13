# phlop/app/valgrind/memcheck.py
#
# memcheck --xml=yes files: errors (invalid reads, uninitialised values, ...)
# and leaks, with their stacks

import os
import re
import shlex
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from phlop.app.valgrind.common import Frame, first_user_frame

LEAK_KINDS = {
    "Leak_DefinitelyLost": "definitely lost",
    "Leak_IndirectlyLost": "indirectly lost",
    "Leak_PossiblyLost": "possibly lost",
    "Leak_StillReachable": "still reachable",
}


@dataclass
class Error:
    kind: str
    what: str
    count: int = 1  # times reported, leaks: 1
    bytes: int = 0  # leaks only
    blocks: int = 0
    stack: list = field(default_factory=list)  # Frame, innermost first
    aux: str = ""  # eg "Address 0x.. is 0 bytes after a block of size 40 alloc'd"
    aux_stack: list = field(default_factory=list)  # that block, or --track-origins

    @property
    def leak(self):
        return self.kind.startswith("Leak_")

    @property
    def site(self):
        return first_user_frame(self.stack)


@dataclass
class Memcheck:
    path: str
    cmd: str = ""
    pid: int = 0
    errors: list = field(default_factory=list)
    complete: bool = True  # the xml parsed whole: else cut short or broken

    @property
    def problems(self):
        """not leaks: invalid reads/writes, uninitialised values, bad frees..."""
        return [e for e in self.errors if not e.leak]

    @property
    def leaks(self):
        return [e for e in self.errors if e.leak]

    def leaked(self, kind):
        return sum(e.bytes for e in self.errors if e.kind == kind)


def _frames(stack):
    out = []
    for f in stack.iter("frame") if stack is not None else []:
        file, d = f.findtext("file", ""), f.findtext("dir", "")
        out.append(
            Frame(
                f.findtext("fn", "???"),
                os.path.join(d, file) if file and d else file,  # dir: identity
                int(f.findtext("line", "0")),
                f.findtext("obj", ""),
            )
        )
    return out


# not used here, and valgrind 3.26 writes it without </still_reachable>
_LEAK_SUMMARY = re.compile(r"<leak_summary>.*?</leak_summary>", re.DOTALL)


def _complete_children(text):
    """the top level elements that were closed: the file can be cut short (the
    process died) or broken further on, everything before that is kept"""
    parser = ET.XMLPullParser(events=("start", "end"))
    depth, kept = 0, []
    try:
        parser.feed(text)
        for event, el in parser.read_events():
            depth += 1 if event == "start" else -1
            if event == "end" and depth == 1:
                kept.append(el)
    except ET.ParseError:
        pass
    root = ET.Element("valgrindoutput")
    root.extend(kept)
    return root


def _root(path):
    """(root, complete)"""
    with open(path, errors="replace") as f:
        text = _LEAK_SUMMARY.sub("", f.read())
    try:
        return ET.fromstring(text), True
    except ET.ParseError:
        return _complete_children(text), False


def parse(path):
    root, complete = _root(path)
    out = Memcheck(path, pid=int(root.findtext("pid", "0")), complete=complete)
    args = root.find("args/argv")
    if args is not None:
        out.cmd = shlex.join(
            [args.findtext("exe", "")] + [a.text or "" for a in args.findall("arg")]
        )

    counts = {  # unique id -> times seen, for non leak errors
        p.findtext("unique"): int(p.findtext("count", "1"))
        for p in root.iterfind("errorcounts/pair")
    }
    for e in root.iterfind("error"):
        xwhat = e.find("xwhat")
        what = e.findtext("what") or (
            xwhat.findtext("text") if xwhat is not None else ""
        )
        stacks = e.findall("stack")
        err = Error(
            kind=e.findtext("kind", ""),
            what=what or "",
            count=counts.get(e.findtext("unique"), 1),
            stack=_frames(stacks[0] if stacks else None),
            aux=e.findtext("auxwhat", ""),
            aux_stack=_frames(stacks[1] if len(stacks) > 1 else None),
        )
        if xwhat is not None:
            err.bytes = int(xwhat.findtext("leakedbytes", "0"))
            err.blocks = int(xwhat.findtext("leakedblocks", "0"))
        out.errors.append(err)
    return out


def is_memcheck_xml(path):
    try:
        with open(path) as f:
            return "<valgrindoutput>" in f.read(512)
    except (OSError, UnicodeDecodeError):
        return False
