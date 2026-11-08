# phlop/app/nvidia/csan.py
#
# compute-sanitizer: run a command under one of its tools with useful defaults,
# keep its files under .phlop/csan, and load the xml back, grouped: the same
# error from 10000 threads is one line with a count
#
# files: <outdir>/csan-<tool>.<run id>.<label>.{xml,log}, label p<pid> or rank<N>
# tools: memcheck (+ leaks), racecheck, initcheck, synccheck
#
# https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html

import glob
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

from phlop.app.valgrind import new_run_id
from phlop.app.valgrind.common import Frame

OUTDIR = Path(".phlop") / "csan"
TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
ERROR_EXITCODE = 99  # errors found, else the program's own
PRINT_LIMIT = 1000  # records saved, groups count up to this

# report order, most serious first
KINDS = ("error", "race", "sync", "uninit", "api", "leak", "note")
_KIND = {
    "Precise": "error",
    "Imprecise": "error",
    "Analysis": "race",
    "Synccheck": "sync",
    "Initcheck": "uninit",
    "Api": "api",
    "Leak": "leak",
    "Sanitizer": "note",
}
# host frames inside the CUDA driver/runtime, skipped to find the caller
_RUNTIME = re.compile(
    r"^(cu[A-Z]\w*|cuda[A-Z]\w*|libcudart_static\w*|__cuda\w*|__device_stub\w*"
    r"|cudaError \w+|void cuda\w*|_start|__libc_\w+)"
)


def tool_options(tool, base, leaks=True):
    opts = [
        f"--tool={tool}",
        f"--error-exitcode={ERROR_EXITCODE}",
        f"--print-limit={PRINT_LIMIT}",
        "--xml",
        f"--save={base}.xml",
        f"--log-file={base}.log",
    ]
    if tool == "memcheck" and leaks:
        opts += ["--leak-check=full"]
    return opts


def command(tool, cmd, extra=(), outdir=OUTDIR, run_id=None, label="p%p"):
    """the compute-sanitizer command line and the files' base path. %p is
    expanded by compute-sanitizer. extra comes last, so it can override"""
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool}, one of {', '.join(TOOLS)}")
    base = Path(outdir) / f"csan-{tool}.{run_id or new_run_id(outdir)}.{label}"
    return ["compute-sanitizer", *tool_options(tool, base), *extra, *cmd], base


def run(tool, cmd, extra=(), outdir=OUTDIR, quiet=False, run_id=None, label="p%p"):
    """returns (exit code, run id): ERROR_EXITCODE on errors, else the program's"""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id(outdir)
    cs, _ = command(tool, cmd, extra, outdir, run_id, label)
    print(f"running: {shlex.join(cs)}", file=sys.stderr)
    pipe = subprocess.DEVNULL if quiet else None
    try:
        rc = subprocess.run(cs, stdout=pipe, stderr=pipe, check=False).returncode
    except FileNotFoundError:
        raise RuntimeError("compute-sanitizer not found on PATH") from None
    return rc, run_id


# --------------------------------------------------------------------- results


@dataclass
class Issue:
    """records that are the same problem: kind, text and place"""

    kind: str
    what: str
    where: Frame = None  # device code, or racecheck's write side
    other: Frame = None  # racecheck's other side
    caller: Frame = None  # first host frame outside the CUDA runtime
    count: int = 0  # records, or racecheck's hazard count
    bytes: int = 0  # leaks
    example: str = ""  # a thread that hit it
    ranks: set = field(default_factory=set)
    severe: bool = False  # a note at error level, see issue()

    def key(self):
        return (self.kind, self.what, self.where, self.other, self.caller)


@dataclass
class Result:
    path: str
    tool: str
    rank: int = None
    issues: list = field(default_factory=list)
    records: int = 0
    complete: bool = True  # the xml was closed: the program didn't die

    @property
    def limited(self):
        return self.records >= PRINT_LIMIT

    @property
    def errors(self):
        return [i for i in self.issues if i.kind not in ("leak", "note")]

    @property
    def leaks(self):
        return [i for i in self.issues if i.kind == "leak"]


def _frame(e):
    if e is None:
        return None
    line = e.findtext("line")
    return Frame(
        e.findtext("func") or "?",
        e.findtext("path") or "",
        int(line) if line and line.isdigit() else 0,
        e.findtext("module") or "",
    )


def _caller(rec):
    frames = [_frame(f) for f in rec.findall("hostStack/frame")]
    frames = [f for f in frames if f and f.fn != "?"]
    for f in frames:
        if not _RUNTIME.match(f.fn) and "libcuda" not in f.obj:
            return f
    return None


def _dim(e):
    return ",".join(e.findtext(a) or "0" for a in "xyz").replace(",0,0", "")


def _who(rec):
    t, b = rec.find("who/threadIdx"), rec.find("who/blockIdx")
    return (
        f"block {_dim(b)} thread {_dim(t)}" if t is not None and b is not None else ""
    )


def _what(rec, kind):
    text = rec.findtext("what/text") or "?"
    if kind == "api":
        api, err = rec.findtext("what/api"), rec.findtext("what/error")
        msg = rec.findtext("what/message")
        return f"{api}: {err}" + (f", {msg}" if msg else "") if api else text
    if kind == "sync" and (t := rec.findtext("what/type")):
        return f"{text}: {t}"
    if kind == "leak":
        return f"leaked {rec.findtext('what/type') or ''} allocation".replace("  ", " ")
    return text


def issue(rec):
    kind = _KIND.get(rec.findtext("kind"), "error")
    i = Issue(kind, _what(rec, kind), count=1, example=_who(rec))
    # a Sanitizer record at error level: a failure of its own, or after others
    i.severe = kind == "note" and (rec.findtext("level") or "").lower() in (
        "error",
        "fatal",
    )
    if kind == "race":
        i.where = _frame(rec.find("what/source/where"))
        i.other = _frame(rec.find("what/destination/where"))
        n = rec.findtext("what/destination/count") or rec.findtext("what/count")
        i.count = int(n) if n and n.isdigit() else 1
    else:
        i.where = _frame(rec.find("where"))
    if kind in ("leak", "api", "note") or i.where is None:
        i.caller = _caller(rec)
    if kind == "leak":
        size = rec.findtext("what/size")
        i.bytes = int(size) if size and size.isdigit() else 0
    return i


def group(issues):
    out = {}
    for i in issues:
        g = out.get(i.key())
        if g is None:  # a copy: the input issues may be grouped again
            out[i.key()] = replace(i, ranks=set(i.ranks))
            continue
        g.count += i.count
        g.bytes += i.bytes
        g.ranks |= i.ranks
    return sorted(out.values(), key=lambda i: (KINDS.index(i.kind), -i.count, -i.bytes))


def _records(path):
    """record elements, as many as were closed when the xml is cut short"""
    out, depth, complete = [], 0, False
    try:
        for event, e in ET.iterparse(path, events=("start", "end")):
            if event == "start":
                depth += 1
                continue
            depth -= 1
            if depth == 1 and e.tag == "record":
                out.append(e)
            elif depth == 0:
                complete = True
    except ET.ParseError:
        pass
    return out, complete


_NAME = re.compile(r"csan-(\w+)\.[^.]+\.(p|rank)(\d+)\.xml$")


def parse(path):
    m = _NAME.search(path)
    tool = m.group(1) if m else "?"
    rank = int(m.group(3)) if m and m.group(2) == "rank" else None
    records, complete = _records(path)
    issues = [issue(r) for r in records]
    for i in issues:
        if rank is not None:
            i.ranks.add(rank)
    return Result(path, tool, rank, group(issues), len(records), complete)


def is_result(path):
    if not path.endswith(".xml"):
        return False
    try:
        with open(path) as f:
            return "<ComputeSanitizerOutput" in f.read(256)
    except (OSError, UnicodeDecodeError):
        return False


def _run_id(path):
    parts = os.path.basename(path).split(".")
    return parts[1] if len(parts) >= 4 else None


def _latest(directory):
    files = [f for f in glob.glob(str(Path(directory) / "csan-*.xml")) if is_result(f)]
    if not files:
        return []
    run_id = _run_id(max(files, key=os.path.getmtime))
    return sorted(f for f in files if _run_id(f) == run_id)


def resolve(spec=None, outdir=OUTDIR):
    """result files from: nothing (the latest run), a run id, a file, a
    directory (its latest run) or a glob"""
    if not spec:
        files = _latest(outdir)
    elif os.path.isdir(spec):
        files = _latest(spec)
    elif os.path.isfile(spec):
        if not is_result(spec):
            raise ValueError(f"{spec}: not a compute-sanitizer xml result")
        files = [spec]
    elif glob.has_magic(spec):
        files = sorted(f for f in glob.glob(spec) if is_result(f))
    else:
        files = sorted(
            f
            for f in glob.glob(str(Path(outdir) / f"csan-*.{spec}.*.xml"))
            if is_result(f)
        )
    if not files:
        raise FileNotFoundError(f"no compute-sanitizer results for {spec or outdir}")
    return files


def results(spec=None, outdir=OUTDIR):
    return [parse(f) for f in resolve(spec, outdir)]
