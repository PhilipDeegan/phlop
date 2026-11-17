# phlop/app/nvidia/nsys.py
#
# Nsight Systems: profile a command, keep nsys's files under .phlop/nsys, and
# load them back (sqlite export, read with the standard library) for short
# reports (nsys_report.py)
#
# files: <outdir>/nsys.<run id>.<label>.{nsys-rep,sqlite}, label p<pid> or rank<N>
# a .nsys-rep from elsewhere (eg a cluster) is exported to sqlite when loaded
#
# https://docs.nvidia.com/nsight-systems/UserGuide/index.html

import bisect
import glob
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from phlop.app.valgrind import new_run_id

OUTDIR = Path(".phlop") / "nsys"
TRACE = "cuda,nvtx"

# nsys enum values, stable across versions (ENUM_CUDA_MEMCPY_OPER, ENUM_CUDA_MEM_KIND)
COPY_KINDS = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P", 11: "UVM H2D", 12: "UVM D2H"}
PAGEABLE = 0
NVTX_RANGES = (59, 60)  # push/pop, start/end
# MPI calls that aren't communication
MPI_SETUP = re.compile(r"^MPI_(Init|Init_thread|Finalize|Comm_\w+|Type_\w+|Op_\w+)$")


def mpi_impl():
    return "openmpi" if "OMPI_COMM_WORLD_RANK" in os.environ else "mpich"


def command(cmd, extra=(), outdir=OUTDIR, run_id=None, label="p%p", trace=TRACE):
    """the nsys command line and the files' base path. %p is the pid, expanded
    by nsys. extra comes last, so it can override the defaults"""
    base = Path(outdir) / f"nsys.{run_id or new_run_id(outdir)}.{label}"
    opts = [
        f"--trace={trace}",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "--export=sqlite",
        f"--output={base}",
    ]
    if "mpi" in trace.split(","):
        opts += [f"--mpi-impl={mpi_impl()}"]
    return ["nsys", "profile", *opts, *extra, *cmd], base


def run(
    cmd, extra=(), outdir=OUTDIR, quiet=False, run_id=None, label="p%p", trace=TRACE
):
    """profile cmd, returns (exit code, run id): nsys passes the program's
    exit code through. nsys's own output goes to <base>.log"""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id(outdir)
    ns, base = command(cmd, extra, outdir, run_id, label, trace)
    print(f"running: {shlex.join(ns)}", file=sys.stderr)
    pipe = subprocess.DEVNULL if quiet else None
    log = str(base).replace("%p", str(os.getpid())) + ".log"
    try:
        with open(log, "w") as lf:
            rc = subprocess.run(ns, stdout=pipe, stderr=lf, check=False).returncode
    except FileNotFoundError:
        raise RuntimeError("nsys not found on PATH") from None
    return rc, run_id


# ------------------------------------------------------------------- intervals


def merge(intervals):
    """sorted, non overlapping [(start, end)]"""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def length(merged):
    return sum(e - s for s, e in merged)


def overlap(a, b):
    """time covered by both of two merged interval lists"""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        total += max(0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


class Coverage:
    """how much of [lo, hi) a merged interval list covers, O(log n) per query"""

    def __init__(self, merged):
        self.starts = [s for s, _ in merged]
        self.merged = merged
        self.prefix = [0]
        for s, e in merged:
            self.prefix.append(self.prefix[-1] + e - s)

    def __call__(self, lo, hi):
        if hi <= lo or not self.merged:
            return 0
        i = max(0, bisect.bisect_right(self.starts, lo) - 1)
        j = bisect.bisect_left(self.starts, hi)
        if i >= j:
            return 0
        total = self.prefix[j] - self.prefix[i]
        total -= max(0, min(lo, self.merged[i][1]) - self.merged[i][0])
        total -= max(0, self.merged[j - 1][1] - max(hi, self.merged[j - 1][0]))
        return total


def gaps(merged, top=3):
    """the longest idle stretches between busy intervals, longest first"""
    out = [
        (merged[k + 1][0] - merged[k][1], merged[k][1]) for k in range(len(merged) - 1)
    ]
    return sorted(out, reverse=True)[:top]


# --------------------------------------------------------------------- results


@dataclass
class Kernel:
    name: str
    count: int = 0
    total: int = 0  # ns
    max: int = 0
    nccl: bool = False
    grid: tuple = ()
    block: tuple = ()
    regs: int = 0
    shared: int = 0  # bytes, static + dynamic
    busy: int = 0  # ns, union of its launches: overlapping ones counted once

    @property
    def avg(self):
        return self.total / self.count if self.count else 0


@dataclass
class Copy:
    kind: str
    pageable: bool
    count: int = 0
    bytes: int = 0
    total: int = 0  # ns

    @property
    def gbps(self):
        return self.bytes / self.total if self.total else 0  # bytes/ns == GB/s


@dataclass
class Call:
    """a host side call (CUDA API, MPI) or an NVTX range, by name"""

    name: str
    count: int = 0
    total: int = 0
    max: int = 0
    gpu: int = 0  # NVTX: GPU busy while the range was open


@dataclass
class Gap:
    length: int
    start: int
    nvtx: str = ""  # innermost NVTX range around it


@dataclass
class Profile:
    path: str
    cmd: str = ""
    rank: int = None
    gpus: list = field(default_factory=list)
    duration: int = 0  # ns, the whole capture
    first: int = 0  # first and last GPU activity
    last: int = 0
    busy: int = 0  # union of kernels, copies, memsets
    compute: int = 0  # union of non NCCL kernels
    nccl: int = 0  # union of NCCL kernels
    nccl_overlap: int = 0  # NCCL kernels running alongside compute
    kernels: list = field(default_factory=list)
    copies: list = field(default_factory=list)
    api: list = field(default_factory=list)
    nvtx: list = field(default_factory=list)
    mpi: list = field(default_factory=list)
    gaps: list = field(default_factory=list)

    @property
    def span(self):
        return self.last - self.first

    @property
    def idle(self):
        return self.span - self.busy

    @property
    def mpi_time(self):
        return sum(c.total for c in self.mpi if not MPI_SETUP.match(c.name))

    @property
    def wait(self):
        """time spent communicating, which on a fast rank is mostly waiting"""
        return self.mpi_time + self.nccl


def _tables(db):
    return {
        r[0] for r in db.execute("select name from sqlite_master where type='table'")
    }


def _strings(db):
    return dict(db.execute("select id, value from StringIds"))


def _api_name(name):
    return re.sub(r"_v\d+$", "", name or "?")


def _kernels(db, strings):
    rows = db.execute(
        "select start, end, demangledName, shortName, gridX, gridY, gridZ, blockX,"
        " blockY, blockZ, registersPerThread, staticSharedMemory + dynamicSharedMemory"
        " from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    by, nccl, compute, spans = {}, [], [], defaultdict(list)
    for s, e, dn, sn, gx, gy, gz, bx, by_, bz, regs, shm in rows:
        name = strings.get(dn) or strings.get(sn) or "?"
        k = by.get(name)
        if k is None:
            short = strings.get(sn) or name
            k = by[name] = Kernel(
                name, nccl=short.startswith("nccl") or "nccl" in name[:16]
            )
            k.grid, k.block, k.regs, k.shared = (
                (gx, gy, gz),
                (bx, by_, bz),
                regs,
                shm or 0,
            )
        k.count, k.total, k.max = k.count + 1, k.total + e - s, max(k.max, e - s)
        (nccl if k.nccl else compute).append((s, e))
        spans[name].append((s, e))
    for name, k in by.items():
        k.busy = length(merge(spans[name]))
    return sorted(by.values(), key=lambda k: -k.total), merge(compute), merge(nccl)


def _copies(db, tables):
    out, intervals = {}, []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for s, e, kind, src, dst, nbytes in db.execute(
            "select start, end, copyKind, srcKind, dstKind, bytes"
            " from CUPTI_ACTIVITY_KIND_MEMCPY"
        ):
            name = COPY_KINDS.get(kind, f"kind {kind}")
            pageable = PAGEABLE in (src, dst) and kind in (1, 2)
            c = out.setdefault((name, pageable), Copy(name, pageable))
            c.count, c.bytes, c.total = c.count + 1, c.bytes + nbytes, c.total + e - s
            intervals.append((s, e))
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        for s, e, nbytes in db.execute(
            "select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMSET"
        ):
            c = out.setdefault(("memset", False), Copy("memset", False))
            c.count, c.bytes, c.total = c.count + 1, c.bytes + nbytes, c.total + e - s
            intervals.append((s, e))
    return sorted(out.values(), key=lambda c: -c.total), merge(intervals)


def _calls(rows, name_of=lambda n: n):
    by = {}
    for s, e, n in rows:
        if e is None:
            continue
        c = by.setdefault(name_of(n), Call(name_of(n)))
        c.count, c.total, c.max = c.count + 1, c.total + e - s, max(c.max, e - s)
    return sorted(by.values(), key=lambda c: -c.total)


def _nvtx(db, strings, coverage):
    rows = [
        (s, e, text or strings.get(tid, "?"))
        for s, e, text, tid in db.execute(
            "select start, end, text, textId from NVTX_EVENTS"
            f" where eventType in {NVTX_RANGES} and end is not null"
        )
    ]
    calls = {c.name: c for c in _calls(rows)}
    for s, e, n in rows:
        calls[n].gpu += coverage(s, e)
    return list(calls.values()), sorted(rows, key=lambda r: r[1] - r[0])


def _innermost(ranges, t):
    """the shortest NVTX range open at time t, ranges sorted by length"""
    return next((n for s, e, n in ranges if s <= t < e), "")


def _mpi(db, strings, tables):
    rows = []
    for t in ("MPI_COLLECTIVES_EVENTS", "MPI_P2P_EVENTS", "MPI_START_WAIT_EVENTS",
              "MPI_OTHER_EVENTS"):  # fmt: skip
        if t in tables:
            rows += db.execute(f"select start, end, textId from {t}").fetchall()
    return _calls(rows, lambda tid: strings.get(tid, "?"))


def _meta(db, tables):
    cmd, rank = "", None
    if "META_DATA_CAPTURE" in tables:
        meta = db.execute(
            "select name, value from META_DATA_CAPTURE where name like 'PROCESS_0:%'"
        ).fetchall()
        exe = [v for n, v in meta if n == "PROCESS_0:COMMAND"]
        args = [v for n, v in meta if n.startswith("PROCESS_0:ARGUMENT_")]
        cmd = " ".join(exe[:1] + args)
    if "MPI_RANKS" in tables:
        r = db.execute("select rank from MPI_RANKS limit 1").fetchone()
        rank = r[0] if r else None
    return cmd, rank


def parse(path):
    """a Profile from an nsys sqlite export"""
    # as_uri percent encodes, a "#" or "?" in the path stays in the file name
    db = sqlite3.connect(Path(path).absolute().as_uri() + "?mode=ro", uri=True)
    try:
        tables, strings = _tables(db), _strings(db)
        p = Profile(path)
        p.cmd, p.rank = _meta(db, tables)
        if p.rank is None and (m := re.search(r"\.rank(\d+)\.sqlite$", path)):
            p.rank = int(m.group(1))
        if "TARGET_INFO_GPU" in tables:
            p.gpus = [r[0] for r in db.execute("select name from TARGET_INFO_GPU")]
        if "ANALYSIS_DETAILS" in tables:
            p.duration = db.execute("select duration from ANALYSIS_DETAILS").fetchone()[
                0
            ]

        compute, nccl = [], []
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            p.kernels, compute, nccl = _kernels(db, strings)
        p.copies, copies = _copies(db, tables)
        busy = merge(compute + nccl + copies)
        p.busy, p.compute, p.nccl = length(busy), length(compute), length(nccl)
        p.nccl_overlap = overlap(compute, nccl)
        if busy:
            p.first, p.last = busy[0][0], busy[-1][1]

        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
            p.api = _calls(
                db.execute(
                    "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
                ).fetchall(),
                lambda n: _api_name(strings.get(n)),
            )
        ranges = []
        if "NVTX_EVENTS" in tables:
            p.nvtx, ranges = _nvtx(db, strings, Coverage(busy))
        p.gaps = [Gap(n, s, _innermost(ranges, s + n // 2)) for n, s in gaps(busy)]
        p.mpi = _mpi(db, strings, tables)
        return p
    finally:
        db.close()


def export(rep):
    """sqlite next to a .nsys-rep, exported when missing or older"""
    out = re.sub(r"\.nsys-rep$", "", rep) + ".sqlite"
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(rep):
        cmd = [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "-o",
            out,
            rep,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out


def _sqlite(path):
    return export(path) if path.endswith(".nsys-rep") else path


def _run_id(path):
    parts = os.path.basename(path).split(".")
    return parts[1] if len(parts) >= 4 and parts[0] == "nsys" else None


def _latest(directory):
    files = glob.glob(str(Path(directory) / "nsys.*.sqlite"))
    files += glob.glob(str(Path(directory) / "nsys.*.nsys-rep"))
    if not files:
        return []
    run_id = _run_id(max(files, key=os.path.getmtime))
    return sorted(f for f in files if _run_id(f) == run_id)


def _dedup(files):
    """a .nsys-rep and its .sqlite are one result: the .nsys-rep, so load()
    re-exports a stale .sqlite"""
    out = []
    for f in files:
        if f.endswith(".sqlite") and os.path.isfile(
            rep := f[: -len(".sqlite")] + ".nsys-rep"
        ):
            f = rep
        if f not in out:
            out.append(f)
    return out


def resolve(spec=None, outdir=OUTDIR):
    """result files from: nothing (the latest run), a run id, a file (.sqlite or
    .nsys-rep), a directory (its latest run) or a glob"""
    is_result = re.compile(r"\.(sqlite|nsys-rep)$").search
    if not spec:
        files = _latest(outdir)
    elif os.path.isdir(spec):
        files = _latest(spec) or sorted(
            f for f in glob.glob(str(Path(spec) / "*")) if is_result(f)
        )
    elif os.path.isfile(spec):
        files = [spec]
    elif glob.has_magic(spec):
        files = sorted(f for f in glob.glob(spec) if is_result(f))
    else:
        files = sorted(
            f for f in glob.glob(str(Path(outdir) / f"nsys.{spec}.*")) if is_result(f)
        )
    if not files:
        raise FileNotFoundError(f"no nsys results for {spec or outdir}")
    return _dedup(files)


def load(path):
    return parse(_sqlite(path))


def results(spec=None, outdir=OUTDIR):
    return [load(f) for f in resolve(spec, outdir)]


def by_rank(profiles):
    return sorted(profiles, key=lambda p: (p.rank is None, p.rank or 0, p.path))


def merged_kernels(profiles):
    """kernel name -> mean total ns per process"""
    out = defaultdict(int)
    for p in profiles:
        for k in p.kernels:
            out[k.name] += k.total
    return {n: t / len(profiles) for n, t in out.items()}
