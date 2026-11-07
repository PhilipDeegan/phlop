# phlop/app/nvidia/ncu.py
#
# Nsight Compute: profile a command's kernels, keep ncu's files under
# .phlop/ncu, and load them back for short reports (ncu_report.py): per kernel,
# what bounds it (compute, memory and which level, or latency), occupancy,
# coalescing, the main warp stall, and ncu's own rules ranked by their
# estimated speedup
#
# by default each distinct launch (kernel, grid, block, shared memory) is
# profiled once (--filter-mode per-launch-config -c 1): a loop launching the
# same kernel 10000 times costs one replay, not 10000
#
# files: <outdir>/ncu.<run id>.<label>.{ncu-repz,log}, label p<pid> or rank<N>
# loading imports the report to csv once (<report>.raw.csv, <report>.rules.csv)
# a report from elsewhere (.ncu-rep or .ncu-repz) works the same
#
# https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
# https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html

import csv
import glob
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from phlop.app.valgrind import new_run_id

OUTDIR = Path(".phlop") / "ncu"
REPORT = re.compile(r"\.ncu-repz?$")
# the detailed set (speed of light, launch, occupancy, memory, source counters)
# and warp stalls: ~30 replays per launch
SECTIONS = ("--set", "detailed", "--section", "WarpStateStats")
PERM_ERROR = "ERR_NVGPUCTRPERM"

# verdict thresholds, % of peak: ncu's "SOL" guidance uses 60/80
BOUND_PCT = 60  # a unit this busy is the limiter
BOTH_PCT = 10  # compute and memory within this of each other: both
EXCESS = 2  # global sectors, or shared wavefronts, over the ideal for the access
OCCUPANCY_RATIO = 0.6  # achieved under this of theoretical: imbalance or tail

# raw metric names, base units (ns, byte/s, %)
METRICS = {
    "time": "gpu__time_duration.sum",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "dram_bw": "dram__bytes.sum.per_second",
    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occupancy_max": "sm__maximum_warps_per_active_cycle_pct",
    "registers": "launch__registers_per_thread",
    "waves": "launch__waves_per_multiprocessor",
    "sms": "device__attribute_multiprocessor_count",
    # coalescing: L2 sectors global accesses needed vs the minimum for their size
    "sectors": "memory_l2_theoretical_sectors_global",
    "sectors_ideal": "memory_l2_theoretical_sectors_global_ideal",
    # bank conflicts: shared memory wavefronts vs the minimum
    "wavefronts": "memory_l1_wavefronts_shared",
    "wavefronts_ideal": "memory_l1_wavefronts_shared_ideal",
    "l1_hit": "l1tex__t_sector_hit_rate.pct",
    "l2_hit": "lts__t_sector_hit_rate.pct",
}
OCCUPANCY_LIMITS = ("registers", "shared_mem", "warps", "blocks", "barriers")
STALL = re.compile(r"^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")
STALL_SKIP = ("selected", "not_selected")  # issuing, not stalled


def command(cmd, extra=(), outdir=OUTDIR, run_id=None, label="p%p", count=1):
    """the ncu command line and the files' base path. %p is the pid, expanded
    by ncu. extra comes last, so it can override the defaults, eg -e="--set full"
    """
    base = Path(outdir) / f"ncu.{run_id or new_run_id(outdir)}.{label}"
    opts = [
        *SECTIONS,
        "--nvtx",
        "--filter-mode=per-launch-config",
        f"--launch-count={count}",
        "--import-source=no",
        "--force-overwrite",
        f"--export={base}",
    ]
    return ["ncu", *opts, *extra, *cmd], base


def run(cmd, extra=(), outdir=OUTDIR, quiet=False, run_id=None, label="p%p", count=1):
    """profile cmd, returns (exit code, run id): ncu passes the program's exit
    code through. ncu's own output goes to <base>.log"""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id(outdir)
    nc, base = command(cmd, extra, outdir, run_id, label, count)
    log = str(base).replace("%p", str(os.getpid())) + ".log"
    nc = [nc[0], f"--log-file={log}", *nc[1:]]
    print(f"running: {shlex.join(nc)}", file=sys.stderr)
    pipe = subprocess.DEVNULL if quiet else None
    try:
        rc = subprocess.run(nc, stdout=pipe, stderr=pipe, check=False).returncode
    except FileNotFoundError:
        raise RuntimeError("ncu not found on PATH") from None
    return rc, run_id


def log_problem(outdir, run_id):
    """why a run wrote no report, from ncu's logs, or None"""
    for log in glob.glob(str(Path(outdir) / f"ncu.{run_id}.*.log")):
        text = Path(log).read_text(errors="replace")
        if PERM_ERROR in text:
            return (
                f"{PERM_ERROR}: GPU performance counters are restricted to admin"
                " users. Set NVreg_RestrictProfilingToAdminUsers=0 for the nvidia"
                " module (/etc/modprobe.d) and reboot, or run as root, see"
                " https://developer.nvidia.com/ERR_NVGPUCTRPERM"
            )
        if "No kernels were profiled" in text:
            names = re.findall(r"^\d+\. (.+)$", text, re.MULTILINE)
            if names:
                return f"no kernel matched, the program's kernels: {', '.join(names)}"
            return "no kernels were profiled, the program launched none"
        errors = re.findall(r"^==ERROR== (.+)$", text, re.MULTILINE)
        if errors:
            return "; ".join(errors)
    return None


# --------------------------------------------------------------------- results


@dataclass
class Rule:
    name: str
    kind: str  # OPT, WRN, INF
    speedup: float  # ncu's estimate, % of the kernel's time
    text: str


@dataclass
class Kernel:
    name: str  # demangled, arguments dropped
    full_name: str
    grid: str
    block: str
    nvtx: str = ""  # innermost push/pop range at launch
    launches: int = 1  # profiled launches merged into this row
    m: dict = field(default_factory=dict)  # METRICS keys, averaged
    limiter: str = ""  # occupancy limited by
    stalls: list = field(default_factory=list)  # [(reason, cycles per issue)]
    rules: list = field(default_factory=list)

    def __getattr__(self, key):  # k.time, k.sm_pct, ...; None when not collected
        if key in METRICS:
            return self.m.get(key)
        raise AttributeError(key)

    @staticmethod
    def _ratio(a, b):
        return a / b if a is not None and b else None

    @property
    def coalescing(self):
        """global memory sectors moved per sector needed, 1 is ideal"""
        return self._ratio(self.sectors, self.sectors_ideal)

    @property
    def bank_conflicts(self):
        """shared memory wavefronts per wavefront needed, 1 is ideal"""
        return self._ratio(self.wavefronts, self.wavefronts_ideal)

    @property
    def bound(self):
        return verdict(self)


@dataclass
class Report:
    path: str
    device: str = ""
    process: str = ""
    rank: int = None
    kernels: list = field(default_factory=list)  # slowest first

    @property
    def time(self):
        return sum((k.time or 0) * k.launches for k in self.kernels)


def number(s):
    """ncu csv cells: '1,048,576', '2.5', '' or 'n/a'"""
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def short_name(name):
    """'void ns::k<float>(int, float *)' -> 'ns::k<float>'"""
    depth, end = 0, len(name)
    for i in range(len(name) - 1, -1, -1):
        c = name[i]
        depth += c == ")"
        depth -= c == "("
        if c == "(" and depth == 0:
            end = i
            break
    head = name[:end].strip()
    return re.sub(r"^void ", "", head) or name


def _nvtx(cell):
    """'4011  "<default domain>:outer:none:..." "<default domain>:inner:..."'
    -> 'inner': domain, range, then 6 payload/colour/message fields"""
    ranges = re.findall(r'"([^"]*)"', cell or "")
    if not ranges:
        return ""
    head = ranges[-1].rsplit(":", 6)[0]
    return head.split(":", 1)[1] if ":" in head else head


def verdict(k):
    """what bounds the kernel, ncu's speed of light reading:
    a unit at >= BOUND_PCT of its peak limits it; neither: latency"""
    sm, mem = k.sm_pct, k.mem_pct
    if sm is None or mem is None:
        return "?"
    if max(sm, mem) >= BOUND_PCT:
        if abs(sm - mem) < BOTH_PCT:
            return "compute+memory"
        if sm > mem:
            return "compute"
        levels = [(k.m.get(f"{u}_pct") or 0, u.upper()) for u in ("dram", "l2", "l1")]
        return f"memory ({max(levels)[1]})"
    if k.waves is not None and k.waves < 1:
        return "latency (small grid)"
    return "latency"


def _merge(rows):
    """profiled launches of one kernel config, metrics averaged"""
    first = rows[0]
    k = Kernel(
        short_name(first["name"]),
        first["name"],
        first["grid"],
        first["block"],
        first["nvtx"],
        len(rows),
    )
    for key in METRICS:
        vals = [r["m"][key] for r in rows if r["m"].get(key) is not None]
        if vals:
            k.m[key] = sum(vals) / len(vals)
    stalls = defaultdict(list)
    for r in rows:
        for s, v in r["stalls"].items():
            stalls[s].append(v)
    k.stalls = sorted(
        ((s, sum(v) / len(v)) for s, v in stalls.items()), key=lambda x: -x[1]
    )
    limits = {u: v for u, v in first["limits"].items() if v is not None}
    if limits:
        k.limiter = min(limits, key=limits.get)
    return k


def parse_raw(path):
    """the raw page, base units: one row per profiled launch.
    returns (device, process, pid, rows by ncu ID)"""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return "", "", None, {}
    head, data = rows[0], rows[2:]  # rows[1] is units
    col = {c: i for i, c in enumerate(head)}
    nvtx_col = next((c for c in head if "Push/Pop_Range" in c), None)
    stall_cols = [(m.group(1), c) for c in head if (m := STALL.match(c))]
    out = {}
    device = process = pid = ""
    for r in data:
        if len(r) != len(head):
            continue

        def get(c, r=r):
            return r[col[c]] if c in col else ""

        device = get("device__attribute_display_name") or get("Device") or device
        process, pid = get("Process Name") or process, get("Process ID") or pid
        out[get("ID")] = {
            "name": get("Kernel Name"),
            "grid": get("Grid Size"),
            "block": get("Block Size"),
            "nvtx": _nvtx(get(nvtx_col)) if nvtx_col else "",
            "m": {k: number(get(c)) for k, c in METRICS.items() if c in col},
            "stalls": {
                s: v
                for s, c in stall_cols
                if s not in STALL_SKIP and (v := number(get(c)))
            },
            "limits": {
                u: number(get(f"launch__occupancy_limit_{u}")) for u in OCCUPANCY_LIMITS
            },
        }
    return device, process, pid, out


def parse_rules(path):
    """the details page's rule rows: ncu ID -> [Rule], OPT/WRN only"""
    out = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            name, kind = r.get("Rule Name"), r.get("Rule Type")
            if not name or kind not in ("OPT", "WRN"):
                continue
            speedup = number(r.get("Estimated Speedup")) or 0.0
            text = " ".join((r.get("Rule Description") or "").split())
            out[r["ID"]].setdefault(name, Rule(name, kind, speedup, text))
    return {i: sorted(v.values(), key=lambda x: -x.speedup) for i, v in out.items()}


def _rank(path):
    m = re.search(r"\.rank(\d+)\.", os.path.basename(path))
    return int(m.group(1)) if m else None


def parse(report, raw, rules):
    device, process, _, rows = parse_raw(raw)
    by_rule = parse_rules(rules) if rules else {}
    groups = defaultdict(list)
    for i, r in rows.items():
        r["rules"] = by_rule.get(i, [])
        groups[(r["name"], r["grid"], r["block"])].append(r)
    kernels = []
    for g in groups.values():
        k = _merge(g)
        seen = {}
        for r in g:
            for rule in r["rules"]:
                if rule.name not in seen or rule.speedup > seen[rule.name].speedup:
                    seen[rule.name] = rule
        k.rules = sorted(seen.values(), key=lambda x: -x.speedup)
        kernels.append(k)
    kernels.sort(key=lambda k: -((k.time or 0) * k.launches))
    return Report(report, device, process, _rank(report), kernels)


def _import(report, page, out):
    cmd = ["ncu", "--import", report, "--csv", "--page", page, "--print-units", "base"]
    cmd += ["--print-rule-details", "yes"] if page == "details" else []
    with open(out + ".tmp", "w") as f:
        try:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True)
        except FileNotFoundError:
            raise RuntimeError(
                "ncu not found on PATH, needed to read reports"
            ) from None
    os.replace(out + ".tmp", out)


def export(report):
    """the csv files next to a report, imported when missing or older"""
    base = REPORT.sub("", report)
    out = []
    for page, suffix in (("raw", ".raw.csv"), ("details", ".rules.csv")):
        f = base + suffix
        if not os.path.exists(f) or os.path.getmtime(f) < os.path.getmtime(report):
            _import(report, page, f)
        out.append(f)
    return out


def load(report):
    raw, rules = export(report)
    return parse(report, raw, rules)


def _run_id(path):
    parts = os.path.basename(path).split(".")
    return parts[1] if len(parts) >= 4 and parts[0] == "ncu" else None


def _latest(directory):
    files = [f for f in glob.glob(str(Path(directory) / "ncu.*")) if REPORT.search(f)]
    if not files:
        return []
    run_id = _run_id(max(files, key=os.path.getmtime))
    return sorted(f for f in files if _run_id(f) == run_id)


def resolve(spec=None, outdir=OUTDIR):
    """reports from: nothing (the latest run), a run id, a file, a directory
    (its latest run) or a glob"""
    if not spec:
        files = _latest(outdir)
    elif os.path.isdir(spec):
        files = _latest(spec) or sorted(
            f for f in glob.glob(str(Path(spec) / "*")) if REPORT.search(f)
        )
    elif os.path.isfile(spec):
        files = [spec]
    elif glob.has_magic(spec):
        files = sorted(f for f in glob.glob(spec) if REPORT.search(f))
    else:
        files = sorted(
            f
            for f in glob.glob(str(Path(outdir) / f"ncu.{spec}.*"))
            if REPORT.search(f)
        )
    if not files:
        raise FileNotFoundError(f"no ncu reports for {spec or outdir}")
    return files


def results(spec=None, outdir=OUTDIR):
    return [load(f) for f in resolve(spec, outdir)]


def by_rank(reports):
    return sorted(reports, key=lambda r: (r.rank is None, r.rank or 0, r.path))


def kernel_times(reports):
    """kernel name -> [total profiled ns per report], for ranks and compares"""
    out = defaultdict(lambda: [0.0] * len(reports))
    for i, r in enumerate(reports):
        for k in r.kernels:
            out[k.name][i] += (k.time or 0) * k.launches
    return dict(out)
