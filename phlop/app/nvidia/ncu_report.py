# phlop/app/nvidia/ncu_report.py
#
# short reports of ncu profiles, the answer first: per kernel what bounds it,
# then hints worth acting on: phlop's own from the metrics, then ncu's rules
# ranked by ncu's estimated speedup. Across MPI ranks: kernels that take
# longer on one rank than the others

import os
import statistics

from phlop.app.nvidia import ncu
from phlop.app.nvidia.nsys_report import _md_label, _md_table, _short, fmt_ns

NEAR_PEAK = 80  # % of DRAM peak: little left to gain without moving less data
LOW_OCCUPANCY = 50  # theoretical occupancy % worth naming its limiter
STRAGGLER = 1.10  # a rank's kernel time over the median rank's
RULES_PER_KERNEL = 3
MIN_SPEEDUP = 1  # % estimated by ncu, below this a rule is only information
# ncu rules that only restate a verdict already given
COMPUTE_RULES = ("HighPipeUtilization", "FPInstructions", "SOLFPRoofline")


def label(r):
    if r.rank is not None:
        return f"rank {r.rank}"
    return os.path.basename(ncu.REPORT.sub("", r.path)).replace("ncu.", "")


def _pct(v):
    return "-" if v is None else f"{min(v, 100):.0f}"


def _x(v):
    return "-" if v is None else f"{v:.1f}x"


def _gbs(v):
    return "-" if v is None else f"{v / 1e9:.0f}"


def _occupancy(k):
    if k.occupancy is None:
        return "-"
    return f"{_pct(k.occupancy)}/{_pct(k.occupancy_max)}"


def _stall(k):
    return k.stalls[0][0] if k.stalls else "-"


def _first_sentence(text, n=160):
    end = text.find(". ")
    s = text if end < 0 else text[: end + 1]
    return _short(s, n)


def kernel_hints(k):
    """phlop's reading of the metrics, most useful first"""
    out = []
    bound = k.bound
    if k.coalescing and k.coalescing > ncu.EXCESS:
        out.append(
            f"global accesses move {k.coalescing:.1f}x the sectors they need:"
            " uncoalesced, neighbouring threads should touch neighbouring addresses"
        )
    if k.bank_conflicts and k.bank_conflicts > ncu.EXCESS:
        out.append(
            f"shared memory needs {k.bank_conflicts:.1f}x the wavefronts it should:"
            " bank conflicts, pad or swizzle the shared arrays"
        )
    if bound == "memory (DRAM)" and (k.dram_pct or 0) >= NEAR_PEAK:
        out.append(
            f"DRAM at {_pct(k.dram_pct)}% of peak, near the hardware limit:"
            " only moving less data helps (fuse kernels, reuse, smaller types)"
        )
    if bound == "compute" and (k.sm_pct or 0) >= NEAR_PEAK:
        out.append(
            f"SM at {_pct(k.sm_pct)}% of peak, near the hardware limit:"
            " fewer or cheaper instructions (algorithm, precision, tensor cores)"
        )
    if bound == "latency (small grid)":
        out.append(
            f"the grid fills {k.waves:.2f} waves of the {k.sms:.0f} SMs:"
            " more blocks, or overlap independent work on streams"
        )
    if k.occupancy_max is not None and k.occupancy_max < LOW_OCCUPANCY and k.limiter:
        out.append(
            f"theoretical occupancy {_pct(k.occupancy_max)}%, limited by"
            f" {k.limiter.replace('_', ' ')}"
            + (f" ({k.registers:.0f} per thread)" if k.limiter == "registers" else "")
        )
    elif (
        k.occupancy is not None
        and k.occupancy_max
        and k.occupancy < ncu.OCCUPANCY_RATIO * k.occupancy_max
    ):
        out.append(
            f"achieved occupancy {_pct(k.occupancy)}% of a possible"
            f" {_pct(k.occupancy_max)}%: uneven blocks, or a tail of the last wave"
        )
    if bound.startswith("latency") and k.stalls:
        s, v = k.stalls[0]
        out += [f"warps mostly stall on {s} ({v:.1f} cycles per issued instruction)"]
    return out


def rules(k):
    """ncu's rules worth showing, by estimated speedup. For a memory bound
    kernel, idle compute pipes and warps waiting on loads restate the verdict"""
    skip = ()
    if k.bound.startswith("memory"):
        skip = COMPUTE_RULES
        if _stall(k) == "long_scoreboard":
            skip += ("CPIStall",)
    keep = [r for r in k.rules if r.name not in skip and r.speedup >= MIN_SPEEDUP]
    return keep[:RULES_PER_KERNEL]


def hints(report, top=10):
    lines = []
    for k in report.kernels[:top]:
        lines += [f"{k.name}: {h}" for h in kernel_hints(k)]
        lines += [
            f"{k.name}: ncu {r.name} (est. {r.speedup:.0f}%): {_first_sentence(r.text)}"
            for r in rules(k)
        ]
    return lines


def headline(r):
    n = len(r.kernels)
    launches = sum(k.launches for k in r.kernels)
    s = f"{r.process or '?'} on {r.device or '?'}, {n} kernel{'s' * (n != 1)}"
    s += f" ({launches} launches)" if launches != n else ""
    return s + f", {fmt_ns(r.time)} profiled"


COLUMNS = ("time", "bound", "SM%", "mem%", "DRAM GB/s", "occ", "coalesce", "stall")


def _row(k):
    time = fmt_ns((k.time or 0) * k.launches)
    return (time, k.bound, _pct(k.sm_pct), _pct(k.mem_pct), _gbs(k.dram_bw),
            _occupancy(k), _x(k.coalescing), _stall(k))  # fmt: skip


def _name(k):
    s = _short(k.name, 50)
    s += f" x{k.launches}" if k.launches > 1 else ""
    return s + (f" [{k.nvtx}]" if k.nvtx else "")


def detail(r, top=10):
    lines = [f"ncu {label(r)}: {headline(r)}"]
    if not r.kernels:
        return lines + ["  no kernels"]
    widths = (8, 18, 4, 4, 9, 7, 8, 16)
    fmt = "  " + " ".join(f"{{:>{w}}}" for w in widths) + "  {}"
    lines += [fmt.format(*COLUMNS, "kernel [NVTX]")]
    for k in r.kernels[:top]:
        lines += [fmt.format(*_row(k), _name(k))]
    if len(r.kernels) > top:
        lines += [f"  ... {len(r.kernels) - top} more, --top"]
    lines += ["  occ: achieved/theoretical occupancy %, coalesce: sectors moved/needed"]
    if h := hints(r, top):
        lines += ["hints:"] + [f"  {s}" for s in h]
    return lines


def stragglers(reports):
    """[(kernel, slow report, ratio to the median of the other ranks)] worst
    first. The others: with 2 ranks the median would include the slow one"""
    out = []
    for name, ts in ncu.kernel_times(reports).items():
        i = max(range(len(ts)), key=lambda j: ts[j])
        others = [t for j, t in enumerate(ts) if t and j != i]
        if not others:
            continue
        med = statistics.median(others)
        if ts[i] / med >= STRAGGLER:
            out.append((name, reports[i], ts[i] / med))
    return sorted(out, key=lambda x: -x[2])


def ranks_lines(reports, top=10):
    lines = [f"  {'':>8} {'profiled':>10} {'kernels':>8}  slowest kernel"]
    for r in reports:
        slow = r.kernels[0].name if r.kernels else "-"
        lines += [
            f"  {label(r):>8} {fmt_ns(r.time):>10} {len(r.kernels):>8}  {_short(slow)}"
        ]
    s = stragglers(reports)
    if not s:
        lines.append(
            "  balanced: no kernel takes over"
            f" {STRAGGLER:.2f}x the other ranks' median on any rank"
        )
    for name, r, ratio in s[:top]:
        lines += [
            f"  straggler: {_short(name)} takes {ratio:.2f}x the other ranks' median on {label(r)}"
        ]
    return lines


def _worst(reports):
    s = stragglers(reports)
    return s[0][1] if s else max(reports, key=lambda r: r.time)


def text(reports, top=10, all=False):
    reports = ncu.by_rank(reports)
    if len(reports) == 1:
        return "\n".join(detail(reports[0], top))
    lines = [f"ncu: {len(reports)} processes"] + ranks_lines(reports, top) + [""]
    for r in reports if all else [_worst(reports)]:
        lines += detail(r, top) + [""]
    return "\n".join(lines).rstrip()


def _delta(a, b):
    return f"{100 * (b - a) / a:+.1f}%" if a else "new"


def _bounds(reports):
    return {k.name: k.bound for r in reports for k in r.kernels}


def compare(a, b, names=("A", "B"), top=10):
    """two sets of reports, eg master vs a branch: per kernel profiled time,
    mean per process, and the verdict when it changed"""
    na, nb = names
    ta = {n: statistics.mean(v) for n, v in ncu.kernel_times(a).items()}
    tb = {n: statistics.mean(v) for n, v in ncu.kernel_times(b).items()}
    ba, bb = _bounds(a), _bounds(b)
    lines = [f"ncu compare {nb} vs {na} (kernel time, means per process)"]
    lines += [f"  {na[:10]:>10} {nb[:10]:>10} {'change':>8}  kernel"]
    keys = sorted(set(ta) | set(tb), key=lambda n: -abs(tb.get(n, 0) - ta.get(n, 0)))
    for n in keys[:top]:
        va, vb = ta.get(n, 0), tb.get(n, 0)
        s = f"  {fmt_ns(va):>10} {fmt_ns(vb):>10} {_delta(va, vb):>8}  {_short(n)}"
        if n in ba and n in bb and ba[n] != bb[n]:
            s += f"  ({ba[n]} -> {bb[n]})"
        lines += [s]
    total_a, total_b = sum(ta.values()), sum(tb.values())
    lines += [
        f"  {fmt_ns(total_a):>10} {fmt_ns(total_b):>10} {_delta(total_a, total_b):>8}  total"
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- markdown


def markdown(reports, top=10, compare_to=None, names=("A", "B")):
    if compare_to is not None:
        return "```\n" + compare(reports, compare_to, names, top) + "\n```"
    reports = ncu.by_rank(reports)
    out = []
    if len(reports) > 1:
        out += [f"### ncu: {len(reports)} processes", "", "```"]
        out += [s[2:] for s in ranks_lines(reports, top)] + ["```", ""]
        reports = [_worst(reports)]
    r = reports[0]
    out += [f"### ncu {label(r)}: {headline(r)}", ""]
    if not r.kernels:
        return "\n".join(out).rstrip()
    out += _md_table(
        (*COLUMNS, "kernel [NVTX]"),
        [(*_row(k), f"`{_name(k)}`") for k in r.kernels[:top]],
    ) + [
        "",
        "occ: achieved/theoretical occupancy %, coalesce: sectors moved/needed",
        "",
    ]
    shown = r.kernels[:top]
    axis = ['"' + _md_label(_short(k.name, 20)) + '"' for k in shown]
    out += [
        "```mermaid",
        "xychart-beta",
        '  title "% of peak: SM (first bar) vs memory (second bar)"',
        f"  x-axis [{', '.join(axis)}]",
        '  y-axis "%" 0 --> 100',
        f"  bar [{', '.join(_pct(k.sm_pct or 0) for k in shown)}]",
        f"  bar [{', '.join(_pct(k.mem_pct or 0) for k in shown)}]",
        "```",
        "",
    ]
    if h := hints(r, top):
        out += ["**hints**", ""] + [f"- {s}" for s in h]
    return "\n".join(out).rstrip()
