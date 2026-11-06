# phlop/app/nvidia/nsys_report.py
#
# short reports of nsys profiles, the answer first: where the GPU time goes,
# how long the GPU sits idle and where, and hints worth acting on. Across MPI
# ranks: the imbalance, and which rank the others wait for

import os
import statistics

from phlop.app.nvidia import nsys
from phlop.app.valgrind.common import format_bytes

# hint thresholds
IDLE_PCT = 20  # GPU idle between its first and last activity
TINY_KERNEL_NS = 10_000  # launch bound below this average...
TINY_KERNEL_COUNT = 1000  # ...for this many launches
SYNC_PCT = 10  # host blocked in synchronise calls, of the capture
OVERLAP_PCT = 50  # NCCL kernels hidden behind compute
WAIT_PCT = 20  # MPI + NCCL, of the GPU span
STRAGGLER = 1.10  # compute over the median rank's
SYNC_CALLS = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize")


def fmt_ns(ns):
    ns = float(ns)
    for unit, div in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if abs(ns) >= div:
            return f"{ns / div:.3g}{unit}"
    return f"{ns:.0f}ns"


def _pct(part, whole):
    return f"{100 * part / whole:.1f}%" if whole else "-"


def _short(name, n=60):
    return name if len(name) <= n else name[: n - 1] + "…"


def label(p):
    if p.rank is not None:
        return f"rank {p.rank}"
    return os.path.basename(p.path).replace("nsys.", "").replace(".sqlite", "")


def _dims(d):
    return "x".join(str(v) for v in d if v and v != 1) or "1"


# ---------------------------------------------------------------------- hints


def hints(p):
    out = []
    if p.span and 100 * p.idle / p.span > IDLE_PCT:
        s = f"GPU idle {_pct(p.idle, p.span)} of its active time ({fmt_ns(p.idle)})"
        if p.gaps:
            g = p.gaps[0]
            s += f", longest gap {fmt_ns(g.length)}"
            s += f" in NVTX '{g.nvtx}'" if g.nvtx else ""
        out += [s + ": host work, syncs or data loading starve it"]
    pageable = [c for c in p.copies if c.pageable]
    if pageable:
        nbytes = sum(c.bytes for c in pageable)
        kinds = "/".join(sorted({c.kind for c in pageable}))
        out += [
            (
                f"{sum(c.count for c in pageable)} {kinds} copies from pageable memory"
                f" ({format_bytes(nbytes)}): pinned memory is faster and can overlap"
                " (cudaMallocHost, pin_memory=True)"
            )
        ]
    for k in p.kernels:
        if k.count >= TINY_KERNEL_COUNT and k.avg < TINY_KERNEL_NS:
            out += [
                (
                    f"{k.count} launches of {_short(k.name, 40)} at {fmt_ns(k.avg)} each:"
                    " launch bound, fuse kernels or use CUDA graphs"
                )
            ]
            break
    sync = sum(c.total for c in p.api if c.name in SYNC_CALLS)
    if p.duration and 100 * sync / p.duration > SYNC_PCT:
        out += [
            f"host blocked {_pct(sync, p.duration)} of the time in synchronise calls"
        ]
    if p.nccl and 100 * p.nccl_overlap / p.nccl < OVERLAP_PCT:
        out += [
            (
                f"NCCL kernels overlap compute only {_pct(p.nccl_overlap, p.nccl)}"
                " of their time: communication is exposed"
            )
        ]
    if p.span and 100 * p.mpi_time / p.span > WAIT_PCT:
        out += [f"{_pct(p.mpi_time, p.span)} of the GPU span in MPI calls"]
    return out


def straggler(profiles):
    """(slowest rank, its compute over the median) when it is enough to matter"""
    if len(profiles) < 2:
        return None, 1.0
    med = statistics.median(p.compute for p in profiles)
    slow = max(profiles, key=lambda p: p.compute)
    ratio = slow.compute / med if med else 1.0
    return (slow if ratio > STRAGGLER else None), ratio


# ----------------------------------------------------------------------- text


def headline(p):
    if not p.span:
        return "no GPU activity captured"
    s = f"GPU busy {_pct(p.busy, p.span)} of {fmt_ns(p.span)}"
    if p.kernels:
        k = p.kernels[0]
        s += f", top kernel {_short(k.name, 40)} {_pct(k.total, p.busy)}"
    if p.wait:
        s += f", comm {fmt_ns(p.wait)}"
    return s


def detail(p, top=10):
    gpu = ", ".join(sorted(set(p.gpus))) or "no GPU info"
    lines = [
        f"nsys  {p.cmd}  ({p.path})",
        f"  {gpu}, captured {fmt_ns(p.duration)}",
        f"  {headline(p)}",
        f"  kernels {fmt_ns(p.compute)}"
        + (
            f", NCCL {fmt_ns(p.nccl)} ({_pct(p.nccl_overlap, p.nccl)} overlapped)"
            if p.nccl
            else ""
        )
        + f", copies {fmt_ns(sum(c.total for c in p.copies))}, idle {fmt_ns(p.idle)}",
    ]
    if p.kernels:
        lines += ["  kernels, by GPU time:"]
        lines += [f"    {'time':>8} {'%':>6} {'count':>7} {'avg':>8} {'max':>8}  name"]
        for k in p.kernels[:top]:
            lines += [
                (
                    f"    {fmt_ns(k.total):>8} {_pct(k.total, p.busy):>6} {k.count:>7}"
                    f" {fmt_ns(k.avg):>8} {fmt_ns(k.max):>8}  {_short(k.name)}"
                    f"  [grid {_dims(k.grid)} block {_dims(k.block)} regs {k.regs}]"
                )
            ]
        if len(p.kernels) > top:
            rest = p.kernels[top:]
            lines += [f"    {fmt_ns(sum(k.total for k in rest)):>8}  {len(rest)} more"]
    if p.copies:
        lines += ["  copies:"]
        for c in p.copies:
            mem = " pageable" if c.pageable else ""
            lines += [
                (
                    f"    {fmt_ns(c.total):>8} {c.count:>7}  {c.kind}{mem} {format_bytes(c.bytes)}"
                    f" at {c.gbps:.1f} GB/s"
                )
            ]
    if p.nvtx:
        lines += ["  NVTX ranges, GPU busy while open:"]
        for r in p.nvtx[:top]:
            lines += [
                (
                    f"    {fmt_ns(r.total):>8} {r.count:>7}  {_short(r.name, 40)}"
                    f"  GPU {_pct(r.gpu, r.total)}"
                )
            ]
    if p.gaps:
        lines += ["  longest GPU gaps: " + ", ".join(
            fmt_ns(g.length) + (f" in '{g.nvtx}'" if g.nvtx else "") for g in p.gaps
        )]  # fmt: skip
    for title, calls in (("CUDA API", p.api), ("MPI", p.mpi)):
        if calls:
            lines += [f"  {title}, host time: " + ", ".join(
                f"{c.name} {fmt_ns(c.total)} ({c.count})" for c in calls[:5]
            )]  # fmt: skip
    if h := hints(p):
        lines += ["  hints:"] + [f"    - {s}" for s in h]
    return lines


def ranks_table(profiles):
    lines = [
        (
            f"  {'':>8} {'captured':>9} {'gpu busy':>9} {'kernels':>9} {'nccl':>8}"
            f" {'mpi':>8} {'idle':>8}"
        )
    ]
    for p in profiles:
        lines += [
            (
                f"  {label(p):>8} {fmt_ns(p.duration):>9} {_pct(p.busy, p.span):>9}"
                f" {fmt_ns(p.compute):>9} {fmt_ns(p.nccl):>8} {fmt_ns(p.mpi_time):>8}"
                f" {fmt_ns(p.idle):>8}"
            )
        ]
    return lines


def balance(profiles):
    slow, ratio = straggler(profiles)
    if not slow:
        return [f"  balanced: slowest rank's kernels {ratio:.2f}x the median"]
    others = [p for p in profiles if p is not slow]
    wait = statistics.mean(p.wait for p in others)
    fastest = min(p.compute for p in profiles)
    vs_fast = f", {slow.compute / fastest:.2f}x the fastest" if fastest else ""
    return [
        (
            f"  straggler: {label(slow)}, kernels {ratio:.2f}x the median rank's{vs_fast};"
            f" the others average {fmt_ns(wait)} in MPI/NCCL vs its {fmt_ns(slow.wait)}"
        )
    ]


def text(profiles, top=10, all=False):
    profiles = nsys.by_rank(profiles)
    if len(profiles) == 1:
        return "\n".join(detail(profiles[0], top))
    slow, _ = straggler(profiles)
    worst = slow or max(profiles, key=lambda p: p.duration)
    lines = [f"nsys: {len(profiles)} processes"] + ranks_table(profiles)
    lines += balance(profiles) + [""]
    for p in profiles if all else [worst]:
        lines += detail(p, top) + [""]
    return "\n".join(lines).rstrip()


def _delta(a, b):
    return f"{100 * (b - a) / a:+.1f}%" if a else "new"


def compare(a, b, names=("A", "B"), top=10):
    """two sets of profiles, eg master vs a branch, per process means"""

    def mean(ps, f):
        return statistics.mean(f(p) for p in ps)

    na, nb = names
    rows = [
        ("captured", lambda p: p.duration),
        ("GPU span", lambda p: p.span),
        ("GPU busy", lambda p: p.busy),
        ("kernels", lambda p: p.compute),
        ("NCCL", lambda p: p.nccl),
        ("MPI", lambda p: p.mpi_time),
        ("idle", lambda p: p.idle),
    ]
    lines = [f"nsys compare {nb} vs {na} (means per process)"]
    lines += [f"  {'':>10} {na[:10]:>10} {nb[:10]:>10} {'change':>8}"]
    for title, f in rows:
        va, vb = mean(a, f), mean(b, f)
        if va or vb:
            lines += [
                f"  {title:>10} {fmt_ns(va):>10} {fmt_ns(vb):>10} {_delta(va, vb):>8}"
            ]
    ka, kb = nsys.merged_kernels(a), nsys.merged_kernels(b)
    names_ = sorted(set(ka) | set(kb), key=lambda n: -abs(kb.get(n, 0) - ka.get(n, 0)))
    if names_:
        lines += ["  kernels, largest changes:"]
        for n in names_[:top]:
            va, vb = ka.get(n, 0), kb.get(n, 0)
            lines += [
                f"    {fmt_ns(va):>10} {fmt_ns(vb):>10} {_delta(va, vb):>8}  {_short(n)}"
            ]
    return "\n".join(lines)


# ------------------------------------------------------------------- markdown


def _md_row(*cells):
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def _md_table(head, rows):
    return [_md_row(*head), _md_row(*["---"] * len(head))] + [_md_row(*r) for r in rows]


def _md_label(s):  # mermaid labels: no quotes or angle brackets
    return "".join(c for c in s if c not in "\"'<>`")


def markdown(profiles, top=10, compare_to=None, names=("A", "B")):
    if compare_to is not None:
        return "```\n" + compare(profiles, compare_to, names, top) + "\n```"
    profiles = nsys.by_rank(profiles)
    out = []
    if len(profiles) > 1:
        out += [f"### nsys: {len(profiles)} processes", ""]
        out += _md_table(
            ("", "captured", "GPU busy", "kernels", "NCCL", "MPI", "idle"),
            [
                (label(p), fmt_ns(p.duration), _pct(p.busy, p.span), fmt_ns(p.compute),
                 fmt_ns(p.nccl), fmt_ns(p.mpi_time), fmt_ns(p.idle))
                for p in profiles
            ],
        )  # fmt: skip
        out += ["", balance(profiles)[0].strip(), ""]
        ranks = ", ".join(_md_label(label(p)) for p in profiles)
        out += [
            "```mermaid",
            "xychart-beta",
            '  title "kernels (first bar) vs MPI/NCCL (second bar), ms"',
            f"  x-axis [{ranks}]",
            '  y-axis "ms"',
            f"  bar [{', '.join(f'{p.compute / 1e6:.2f}' for p in profiles)}]",
            f"  bar [{', '.join(f'{p.wait / 1e6:.2f}' for p in profiles)}]",
            "```",
            "",
        ]
        slow, _ = straggler(profiles)
        profiles = [slow or max(profiles, key=lambda p: p.duration)]
    p = profiles[0]
    out += [f"### nsys {label(p)}: {headline(p)}", ""]
    if p.kernels:
        out += _md_table(
            ("time", "% busy", "count", "avg", "kernel"),
            [
                (fmt_ns(k.total), _pct(k.total, p.busy), k.count, fmt_ns(k.avg),
                 f"`{_short(k.name)}`")
                for k in p.kernels[:top]
            ],
        ) + [""]  # fmt: skip
        out += ["```mermaid", "pie title GPU time"]
        out += [
            f'  "{_md_label(_short(k.name, 30))}" : {k.total / 1e6:.3f}'
            for k in p.kernels[:6]
        ]
        rest = sum(k.total for k in p.kernels[6:]) + sum(c.total for c in p.copies)
        out += [f'  "copies and the rest" : {rest / 1e6:.3f}'] if rest else []
        out += ["```", ""]
    if p.nvtx:
        out += _md_table(
            ("NVTX range", "time", "count", "GPU busy while open"),
            [
                (r.name, fmt_ns(r.total), r.count, _pct(r.gpu, r.total))
                for r in p.nvtx[:top]
            ],
        ) + [""]
    if h := hints(p):
        out += ["**hints**", ""] + [f"- {s}" for s in h]
    return "\n".join(out).rstrip()
