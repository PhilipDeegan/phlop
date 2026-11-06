# phlop/app/valgrind/report.py
#
# short reports of valgrind results, the answer first: text for the terminal,
# markdown (tables, mermaid charts) to paste into a PR, plots via matplotlib

import os
import re
from collections import defaultdict

from phlop.app.valgrind import massif as ms
from phlop.app.valgrind.common import format_bytes, is_std
from phlop.app.valgrind.memcheck import LEAK_KINDS, Memcheck

TIME_UNITS = {"i": "instructions", "B": "bytes allocated", "ms": "ms"}
_LABEL = re.compile(r"\.(p|rank)(\d+)\.\w+$")


def label(path):
    """pid 1234 / rank 3 from phlop's file names, else the file name"""
    if m := _LABEL.search(path):
        return f"{'pid' if m.group(1) == 'p' else 'rank'} {m.group(2)}"
    return os.path.basename(path)


def _mib(b):
    return f"{b / 2**20:.1f}"


def _pct(part, whole):
    return f"{100 * part / whole:.1f}%" if whole else ""


def _when(m, snap):
    unit = TIME_UNITS.get(m.time_unit, m.time_unit)
    end = m.snapshots[-1].time
    if m.time_unit == "ms":
        return f"at {snap.time:.0f}ms of {end:.0f}ms"
    return f"at {snap.time:.3g} of {end:.3g} {unit}"


def _chain(frames):
    return " ← ".join(str(f) for f in frames)


def _caller_lines(site, top=3, depth=3):
    """where an allocation site was called from, largest first"""
    out = []
    for c in site.callers[:top]:
        if c.frame is None:
            out.append((c.bytes, f"{c.below} callers below the threshold"))
        else:
            out.append((c.bytes, _chain([c.frame] + ms.chain(c, depth - 1))))
    return out


def _places(n):
    return f"{n} place{'s' if n != 1 else ''}"


def _below_hint(m):
    t = m.threshold
    return f"below massif's threshold of {t:g}%, add -e=--threshold={t / 10:g} for more"


# --------------------------------------------------------------------- massif


def massif_headline(m):
    peak, tree = m.peak, m.peak_tree
    s = f"peak {format_bytes(peak.total)} {_when(m, peak)}"
    sites = ms.sites(tree.tree) if tree else []
    if sites and sites[0].frame:
        s += f", {_pct(sites[0].bytes, tree.heap)} from {sites[0].frame}"
    return s


def massif_detail(m, top=5):
    peak, tree = m.peak, m.peak_tree
    lines = [f"massif  {m.cmd}  ({m.path})", f"  {massif_headline(m)}"]
    parts = [f"heap {format_bytes(peak.heap)}", f"allocator {format_bytes(peak.extra)}"]
    if peak.stacks:
        parts += [f"stacks {format_bytes(peak.stacks)}"]
    lines += ["  " + ", ".join(parts)]
    if not tree:
        return lines + ["  no detailed snapshot, nothing allocated?"]
    if tree.kind != "peak" and tree is not peak:
        lines += [f"  allocation sites from the snapshot {_when(m, tree)}"]
    lines += ["  at the peak, by allocation site (C++ runtime frames collapsed):"]
    for site in ms.sites(tree.tree)[:top] + [
        s for s in ms.sites(tree.tree)[top:] if s.frame is None
    ]:
        head = f"    {format_bytes(site.bytes):>9} {_pct(site.bytes, tree.heap):>6}  "
        if site.frame is None:
            lines += [head + f"{_places(site.below)} {_below_hint(m)}"]
            continue
        lines += [head + str(site.frame)]
        for nbytes, chain in _caller_lines(site):
            lines += [f"{'':20}{format_bytes(nbytes):>9} from {chain}"]
    return lines


def massif_text(results, top=5, all=False):
    if len(results) == 1:
        return "\n".join(massif_detail(results[0], top))
    by_peak = sorted(results, key=lambda m: m.peak.total)
    worst = by_peak[-1]
    lines = [
        f"massif: {len(results)} processes, peak per process (sorted) "
        + " · ".join(_mib(m.peak.total) for m in by_peak)
        + f" MiB, worst {label(worst.path)}",
    ]
    for m in reversed(by_peak):
        lines += [f"  {label(m.path):>12}  {massif_headline(m)}"]
    lines += [""]
    for m in reversed(by_peak) if all else [worst]:
        lines += massif_detail(m, top) + [""]
    return "\n".join(lines).rstrip()


def _frames_at(tree):
    """function -> bytes, the most it holds anywhere in a tree"""
    out = defaultdict(int)

    def walk(n):
        if n.frame and not is_std(n.frame):
            out[n.frame] = max(out[n.frame], n.bytes)
        for c in n.children:
            walk(c)

    walk(tree)
    return out


def massif_compare(a, b, a_name="A", b_name="B", top=5):
    """two sets of massif files, eg master vs a branch"""
    wa, wb = (max(x, key=lambda m: m.peak.total) for x in (a, b))
    diff = wb.peak.total - wa.peak.total
    rel = f" ({_pct(diff, wa.peak.total)})" if wa.peak.total else ""
    lines = [
        f"massif compare: worst peak {b_name} - {a_name} = {format_bytes(diff, sign=True)}{rel}",
        "  peak per process (sorted), MiB",
    ]
    width = max(len(a_name), len(b_name))
    for name, x in ((a_name, a), (b_name, b)):
        peaks = sorted(m.peak.total for m in x)
        lines += [f"    {name:<{width}}  " + " · ".join(_mib(p) for p in peaks)]
    for (name, w), (other, x) in (
        ((a_name, wa), (b_name, b)),
        ((b_name, wb), (a_name, a)),
    ):
        seen = set().union(*(m.frames() for m in x))
        tree = w.peak_tree
        only = (
            [(f, n) for f, n in _frames_at(tree.tree).items() if f.fn not in seen]
            if tree
            else []
        )
        only.sort(key=lambda p: -p[1])
        if only:
            lines += [f"  at {name}'s worst peak, in no {other} profile:"]
            lines += [f"    {format_bytes(n):>9}  {f}" for f, n in only[:top]]
    return "\n".join(lines)


# ------------------------------------------------------------------- memcheck


def _merge(errors):
    """same kind at the same place: one line, counts and bytes summed"""
    out = {}
    for e in errors:
        key = (e.kind, e.what if not e.leak else "", e.site)
        if key in out:
            out[key].count += e.count
            out[key].bytes += e.bytes
            out[key].blocks += e.blocks
        else:
            out[key] = type(e)(**vars(e))
    return sorted(out.values(), key=lambda e: (-e.bytes, -e.count))


def memcheck_headline(m):
    n = sum(e.count for e in m.problems)
    leaks = [
        f"{LEAK_KINDS[k]} {format_bytes(m.leaked(k))}"
        for k in LEAK_KINDS
        if m.leaked(k) and k != "Leak_StillReachable"
    ]
    if not n and not leaks:
        return "clean: no errors, no leaks"
    if not n:
        return ", ".join(leaks) + ", no other errors"
    return f"{n} error{'s' if n != 1 else ''}, " + (
        ", ".join(leaks) if leaks else "no leaks"
    )


def _site_chain(stack, depth=3):
    frames = [f for f in stack if not is_std(f) and "vg_replace" not in f.file]
    return _chain(frames[:depth]) if frames else _chain(stack[:depth])


def memcheck_detail(m, top=10):
    lines = [f"memcheck  {m.cmd}  ({m.path})", f"  {memcheck_headline(m)}"]
    problems, leaks = _merge(m.problems), _merge(m.leaks)
    if problems:
        lines += ["  errors:"]
        for e in problems[:top]:
            lines += [
                f"    {e.count:>5}x  {e.what}",
                f"{'':12}at {_site_chain(e.stack)}",
            ]
            if e.aux and e.aux_stack:
                lines += [f"{'':12}{e.aux}: {_site_chain(e.aux_stack)}"]
    if leaks:
        lines += ["  leaks:"]
        for e in leaks[:top]:
            blocks = f"{e.blocks} block{'s' if e.blocks != 1 else ''}"
            what = LEAK_KINDS.get(e.kind, e.kind)
            lines += [
                f"    {format_bytes(e.bytes):>9} {what} in {blocks}",
                f"{'':12}from {_site_chain(e.stack)}",
            ]
    more = len(problems) + len(leaks) - min(len(problems), top) - min(len(leaks), top)
    if more > 0:
        lines += [f"  {more} more, see --top"]
    return lines


def memcheck_text(results, top=10, all=False):
    if len(results) == 1:
        return "\n".join(memcheck_detail(results[0], top))
    lines = [f"memcheck: {len(results)} processes"]
    for m in results:
        lines += [f"  {label(m.path):>12}  {memcheck_headline(m)}"]
    for m in results:
        if all or m.errors:
            lines += [""] + memcheck_detail(m, top)
    return "\n".join(lines)


def memcheck_compare(a, b, a_name="A", b_name="B", top=10):
    def keys(x):
        return {(e.kind, e.site) for m in x for e in m.errors}

    ka, kb = keys(a), keys(b)
    lines = [
        f"memcheck compare: {len(ka)} kinds/sites in {a_name}, {len(kb)} in {b_name}"
    ]
    for name, only in ((a_name, ka - kb), (b_name, kb - ka)):
        if only:
            lines += [f"  only in {name}:"]
            lines += [f"    {k} at {s}" for k, s in sorted(only, key=str)[:top]]
    if ka == kb:
        lines += ["  the same errors and leaks in both"]
    return "\n".join(lines)


# ------------------------------------------------------------------- markdown


def _md_label(s):  # mermaid labels: no quotes or angle brackets
    return str(s).replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")


def _palette():
    palette = ", ".join(COLORS)
    return (
        '%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "'
        + palette
        + '"}}}}%%'
    )


def _legend(compare, names):
    return f" (brown: {names[0]}, teal: {names[1]})" if compare else ""


def _resample(m, t0=0, t1=None, n=40):
    """mermaid line charts need evenly spaced points: memory at each"""
    t1 = m.snapshots[-1].time if t1 is None else t1
    out, i = [], 0
    for k in range(n + 1):
        t = t0 + (t1 - t0) * k / n
        while i + 1 < len(m.snapshots) and m.snapshots[i + 1].time <= t:
            i += 1
        out.append(m.snapshots[i].total)
    return out


def massif_markdown(results, top=5, compare=None, names=("A", "B")):
    md = []
    sets = [results] + ([compare] if compare else [])
    worst = [max(x, key=lambda m: m.peak.total) for x in sets]
    if compare:
        md += [
            f"**{massif_compare(results, compare, *names, top).splitlines()[0]}**",
            "",
        ]
    else:
        md += [f"**{massif_headline(worst[0])}**", ""]

    if compare or len(results) > 1:
        rows = max(len(x) for x in sets)
        md += [
            "| " + " | ".join(["process (sorted)"] + list(names[: len(sets)])) + " |"
        ]
        md += ["|---" * (len(sets) + 1) + "|"]
        for i in range(rows):
            cells = []
            for x in sets:
                peaks = sorted(m.peak.total for m in x)
                cells += [f"{_mib(peaks[i])} MiB" if i < len(peaks) else ""]
            md += [f"| {i} | " + " | ".join(cells) + " |"]
        md += ["", "```mermaid", _palette(), "xychart-beta"]
        md += [f'  title "Peak memory per process, MiB{_legend(compare, names)}"']
        md += ["  x-axis [" + ", ".join(f'"{i}"' for i in range(rows)) + "]"]
        md += ['  y-axis "MiB"']
        for x in sets:
            peaks = sorted(m.peak.total for m in x)
            md += ["  bar [" + ", ".join(_mib(p) for p in peaks) + "]"]
        md += ["```", ""]

    m = worst[0]
    unit = TIME_UNITS.get(m.time_unit, m.time_unit)
    t0, t1 = _start(sets), max(w.snapshots[-1].time for w in worst)
    md += ["```mermaid", _palette(), "xychart-beta"]
    title = "worst process" if len(results) > 1 or compare else label(m.path)
    md += [
        f'  title "Memory over time, {_md_label(title)}, MiB{_legend(compare, names)}"'
    ]
    md += [f'  x-axis "{unit}" {t0:.0f} --> {t1:.0f}', '  y-axis "MiB"']
    for w in worst:
        md += ["  line [" + ", ".join(_mib(v) for v in _resample(w, t0, t1)) + "]"]
    md += ["```", ""]

    for name, w in zip(names, worst):
        tree = w.peak_tree
        if not tree:
            continue
        sites = ms.sites(tree.tree)
        title = f"{name}: " if compare else ""
        md += [
            f"{title}at the peak of {label(w.path)}, {format_bytes(tree.heap)} heap",
            "",
        ]
        md += ["| bytes | % | allocation site | called from |", "|---:|---:|---|---|"]
        for s in sites[:top]:
            if s.frame is None:
                where = f"{_places(s.below)} {_below_hint(w)}"
                md += [
                    f"| {format_bytes(s.bytes)} | {_pct(s.bytes, tree.heap)} | {where} | |"
                ]
                continue
            callers = "<br>".join(
                f"{format_bytes(n)} {c}" for n, c in _caller_lines(s, depth=2)
            )
            pct = _pct(s.bytes, tree.heap)
            md += [f"| {format_bytes(s.bytes)} | {pct} | `{s.frame}` | {callers} |"]
        top_site = next((s for s in sites if s.frame), None)
        if top_site and top_site.callers:
            path = [top_site.frame, top_site.callers[0].frame]
            path += ms.chain(top_site.callers[0], 4)
            path = [f for f in path if f]
            md += ["", "```mermaid", "flowchart BT"]
            for i, f in enumerate(path):
                md += [f'  n{i}["{_md_label(f.fn)}<br/>{_md_label(f.where())}"]']
            md += [f"  n{i + 1} --> n{i}" for i in range(len(path) - 1)]
            md += ["  style n0 fill:#f4eadf,stroke:#c0803f", "```"]
        md += [""]
    if compare:
        only = massif_compare(results, compare, *names, top).splitlines()
        only = (
            only[next(i for i, l in enumerate(only) if l.startswith("  at ")) :]
            if any(l.startswith("  at ") for l in only)
            else []
        )
        if only:
            md += ["```", *only, "```"]
    return "\n".join(md)


def memcheck_markdown(results, top=10, compare=None, names=("A", "B")):
    if compare:
        return "```\n" + memcheck_compare(results, compare, *names, top) + "\n```"
    md = []
    for m in results:
        md += [f"**{label(m.path)}: {memcheck_headline(m)}**", ""]
        errs = _merge(m.problems)[:top] + _merge(m.leaks)[:top]
        if errs:
            md += ["| count / bytes | what | where |", "|---:|---|---|"]
            for e in errs:
                n = format_bytes(e.bytes) if e.leak else f"{e.count}x"
                what = LEAK_KINDS.get(e.kind, e.what)
                md += [f"| {n} | {what} | `{_site_chain(e.stack)}` |"]
            md += [""]
    return "\n".join(md)


# ----------------------------------------------------------------------- plot


COLORS = ("#c0803f", "#0d6b73")  # A, B: brown, teal


def _start(sets):
    """skip valgrind's startup: from just before the first allocation"""
    first = [next((s.time for s in m.snapshots if s.total), 0) for x in sets for m in x]
    end = max(m.snapshots[-1].time for x in sets for m in x)
    return max(0, min(first) - 0.02 * end)


def _top_sites(sets, top):
    """per set, site name -> bytes at its worst peak, and the union's order"""
    per_set, order = [], {}
    for x in sets:
        tree = max(x, key=lambda m: m.peak.total).peak_tree
        sites = {}
        for s in ms.sites(tree.tree)[:top] if tree else []:
            name = str(s.frame) if s.frame else "below threshold"
            sites[name] = s.bytes
            order[name] = max(order.get(name, 0), s.bytes)
        per_set.append(sites)
    return per_set, sorted(order, key=lambda n: order[n])


def massif_plot(path, results, top=5, compare=None, names=("A", "B")):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("--plot needs matplotlib: pip install matplotlib") from None

    sets = [results] + ([compare] if compare else [])
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 2]}
    )
    for k, x in enumerate(sets):
        by_peak = sorted(x, key=lambda m: -m.peak.total)
        for i, m in enumerate(by_peak):  # the worst process bold, others faint
            t = [s.time for s in m.snapshots]
            v = [s.total / 2**20 for s in m.snapshots]
            name = names[k] if compare else "process"
            ax.step(
                t,
                v,
                where="post",
                color=COLORS[k],
                lw=2 if i == 0 else 0.8,
                alpha=1 if i == 0 else 0.4,
                label=f"{name}, worst: {label(m.path)}" if i == 0 else None,
            )
        p = by_peak[0].peak
        ax.plot([p.time], [p.total / 2**20], "o", color=COLORS[k])
        ax.annotate(
            format_bytes(p.total),
            (p.time, p.total / 2**20),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    m0 = sets[0][0]
    ax.set_xlim(left=_start(sets))
    ax.set_xlabel(TIME_UNITS.get(m0.time_unit, m0.time_unit))
    ax.set_ylabel("MiB")
    n = sum(len(x) for x in sets)
    ax.set_title(
        f"memory over time (heap + allocator), {n} process{'es' if n > 1 else ''}"
    )
    ax.legend(fontsize=8)

    per_set, order = _top_sites(sets, top)
    height = 0.8 / len(sets)
    for k, sites in enumerate(per_set):
        y = [i + (len(sets) - 1 - k) * height for i in range(len(order))]
        bx.barh(
            y,
            [sites.get(o, 0) / 2**20 for o in order],
            height=height,
            color=COLORS[k],
            label=names[k] if compare else None,
        )
    bx.set_yticks([i + height * (len(sets) - 1) / 2 for i in range(len(order))])
    bx.set_yticklabels(
        [o if len(o) <= 50 else o[:47] + "..." for o in order], fontsize=7
    )
    bx.set_xlabel("MiB")
    bx.set_title("at the worst peak, by allocation site")
    if compare:
        bx.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- entry


def kind(results):
    kinds = {type(r) for r in results}
    if len(kinds) > 1:
        raise ValueError("mixed massif and memcheck files, report them separately")
    return "memcheck" if kinds == {Memcheck} else "massif"


def text(results, top=None, all=False, compare=None, names=("A", "B")):
    if kind(results) == "massif":
        top = top or 5
        if compare:
            return massif_compare(results, compare, *names, top)
        return massif_text(results, top, all)
    top = top or 10
    if compare:
        return memcheck_compare(results, compare, *names, top)
    return memcheck_text(results, top, all)


def markdown(results, top=None, compare=None, names=("A", "B")):
    if kind(results) == "massif":
        return massif_markdown(results, top or 5, compare, names)
    return memcheck_markdown(results, top or 10, compare, names)


def plot(path, results, top=None, compare=None, names=("A", "B")):
    if kind(results) != "massif":
        raise ValueError("--plot is for massif results")
    return massif_plot(path, results, top or 5, compare, names)
