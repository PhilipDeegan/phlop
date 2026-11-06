# phlop/app/nvidia/csan_report.py
#
# short compute-sanitizer reports, the answer first: grouped errors with the
# kernel source line, an example thread and the host caller, then leaks.
# Across processes the same error is one line, with the ranks that hit it

import os

from phlop.app.nvidia import csan
from phlop.app.valgrind.common import format_bytes

TITLES = {
    "error": "memory errors",
    "race": "shared memory races",
    "sync": "barrier errors",
    "uninit": "uninitialised reads",
    "api": "CUDA API errors",
    "leak": "leaks",
    "note": "notes",
}


def _where(f):
    return str(f) if f else "?"


def _ranks(ranks):
    """0-3,7"""
    out, rs = [], sorted(ranks)
    i = 0
    while i < len(rs):
        j = i
        while j + 1 < len(rs) and rs[j + 1] == rs[j] + 1:
            j += 1
        out.append(str(rs[i]) if i == j else f"{rs[i]}-{rs[j]}")
        i = j + 1
    return ",".join(out)


def _count(n, limited):
    return f"{n}+" if limited else str(n)


def issue_line(i, limited=False, nranks=0):
    head = f"{_count(i.count, limited and i.kind != 'race'):>6}x  {i.what}"
    if i.kind == "leak":
        head = f"{format_bytes(i.bytes):>9} {i.count:>4}x  allocated"
    lines = [head]
    if i.kind == "race":
        lines += [f"{'':9}{_where(i.where)} vs {_where(i.other)}"]
    elif i.where:
        lines += [f"{'':9}at {i.where}" + (f", eg {i.example}" if i.example else "")]
    if i.caller:
        lines[
            -1 if i.kind != "leak" else 0
        ] += f"{'' if i.kind == 'leak' else ','} from {i.caller}"
    if nranks > 1 and i.ranks:
        lines[-1] += f"  [ranks {_ranks(i.ranks)}]"
    return lines


def _faulted(issues):
    return any(i.kind == "api" and "LaunchFailure" in i.what for i in issues) or any(
        i.kind == "note" and "Sticky" in i.what for i in issues
    )


def hints(results, issues):
    out = []
    kinds = {i.kind for i in issues}
    if _faulted(issues):
        out += [
            (
                "a kernel faulted: CUDA errors are sticky after that, so later API"
                " errors (and leaks from frees that failed) follow from the first error"
            )
        ]
    if "uninit" in kinds:
        out += [
            "uninitialised reads: memset/copy into the buffer before the kernel reads it"
        ]
    if "race" in kinds:
        out += ["shared memory races: a __syncthreads() between the write and the read"]
    if "sync" in kinds:
        out += [
            "barrier in divergent code: every thread of the block must reach __syncthreads()"
        ]
    if any(r.limited for r in results):
        out += [
            f"stopped recording at {csan.PRINT_LIMIT} records, counts are lower bounds"
        ]
    if not all(r.complete for r in results):
        out += [
            "a result file was cut short, the process died: these are the records written"
        ]
    if kinds - {"note"} and not any(i.where and i.where.file for i in issues):
        out += ["no source lines: build with -lineinfo (device) and -g (host)"]
    return out


def headline(results, issues):
    errors = [i for i in issues if i.kind not in ("leak", "note")]
    leaks = [i for i in issues if i.kind == "leak"]
    tools = "/".join(sorted({r.tool for r in results}))
    n = sum(i.count for i in errors)
    limited = any(r.limited for r in results)
    s = f"compute-sanitizer {tools}: "
    if not errors and not leaks:
        return s + "clean"
    parts = []
    if errors:
        where = "1 place" if len(errors) == 1 else f"{len(errors)} places"
        parts += [f"{_count(n, limited)} errors in {where}"]
    if leaks:
        parts += [
            f"{sum(i.count for i in leaks)} leaks, {format_bytes(sum(i.bytes for i in leaks))}"
        ]
    return s + ", ".join(parts)


def text(results, top=10):
    issues = csan.group(i for r in results for i in r.issues)
    limited = any(r.limited for r in results)
    lines = [headline(results, issues)]
    if len(results) > 1:
        ranks = sorted(r.rank for r in results if r.rank is not None)
        bad = sorted({k for i in issues if i.kind != "note" for k in i.ranks})
        lines += [
            f"  {len(results)} processes"
            + (f", issues on ranks {_ranks(bad)}" if ranks and bad else "")
        ]
    else:
        lines += [f"  ({results[0].path})"]
    for kind in csan.KINDS:
        of = [i for i in issues if i.kind == kind]
        if not of or kind == "note":
            continue
        lines += [f"  {TITLES[kind]}:"]
        for i in of[:top]:
            lines += ["    " + s for s in issue_line(i, limited, len(results))]
        if len(of) > top:
            lines += [f"    {len(of) - top} more"]
    if h := hints(results, issues):
        lines += ["  hints:"] + [f"    - {s}" for s in h]
    return "\n".join(lines)


def markdown(results, top=10):
    issues = csan.group(i for r in results for i in r.issues)
    limited = any(r.limited for r in results)
    out = [f"### {headline(results, issues)}", ""]
    rows = [i for i in issues if i.kind != "note"][: top * 2]
    if rows:
        multi = len(results) > 1
        head = "| kind | count | what | where | from |" + (" ranks |" if multi else "")
        out += [head, "|" + "---|" * (5 + multi)]
        for i in rows:
            where = (
                f"{_where(i.where)} vs {_where(i.other)}"
                if i.kind == "race"
                else (_where(i.where) if i.where else "")
            )
            count = (
                format_bytes(i.bytes) if i.kind == "leak" else _count(i.count, limited)
            )
            cells = [TITLES[i.kind], count, i.what, f"`{where}`" if where else "",
                     f"`{i.caller}`" if i.caller else ""]  # fmt: skip
            if multi:
                cells += [_ranks(i.ranks)]
            out += ["| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"]
        out += [""]
    if h := hints(results, issues):
        out += ["**hints**", ""] + [f"- {s}" for s in h]
    return "\n".join(out).rstrip()


def files_line(outdir, tool, run_id):
    return (
        f"files: {os.path.join(outdir, f'csan-{tool}.{run_id}')}.*  (--show {run_id})"
    )
