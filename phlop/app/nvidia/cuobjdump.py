# phlop/app/nvidia/cuobjdump.py
#
# per kernel resource use of built binaries, from cuobjdump: registers, stack,
# shared and local memory, and the largest block size each kernel can launch
# with. No GPU needed: it reads the binary
#
#   python3 -m phlop.app.nvidia.cuobjdump bin/app                 # most registers first
#   python3 -m phlop.app.nvidia.cuobjdump -t 512 bin/app          # kernels that can't launch at 512
#   python3 -m phlop.app.nvidia.cuobjdump dbg=bin/a rel=bin/b     # compare builds
#
# registers are allocated per warp, rounded up to an allocation unit, out of a
# per block budget: a kernel using R registers per thread launches at most
#   floor(regs_per_block / (ceil(R * 32 / unit) * unit)) * 32 threads
# capped at the block maximum. Launching more fails with "too many resources
# requested for launch". Defaults (65536, 256, 1024) hold for current GPUs,
# the options override them
#
# LOCAL > 0 is per thread local memory: register spills or local arrays,
# slow, worth a look in -Xptxas -v
#
# https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

WARP = 32
REGS_PER_BLOCK = 65536
ALLOC_UNIT = 256
MAX_BLOCK = 1024

ARCH = re.compile(r"arch = (sm_\d+\w*)")
FUNCTION = re.compile(r"Function (\S+):")
RESOURCES = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")


@dataclass
class Kernel:
    name: str  # demangled
    mangled: str
    arch: str
    reg: int
    stack: int
    shared: int
    local: int


def max_threads(
    reg, regs_per_block=REGS_PER_BLOCK, unit=ALLOC_UNIT, max_block=MAX_BLOCK
):
    """the largest block a kernel using reg registers per thread can launch"""
    if reg == 0:
        return max_block
    per_warp = -(-reg * WARP // unit) * unit  # rounded up to the unit
    return min(max_block, regs_per_block // per_warp * WARP)


def pow2_floor(n):
    return 1 << (n.bit_length() - 1) if n > 0 else 0


def demangle(names):
    """llvm-cxxfilt when there: binutils c++filt leaves C++20 manglings, eg
    auto template parameters, as they are"""
    if not names:
        return {}
    exe = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if not exe:
        return {n: n for n in names}
    out = subprocess.run(
        [exe], input="\n".join(names), capture_output=True, text=True, check=False
    ).stdout.splitlines()
    return dict(zip(names, out)) if len(out) == len(names) else {n: n for n in names}


def parse(text):
    """cuobjdump --dump-resource-usage output -> [Kernel], names mangled"""
    kernels, arch, fn = [], "?", None
    for line in text.splitlines():
        if m := ARCH.search(line):
            arch = m.group(1)
        elif m := FUNCTION.search(line):
            fn = m.group(1)
        elif fn and (m := RESOURCES.search(line)):
            kernels.append(Kernel(fn, fn, arch, *map(int, m.groups())))
            fn = None
    return kernels


def kernels(binary):
    """the binary's kernels, demangled, most registers first"""
    exe = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    try:
        res = subprocess.run(
            [exe, "--dump-resource-usage", binary],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, PermissionError):
        raise RuntimeError(
            "cuobjdump not found on PATH or in /usr/local/cuda"
        ) from None
    if res.returncode != 0:
        msg = (res.stderr or res.stdout).strip().replace("cuobjdump info    : ", "")
        raise RuntimeError(f"cuobjdump failed on {binary}: {msg}")
    out = parse(res.stdout)
    names = demangle(sorted({k.mangled for k in out}))
    for k in out:
        k.name = names.get(k.mangled, k.mangled)
    return sorted(out, key=lambda k: (-k.reg, k.name, k.arch))


def shorten(name, depth):
    """template arguments nested deeper than depth as <…>"""
    out, level = [], 0
    for c in name:
        if c == "<":
            level += 1
            if level == depth + 1:
                out.append("<…>")
            if level > depth:
                continue
        elif c == ">":
            level -= 1
            if level >= depth:
                continue
        elif level > depth:
            continue
        out.append(c)
    return "".join(out)


def cli_args_parser():
    p = argparse.ArgumentParser(
        description="per kernel registers, memory and largest block size, from cuobjdump",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("binaries", nargs="+", help="path, or label=path to compare builds")
    p.add_argument("-t", "--threads", type=int,
                   help="only kernels that can't launch with this many threads per block")  # fmt: skip
    p.add_argument(
        "-n", "--top", type=int, default=20, help="rows per binary, 0 for all"
    )
    p.add_argument("-d", "--depth", type=int, default=1, help="template depth shown")
    p.add_argument("-f", "--filter", default="", help="regex on the demangled name")
    p.add_argument("--regs-per-block", type=int, default=REGS_PER_BLOCK)
    p.add_argument("--alloc-unit", type=int, default=ALLOC_UNIT,
                   help="registers are allocated per warp in multiples of this")  # fmt: skip
    p.add_argument("--max-block", type=int, default=MAX_BLOCK)
    return p


def report(cli_args):
    limit = lambda reg: max_threads(
        reg, cli_args.regs_per_block, cli_args.alloc_unit, cli_args.max_block
    )
    flt = re.compile(cli_args.filter) if cli_args.filter else None
    lines, summary = [], []
    for spec in cli_args.binaries:
        # label=path, the label up to the first "=": a path may hold one too
        label, eq, path = spec.partition("=")
        if not eq or os.path.exists(spec):
            label = path = spec
        ks = kernels(path)
        if flt:
            ks = [k for k in ks if flt.search(k.name)]
        worst = min((limit(k.reg) for k in ks), default=cli_args.max_block)
        summary += [(label, len(ks), max((k.reg for k in ks), default=0), worst,
                     sum(k.local > 0 for k in ks))]  # fmt: skip
        rows = (
            [k for k in ks if limit(k.reg) < cli_args.threads]
            if cli_args.threads
            else ks
        )
        lines += ["", f"== {label} ({path}): {len(ks)} kernels"]
        if cli_args.threads:
            lines[-1] += f", {len(rows)} can't launch {cli_args.threads} threads"
        lines += [
            f"{'REG':>4} {'STACK':>6} {'SHARED':>6} {'LOCAL':>6} {'MAXTHR':>6}  {'arch':<8} kernel"
        ]
        for k in rows[: cli_args.top or None]:
            lines.append(
                f"{k.reg:>4} {k.stack:>6} {k.shared:>6} {k.local:>6} {limit(k.reg):>6}"
                f"  {k.arch:<8} {shorten(k.name, cli_args.depth)}"
            )
        if cli_args.top and len(rows) > cli_args.top:
            lines += [f"  ... {len(rows) - cli_args.top} more, -n 0 for all"]
    lines += ["", "== summary: the largest block every kernel can launch with"]
    lines += [
        f"{'kernels':>8} {'maxREG':>6} {'maxthr':>6} {'pow2':>5} {'LOCAL>0':>7}  binary"
    ]
    for label, n, reg, worst, local in summary:
        lines += [
            f"{n:>8} {reg:>6} {worst:>6} {pow2_floor(worst):>5} {local:>7}  {label}"
        ]
    return "\n".join(lines).lstrip()


def main():
    parser = cli_args_parser()
    cli_args = parser.parse_args()
    if cli_args.top < 0:
        parser.error("-n/--top: 0 or more")
    if cli_args.threads is not None and cli_args.threads <= 0:
        parser.error("-t/--threads: 1 or more")
    try:
        print(report(cli_args))
    except (RuntimeError, OSError) as e:
        print(f"phlop cuobjdump: {e}", file=sys.stderr)
        sys.exit(1)
    except re.error as e:
        print(f"phlop cuobjdump: --filter: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
