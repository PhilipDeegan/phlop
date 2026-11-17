# phlop/app/valgrind/__init__.py
#
# run a command under a valgrind tool with useful defaults, keep valgrind's
# files under .phlop/valgrind, and load them back for reports (report.py)
#
# files: <outdir>/<tool>.<run id>.<label>.<ext>, label p<pid> or rank<N>
#  memcheck: .xml (parsed), .log (valgrind's text)   massif: .out, .log
#  dhat: .json   callgrind/cachegrind: .out   others: .log

import glob
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from phlop.app.valgrind import massif, memcheck

OUTDIR = Path(".phlop") / "valgrind"
ERROR_EXITCODE = 99  # memcheck found errors or leaks, else the program's own


def tool_options(tool, base):
    """valgrind's options for a tool, base is the output path minus extension"""
    log = f"--log-file={base}.log"
    if tool == "memcheck":
        return [
            "--leak-check=full",
            "--track-origins=yes",
            f"--error-exitcode={ERROR_EXITCODE}",
            "--xml=yes",
            f"--xml-file={base}.xml",
            log,
        ]
    if tool == "massif":
        return ["--time-unit=ms", f"--massif-out-file={base}.out", log]
    if tool == "dhat":
        return [f"--dhat-out-file={base}.json", log]
    if tool in ("callgrind", "cachegrind"):
        return [f"--{tool}-out-file={base}.out", log]
    return [log]


def new_run_id(outdir=OUTDIR):
    run_id = time.strftime("%Y%m%d-%H%M%S")
    n, out = 0, run_id
    while glob.glob(str(Path(outdir) / f"*.{out}.*")):
        n += 1
        out = f"{run_id}-{n}"
    return out


def command(tool, cmd, extra=(), outdir=OUTDIR, run_id=None, label="p%p"):
    """the valgrind command line, and the files' base path. %p is the pid,
    expanded by valgrind. extra comes last, so it can override the defaults"""
    base = Path(outdir) / f"{tool}.{run_id or new_run_id(outdir)}.{label}"
    return ["valgrind", f"--tool={tool}", *tool_options(tool, base), *extra, *cmd], base


def run(tool, cmd, extra=(), outdir=OUTDIR, quiet=False, run_id=None, label="p%p"):
    """run cmd under valgrind, returns (exit code, run id). The program's
    output is shown unless quiet, valgrind's own goes to the .log files"""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id(outdir)
    vg, _ = command(tool, cmd, extra, outdir, run_id, label)
    print(f"running: {shlex.join(vg)}", file=sys.stderr)
    pipe = subprocess.DEVNULL if quiet else None
    try:
        rc = subprocess.run(vg, stdout=pipe, stderr=pipe, check=False).returncode
    except FileNotFoundError:
        raise RuntimeError("valgrind not found on PATH") from None
    return rc, run_id


# ------------------------------------------------------------------- results


def is_massif(path):
    try:
        with open(path) as f:
            return any(f.readline().startswith("time_unit:") for _ in range(3))
    except (OSError, UnicodeDecodeError):
        return False


def is_result(path):
    return (path.endswith(".out") and is_massif(path)) or (
        path.endswith(".xml") and memcheck.is_memcheck_xml(path)
    )


def load(path):
    return massif.parse(path) if is_massif(path) else memcheck.parse(path)


def _run_id(path):
    parts = os.path.basename(path).split(".")
    return parts[1] if len(parts) >= 4 else None


def _latest(directory):
    files = [f for f in glob.glob(str(Path(directory) / "*.*")) if is_result(f)]
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
        files = [spec]
    elif glob.has_magic(spec):
        files = sorted(f for f in glob.glob(spec) if is_result(f))
    else:
        files = sorted(
            f for f in glob.glob(str(Path(outdir) / f"*.{spec}.*")) if is_result(f)
        )
    if not files:
        raise FileNotFoundError(f"no massif/memcheck results for {spec or outdir}")
    return files


def results(spec=None, outdir=OUTDIR):
    return [load(f) for f in resolve(spec, outdir)]
