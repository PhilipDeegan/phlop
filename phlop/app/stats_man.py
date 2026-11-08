# phlop/app/stats_man.py

import argparse
import logging
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing import Process

import psutil
import yaml

from phlop.app.nvidia import smi
from phlop.dict import ValDict
from phlop.proc import run_raw

logger = logging.getLogger(__name__)

_default_interval = 2
_first_sample_delay = 0.1  # first cpu_percent window, later ones are `interval`
HEADERS = ["fds", "mem_used_mb", "cpu_usage"]
# per GPU, prefixed gpu<index>_: smi.Sample fields, slowdowns joined by +
GPU_HEADERS = [
    "util",
    "mem_mb",
    "proc_mem_mb",
    "power_w",
    "sm_mhz",
    "temp_c",
    "slowdown",
]
SLOWDOWN_PCT = 5  # of samples, worth naming in the summary
LOW_UTIL = 50  # % of time a kernel ran, worth naming in the summary


def columns(gpus=()):
    return HEADERS + [f"gpu{g.index}_{h}" for g in gpus for h in GPU_HEADERS]


@dataclass
class ProcessCaptureInfo:
    cpu_load: list = field(default_factory=list)
    fds: list = field(default_factory=list)
    mem_usage: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    gpu: dict = field(default_factory=dict)  # index -> [smi.Sample]

    def append(self, stats):
        self.fds.append(stats["fds"])
        self.mem_usage.append(stats["mem_used_mb"])
        self.cpu_load.append(stats["cpu_usage"])
        self.timestamps.append(datetime.now(timezone.utc).timestamp())
        for s in stats.get("gpu", []):
            self.gpu.setdefault(s.index, []).append(s)


def cli_args_parser(description=""):
    _help = ValDict(
        quiet="Redirect output to /dev/null",
        interval="Seconds between process stat capture",
        yaml="write yaml file during execution",
        summary="prints summary on end",
        gpu="sample GPUs with nvidia-smi: default when it's on PATH, only"
        " those in CUDA_VISIBLE_DEVICES when set",
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument(
        "-q", "--quiet", action="store_true", default=False, help=_help.quiet
    )
    parser.add_argument(
        "-i", "--interval", default=_default_interval, help=_help.interval
    )
    parser.add_argument("-y", "--yaml", default=None, help=_help.yaml)
    parser.add_argument(
        "-s",
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_help.summary,
    )
    parser.add_argument(
        "--gpu", action=argparse.BooleanOptionalAction, default=None, help=_help.gpu
    )
    return parser


def verify_cli_args(cli_args):
    try:
        cli_args.interval = int(cli_args.interval)
    except ValueError:
        raise ValueError("Interval must be an integer") from None
    if cli_args.yaml:
        cli_args.summary = False
    sys.argv = [sys.argv[0]]  # drop everything!
    return cli_args


def check_pid(pid):
    """Check For the existence of a unix pid."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def bytes_as_mb(n_bytes):
    return int(n_bytes / 1024**2)


def timestamp_now():
    return datetime.now(timezone.utc).isoformat()


def gpus_to_sample(flag=None):
    """smi.Gpu list: none for --no-gpu; for --gpu or by default the visible
    ones, warning when --gpu finds none"""
    if flag is False:
        return []
    gpus = smi.visible(smi.gpus())
    if flag and not gpus:
        logger.warning("phlop stats_man: --gpu, but no GPU found by nvidia-smi")
    return gpus


class _Sampler:
    """One psutil.Process for the whole run: cpu_percent(None) then measures
    the time since the previous sample rather than blocking on a short window"""

    def __init__(self, pid, gpus=()):
        self.proc = psutil.Process(pid)
        self.proc.cpu_percent(None)  # starts the first window
        self.gpus = list(gpus)
        self.gpu_failed = False

    def gpu_stats(self):
        """smi.Samples, process memory summed over the process and its
        children; empty Samples when nvidia-smi fails, warned once"""
        pids = {self.proc.pid}
        try:
            pids |= {c.pid for c in self.proc.children(recursive=True)}
        except psutil.Error:
            pass
        try:
            return smi.sample([g.index for g in self.gpus], pids)
        except (subprocess.SubprocessError, OSError) as e:
            if not self.gpu_failed:
                logger.warning(f"phlop stats_man: nvidia-smi failed: {e}")
            self.gpu_failed = True
            return [smi.Sample.failed(g.index) for g in self.gpus]

    def __call__(self):
        p = self.proc
        if p.status() == psutil.STATUS_ZOMBIE:  # exited, not yet reaped
            raise psutil.NoSuchProcess(p.pid)
        with p.oneshot():
            fds = p.num_fds() if hasattr(p, "num_fds") else p.num_handles()
            stats = {
                "fds": fds,
                "mem_used_mb": bytes_as_mb(p.memory_info().rss),
                "cpu_usage": p.cpu_percent(None),
            }
        if self.gpus:
            stats["gpu"] = self.gpu_stats()
        return stats


def now(pid, gpus=()):
    """single snapshot, cpu over a 0.1s window"""
    sampler = _Sampler(pid, gpus)
    time.sleep(_first_sample_delay)
    return sampler()


def _sample_until_gone(pid, interval, on_sample, stop, gpus=()):
    """call on_sample every `interval` seconds until pid exits or `stop` is set"""
    try:
        sampler = _Sampler(pid, gpus)
        delay = min(_first_sample_delay, interval)
        while not stop(delay):
            on_sample(sampler())
            delay = interval
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def _gpu_values(samples):
    out = []
    for s in samples:
        slow = ("+".join(s.slowdowns) or "-") if s.ok else ""
        vals = (s.util, s.mem_mb, s.proc_mem_mb, s.power_w, s.sm_mhz, s.temp_c, slow)
        out += ["" if v is None else str(v) for v in vals]
    return out


def append_yaml(file, stats):
    vals = [str(stats[key]) for key in HEADERS] + _gpu_values(stats.get("gpu", []))
    vals = ",".join(vals)
    try:
        with open(file, "a") as f:
            f.write(f"- v: {vals}\n")
    except OSError as e:
        logger.error(f"phlop detected IO error: {e}")


def init_yaml(cli_args, pid, info, gpus=()):
    file = cli_args.yaml
    if gpus:
        info = {**info, "gpus": [{"index": g.index, "name": g.name, "uuid": g.uuid,
                                  "memory_total_mb": g.memory_total} for g in gpus]}  # fmt: skip
    headers = {"headers": columns(gpus)}
    cli = {"cli_args": {"interval": cli_args.interval}}

    try:
        with open(file, "w") as f:
            f.write("---\n")
            yaml.dump(info, f, default_flow_style=False)
            yaml.dump(cli, f, default_flow_style=False)
            yaml.dump(headers, f, default_flow_style=False)
            f.write(f"start: {timestamp_now()} \n")
            f.write("snapshots:\n")
    except OSError as e:
        logger.error(f"phlop detected IO error: {e}")


def end_yaml(file):
    try:
        with open(file, "a") as f:
            f.write(f"end: {timestamp_now()} \n")
    except OSError as e:
        logger.error(f"phlop detected IO error: {e}")


class RuntimeStatsManager:
    """Runs cli_args.remaining, sampling it from a thread until it exits.
    join() waits for it and returns self, with .data if summary is on."""

    def __init__(self, cli_args, info=None):
        self.cli_args = cli_args
        self.proc = run_raw(cli_args.remaining, quiet=cli_args.quiet)
        self.pid = self.proc.pid
        self.data = ProcessCaptureInfo() if cli_args.summary else None
        self.gpus = gpus_to_sample(getattr(cli_args, "gpu", None))

        if cli_args.yaml:
            init_yaml(cli_args, self.pid, info or {}, self.gpus)

        def on_sample(stats):
            if self.data is not None:
                self.data.append(stats)
            if cli_args.yaml:
                append_yaml(cli_args.yaml, stats)

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=_sample_until_gone,
            args=(self.pid, cli_args.interval, on_sample, self._stop.wait, self.gpus),
            daemon=True,
        )
        self._thread.start()

    def __del__(self):
        if getattr(self, "pid", 0) and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            self.join()

    def join(self):
        if not self.pid:
            return self
        try:
            self.proc.wait()  # reaps it, so the sampler sees it gone
        except KeyboardInterrupt:  # the child got the SIGINT too
            self.proc.wait()
        self._stop.set()
        self._thread.join()
        if self.cli_args.yaml:
            end_yaml(self.cli_args.yaml)
        self.pid = 0
        return self


def _attached_run(pid, yaml_file, interval, gpus=()):
    """separate process, so sampling isn't held up by pid's GIL"""
    stop = threading.Event()
    # SIGINT from kill(), SIGTERM from multiprocessing ending daemons at exit
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        _sample_until_gone(
            pid, interval, lambda stats: append_yaml(yaml_file, stats), stop.wait, gpus
        )
    finally:
        end_yaml(yaml_file)


class AttachableRuntimeStatsManager:
    """Samples the calling process itself, from a separate daemon process so
    sampling is not held up by this process's GIL. start(), then kill()/join()."""

    def __init__(self, cli_args, info=None):
        self.pid = os.getpid()
        self.cli_args = cli_args
        self.gpus = gpus_to_sample(getattr(cli_args, "gpu", None))
        init_yaml(self.cli_args, self.pid, info or {}, self.gpus)
        self.data = {}
        self.p = Process(
            target=_attached_run,
            args=(self.pid, cli_args.yaml, cli_args.interval, self.gpus),
        )

    def start(self):
        # daemon: else exiting would wait on the sampler, which samples the
        # still-alive exiting process forever
        self.p.daemon = True
        self.p.start()
        return self

    def kill(self):
        if self and self.p and check_pid(self.p.pid):
            os.kill(self.p.pid, signal.SIGINT)
        return self

    def join(self):
        if not self.p.pid:
            return
        self.p.join()
        self.pid = 0
        return self


def print_summary(statsman):
    data = statsman.data
    print("summary:")
    if not data or not data.cpu_load:
        print("\tno samples, the process exited before the first one")
        return
    print("\tCPU avg:", statistics.fmean(data.cpu_load))
    print("\tCPU max:", max(data.cpu_load))

    print("\tFDS avg:", statistics.fmean(data.fds))
    print("\tFDS max:", max(data.fds))

    print("\tMEM avg:", statistics.fmean(data.mem_usage))
    print("\tMEM max:", max(data.mem_usage))
    for line in gpu_summary(data.gpu):
        print("\t" + line)


def _stat(samples, attr, fn):
    vals = [v for s in samples if (v := getattr(s, attr)) is not None]
    return fn(vals) if vals else None


def _fmt(v, unit=""):
    return "-" if v is None else f"{v:.0f}{unit}"


def gpu_summary(by_gpu):
    """per GPU: utilisation, memory, power, clocks, temperature, and the clock
    slowdowns active in more than SLOWDOWN_PCT of samples"""
    lines = []
    for index, samples in sorted(by_gpu.items()):
        mean, peak, low = statistics.fmean, max, min
        util = _stat(samples, "util", mean)
        lines.append(
            f"GPU {index}: util avg {_fmt(util, '%')}"
            f" max {_fmt(_stat(samples, 'util', peak), '%')},"
            f" process mem max {_fmt(_stat(samples, 'proc_mem_mb', peak), 'MB')}"
            f" (device {_fmt(_stat(samples, 'mem_mb', peak), 'MB')})"
        )
        lines.append(
            f"GPU {index}: power avg {_fmt(_stat(samples, 'power_w', mean), 'W')}"
            f" max {_fmt(_stat(samples, 'power_w', peak), 'W')},"
            f" SM clock min {_fmt(_stat(samples, 'sm_mhz', low), 'MHz')},"
            f" temp max {_fmt(_stat(samples, 'temp_c', peak), 'C')}"
        )
        if util is not None and util < LOW_UTIL:
            lines.append(
                f"GPU {index}: no kernel running {100 - util:.0f}% of the time,"
                " nsys shows where the gaps are"
            )
        ok = [s for s in samples if s.ok]  # failed queries: unknown, not unslowed
        for reason in smi.SLOWDOWNS:
            n = sum(reason in s.slowdowns for s in ok)
            if ok and 100 * n / len(ok) > SLOWDOWN_PCT:
                lines.append(
                    f"GPU {index}: clocks slowed by {reason} in {n} of"
                    f" {len(ok)} samples ({100 * n / len(ok):.0f}%)"
                )
    return lines
