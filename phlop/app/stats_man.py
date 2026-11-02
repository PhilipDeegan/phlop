# phlop/app/stats_man.py

import argparse
import logging
import os
import signal
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing import Process

import psutil
import yaml

from phlop.dict import ValDict
from phlop.proc import run_raw

logger = logging.getLogger(__name__)

_default_interval = 2
_first_sample_delay = 0.1  # first cpu_percent window, later ones are `interval`
HEADERS = ["fds", "mem_used_mb", "cpu_usage"]


@dataclass
class ProcessCaptureInfo:
    cpu_load: list = field(default_factory=list)
    fds: list = field(default_factory=list)
    mem_usage: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)

    def append(self, stats):
        self.fds.append(stats["fds"])
        self.mem_usage.append(stats["mem_used_mb"])
        self.cpu_load.append(stats["cpu_usage"])
        self.timestamps.append(datetime.now(timezone.utc).timestamp())


def cli_args_parser(description=""):
    _help = ValDict(
        quiet="Redirect output to /dev/null",
        interval="Seconds between process stat capture",
        yaml="write yaml file during execution",
        summary="prints summary on end",
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


class _Sampler:
    """One psutil.Process for the whole run: cpu_percent(None) then measures
    the time since the previous sample rather than blocking on a short window"""

    def __init__(self, pid):
        self.proc = psutil.Process(pid)
        self.proc.cpu_percent(None)  # starts the first window

    def __call__(self):
        p = self.proc
        if p.status() == psutil.STATUS_ZOMBIE:  # exited, not yet reaped
            raise psutil.NoSuchProcess(p.pid)
        with p.oneshot():
            fds = p.num_fds() if hasattr(p, "num_fds") else p.num_handles()
            return {
                "fds": fds,
                "mem_used_mb": bytes_as_mb(p.memory_info().rss),
                "cpu_usage": p.cpu_percent(None),
            }


def now(pid):
    """single snapshot, cpu over a 0.1s window"""
    sampler = _Sampler(pid)
    time.sleep(_first_sample_delay)
    return sampler()


def _sample_until_gone(pid, interval, on_sample, stop):
    """call on_sample every `interval` seconds until pid exits or `stop` is set"""
    try:
        sampler = _Sampler(pid)
        delay = min(_first_sample_delay, interval)
        while not stop(delay):
            on_sample(sampler())
            delay = interval
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def append_yaml(file, stats):
    vals = ",".join(str(stats[key]) for key in HEADERS)
    try:
        with open(file, "a") as f:
            f.write(f"- v: {vals}\n")
    except OSError as e:
        logger.error(f"phlop detected IO error: {e}")


def init_yaml(cli_args, pid, info):
    file = cli_args.yaml
    headers = {"headers": list(HEADERS)}
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

        if cli_args.yaml:
            init_yaml(cli_args, self.pid, info or {})

        def on_sample(stats):
            if self.data is not None:
                self.data.append(stats)
            if cli_args.yaml:
                append_yaml(cli_args.yaml, stats)

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=_sample_until_gone,
            args=(self.pid, cli_args.interval, on_sample, self._stop.wait),
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


def _attached_run(pid, yaml_file, interval):
    """separate process, so sampling isn't held up by pid's GIL"""
    stop = threading.Event()
    # SIGINT from kill(), SIGTERM from multiprocessing ending daemons at exit
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        _sample_until_gone(
            pid, interval, lambda stats: append_yaml(yaml_file, stats), stop.wait
        )
    finally:
        end_yaml(yaml_file)


class AttachableRuntimeStatsManager:
    """Samples the calling process itself, from a separate daemon process so
    sampling is not held up by this process's GIL. start(), then kill()/join()."""

    def __init__(self, cli_args, info=None):
        self.pid = os.getpid()
        self.cli_args = cli_args
        init_yaml(self.cli_args, self.pid, info or {})
        self.data = {}
        self.p = Process(
            target=_attached_run,
            args=(self.pid, cli_args.yaml, cli_args.interval),
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
