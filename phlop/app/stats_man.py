#
#
#
#
#


import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Process, Queue

import numpy as np
import psutil
import yaml
from phlop.dict import ValDict
from phlop.proc import run_raw

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

_default_interval = 2


@dataclass
class ProcessCaptureInfo:
    cpu_load: list = field(default_factory=lambda: [])
    fds: list = field(default_factory=lambda: [])
    mem_usage: list = field(default_factory=lambda: [])
    timestamps: list = field(default_factory=lambda: [])


def cli_args_parser():
    import argparse

    _help = ValDict(
        quiet="Redirect output to /dev/null",
        interval="Seconds between process stat capture",
        yaml="write yaml file during execution",
        summary="prints summary on end",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument(
        "-q", "--quiet", action="store_true", default=False, help=_help.quiet
    )
    parser.add_argument(
        "-i", "--interval", default=_default_interval, help=_help.interval
    )
    parser.add_argument("-y", "--yaml", default=None, help=_help.interval)
    parser.add_argument("-s", "--summary", action="store_true", default=True)
    return parser


def verify_cli_args(cli_args):
    try:
        cli_args.interval = int(cli_args.interval)
        if cli_args.yaml:
            cli_args.summary = False
    except ValueError:
        raise ValueError("Interval must be an integer")
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


def signal_handler(sig, frame):
    sys.exit(0)


def bytes_as_mb(n_bytes):
    return int(n_bytes / 1024**2)


def timestamp_now():
    return datetime.utcnow().isoformat()


def now(pid):
    proc = psutil.Process(pid=pid)
    open_file = len(proc.open_files())
    mem_used_mb = bytes_as_mb(proc.memory_info().rss)
    cpu_usage = proc.cpu_percent(interval=0.1)
    return dict(open_file=open_file, mem_used_mb=mem_used_mb, cpu_usage=cpu_usage)


def capture_now(pid, data):
    now = datetime.utcnow().timestamp()
    proc = psutil.Process(pid=pid)
    data.fds += [len(proc.open_files())]
    data.mem_usage += [bytes_as_mb(proc.memory_info().rss)]
    data.cpu_load += [proc.cpu_percent(interval=0.1)]
    data.timestamps += [now]


def append_yaml(file, pid):
    keys = list(now(pid).keys())
    stats = now(pid)
    vals = {"v": ",".join([str(stats[key]) for key in keys])}
    sdump = "- " + yaml.dump(vals, indent=2)
    with open(file, "a") as f:
        f.write(sdump)


def init_yaml(cli_args, pid, info):
    file = cli_args.yaml
    headers = {"headers": [str(i) for i in list(now(pid).keys())]}
    cli = {"cli_args": dict(interval=cli_args.interval)}
    with open(file, "w") as f:
        f.write("---\n")
        yaml.dump(info, f, default_flow_style=False)
        yaml.dump(cli, f, default_flow_style=False)
        yaml.dump(headers, f, default_flow_style=False)
        f.write(f"start: {timestamp_now()} \n")
        f.write(f"snapshots:\n")


def end_yaml(file):
    with open(file, "a") as f:
        f.write(f"end: {timestamp_now()} \n")


class RuntimeStatsManager:
    def __init__(self, cli_args, info={}):
        self.proc = run_raw(cli_args.remaining, quiet=cli_args.quiet)
        self.pid = self.proc.pid
        self.cli_args = cli_args

        if self.cli_args.yaml:
            init_yaml(self.cli_args, self.pid, info)

        self.pqueue = Queue()
        self.data = {}
        self.p = Process(target=RuntimeStatsManager._run, args=(self,))
        self.p.start()
        self.data = None
        if cli_args.summary:
            self.data = self.pqueue.get()

    def __del__(self):
        if check_pid(self.pid):
            os.kill(self.pid, signal.SIGINT)
        self.join()

    def join(self):
        if not self.pid:
            return
        while self.p.exitcode is None and check_pid(self.p.pid):
            time.sleep(1)
        self.pid = 0
        return self

    @staticmethod
    def _run(this):
        if this.cli_args.summary:
            data = ProcessCaptureInfo()
            signal.signal(signal.SIGINT, signal_handler)
            while check_pid(this.pid):
                try:
                    capture_now(this.pid, data)
                except psutil.AccessDenied:
                    break
                time.sleep(this.cli_args.interval)
            this.pqueue.put(data)
        else:
            while check_pid(this.pid):
                try:
                    append_yaml(this.cli_args.yaml, this.pid)
                    time.sleep(this.cli_args.interval)
                except psutil.AccessDenied:
                    break
            end_yaml(this.cli_args.yaml)


def attach_to_this_process():
    return RuntimeStatsManager(os.getpid())


def print_summary(statsman):
    print("summary:")
    print("\tCPU avg:", np.average(statsman.data.cpu_load))
    print("\tCPU max:", np.max(statsman.data.cpu_load))

    print("\tFDS avg:", np.average(statsman.data.fds))
    print("\tFDS max:", np.max(statsman.data.fds))

    print("\tMEM avg:", np.average(statsman.data.mem_usage))
    print("\tMEM max:", np.max(statsman.data.mem_usage))
