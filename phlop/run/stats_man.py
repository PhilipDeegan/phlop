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
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    parser.add_argument(
        "-q", "--quiet", action="store_true", default=False, help=_help.quiet
    )
    parser.add_argument(
        "-i", "--interval", default=_default_interval, help=_help.interval
    )
    return parser


def verify_cli_args(cli_args):
    try:
        cli_args.interval = int(cli_args.interval)
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


def capture_now(pid, data):
    now = datetime.utcnow().timestamp()
    proc = psutil.Process(pid=pid)
    data.cpu_load += [proc.cpu_percent(interval=0.1)]
    data.fds += [len(proc.open_files())]
    data.mem_usage += [bytes_as_mb(proc.memory_info().rss)]
    data.timestamps += [now]


class RuntimeStatsManager:
    def __init__(self, pid, interval=_default_interval):
        self.pid = pid
        self.interval = interval
        self.pqueue = Queue()
        self.data = {}
        self.p = Process(target=RuntimeStatsManager._run, args=(self,))
        self.p.start()
        self.data = self.pqueue.get()

    def __del__(self):
        if check_pid(self.p.pid):
            os.kill(self.p.pid, signal.SIGINT)
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
        data = ProcessCaptureInfo()
        signal.signal(signal.SIGINT, signal_handler)
        while check_pid(this.pid):
            try:
                capture_now(this.pid, data)
            except psutil.AccessDenied:
                break
            time.sleep(this.interval)
        this.pqueue.put(data)


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


def main():
    parser = cli_args_parser()
    cli_args = verify_cli_args(parser.parse_args())
    try:
        proc = run_raw(cli_args.remaining, quiet=cli_args.quiet)
        statsman = RuntimeStatsManager(proc.pid, cli_args.interval).join()
        print_summary(statsman)
    except (Exception, SystemExit) as e:
        logger.exception(e)
        parser.print_help()
    except:
        e = sys.exc_info()[0]
        print(f"Error: Unknown Error {e}")


if __name__ == "__main__":
    main()
