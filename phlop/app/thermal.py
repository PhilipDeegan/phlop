# phlop/app/thermal.py
#
# Keeps a machine from running hot for long: commands paused while the CPU is
#  too hot, or (the test runner, see phlop.procs.parallel_processor) no new
#  jobs started while it is.
#
#   python3 -m phlop.app.thermal [--high 88] [--low 78] [--every 1] -- <command> [args...]
#
# The command runs in its own session. Every --every seconds the CPU
#  temperature is read (psutil: the hottest of k10temp's, coretemp's or
#  zenpower's sensors); at or above --high every process of the command is
#  stopped (SIGSTOP), at or below --low the stopped ones are continued
#  (SIGCONT). The command's processes are its descendants and anything else in
#  its session, found again at each pause until no new one appears: MPI
#  launchers put each rank in a process group of its own (Open MPI's prterun
#  does), and ranks spawn helpers (phlop.app.stats_man's per rank watchers), so
#  the command's process group alone would miss them. Exits with the command's
#  code. Ctrl-C / SIGTERM: everything stopped is continued, then the command's
#  process group gets the signal (an MPI launcher forwards it to its ranks).
#  Without psutil, or without a sensor (eg macOS), the command runs unguarded.
#
# PHLOP_THERMAL_MAX (C, unset: off) and PHLOP_THERMAL_RESUME (C, default 10
#  below it): what phlop.procs.parallel_processor holds new jobs at, see Hold.

import argparse
import os
import signal
import subprocess
import sys
import time

SENSORS = ("k10temp", "coretemp", "zenpower")


def cpu_temperature():
    """the hottest CPU sensor in C, None without psutil or a sensor"""
    try:
        import psutil

        sensors = psutil.sensors_temperatures()  # linux and freebsd only
    except (ImportError, AttributeError):
        return None
    for name in SENSORS:
        if sensors.get(name):
            return max(s.current for s in sensors[name])
    return None


class Hold:
    """whether to hold back new work: at or above high until at or below low,
    from the environment by default (PHLOP_THERMAL_MAX, PHLOP_THERMAL_RESUME):
    never when PHLOP_THERMAL_MAX is unset or there is no sensor"""

    def __init__(self, high=None, low=None, temperature=None):
        if high is None and os.environ.get("PHLOP_THERMAL_MAX"):
            high = float(os.environ["PHLOP_THERMAL_MAX"])
        if low is None and high is not None:
            low = float(os.environ.get("PHLOP_THERMAL_RESUME", high - 10))
        self.high, self.low = high, low
        self.temperature = temperature or cpu_temperature
        self.holding = False

    @property
    def enabled(self):
        return self.high is not None

    def __call__(self):
        """true while too hot (reads the sensor)"""
        return self.update(self.temperature()) if self.enabled else False

    def update(self, t):
        """true while too hot, from a reading t (None: no sensor, never)"""
        if not self.enabled or t is None:
            return False
        if not self.holding and t >= self.high:
            self.holding = True
        elif self.holding and t <= self.low:
            self.holding = False
        return self.holding


class Governor:
    """a command run with its processes stopped while too hot"""

    def __init__(
        self, command, high=88.0, low=78.0, every=1.0, temperature=None, log=None
    ):
        if low >= high:
            raise ValueError("low must be below high")
        self.command, self.every = command, every
        self.hold = Hold(high, low, temperature)
        self.log = log or (
            lambda msg: print(f"thermal: {msg}", file=sys.stderr, flush=True)
        )
        self.stopped = []  # the processes stopped, to continue exactly these
        self.pauses, self.paused_for, self.hottest = 0, 0.0, None
        self.child = None

    def members(self):
        """the command's processes now: its descendants and the rest of its session"""
        import psutil

        found = {}
        try:
            top = psutil.Process(self.child.pid)
            for p in [top] + top.children(recursive=True):
                found[p.pid] = p
        except psutil.NoSuchProcess:
            pass
        for p in psutil.process_iter():
            try:
                if p.pid not in found and os.getsid(p.pid) == self.child.pid:
                    found[p.pid] = p
            except (ProcessLookupError, PermissionError, psutil.Error):
                pass
        return list(found.values())

    @staticmethod
    def send(processes, sig):
        import psutil

        for p in processes:
            try:
                p.send_signal(sig)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def pause(self):
        """stopped until no new process appears (one forked between a scan
        and its parent's stop, eg a build's next compiler)"""
        self.stopped.clear()
        while new := [
            p for p in self.members() if p.pid not in {q.pid for q in self.stopped}
        ]:
            self.send(new, signal.SIGSTOP)
            self.stopped.extend(new)

    def resume(self):
        self.send(self.stopped, signal.SIGCONT)
        self.stopped.clear()

    def forward(self, sig, _frame):
        self.resume()
        try:
            os.killpg(self.child.pid, sig)
        except ProcessLookupError:
            pass

    def run(self):
        """the command's exit code"""
        guarded = self.hold.temperature() is not None
        if not guarded:
            self.log("no CPU temperature sensor (or no psutil), running unguarded")
        self.child = subprocess.Popen(self.command, start_new_session=True)
        previous = {
            s: signal.signal(s, self.forward) for s in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            since = None
            while guarded and self.child.poll() is None:
                t = self.hold.temperature()
                if t is not None:
                    self.hottest = t if self.hottest is None else max(self.hottest, t)
                was = self.hold.holding
                if self.hold.update(t) and not was:
                    self.pause()
                    self.pauses, since = self.pauses + 1, time.monotonic()
                    self.log(f"{t:.1f} C, paused {len(self.stopped)} processes")
                elif was and not self.hold.holding:
                    self.resume()
                    self.paused_for += time.monotonic() - since
                    self.log(f"{t:.1f} C, resumed")
                try:
                    self.child.wait(timeout=self.every)
                except subprocess.TimeoutExpired:
                    pass
            if self.stopped:  # the command ended while stopped: leave nothing stopped
                self.resume()
            if self.pauses:
                self.log(
                    f"hottest {self.hottest:.1f} C, {self.pauses} pauses,"
                    f" {self.paused_for:.0f} s paused"
                )
            return self.child.wait()
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


def cli_args_parser():
    parser = argparse.ArgumentParser(
        description="runs a command, paused while the CPU is too hot"
    )
    parser.add_argument(
        "--high", type=float, default=88.0, help="pause at or above (C)"
    )
    parser.add_argument(
        "--low", type=float, default=78.0, help="resume at or below (C)"
    )
    parser.add_argument(
        "--every", type=float, default=1.0, help="seconds between readings"
    )
    return parser


def main(argv):
    parser = cli_args_parser()
    if "--" not in argv:
        parser.error("the command goes after --")
    at = argv.index("--")
    args, command = parser.parse_args(argv[:at]), argv[at + 1 :]
    if not command:
        parser.error("no command")
    if args.low >= args.high:
        parser.error("--low must be below --high")
    return Governor(command, args.high, args.low, args.every).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
