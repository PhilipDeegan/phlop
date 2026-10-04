# phlop/procs/runtimer.py

from __future__ import annotations

import os
import subprocess
import time
from collections import deque
from contextlib import ExitStack, contextmanager, suppress

from phlop.os import pushd, write_to_file
from phlop.string import decode_bytes

STALL_KILL_GRACE = 30  # seconds between terminate and kill of a stalled process
STALL_POLL_MAX = 60  # max seconds between stall checks
BUSY_CPU_FRACTION = 0.5  # of a single cpu, at or above is considered busy


def _files_signature(paths):
    sig = []
    for path in paths:
        try:
            st = os.stat(path)
            sig.append((st.st_size, st.st_mtime_ns))
        except OSError:
            sig.append(None)
    return tuple(sig)


def _tree_cpu_seconds(proc):
    """cpu seconds of proc and its live descendants, including their reaped children"""
    total = 0.0
    for each in [proc, *proc.children(recursive=True)]:
        with suppress(Exception):  # may already be gone
            t = each.cpu_times()
            total += t.user + t.system
            total += getattr(t, "children_user", 0) + getattr(t, "children_system", 0)
    return total


def _psutil_process(p):
    try:
        import psutil

        return psutil.Process(p.pid)
    except Exception:  # noqa: BLE001 - psutil is optional
        return None


class _StallMonitor:
    """Nothing is stalled until there's been no output for `timeout` seconds,
    without log files that's since the start. Then without psutil it's stalled,
    with psutil only if under BUSY_CPU_FRACTION cpu over the last `timeout`
    seconds, or if there's been no output for `busy_timeout` seconds."""

    def __init__(self, log_paths, proc, timeout, busy_timeout):
        self.log_paths, self.proc = log_paths, proc
        self.timeout, self.busy_timeout = timeout, busy_timeout
        self.what = "no output" if log_paths else "no log files"
        self.sig, self.last_change = _files_signature(log_paths), time.monotonic()
        self.samples = deque()
        if proc:
            self.samples.append((self.last_change, _tree_cpu_seconds(proc)))

    def stalled(self, now):
        sig = _files_signature(self.log_paths)
        if sig != self.sig:
            self.sig, self.last_change = sig, now
        quiet = now - self.last_change

        if self.proc:
            self.samples.append((now, _tree_cpu_seconds(self.proc)))
            # keep the newest sample at least `timeout` old as the window start
            while len(self.samples) > 1 and now - self.samples[1][0] >= self.timeout:
                self.samples.popleft()

        if quiet < self.timeout:
            return None
        if not self.proc:
            return f"{self.what} for {self.timeout:g}s"

        t0, cpu0 = self.samples[0]
        if now - t0 >= self.timeout:
            cpu_fraction = (self.samples[-1][1] - cpu0) / (now - t0)
            if cpu_fraction < BUSY_CPU_FRACTION:
                return f"{self.what} for {self.timeout:g}s, {cpu_fraction:.0%} cpu"
        if quiet >= self.busy_timeout:
            return f"{self.what} for {self.busy_timeout:g}s, busy"
        return None


def _kill_tree(p, grace):
    """terminate p and, if psutil is available, its descendants (e.g. mpi ranks),
    then kill whichever of them are still alive after `grace` seconds"""
    try:
        import psutil

        children = psutil.Process(p.pid).children(recursive=True)
    except Exception:  # noqa: BLE001 - psutil is optional, best effort
        children = []

    def signal_all(procs, fn_name):
        for proc in procs:
            with suppress(Exception):  # may already be gone
                getattr(proc, fn_name)()

    deadline = time.monotonic() + grace
    signal_all([p, *children], "terminate")
    with suppress(subprocess.TimeoutExpired):
        p.wait(timeout=grace)  # not via psutil, Popen must reap p for its exit code
    if children:  # p may exit while children ignoring SIGTERM live on
        remaining = max(0, deadline - time.monotonic())
        _, children = psutil.wait_procs(children, timeout=remaining)
    signal_all([p, *children], "kill")  # no-op for p if it has exited


@contextmanager
def _opened_streams(kwargs):
    with ExitStack() as stack:
        resolved = dict(kwargs)
        for key in ("stdout", "stderr"):
            if callable(resolved[key]):
                resolved[key] = stack.enter_context(resolved[key]())
        yield resolved


class RunTimer:
    def __init__(
        self,
        cmd,
        shell=False,
        capture_output=True,
        check=False,
        print_cmd=True,
        env: dict | None = None,  # dict[str, str] # eventually
        working_dir=None,
        log_file_path=None,
        logging=2,
        popen=True,
        stall_timeout=None,  # seconds, see _StallMonitor (popen only)
        stall_busy_timeout=None,  # seconds, see _StallMonitor
        **kwargs,
    ):
        self.cmd = cmd
        self.stdout = ""
        self.stderr = ""
        self.run_time = None
        self.logging = logging
        self.log_file_path = log_file_path
        self.capture_output = capture_output
        self.stall_timeout = stall_timeout
        self.stall_busy_timeout = stall_busy_timeout or stall_timeout
        self.stalled = None  # reason the process was killed as stalled
        self.log_paths = []
        benv = os.environ.copy()
        benv.update(env or {})
        ekwargs = {
            "shell": shell,
            "env": benv,
            "close_fds": True,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if not capture_output and log_file_path:
            stdout_path = f"{log_file_path}.stdout"
            stderr_path = f"{log_file_path}.stderr"
            self.log_paths = [stdout_path, stderr_path]
            ekwargs.update(
                {
                    "stdout": lambda: open(stdout_path, "w"),  # noqa: SIM115
                    "stderr": lambda: open(stderr_path, "w"),  # noqa: SIM115
                }
            )
        elif capture_output:
            ekwargs.update(
                {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE},
            )

        def go():
            if popen:
                self._popen(**ekwargs, **kwargs)
            else:
                self._run(
                    check=check, capture_output=capture_output, **ekwargs, **kwargs
                )

        if working_dir:
            with pushd(working_dir):
                go()
        else:
            go()

    def _locals(self):
        return self.capture_output, self.log_file_path, self.logging

    def _run(self, **kwargs):
        capture_output, log_file_path, logging = self._locals()
        check = kwargs.pop("check", False)
        try:
            start = time.time()
            with _opened_streams(kwargs) as stream_kwargs:
                self.run = subprocess.run(self.cmd, check=check, **stream_kwargs)
            self.run_time = time.time() - start
            self.exitcode = self.run.returncode
            if logging == 2 and capture_output:
                self.stdout = decode_bytes(self.run.stdout)
                self.stderr = decode_bytes(self.run.stderr)
        except subprocess.CalledProcessError as e:
            # only triggers on failure if check=True
            self.run_time = time.time() - start
            self.exitcode = e.returncode
            if logging >= 1 and capture_output:
                self.stdout = decode_bytes(e.stdout)
                self.stderr = decode_bytes(e.stderr)
                logging = 2  # force logging as exception occurred
        if logging == 2 and capture_output and log_file_path:
            write_to_file(f"{log_file_path}.stdout", self.stdout)
            write_to_file(f"{log_file_path}.stderr", self.stderr)

    def _popen(self, **kwargs):
        capture_output, log_file_path, logging = self._locals()
        start = time.time()
        with _opened_streams(kwargs) as stream_kwargs:
            p = subprocess.Popen(self.cmd, **stream_kwargs)
            self.stdout, self.stderr = self._communicate(p)
            self.run_time = time.time() - start
            self.exitcode = p.returncode
            if capture_output:
                p.stdout.close()
                p.stderr.close()
        p = None

        if self.exitcode > 0 and capture_output:
            logging = 2  # force logging as exception occurred
        if logging == 2 and capture_output:
            self.stdout = decode_bytes(self.stdout)
            self.stderr = decode_bytes(self.stderr)
        if logging == 2 and capture_output and log_file_path:
            write_to_file(f"{log_file_path}.stdout", self.stdout)
            write_to_file(f"{log_file_path}.stderr", self.stderr)

    def _stall_monitor(self, p):
        if not self.stall_timeout:
            return None
        proc = _psutil_process(p)
        if not (self.log_paths or proc):
            return None
        return _StallMonitor(
            self.log_paths, proc, self.stall_timeout, self.stall_busy_timeout
        )

    def _communicate(self, p):
        """communicate, but kill p if it appears stalled, see _stall_monitor"""
        monitor = self._stall_monitor(p)
        if monitor is None:
            return p.communicate()

        while True:
            try:
                return p.communicate(timeout=min(self.stall_timeout, STALL_POLL_MAX))
            except subprocess.TimeoutExpired:
                self.stalled = monitor.stalled(time.monotonic())
                if self.stalled:
                    _kill_tree(p, STALL_KILL_GRACE)
                    try:
                        return p.communicate(timeout=STALL_KILL_GRACE)
                    except subprocess.TimeoutExpired:
                        # a descendant outside the killed tree holds the pipes open
                        with suppress(subprocess.TimeoutExpired):
                            p.wait(timeout=STALL_KILL_GRACE)
                        empty = "" if self.capture_output else None
                        return empty, empty

    def out(self, ignore_exit_code=False):
        if not ignore_exit_code and self.exitcode > 0:
            raise RuntimeError(f"phlop.RunTimer error: {self.stderr}")
        return self.stdout
