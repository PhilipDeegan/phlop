# phlop/procs/runtimer.py

from __future__ import annotations

import os
import subprocess
import time
from contextlib import ExitStack, contextmanager

from phlop.os import pushd, write_to_file
from phlop.string import decode_bytes


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
        **kwargs,
    ):
        self.cmd = cmd
        self.stdout = ""
        self.stderr = ""
        self.run_time = None
        self.logging = logging
        self.log_file_path = log_file_path
        self.capture_output = capture_output
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
            self.stdout, self.stderr = p.communicate()
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

    def out(self, ignore_exit_code=False):
        if not ignore_exit_code and self.exitcode > 0:
            raise RuntimeError(f"phlop.RunTimer error: {self.stderr}")
        return self.stdout
