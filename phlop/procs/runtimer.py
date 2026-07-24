# phlop/procs/runtimer.py

from __future__ import annotations

import os
import subprocess
import time

from phlop.os import pushd, write_to_file
from phlop.string import decode_bytes


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
            # handles must outlive this scope: passed to Popen/run and closed there
            stdout_file = open(f"{log_file_path}.stdout", "w")  # noqa: SIM115
            stderr_file = open(f"{log_file_path}.stderr", "w")  # noqa: SIM115
            ekwargs.update({"stdout": stdout_file, "stderr": stderr_file})
        else:
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
            self.run = subprocess.run(self.cmd, check=check, **kwargs)
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
        p = subprocess.Popen(self.cmd, **kwargs)
        self.stdout, self.stderr = p.communicate()
        self.run_time = time.time() - start
        self.exitcode = p.returncode
        if not capture_output and log_file_path:
            kwargs["stdout"].close()
            kwargs["stderr"].close()
        elif capture_output:
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
