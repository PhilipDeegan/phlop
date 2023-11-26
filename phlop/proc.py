import os
import subprocess
import time

from phlop.os import pushd, write_to_file
from phlop.string import decode_bytes


class RunTimer:
    def __init__(
        self,
        cmd,
        shell=True,
        capture_output=True,
        check=False,
        print_cmd=True,
        env: dict = {},  # dict[str, str] # eventually
        working_dir=None,
        log_file_path=None,
        **kwargs,
    ):
        self.cmd = cmd
        start = time.time()

        self.run_time = time.time() - start
        self.stdout = ""
        benv = os.environ.copy()
        benv.update(env)

        ekwargs = {}
        if not capture_output and log_file_path:
            ekwargs.update(
                dict(
                    stdout=open(f"{log_file_path}.stdout", "w"),
                    stderr=open(f"{log_file_path}.stderr", "w"),
                ),
            )

        def run():
            try:
                self.run = subprocess.run(
                    self.cmd,
                    shell=shell,
                    check=check,
                    env=benv,
                    capture_output=capture_output,
                    **kwargs,
                    **ekwargs,
                )
                self.run_time = time.time() - start
                self.exitcode = self.run.returncode
                if capture_output:
                    self.stdout = decode_bytes(self.run.stdout)
                    self.stderr = decode_bytes(self.run.stderr)
            except (
                subprocess.CalledProcessError
            ) as e:  # only triggers on failure if check=True
                self.exitcode = e.returncode
                self.run_time = time.time() - start
                if capture_output:
                    self.stdout = decode_bytes(e.stdout)
                    self.stderr = decode_bytes(e.stderr)
            if capture_output and log_file_path:
                write_to_file(f"{log_file_path}.stdout", self.stdout)
                write_to_file(f"{log_file_path}.stderr", self.stderr)

        if working_dir:
            with pushd(working_dir):
                run()
        else:
            run()

    def out(self, ignore_exit_code=False):
        if not ignore_exit_code and self.exitcode > 0:
            raise RuntimeError(f"phlop.RunTimer error: {self.stderr}")
        return self.stdout


class ProcessNonZeroExitCode(RuntimeError):
    ...


def run(cmd, shell=True, capture_output=True, check=False, print_cmd=True, **kwargs):
    """https://docs.python.org/3/library/subprocess.html"""
    if print_cmd:
        print(f"running: {cmd}")

    return RunTimer(
        cmd, shell=shell, capture_output=capture_output, check=check, **kwargs
    )


def run_mp(cmds, N_CORES=None, **kwargs):
    """
    spawns N_CORES threads (default=len(cmds)) running commands and waiting for results
    https://docs.python.org/3/library/concurrent.futures.html
    """
    import concurrent.futures

    if N_CORES is None:
        N_CORES = len(cmds)

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CORES) as executor:
        jobs = [executor.submit(run, cmd, **kwargs) for cmd in cmds]
        results = []
        for future in concurrent.futures.as_completed(jobs):
            try:
                results += [future.result()]
                if future.exception() is not None:
                    raise future.exception()
            except Exception as exc:
                if kwargs.get("check", False):
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise exc
                else:
                    print(f"run_mp generated an exception: {exc}")
        return results


def binary_exists_on_path(bin):
    """
    https://linux.die.net/man/1/which
    """
    raise ValueError("do better")
    return run(f"which {bin}").exitcode == 0
