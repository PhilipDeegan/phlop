import subprocess
import time

from phlop.string import decode_bytes


class RunTimer:
    def __init__(
        self,
        cmd,
        shell=True,
        capture_output=True,
        check=False,
        print_cmd=True,
        **kwargs,
    ):
        self.cmd = cmd
        start = time.time()

        self.run_time = time.time() - start
        self.stdout = ""

        try:
            self.run = subprocess.run(
                self.cmd,
                shell=shell,
                capture_output=capture_output,
                check=check,
                **kwargs,
            )
            self.run_time = time.time() - start
            self.stdout = self.run.stdout
            self.stderr = self.run.stderr
            self.exitcode = self.run.returncode
        except (
            subprocess.CalledProcessError
        ) as e:  # only triggers on failure if check=True
            self.exitcode = e.returncode
            self.stdout = decode_bytes(e.stdout)
            self.stderr = decode_bytes(e.stderr)
            self.run_time = time.time() - start


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
    return run(f"which {bin}").returncode == 0
