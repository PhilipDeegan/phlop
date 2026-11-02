# phlop/os.py

import contextlib
import os
import platform
from pathlib import Path


def scan_dir(path, files_only=False, dirs_only=False, drop=None):
    assert os.path.exists(path)
    drop = drop or []
    checks = [
        lambda entry: not files_only or (files_only and entry.is_file()),
        lambda entry: not dirs_only or (dirs_only and entry.is_dir()),
        lambda entry: entry.name not in drop,
    ]
    return [
        entry.name
        for entry in os.scandir(path)
        if all(check(entry) for check in checks)
    ]


@contextlib.contextmanager
def pushd(new_cwd):
    if not os.path.exists(new_cwd):
        raise RuntimeError("phlop.os.pushd: new_cwd does not exist")

    cwd = os.getcwd()
    os.chdir(new_cwd)
    try:
        yield
    finally:
        os.chdir(cwd)


def write_to_file(file, contents, mode="w", skip_if_empty=True):
    if contents or not skip_if_empty:
        try:
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            with open(file, mode) as f:
                f.write(contents)
        except OSError as e:
            raise RuntimeError(f"Failed to write to file {file}: {e}")


def read_file(file):
    try:
        with open(file, "r") as f:
            return f.read()
    except OSError as e:
        raise RuntimeError(f"Failed to read file {file}: {e}")


def read_last_lines_of(file, n=10):
    try:
        size = os.path.getsize(file)
        if size == 0:
            return []
        with open(file, "rb") as f:
            f.seek(max(0, size - n * 200))
            data = f.read()
        return data.decode("utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _ceil_div(a, b):
    return max(1, -(-a // b))


def cgroup_cpu_limit():
    """cpus allowed by a cgroup CPU quota (docker --cpus, k8s limits, systemd
    CPUQuota), None if unlimited or unknown. v2: the tightest cpu.max from this
    process's cgroup up to the root, v1: cpu.cfs_quota_us / cpu.cfs_period_us"""
    limit = None
    try:
        with open("/proc/self/cgroup") as f:
            cgroup = next(ln.split("::", 1)[1] for ln in f if ln.startswith("0::"))
        path = Path("/sys/fs/cgroup") / cgroup.strip().lstrip("/")
        for d in [path, *path.parents]:
            cpu_max = d / "cpu.max"
            if cpu_max.is_file():
                quota, period = cpu_max.read_text().split()[:2]
                if quota != "max":
                    n = _ceil_div(int(quota), int(period))
                    limit = n if limit is None else min(limit, n)
            if d == Path("/sys/fs/cgroup"):
                break
        return limit
    except (OSError, ValueError, StopIteration):
        pass
    try:
        v1 = Path("/sys/fs/cgroup/cpu")
        quota = int((v1 / "cpu.cfs_quota_us").read_text())
        period = int((v1 / "cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return _ceil_div(quota, period)
    except (OSError, ValueError):
        pass
    return None


def usable_cores():
    """cores this process may run on: affinity (taskset/slurm), capped by any
    cgroup CPU quota (containers), which os.process_cpu_count ignores"""
    if hasattr(os, "process_cpu_count"):  # python >= 3.13
        cores = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):
        cores = len(os.sched_getaffinity(0)) or 1
    else:
        cores = os.cpu_count() or 1
    limit = cgroup_cpu_limit()
    return cores if limit is None else min(cores, limit)


def env_sep():
    return ";" if any(platform.win32_ver()) else ":"
