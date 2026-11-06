# phlop/app/nvidia/smi.py
#
# GPU samples from nvidia-smi (NVML underneath), for stats_man: per GPU
# utilisation, memory (the device's, and the sampled processes'), power, SM
# clock, temperature and active clock slowdowns (power cap, thermal, ...)
#
# utilization.gpu is the share of time any kernel ran, not how busy the SMs
# were: one tiny kernel always running reads 100%. ncu answers how busy
#
# two nvidia-smi calls per sample, ~30ms. Only the GPUs in
# CUDA_VISIBLE_DEVICES (indices or UUIDs, as slurm sets it), else all
#
# https://docs.nvidia.com/deploy/nvidia-smi/index.html

import os
import shutil
import subprocess
from dataclasses import dataclass, field

# slowdowns worth reporting: the GPU wanted to run faster and couldn't
SLOWDOWNS = (
    "sw_power_cap",
    "hw_slowdown",
    "hw_power_brake_slowdown",
    "sw_thermal_slowdown",
    "hw_thermal_slowdown",
)
FIELDS = (
    "index",
    "uuid",
    "utilization.gpu",
    "memory.used",
    "power.draw",
    "clocks.sm",
    "temperature.gpu",
    *(f"clocks_event_reasons.{s}" for s in SLOWDOWNS),
)
TIMEOUT = 10  # seconds, nvidia-smi can hang on a GPU that fell off the bus


@dataclass
class Gpu:
    index: int
    uuid: str
    name: str = ""
    memory_total: float = 0  # MB
    clock_max: float = 0  # MHz, SM


@dataclass
class Sample:
    """one GPU at one time. None: not supported, eg power on some boards"""

    index: int
    util: float = None  # % of time a kernel ran
    mem_mb: float = None  # the whole device
    proc_mem_mb: float = 0  # the sampled processes
    power_w: float = None
    sm_mhz: float = None
    temp_c: float = None
    slowdowns: list = field(default_factory=list)


def available():
    return shutil.which("nvidia-smi") is not None


def _query(args):
    out = subprocess.run(
        ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line]


def _number(s):
    """'26.11' -> 26.11, '[N/A]' or '[Not Supported]' -> None"""
    try:
        return float(s)
    except ValueError:
        return None


def visible(gpus, env=None):
    """the GPUs CUDA_VISIBLE_DEVICES lets a process use, all when unset"""
    env = os.environ.get("CUDA_VISIBLE_DEVICES") if env is None else env
    if env is None or env.strip() == "":
        return gpus if env is None else []  # "" hides every GPU
    out = []
    for tok in (t.strip() for t in env.split(",")):
        if tok.isdigit():
            out += [g for g in gpus if g.index == int(tok)]
        else:  # GPU-<uuid>, or a unique prefix of it
            out += [g for g in gpus if g.uuid.startswith(tok)]
    return out


def gpus():
    """the GPUs on this node, [] without nvidia-smi or a working driver"""
    if not available():
        return []
    try:
        rows = _query(["--query-gpu=index,uuid,name,memory.total,clocks.max.sm"])
    except (subprocess.SubprocessError, OSError):
        return []
    return [
        Gpu(int(r[0]), r[1], r[2], _number(r[3]) or 0, _number(r[4]) or 0) for r in rows
    ]


def parse(gpu_rows, app_rows, indices, pids):
    """Samples for the GPUs in indices from --query-gpu=FIELDS rows and
    --query-compute-apps=pid,gpu_uuid,used_memory rows"""
    samples, by_uuid = {}, {}
    for r in gpu_rows:
        if len(r) != len(FIELDS) or int(r[0]) not in indices:
            continue
        nums = [_number(v) for v in r[2:7]]
        s = Sample(int(r[0]), *nums[:2], 0, *nums[2:])
        s.slowdowns = [n for n, v in zip(SLOWDOWNS, r[7:]) if v == "Active"]
        samples[s.index] = s
        by_uuid[r[1]] = s
    for pid, uuid, used in (r for r in app_rows if len(r) == 3):
        if pid.isdigit() and int(pid) in pids and uuid in by_uuid:
            by_uuid[uuid].proc_mem_mb += _number(used) or 0
    return [samples[i] for i in sorted(samples)]


def sample(indices, pids=()):
    """Samples for the GPUs in indices, proc_mem_mb summed over pids.
    Raises subprocess.SubprocessError or OSError if nvidia-smi fails"""
    gpu_rows = _query([f"--query-gpu={','.join(FIELDS)}"])
    app_rows = _query(["--query-compute-apps=pid,gpu_uuid,used_memory"])
    return parse(gpu_rows, app_rows, set(indices), set(pids))
