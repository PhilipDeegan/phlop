# quick guide

#### installation

```shell
python3 -m pip install phlop -U
```

#### cmake test runner for cmake project with 20 cores

```shell
python3 -m phlop.run.test_cases -c 20 -i build --cmake
```

#### etc

```shell
python3 -m phlop
Available:
    phlop.app
    phlop.os
    phlop.proc
    phlop.reflection
    phlop.run
    phlop.string
    phlop.testing
```

```shell
python3 -m phlop.run
Available:
    phlop.run.mpirun_stats_man -h
    phlop.run.perf -h
    phlop.run.stats_man -h
    phlop.run.test_cases -h
    phlop.run.valgrind -h
    phlop.run.mpi_valgrind -h
    phlop.run.nsys -h
    phlop.run.mpi_nsys -h
    phlop.run.csan -h
    phlop.run.mpi_csan -h
    phlop.run.ncu -h
    phlop.run.mpi_ncu -h
```

```shell
# valgrind with a short answer, files kept under .phlop/valgrind
python3 -m phlop.run.valgrind ./app args            # memcheck: errors and leaks, where
python3 -m phlop.run.valgrind -t massif ./app args  # heap peak, and what holds it
python3 -m phlop.run.valgrind --show --md -         # the latest run again, as markdown
python3 -m phlop.run.valgrind --compare RUN1 RUN2 --plot cmp.png  # eg master vs a branch
mpirun -n 4 python3 -m phlop.run.mpi_valgrind -t massif ./app  # per rank, one report
```

```shell
# Nsight Systems with a short answer, files kept under .phlop/nsys: where GPU time
#  goes, GPU idle gaps and the NVTX range they fall in, pageable copies, exposed
#  NCCL, then hints
python3 -m phlop.run.nsys ./app args
python3 -m phlop.run.nsys -t cuda,nvtx,osrt,cublas ./app   # nsys --trace
python3 -m phlop.run.nsys --show job.nsys-rep --md -       # a profile from a cluster
python3 -m phlop.run.nsys --compare RUN1 RUN2              # eg master vs a branch
mpirun -n 4 python3 -m phlop.run.mpi_nsys ./app  # per rank, MPI traced: the straggler
srun python3 -m phlop.run.mpi_nsys ./app         #  and how long the others wait for it
```

```shell
# compute-sanitizer, grouped: the same error from 10000 threads is one line, with
#  the kernel's source line, an example thread and the host caller. Exit code 99
#  on errors. Build with -lineinfo (device lines) and -g (host lines)
python3 -m phlop.run.csan ./app args                 # memcheck, with leaks
python3 -m phlop.run.csan -t racecheck ./app         # or initcheck, synccheck
mpirun -n 4 python3 -m phlop.run.mpi_csan ./app      # each error once, with its ranks
```

```shell
# Nsight Compute, per kernel: what bounds it (compute, memory and which level,
#  or latency), occupancy, coalescing, the main warp stall, then hints: phlop's
#  and ncu's own rules ranked by estimated speedup. Each distinct launch is
#  profiled once, but ~30 replays each: the desktop may stall on a display GPU
python3 -m phlop.run.ncu ./app args
python3 -m phlop.run.ncu -k 'gemm|axpy' -c 3 ./app   # kernel regex, launches each
python3 -m phlop.run.ncu --show job.ncu-rep --md -   # a report from a cluster
python3 -m phlop.run.ncu --compare RUN1 RUN2         # eg master vs a branch
mpirun -n 4 python3 -m phlop.run.mpi_ncu -k work ./app  # kernels slower on one rank
# ERR_NVGPUCTRPERM: options nvidia NVreg_RestrictProfilingToAdminUsers=0 in
#  /etc/modprobe.d, reboot
```

```shell
# hardware counter analysis via libpfm event sets (./sh/setup_pfm.sh first)
python3 -m phlop.app.pfm.stat --list
python3 -m phlop.app.pfm.stat -s topdown,vector,flops -- ./app args
# unprivileged counting needs kernel.perf_event_paranoid <= 2, warnings say how
```

```shell
# per kernel registers, stack, shared and local memory from a built binary (no
#  GPU needed), and the largest block size each kernel can launch with
python3 -m phlop.app.nvidia.cuobjdump bin/app
python3 -m phlop.app.nvidia.cuobjdump -t 512 bin/app            # kernels that can't launch 512
python3 -m phlop.app.nvidia.cuobjdump dbg=bin/a rel=bin/b       # compare builds
```

```cpp
// per scope hardware counters: scope_timer.hpp, threaded_scope_timer.hpp, mpi_scope_timer.hpp
// threaded: each thread counts only itself, opening its counters on first use
PHLOP_SCOPE("solve", time, perf_all);   // time + PHLOP_PERF_ALL events
PHLOP_SCOPE("step", time, perf_light);  // time + PHLOP_PERF_LIGHT events
PHLOP_SCOPE("io");                      // PHLOP_SCOPE_DEFAULT, time
// the calling thread's I/O over the scope, /proc/thread-self/io (linux), when
// enabled: phlop::scope_timer().io() or PHLOP_SCOPE_IO=1. Bytes and calls through
// read/write, bytes to/from the block layer, shown by: scope_timer times <file>
PHLOP_SCOPE("h5_write", time, io);
```

```cpp
// mpi_scope_timer.hpp: MPI_Wtime may differ per rank, a barrier gives a common point
MPI_Init(&argc, &argv);
phlop::mpi::sync_clock();  // collective, after MPI_Init
// ... run ...
phlop::mpi::sync_clock();  // before MPI_Finalize: corrects drift between the two
// variance across ranks then compares starts from the sync, not each rank's first scope:
// python3 -m phlop.timing.scope_timer variance 'phlop_scope_timers.*.bin'
```

```shell
# -DPHLOP_NVTX: every PHLOP_SCOPE / PHLOP_SCOPE_TIMER is also an NVTX range in domain
#  "phlop", so scopes show on the Nsight Systems timeline next to their kernels
g++ -DPHLOP_NVTX -I/usr/local/cuda/include ... -ldl   # nvtx3 is header only
nsys profile --trace=cuda,nvtx ./app
ncu --nvtx --nvtx-include "phlop@step/" ./app         # only kernels inside scope "step"
```

```shell
# choose the event sets per level, then run as usual
eval "$(python3 -m phlop.app.pfm.stat --env perf_light=light --env perf_all=topdown)"
./app  # writes counts into the scope timer file, read with:
python3 -m phlop.timing.scope_timer perf .phlop_scope_times.txt -v  # scope_timer.hpp
python3 -m phlop.timing.scope_timer perf .phlop_scope_times.bin -v  # threaded, same format
# opt in: count kernel space too (eg I/O scopes), needs perf_event_paranoid <= 1
eval "$(python3 -m phlop.app.pfm.stat --env perf_light=light --kernel)"
```

```shell
python3 -m phlop.run.test_cases -h
usage: test_cases.py [-h] [--cmake] [-c CORES] [-i INPUT] [-p] [--prefix PREFIX] [--postfix POSTFIX] [--dump DUMP] [--load LOAD] [-r REGEX] [-R] [--rerun RERUN] [--logging LOGGING]

Flexible parallel test runner

options:
  -h, --help            show this help message and exit
  --cmake               Enable cmake build config tests extraction (default: False)
  -c CORES, --cores CORES
                        Parallism core/thread count (default: 1)
  -i INPUT, --input INPUT
                        Input file or directory. (default: .)
  -p, --print_only      Print only, no execution (default: False)
  --prefix PREFIX       Prepend string to execution string (default: )
  --postfix POSTFIX     Append string to execution string (default: )
  --dump DUMP           Dump discovered tests to filepath (hex-encoded dill pickle), no execution (default: None)
  --load LOAD           globbing filepath for files exported from dump (default: None)
  -r REGEX, --regex REGEX
                        Filter out non-matching execution strings (default: None)
  -R, --reverse         reverse order - higher core count tests preferred (default: False)
  --rerun RERUN         number of times to re-execute discovered tests (default: 1)
  --logging LOGGING     0=off, 1=on non zero exit code, 2=always (default: 1)
```
