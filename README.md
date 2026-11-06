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
# hardware counter analysis via libpfm event sets (./sh/setup_pfm.sh first)
python3 -m phlop.app.pfm.stat --list
python3 -m phlop.app.pfm.stat -s topdown,vector,flops -- ./app args
# unprivileged counting needs kernel.perf_event_paranoid <= 2, warnings say how
```

```cpp
// per scope hardware counters: scope_timer.hpp, threaded_scope_timer.hpp, mpi_scope_timer.hpp
// threaded: each thread counts only itself, opening its counters on first use
PHLOP_SCOPE("solve", time, perf_all);   // time + PHLOP_PERF_ALL events
PHLOP_SCOPE("step", time, perf_light);  // time + PHLOP_PERF_LIGHT events
PHLOP_SCOPE("io");                      // PHLOP_SCOPE_DEFAULT, time
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
