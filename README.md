# quick guide

#### installation

```shell
python3 -m pip install phlop -U
```

#### cmake test runner for cmake project with 20 cores

```shell
python3 -m phlop.run.test_cases -c 20 -d build --cmake
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
```

```shell
usage: test_cases.py [-h] [--cmake] [-c CORES] [-i INPUT] [-p] [--prefix PREFIX] [--postfix POSTFIX] [--dump [DUMP]] [--load LOAD] [-r REGEX] [-R] [--rerun RERUN] [--logging LOGGING]

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
  --dump [DUMP]         Dump discovered tests as YAML to filepath, no execution (default: None)
  --load LOAD           globbing filepath for files exported from dump (default: None)
  -r REGEX, --regex REGEX
                        Filter out non-matching execution strings (default: None)
  -R, --reverse         reverse order - higher core count tests preferred (default: False)
  --rerun RERUN         number of times to re-execute discovered tests (default: 1)
  --logging LOGGING     0=off, 1=on non zero exit code, 2=always (default: 1)
```
