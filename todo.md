# todo.md

## phlop.proc.run_mp: bound default concurrency

`run_mp(cmds, N_CORES=None)` defaults `N_CORES` to `len(cmds)`
(`phlop/proc.py`), so every command starts at once: 500 commands means 500
concurrent processes. Default to `phlop.os.usable_cores()` instead.
