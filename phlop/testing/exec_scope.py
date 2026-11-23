# phlop/testing/exec_scope.py
#
# Scopes and phases of a .phlop.exec.yaml, see load_config_tests in
#  phlop/testing/test_cases.py.
#
# A scope is a dict of SCOPE_KEYS; a nested scope (a phase, a file chained to by
#  `paths:`, a subdirectory's file, a variant) inherits the scope above it, mkn
#  style: an inner key replaces the outer one, `${key}` in it is the outer value
#  (`prefix: ${prefix} mpirun -n 2`); env per variable (`PATH: /x:${PATH}`, the
#  process environment when no outer scope sets it); a "${tags}" element in
#  `tags` splices the outer list in.
#
# A job's phase is a list of (list_id, ordinal, name, ignore_errors), outermost
#  first: one per enclosing scope that has `phases:`, ordinal 0 its unphased
#  entries, 1.. its phases. phase_fixtures() turns them into fixtures: a job waits
#  for the phase before its own, and fails unrun if a phase before it failed, back
#  to the nearest one with ignore_errors.

import os

SCOPE_KEYS = (
    "env",
    "prefix",
    "postfix",
    "tags",
    "cores",
    "threads",
    "working_dir",
    "ignore_errors",
)
CHAIN_KEYS = ("paths", "path", "phases", "name")
RESERVED_KEYS = SCOPE_KEYS + CHAIN_KEYS


def _outer(key, value, outer):
    """`${key}` in a string value replaced by `outer`"""
    if isinstance(value, str):
        return value.replace("${" + key + "}", "" if outer is None else str(outer))
    return value


def scope_of(body):
    """the scope keys of a yaml mapping"""
    return {k: body[k] for k in SCOPE_KEYS if k in body}


def nest(outer, inner):
    """`inner` (a scope or a variant) over `outer`, see the top of this file.
    Keys of `inner` that are not scope keys (fixtures) are kept as they are"""
    out = dict(outer)
    for key, value in inner.items():
        if key == "env":
            env = dict(outer.get("env") or {})
            for name, val in (value or {}).items():
                env[name] = _outer(name, val, env.get(name, os.environ.get(name)))
            out[key] = env
        elif key == "tags":
            if isinstance(value, str):
                value = [value]
            tags = []
            for tag in value or []:
                tags += list(outer.get("tags") or []) if tag == "${tags}" else [tag]
            out[key] = tags
        else:
            out[key] = _outer(key, value, outer.get(key))
    return out


def phase_label(phase):
    """`core/sub` for a job in phase sub of phase core, None if unphased"""
    names = [name for _, ordinal, name, _ in phase or () if ordinal]
    return "/".join(names) or None


def _fixture(list_id, ordinal):
    return f"phase {list_id}#{ordinal}"


def phase_fixtures(jobs):
    """the fixtures ordering `jobs` by their phases (see the top of this file),
    added to their own. Phases no job is in are not waited for"""
    lists = {}  # list_id -> {ordinal: ignore_errors}
    for job in jobs:
        for list_id, ordinal, _, ignore in getattr(job, "phase", None) or ():
            lists.setdefault(list_id, {})[ordinal] = ignore
    for job in jobs:
        provides, requires, cleans = (
            set(getattr(job, k, None) or ()) for k in ("provides", "requires", "cleans")
        )
        for list_id, ordinal, _, _ in getattr(job, "phase", None) or ():
            provides.add(_fixture(list_id, ordinal))
            ignores = lists[list_id]
            earlier = sorted(m for m in ignores if m < ordinal)
            if not earlier:
                continue
            last = earlier[-1]  # waited for, whatever its outcome if ignore_errors
            (cleans if ignores[last] else requires).add(_fixture(list_id, last))
            strict = [m for m in earlier if not ignores[m]]
            if strict:  # its failure stops this one
                requires.add(_fixture(list_id, strict[-1]))
        job.provides, job.requires, job.cleans = provides, requires, cleans
    return jobs
