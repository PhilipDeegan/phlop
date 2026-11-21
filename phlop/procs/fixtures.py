# phlop/procs/fixtures.py
#
# Dependencies between jobs, as ctest fixtures (FIXTURES_SETUP / FIXTURES_REQUIRED /
#  FIXTURES_CLEANUP): a job requiring a fixture waits for every job providing it and
#  fails without running if one of them failed; a job cleaning a fixture waits for
#  every job providing or requiring it, and runs whatever their outcome. A required
#  fixture no job provides is ignored. Jobs carry the fixture names as plain sets
#  (`provides`, `requires`, `cleans`), the processor knows nothing of ctest.


def fixtures_of(job):
    """(provides, requires, cleans) of a job: jobs unpickled from a dump older than
    the fields have none"""
    return tuple(
        set(getattr(job, k, None) or ()) for k in ("provides", "requires", "cleans")
    )


def has_fixtures(job):
    return any(fixtures_of(job))


class FixtureCycleError(ValueError): ...


class FixtureGraph:
    """Which jobs (by index) each job waits for, evaluated as jobs finish: a job is
    ready once every job it waits for is done, not in waves"""

    def __init__(self, jobs):
        n = len(jobs)
        sets = [fixtures_of(job) for job in jobs]
        providers, requirers = {}, {}
        for i, (provides, requires, _) in enumerate(sets):
            for f in provides:
                providers.setdefault(f, []).append(i)
            for f in requires:
                requirers.setdefault(f, []).append(i)

        # deps[i]: {j: fixture} - a failed j fails i with that fixture's name,
        #  {j: None} - i (a cleanup) only waits for j
        self.deps = [{} for _ in range(n)]
        for i, (_, requires, cleans) in enumerate(sets):
            for f in sorted(requires):
                for j in providers.get(f, ()):
                    if j != i:
                        self.deps[i].setdefault(j, f)
            for f in sorted(cleans):
                for j in providers.get(f, []) + requirers.get(f, []):
                    if j != i:
                        self.deps[i].setdefault(j, None)

        self.dependents = [[] for _ in range(n)]
        for i, deps in enumerate(self.deps):
            for j in deps:
                self.dependents[j].append(i)

        self._check_cycles(jobs)
        self.waiting = [len(deps) for deps in self.deps]
        self.jobs = jobs
        self.doomed = {}  # idx -> why it will not run
        self.priority = [self._n_dependents(i) for i in range(n)]

    def _check_cycles(self, jobs):
        WHITE, GREY, BLACK = 0, 1, 2
        color = [WHITE] * len(self.deps)
        for root in range(len(self.deps)):
            if color[root] != WHITE:
                continue
            color[root] = GREY
            path = [root]
            stack = [iter(self.deps[root])]
            while stack:
                j = next(stack[-1], None)
                if j is None:
                    color[path.pop()] = BLACK
                    stack.pop()
                elif color[j] == GREY:
                    cycle = path[path.index(j) :] + [j]
                    raise FixtureCycleError(
                        "fixture cycle: " + " -> ".join(jobs[k].id for k in cycle)
                    )
                elif color[j] == WHITE:
                    color[j] = GREY
                    path.append(j)
                    stack.append(iter(self.deps[j]))

    def _n_dependents(self, i):
        """jobs waiting on i, directly or not"""
        if not self.dependents[i]:
            return 0
        seen, todo = set(), list(self.dependents[i])
        while todo:
            k = todo.pop()
            if k not in seen:
                seen.add(k)
                todo += self.dependents[k]
        return len(seen)

    def ready(self, i):
        return self.waiting[i] == 0

    def done(self, j, ok):
        """j finished (ok) or failed or will not run (not ok)"""
        for i in self.dependents[j]:
            self.waiting[i] -= 1
            fixture = self.deps[i][j]
            if not ok and fixture is not None:
                self.doomed.setdefault(i, self._why(j, fixture))

    def _why(self, j, fixture):
        """a failed provider's log, or why it did not run itself"""
        if j in self.doomed:
            return f"fixture {fixture} failed ({self.jobs[j].id}: {self.doomed[j]})"
        log = getattr(self.jobs[j], "log_file_path", None)
        logs = f", see {log}.stdout/.stderr" if log else ""
        return f"fixture {fixture} failed ({self.jobs[j].id}{logs})"


def with_providers(kept, candidates):
    """`kept` plus, from `candidates`, the providers of every fixture a kept job
    requires and the cleanups of every fixture used, transitively (as `ctest -R`)"""
    kept = list(kept)
    kept_ids = {id(job) for job in kept}
    rest = [job for job in candidates if id(job) not in kept_ids]
    while True:
        required, used = set(), set()
        for job in kept:
            provides, requires, _ = fixtures_of(job)
            required |= requires
            used |= provides | requires
        add = []
        for job in rest:
            provides, _, cleans = fixtures_of(job)
            if provides & required or cleans & used:
                add.append(job)
        if not add:
            return kept
        add_ids = {id(job) for job in add}
        kept += add
        rest = [job for job in rest if id(job) not in add_ids]
