import time
import unittest
from multiprocessing import Process, Queue, cpu_count

from phlop.proc import ProcessNonZeroExitCode, run


class TestCaseFailure(Exception):
    ...


def test_cmd(clazz, test_id, cores):
    return f"python3 -m {clazz.__module__} {clazz.__name__}.{test_id}"


class TestBatch:
    def __init__(self, tests, cores=1):
        self.tests = tests
        self.cores = cores


def load_test_cases_in(classes, cores=1, test_cmd_fn=None):
    test_cmd_fn = test_cmd_fn if test_cmd_fn else test_cmd

    tests, loader = [], unittest.TestLoader()
    for test_class in classes:
        for suite in loader.loadTestsFromTestCase(test_class):
            tests += [test_cmd_fn(type(suite), suite._testMethodName, cores)]

    return TestBatch(tests, cores)


class CallableTest:
    def __init__(self, batch_index, cmd):
        self.batch_index = batch_index
        self.cmd = cmd
        self.run = None

    def __call__(self, **kwargs):
        self.run = run(
            self.cmd.split(),
            shell=False,
            capture_output=True,
            check=True,
            print_cmd=False,
        )
        if self.run.exitcode != 0:
            print(self.run.stderr)
        return self


class CoreCount:
    def __init__(self, cores_avail):
        self.cores_avail = cores_avail
        self.proces = []
        self.fin = []


def runner(runnable, queue):
    queue.put(runnable())


def print_tests(batches):
    for batch in batches:
        for test in batch.tests:
            print(test)


def process(batches, n_cores=None, print_only=False, fail_fast=False):
    if print_only:
        print_tests(batches)
        return

    if not isinstance(batches, list):
        batches = [batches]

    n_cores = n_cores if n_cores else cpu_count()

    cc = CoreCount(n_cores)
    assert cc.cores_avail >= max([batch.cores for batch in batches])
    cc.procs = [[] for batch in batches]
    cc.fin = [0 for batch in batches]
    pqueue = Queue()

    def launch_tests():
        for batch_index, batch in enumerate(batches):
            offset = len(cc.procs[batch_index])
            for test_index, test in enumerate(batch.tests[offset:]):
                test_index += offset
                if batch.cores <= cc.cores_avail:
                    test = CallableTest(
                        batch_index, batches[batch_index].tests[test_index]
                    )
                    cc.cores_avail -= batch.cores
                    cc.procs[batch_index] += [
                        Process(
                            target=runner,
                            args=(
                                test,
                                (pqueue),
                            ),
                        )
                    ]
                    cc.procs[batch_index][-1].daemon = True
                    cc.procs[batch_index][-1].start()

    def finished():
        b = True
        for batch_index, batch in enumerate(batches):
            b &= cc.fin[batch_index] == len(batch.tests)
        return b

    def waiter(queue):
        fail = 0
        while True:
            proc = queue.get()
            time.sleep(0.01)  # don't throttle!
            if isinstance(proc, CallableTest):
                status = "finished" if proc.run.exitcode == 0 else "FAILED"
                fail += proc.run.exitcode
                if fail_fast and fail > 0:
                    raise TestCaseFailure("Some tests have failed")
                print(proc.cmd, f"{status} in {proc.run.run_time:.2f} seconds")
                cc.cores_avail += batches[proc.batch_index].cores
                cc.fin[proc.batch_index] += 1
                launch_tests()
                if finished():
                    if fail > 0:
                        raise TestCaseFailure("Some tests have failed")
                    break

    launch_tests()
    waiter(pqueue)
