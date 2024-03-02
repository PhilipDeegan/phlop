#
#
#
#
#


import time
from enum import Enum
from multiprocessing import Process, Queue, cpu_count

from phlop.proc import ProcessNonZeroExitCode, run
from phlop.testing.test_cases import *


class TestCaseFailure(Exception):
    ...


class LoggingMode(Enum):
    OFF = 0
    ON_NON_ZERO_EXIT_CODE = 1
    ALWAYS = 2


class CallableTest:
    def __init__(self, batch_index, test_case, logging):
        self.batch_index = batch_index
        self.test_case = test_case
        self.run = None
        self.logging = logging

    def __call__(self, **kwargs):
        self.run = run(
            self.test_case.cmd.split(),
            shell=False,
            capture_output=True,
            check=True,
            print_cmd=False,
            env=self.test_case.env,
            working_dir=self.test_case.working_dir,
            log_file_path=self.test_case.log_file_path,
            logging=self.logging,
        )
        if self.run.stderr and self.run.exitcode != 0:
            print(self.run.stderr)
        return self


class CoreCount:
    def __init__(self, cores_avail):
        self.cores_avail = cores_avail
        self.procs = []
        self.fin = []


def runner(runnable, queue):
    queue.put(runnable())


def print_tests(batches):
    for batch in batches:
        for test in batch.tests:
            print(test.cmd)


def process(
    batches, n_cores=None, print_only=False, fail_fast=False, logging: LoggingMode = 1
):
    if not isinstance(batches, list):
        batches = [batches]
    if sum([len(t.tests) for t in batches]) == 0:
        return  # nothing to do

    if print_only:
        print_tests(batches)
        return

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
                        batch_index, batches[batch_index].tests[test_index], logging
                    )
                    cc.cores_avail -= batch.cores
                    cc.procs[batch_index] += [
                        Process(
                            target=runner,
                            args=(test, (pqueue)),
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
                print(
                    proc.test_case.cmd, f"{status} in {proc.run.run_time:.2f} seconds"
                )
                cc.cores_avail += batches[proc.batch_index].cores
                cc.fin[proc.batch_index] += 1
                if finished():
                    if fail > 0:
                        raise TestCaseFailure("Some tests have failed")
                    break
                launch_tests()

    launch_tests()
    waiter(pqueue)
