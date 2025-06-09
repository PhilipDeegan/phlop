#
#
#
#
#

import json
import os
from enum import Enum
from multiprocessing import Process, Queue, cpu_count

from phlop.logger import getLogger
from phlop.os import read_file, read_last_lines_of
from phlop.proc import run

timeout = 60 * 60
logger = getLogger(__name__)


FAIL_FAST = bool(json.loads(os.environ.get("PHLOP_FAIL_FAST", "false")))


class TestCaseFailure(Exception): ...


class LoggingMode(Enum):
    OFF = 0
    ON_NON_ZERO_EXIT_CODE = 1
    ALWAYS = 2


class CallableTest:
    def __init__(self, batch_index, test_case, logging):
        self.batch_index = batch_index
        self.test_case = test_case
        self.logging = logging
        self.run = None

    def __call__(self, **kwargs):
        self.run = run(
            self.test_case.cmd.split(),
            shell=False,
            capture_output=False,
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

    def print_log_files(self):
        print(self.run.stdout)
        print(self.run.stderr)
        if self.test_case.log_file_path:
            print(read_file(f"{self.test_case.log_file_path}.stdout"))
            print(read_file(f"{self.test_case.log_file_path}.stdout"))


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


def print_pending(cc, batches, logging):
    print("pending jobs start")
    for batch_index, batch in enumerate(batches):
        for pid, proc in enumerate(cc.procs[batch_index]):
            if not proc.is_alive():
                continue
            test_case = batch.tests[pid]
            print("cmd: ", test_case.cmd)
            if test_case.log_file_path:
                stdout = read_last_lines_of(f"{test_case.log_file_path}.stdout")
                if stdout:
                    print(f"stdout:{os.linesep}", " ".join(stdout))
                stderr = read_last_lines_of(f"{test_case.log_file_path}.stderr")
                if stderr:
                    print(f"stderr:{os.linesep}", " ".join(stderr))
    print("pending jobs end")


def process(
    batches, n_cores=None, print_only=False, fail_fast=None, logging: LoggingMode = 1
):
    if not isinstance(batches, list):
        batches = [batches]
    if sum([len(t.tests) for t in batches]) == 0:
        return  # nothing to do

    if print_only:
        print_tests(batches)
        return

    fail_fast = fail_fast if fail_fast is not None else FAIL_FAST

    n_cores = n_cores if n_cores else cpu_count()

    cc = CoreCount(n_cores)
    assert cc.cores_avail >= max([batch.cores for batch in batches])
    cc.procs = [[] for batch in batches]
    cc.fin = [0 for batch in batches]
    pqueue = Queue()

    def launch_tests():
        for batch_index, batch in enumerate(batches):
            offset = len(cc.procs[batch_index])
            for test_index, test_case in enumerate(batch.tests[offset:]):
                test_index += offset
                if batch.cores <= cc.cores_avail:
                    test = CallableTest(batch_index, test_case, logging)
                    cc.cores_avail -= batch.cores
                    cc.procs[batch_index] += [
                        Process(target=runner, args=(test, pqueue))
                    ]
                    cc.procs[batch_index][-1].daemon = True
                    cc.procs[batch_index][-1].start()

    def finished():
        for batch_index, batch in enumerate(batches):
            if cc.fin[batch_index] != len(batch.tests):
                return False
        return True

    def waiter(queue):
        fail = 0
        while True:
            proc = None
            try:
                proc = queue.get(timeout=timeout)

            except Exception:
                logger.info("Queue timeout - polling")
                print_pending(cc, batches, logging)
                continue

            if isinstance(proc, CallableTest):
                status = "finished" if proc.run.exitcode == 0 else "FAILED"
                print(
                    proc.test_case.cmd, f"{status} in {proc.run.run_time:.2f} seconds"
                )
                if proc.run.exitcode != 0:
                    proc.print_log_files()
                    if fail_fast:
                        raise TestCaseFailure("Some tests have failed")
                fail += proc.run.exitcode
                cc.cores_avail += batches[proc.batch_index].cores
                cc.fin[proc.batch_index] += 1
                if finished():
                    if fail > 0:
                        raise TestCaseFailure("Some tests have failed")
                    break
                launch_tests()

    launch_tests()
    waiter(pqueue)
