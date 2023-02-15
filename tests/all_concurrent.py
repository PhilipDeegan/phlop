"""
  This script exists to minimize testing time by running all simulation/tests
    concurrently without needing to wait for any particular file or set of tests
"""

import json
import os
import unittest
from multiprocessing import cpu_count
from pathlib import Path

from phlop.reflection import classes_in_directory
from phlop.testing.parallel_processor import load_test_cases_in, process

N_CORES = int(os.environ["N_CORES"]) if "N_CORES" in os.environ else cpu_count()
PRINT = bool(json.loads(os.environ.get("PRINT", "false")))


def get_test_classes():
    return load_test_cases_in(
        classes_in_directory(str(Path("tests") / "phlop"), unittest.TestCase)
    )


if __name__ == "__main__":
    process(get_test_classes(), n_cores=N_CORES, print_only=PRINT)
