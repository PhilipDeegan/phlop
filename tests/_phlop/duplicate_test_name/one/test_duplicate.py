# tests/_phlop/duplicate_test_name/one/test_duplicate.py
# same file (and sibling helper) names as ../two/, different class
# https://github.com/PhilipDeegan/phlop/issues/59

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # as a ctest WORKING_DIRECTORY would

from helper import ONE, WHO  # noqa: F401 # pylint: disable=import-error


class DuplicateTestNameOneTest(unittest.TestCase):
    def test_run(self):
        self.assertEqual(WHO, "one")


if __name__ == "__main__":
    unittest.main()
