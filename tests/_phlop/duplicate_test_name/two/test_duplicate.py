# tests/_phlop/duplicate_test_name/two/test_duplicate.py
# same file (and sibling helper) names as ../one/, different class
# https://github.com/PhilipDeegan/phlop/issues/59

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # as a ctest WORKING_DIRECTORY would

from helper import TWO, WHO  # noqa: F401 # pylint: disable=import-error


class DuplicateTestNameTwoTest(unittest.TestCase):
    def test_run(self):
        self.assertEqual(WHO, "two")


if __name__ == "__main__":
    unittest.main()
