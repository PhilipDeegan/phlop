#
#

import os
import sys
import unittest


class PhlopGitTestCasesTest(unittest.TestCase):
    #

    def test_fails(self):
        """Force some test failures conditionally to see exit codes in bash etc"""
        force_failure = int(os.environ.get("PHLOP_FORCE_TEST_CASE_FAILURE", 0))

        if force_failure == 1:
            raise RuntimeError("Fail")
        if force_failure == 2:
            sys.exit(1)
        if force_failure == 3:
            self.assertEqual(1, 2)
        if force_failure == 4:
            self.fail("force_failure")


if __name__ == "__main__":
    unittest.main()
