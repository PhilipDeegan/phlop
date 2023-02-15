import unittest

import phlop.app.git


class PykulGitTest(unittest.TestCase):
    def test_current_branch(self):
        phlop.app.git.current_branch()

    def test_hashes(self):
        phlop.app.git.hashes()


if __name__ == "__main__":
    unittest.main()
