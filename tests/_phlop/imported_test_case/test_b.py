# tests/_phlop/imported_test_case/test_b.py

import unittest

from .base import ImportedBaseTest


class ImportedBTest(ImportedBaseTest):
    def test_own(self):
        pass


if __name__ == "__main__":
    unittest.main()
