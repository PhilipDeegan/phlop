# tests/_phlop/imported_test_case/base.py
# imported by test_a.py and test_b.py - its own tests must be collected once, here


import unittest


class ImportedBaseTest(unittest.TestCase):
    def test_shared(self):
        pass
