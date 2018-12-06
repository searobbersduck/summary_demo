# !/usr/bin/env python3

from mathfunc import *
import unittest
from parameterized import parameterized

import os

class TestMathFunc(unittest.TestCase):
    @parameterized.expand([
        (2, 1, 2),
        (2, 3, 8),
        (1, 9, 1),
        (0, 9, 0),
    ])
    def test_add(self, a1, a2, a3):
        self.assertEqual(a1, p_add(a2, a3))

    @parameterized.expand([[1,2], [2,3]])
    def test_list(self, a, b):
        self.assertEqual(a, [1, 2])

if __name__ == '__main__':
    unittest.main()