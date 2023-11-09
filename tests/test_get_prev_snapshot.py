import unittest
import numpy as np
import sys
import os
from auriga.support import get_name_of_previous_snapshot


class ParserTests(unittest.TestCase):
    """
    Some tests for a method in support.py.
    """

    def test_originals(self):
        for i in range(1, 31):
            this_simulation = f"au{i}_or_l4_s127"
            prev_simulation = get_name_of_previous_snapshot(this_simulation)
            self.assertEqual(prev_simulation, f"au{i}_or_l4_s126")
    
    def test_au6_or(self):
        for i in range(1, 127):
            this_simulation = f"au6_or_l4_s{i}"
            prev_simulation = get_name_of_previous_snapshot(this_simulation)
            self.assertEqual(prev_simulation, f"au6_or_l4_s{i - 1}")

    def test_au6_re(self):
        for i in range(1, 251):
            this_simulation = f"au6_or_l4_s{i}"
            prev_simulation = get_name_of_previous_snapshot(this_simulation)
            self.assertEqual(prev_simulation, f"au6_or_l4_s{i - 1}")

    def test_first_snapshot(self):
        this_simulation = "au6_or_l4_s0"
        self.assertRaises(ValueError, get_name_of_previous_snapshot,
                          this_simulation)

    def test_wrong_format(self):
        this_simulation = "au6_or_l4"
        self.assertRaises(ValueError, get_name_of_previous_snapshot,
                          this_simulation)

if __name__ == '__main__':
    unittest.main()
