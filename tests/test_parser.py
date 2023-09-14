import unittest
import numpy as np
import sys
import os
from auriga.parser import parse


class ParserTests(unittest.TestCase):
    """
    Some tests for the methods defined in parser.py
    """

    def test_parser_std(self):
        galaxy, rerun, resolution = parse(simulation="au6_or_l4")
        self.assertEqual(galaxy, 6)
        self.assertFalse(rerun)
        self.assertEqual(resolution, 4)

    def test_parser_1(self):
        galaxy, rerun, resolution, snapshot = parse(simulation="au19_re_l3_s5")
        self.assertEqual(galaxy, 19)
        self.assertTrue(rerun)
        self.assertEqual(resolution, 3)
        self.assertEqual(snapshot, 5)

    def test_parser_2(self):
        galaxy, rerun, resolution, snapshot = parse(
            simulation="au30_or_l2_s127")
        self.assertEqual(galaxy, 30)
        self.assertFalse(rerun)
        self.assertEqual(resolution, 2)
        self.assertEqual(snapshot, 127)

    def test_parser_3(self):
        galaxy, rerun, resolution, snapshot = parse(
            simulation="au24_re_l4_s251")
        self.assertEqual(galaxy, 24)
        self.assertTrue(rerun)
        self.assertEqual(resolution, 4)
        self.assertEqual(snapshot, 251)

    def test_parser_4(self):
        galaxy, rerun, resolution = parse(simulation="au1_or_l4")
        self.assertEqual(galaxy, 1)
        self.assertFalse(rerun)
        self.assertEqual(resolution, 4)


if __name__ == '__main__':
    unittest.main()
