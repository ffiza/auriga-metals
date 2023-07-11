import unittest
import numpy as np
import sys
import os
from auriga.support import find_idx_ksmallest


class SupportMethodsTests(unittest.TestCase):
    """
    Some tests for the methods defined in support.py
    """

    def test_find_ksmallest_1(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        k: int = 7
        ksmallest = find_idx_ksmallest(arr=arr, k=k)
        self.assertCountEqual(list(arr[:k]), list(arr[ksmallest]))

    def test_find_ksmallest_2(self):
        arr = np.array([-1.0, 2.2, 3, 5.5, 6, 7, 8, 9, 2.2, 11, 12, 14])
        k: int = 4
        res = np.array([-1.0, 2.2, 2.2, 3])
        ksmallest = find_idx_ksmallest(arr=arr, k=k)
        self.assertCountEqual(list(res), list(arr[ksmallest]))

    def test_find_ksmallest_3(self):
        arr = np.array([-1.0, -1.0, 3, 5.5, 6, -1.0, 8, 9, -1.0, 11, 12, 14])
        k: int = 3
        res = np.array([-1.0, -1.0, -1.0])
        ksmallest = find_idx_ksmallest(arr=arr, k=k)
        self.assertCountEqual(list(res), list(arr[ksmallest]))

    def test_find_ksmallest_4(self):
        arr = np.array([-1.0, -1.0, 3, 5.5, 6, -1.0, 8, 9, -1.0, 11, 12, 14])
        k: int = 0
        res = np.array([])
        ksmallest = find_idx_ksmallest(arr=arr, k=k)
        self.assertCountEqual(list(res), list(arr[ksmallest]))


if __name__ == '__main__':
    unittest.main()
