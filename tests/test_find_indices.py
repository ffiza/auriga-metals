import unittest
import numpy as np
from auriga.support import find_indices


class FindIndicesTest(unittest.TestCase):
    """
    Some tests for the `find_indices` function defined in `support.py`.
    """

    def test_1(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7])
        b = np.array([2, 7, 8, 3, 2, 19, 22, 14])
        indices = find_indices(a=a, b=b)
        results = np.array([1, 6, -1, 2, 1, -1, -1, -1])
        self.assertTrue((indices == results).all())

    def test_2(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7])
        b = np.array([19, 22, -5])
        indices = find_indices(a=a, b=b)
        results = np.array([-1, -1, -1])
        self.assertTrue((indices == results).all())

    def test_3(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7])
        b = np.array([19, 22, -5])
        indices = find_indices(a=a, b=b, invalid_specifier=2)
        results = np.array([2, 2, 2])
        self.assertTrue((indices == results).all())

    def test_4(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7])
        b = np.array([2, 7, 8, 3, 2, 19, 22, 14])
        indices = find_indices(a=a, b=b)
        self.assertTrue(len(b) == len(indices))

    def test_5(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7])
        b = np.array([2, 7, 8, 3, 2, 19, 22, 14])
        indices = find_indices(a=a, b=b)
        self.assertTrue((a[indices[indices >= 0]] == b[indices >= 0]).all())

    def test_6(self):
        a = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
        b = np.array([2, 43, -2, -27, 8, 3, 2, 19, 22, 14])
        indices = find_indices(a=a, b=b)
        self.assertTrue((a[indices[indices >= 0]] == b[indices >= 0]).all())


if __name__ == '__main__':
    unittest.main()
