import unittest
import numpy as np
from auriga.snapshot import Snapshot


class TestStellarInSituTag(unittest.TestCase):
    """
    Some tests for the method `tag_in_situ_stars` defined in `snapshot.py`.
    """
    def test_non_stellar_type(self):
        s = Snapshot(simulation="au6_or_l4_s127", loadonlytype=[0, 1, 2, 3, 5])
        s.tag_in_situ_stars()
        self.assertTrue(
            (s.is_in_situ.max() == -1) & (s.is_in_situ.min() == -1))

    def test_star_in_au6(self):
        s = Snapshot(simulation="au6_or_l4_s127", loadonlytype=[4])
        s.tag_in_situ_stars()

        star_idx = np.where(s.type == 4)[0][0]

        # Check where this star was born
        origin_idxs = s.stellar_origin_idx[star_idx]

        # Main object idx for Au6 is (0, 0) always
        if ((origin_idxs[0] == 0) & (origin_idxs[1] == 0)):
            self.assertTrue(s.is_in_situ[star_idx])
        else:
            self.assertFalse(s.is_in_situ[star_idx])


if __name__ == '__main__':
    unittest.main()
