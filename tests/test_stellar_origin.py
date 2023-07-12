import unittest
from auriga.snapshot import Snapshot


class TestStellarOriginCalculation(unittest.TestCase):
    """
    Some tests for the method `add_stellar_origin` defined in `snapshot.py`.
    """

    def test_1(self):
        s = Snapshot(simulation="au6_or_l4_s127", loadonlytype=[4])
        s.add_stellar_origin()
        self.assertTrue(
            s.stellar_origin_idx[s.stellar_formation_time <= 0].max() == -1)
        self.assertTrue(
            s.stellar_origin_idx[s.stellar_formation_time <= 0].min() == -1)


if __name__ == '__main__':
    unittest.main()
