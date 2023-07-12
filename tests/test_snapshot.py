import unittest
from auriga.snapshot import Snapshot


class SnapshotTests(unittest.TestCase):
    """
    Some tests for the Snapshot class defined in snapshot.py.
    """

    def test_load_snapshot_exp_facts(self):
        s = Snapshot(simulation="au6_or_l4_s100", loadonlytype=[4])
        exp_facts = s._load_snapshot_exp_facts()
        self.assertEqual(len(exp_facts), s.snapnum + 1)

    def test_stellar_formation_snapshot(self):
        s = Snapshot(simulation="au6_or_l4_s100", loadonlytype=[4])
        stellar_formation_snapshot = \
            s._calculate_stellar_formation_snapshot()
        self.assertTrue(stellar_formation_snapshot[0] == 81)
        self.assertTrue(stellar_formation_snapshot[1] == 53)
        self.assertTrue(stellar_formation_snapshot[2] == 56)
        self.assertTrue(stellar_formation_snapshot[3] == 49)
        self.assertTrue(stellar_formation_snapshot[4] == 39)
        self.assertTrue(stellar_formation_snapshot[5] == 59)
        self.assertTrue(stellar_formation_snapshot[8] == 60)

    def test_stellar_formation_snapshot_max(self):
        s = Snapshot(simulation="au6_or_l4_s100", loadonlytype=[4])
        stellar_formation_snapshot = \
            s._calculate_stellar_formation_snapshot()
        self.assertTrue(stellar_formation_snapshot.max() <= 100)

        s = Snapshot(simulation="au6_or_l4_s99", loadonlytype=[4])
        stellar_formation_snapshot = \
            s._calculate_stellar_formation_snapshot()
        self.assertTrue(stellar_formation_snapshot.max() <= 99)

    def test_stellar_formation_snapshot_for_non_stellar_part(self):
        s = Snapshot(simulation="au6_or_l4_s100", loadonlytype=[0, 4])
        stellar_formation_snapshot = \
            s._calculate_stellar_formation_snapshot()
        self.assertTrue(stellar_formation_snapshot[s.type == 0].max() == -1)
        self.assertTrue(stellar_formation_snapshot[s.type == 0].min() == -1)


if __name__ == '__main__':
    unittest.main()
