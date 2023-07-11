import unittest
from auriga.snapshot import Snapshot


class SnapshotTests(unittest.TestCase):
    """
    Some tests for the Snapshot class defined in snapshot.py.
    """

    def test_load_snapshot_exp_facts(self):
        s = Snapshot(simulation="au6_or_l4_s100",
                     loadonlytype=[4])
        s._load_snapshot_exp_facts()
        self.assertEqual(len(s._expansion_factors), s.snapnum + 1)


if __name__ == '__main__':
    unittest.main()
