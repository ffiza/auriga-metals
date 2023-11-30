import unittest
from auriga.physics import Physics


class MagnitudeToLuminosityTest(unittest.TestCase):
    """
    Some tests for the `magnitudes_to_luminosity` function defined in
    the class Physics of `physics.py`.
    """

    def test_1(self):
        physics = Physics()
        solar_luminosity = physics.magnitudes_to_luminosities(m=3.828E26)
        self.assertAlmostEqual(physics.solar_luminosity, solar_luminosity,
                               places=20)


if __name__ == '__main__':
    unittest.main()
