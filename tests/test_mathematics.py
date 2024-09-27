import unittest

from auriga.mathematics import round_to_1, get_decimal_places


class MathTests(unittest.TestCase):
    def test_rounding_int(self):
        x = 1234
        self.assertAlmostEqual(1000.0, round_to_1(x))
    
    def test_rounding_01(self):
        x = 0.00364
        self.assertAlmostEqual(0.004, round_to_1(x))
    
    def test_rounding_02(self):
        x = 0.000000012222
        self.assertAlmostEqual(0.00000001, round_to_1(x))
    
    def test_rounding_03(self):
        x = 1728.21
        self.assertAlmostEqual(2000.0, round_to_1(x))
    
    def test_rounding_04(self):
        x = -0.00364
        self.assertAlmostEqual(-0.004, round_to_1(x))    

    def test_decimals_int(self):
        x = 10
        self.assertEqual(1, get_decimal_places(x))
    
    def test_decimals_01(self):
        x = 10.0
        self.assertEqual(1, get_decimal_places(x))
    
    def test_decimals_02(self):
        x = 10.003
        self.assertEqual(3, get_decimal_places(x))

    def test_decimals_03(self):
        x = 0.004704330001
        self.assertEqual(12, get_decimal_places(x))

    def test_decimals_04(self):
        x = 93.9274054
        self.assertEqual(7, get_decimal_places(x))

    def test_decimals_05(self):
        x = -4.38263
        self.assertEqual(5, get_decimal_places(x))
    
    def test_decimals_06(self):
        x = -0.0747830001
        self.assertEqual(10, get_decimal_places(x))


if __name__ == '__main__':
    unittest.main()
