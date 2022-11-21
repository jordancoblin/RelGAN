import numpy as np
import numpy.testing as np_test
import unittest
import sparsemax

class TestSparsemax(unittest.TestCase):
    def test_sparsemax(self):
        z = np.array([2.5, 0.2, 0.1, 3, 0.1, 2.5])
        s = sparsemax.sparsemax(z)

        expected = np.array([0.1667, 0.0000, 0.0000, 0.6667, 0.0000, 0.1667])
        np_test.assert_array_almost_equal(s, expected, decimal=4)

if __name__ == '__main__':
    unittest.main()