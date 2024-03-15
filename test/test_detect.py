import unittest 
from numpy.testing import assert_array_equal
import numpy as np
import das4whales as dw


class TestMyFunction(unittest.TestCase):
    def test_gen_linear_chirp(self):
        y = dw.detect.gen_linear_chirp(10, 20, 1, 100)
        self.assertEqual(y.shape, (100,))
        self.assertAlmostEqual(y[0], 10)
        self.assertAlmostEqual(y[-1], 20)

    def test_gen_hyperbolic_chirp(self):
        y = dw.detect.gen_hyperbolic_chirp(10, 20, 1, 100)
        self.assertEqual(y.shape, (100,))
        self.assertAlmostEqual(y[0], 10)
        self.assertAlmostEqual(y[-1], 20)

    def test_gen_template_fincall(self):
        y = dw.detect.gen_template_fincall(np.arange(100) / 100, 100, window=False)
        self.assertEqual(y.shape, (100,))
        self.assertAlmostEqual(y[0], 0)
        self.assertAlmostEqual(y[-1], 0)

    def test_shift_xcorr(self):
        x = np.arange(10)
        y = np.arange(5, 15)
        corr = dw.detect.shift_xcorr(x, y)
        self.assertEqual(corr.shape, (10,))
        self.assertAlmostEqual(corr[0], 0)
        self.assertAlmostEqual(corr[-1], 14)

    def test_shift_nxcorr(self):
        x = np.arange(10)
        y = np.arange(5, 15)
        corr = dw.detect.shift_nxcorr(x, y)
        self.assertEqual(corr.shape, (10,))
        self.assertAlmostEqual(corr[0], 0)
        self.assertAlmostEqual(corr[-1], 14)

    def test_compute_cross_correlogram(self):
        data = np.random.rand(10, 100)
        template = np.random.rand(100)
        corr_m = dw.detect.compute_cross_correlogram(data, template)
        self.assertEqual(corr_m.shape, (10, 100))

    def test_pick_times(self):
        corr_m = np.random.rand(10, 100)
        peaks_indexes_m = dw.detect.pick_times(corr_m, 100)
        self.assertEqual(peaks_indexes_m[0].shape, (10,))

    def test_convert_pick_times(self):
        peaks_indexes_m = [np.arange(10), np.arange(10)]
        peaks_indexes_tp = dw.detect.convert_pick_times(peaks_indexes_m)
        assert_array_equal(peaks_indexes_tp[0], np.repeat(np.arange(10), 10))
        assert_array_equal(peaks_indexes_tp[1], np.tile(np.arange(10), 10))


if __name__ == '__main__':
    unittest.main()