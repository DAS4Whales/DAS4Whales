import numpy as np
import pytest
from das4whales.detect import gen_linear_chirp, gen_hyperbolic_chirp, gen_template_fincall, shift_xcorr, shift_nxcorr, compute_cross_correlogram, pick_times, convert_pick_times

def test_gen_linear_chirp():
    # Test case 1
    fmin = 100
    fmax = 1000
    duration = 1
    sampling_rate = 44100
    result = gen_linear_chirp(fmin, fmax, duration, sampling_rate)
    assert len(result) == duration * sampling_rate

    # Test case 2
    # Add more test cases here

def test_gen_hyperbolic_chirp():
    # Test case 1
    fmin = 100
    fmax = 1000
    duration = 1
    sampling_rate = 44100
    result = gen_hyperbolic_chirp(fmin, fmax, duration, sampling_rate)
    assert len(result) == duration * sampling_rate

    # Test case 2
    # Add more test cases here

def test_gen_template_fincall():
    # Test case 1
    time = np.linspace(0, 1, 44100)
    fs = 44100
    fmin = 15
    fmax = 25
    duration = 1
    window = True
    result = gen_template_fincall(time, fs, fmin, fmax, duration, window)
    assert len(result) == len(time)

    # Test case 2
    # Add more test cases here

# Add more test functions for the remaining functions in detect.py
def test_shift_xcorr():
    # Test case 1
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])
    result = shift_xcorr(x, y)
    assert len(result) == len(x)

    # Test case 2
    # Add more test cases here

def test_shift_nxcorr():
    # Test case 1
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])
    result = shift_nxcorr(x, y)
    assert len(result) == len(x)

    # Test case 2
    # Add more test cases here

def test_compute_cross_correlogram():
    # Test case 1
    x = np.array([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]])
    y = np.array([5, 4, 3, 2, 1])
    result = compute_cross_correlogram(x, y)
    assert len(result) == len(x)

    # Test case 2
    # Add more test cases here

def test_pick_times():
    # Test case 1
    x = np.array([[1, 2, 3, 2, 1],[1, 2, 3, 2, 1]])
    threshold = 3
    import scipy.signal as sp
    print(abs(sp.hilbert(x[0])))
    result = pick_times(x, threshold)
    print(result)
    assert len(result) == 2

    # Test case 2
    # Add more test cases here

def test_convert_pick_times():
    # Test case 1
    x = np.array([[1, 2, 3, 2, 1],[1, 2, 3, 2, 1]])
    threshold = 3
    result = convert_pick_times(x)
    print(result)
    assert len(result) == 2

    # Test case 2
    # Add more test cases here

if __name__ == '__main__':
    pytest.main()
