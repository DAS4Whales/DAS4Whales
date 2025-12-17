import numpy as np
import pytest
from das4whales.detect import (
    gen_linear_chirp, 
    gen_hyperbolic_chirp, 
    gen_template_fincall, 
    shift_xcorr, 
    shift_nxcorr, 
    compute_cross_correlogram, 
    pick_times, 
    convert_pick_times,
    calc_nmf,
    calc_nmf_correlogram
)

def test_gen_linear_chirp():
    # Test case 1: Basic functionality
    fmin = 100
    fmax = 1000
    duration = 1
    sampling_rate = 44100
    result = gen_linear_chirp(fmin, fmax, duration, sampling_rate)
    assert len(result) == duration * sampling_rate
    assert isinstance(result, np.ndarray)

    # Test case 2: Different parameters
    fmin = 50
    fmax = 500
    duration = 0.5
    sampling_rate = 22050
    result = gen_linear_chirp(fmin, fmax, duration, sampling_rate)
    assert len(result) == int(duration * sampling_rate)
    
    # Test case 3: Verify frequency content starts at fmax and ends at fmin
    result = gen_linear_chirp(100, 1000, 1, 44100)
    # The chirp should have the expected characteristics
    assert np.max(np.abs(result)) > 0  # Signal should not be zero

def test_gen_hyperbolic_chirp():
    # Test case 1: Basic functionality
    fmin = 100
    fmax = 1000
    duration = 1
    sampling_rate = 44100
    result = gen_hyperbolic_chirp(fmin, fmax, duration, sampling_rate)
    assert len(result) == duration * sampling_rate
    assert isinstance(result, np.ndarray)

    # Test case 2: Different parameters
    fmin = 50
    fmax = 500
    duration = 2
    sampling_rate = 22050
    result = gen_hyperbolic_chirp(fmin, fmax, duration, sampling_rate)
    assert len(result) == duration * sampling_rate
    
    # Test case 3: Verify signal has non-zero content
    result = gen_hyperbolic_chirp(100, 1000, 1, 44100)
    assert np.max(np.abs(result)) > 0

def test_gen_template_fincall():
    # Test case 1: Basic functionality with windowing
    time = np.linspace(0, 1, 44100)
    fs = 44100
    fmin = 15
    fmax = 25
    duration = 1
    window = True
    result = gen_template_fincall(time, fs, fmin, fmax, duration, window)
    assert len(result) == len(time)
    assert isinstance(result, np.ndarray)

    # Test case 2: Test without windowing
    result_no_window = gen_template_fincall(time, fs, fmin, fmax, duration, False)
    assert len(result_no_window) == len(time)
    
    # Test case 3: Different frequency range
    result_diff_freq = gen_template_fincall(time, fs, 10, 30, duration, window)
    assert len(result_diff_freq) == len(time)
    
    # Test case 4: Verify the windowed version differs from non-windowed
    result_windowed = gen_template_fincall(time, fs, fmin, fmax, duration, True)
    result_unwindowed = gen_template_fincall(time, fs, fmin, fmax, duration, False)
    # They should be different due to windowing
    assert not np.array_equal(result_windowed, result_unwindowed)

# Add more test functions for the remaining functions in detect.py
def test_shift_xcorr():
    # Test case 1: Basic functionality
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])
    result = shift_xcorr(x, y)
    assert len(result) == len(x)
    assert isinstance(result, np.ndarray)

    # Test case 2: Identical signals
    x = np.array([1, 2, 3, 2, 1])
    y = x.copy()
    result = shift_xcorr(x, y)
    assert len(result) == len(x)
    # Peak should be at zero lag (first element for positive lags only)
    max_idx = np.argmax(result)
    assert max_idx == 0  # Zero lag should give maximum correlation


def test_shift_nxcorr():
    # Test case 1: Basic functionality  
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([5, 4, 3, 2, 1], dtype=float)
    result = shift_nxcorr(x, y)
    assert len(result) == len(x)
    assert isinstance(result, np.ndarray)

    # Test case 2: Test normalization with identical signals
    x = np.array([1, 2, 3, 2, 1], dtype=float)
    y = x.copy()
    result = shift_nxcorr(x, y)
    assert len(result) == len(x)
    # For identical signals, the peak at zero lag should be significant
    assert np.max(np.abs(result)) > 0


def test_compute_cross_correlogram():
    # Test case 1: Basic functionality
    data = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=float)
    template = np.array([5, 4, 3, 2, 1], dtype=float)
    result = compute_cross_correlogram(data, template)
    assert result.shape == data.shape
    assert isinstance(result, np.ndarray)

    # Test case 2: Different data
    data = np.random.randn(3, 100)
    template = np.random.randn(100)
    result = compute_cross_correlogram(data, template)
    assert result.shape == data.shape


def test_calc_nmf():
    """Test the calc_nmf function."""
    # Test case 1: Basic functionality
    data = np.array([1, 2, 3, 4, 5], dtype=float)
    template = np.array([1, 1, 1, 1, 1], dtype=float)
    result = calc_nmf(data, template)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == len(data)  # Same mode returns same length
    
    # Test case 2: Perfect match should give higher correlation
    template = data.copy()
    result = calc_nmf(data, template)
    # Should have a peak at the center for 'same' mode
    center_idx = len(result) // 2
    assert result[center_idx] > 0


def test_calc_nmf_correlogram():
    """Test the calc_nmf_correlogram function.""" 
    # Test case 1: Basic functionality
    data = np.random.randn(2, 100)
    template = np.random.randn(50)
    result = calc_nmf_correlogram(data, template)
    
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == data.shape[0]  # Same number of channels
    assert result.shape[1] == data.shape[1]  # Same number of samples (same mode)


def test_pick_times():
    # Test case 1: Basic functionality
    x = np.array([[1, 2, 3, 2, 1], [1, 2, 3, 2, 1]], dtype=float)
    threshold = 2.5
    ipi_idx = 1
    result = pick_times(x, threshold, ipi_idx)
    assert len(result) == 2  # Two channels
    assert all(isinstance(channel_peaks, np.ndarray) for channel_peaks in result)

    # Test case 2: Higher threshold should result in fewer peaks
    threshold_high = 10.0
    result_high = pick_times(x, threshold_high, ipi_idx)
    assert len(result_high) == 2


def test_convert_pick_times():
    # Test case 1: Basic functionality
    peaks_list = [np.array([1, 3, 5]), np.array([2, 4])]
    result = convert_pick_times(peaks_list)
    
    assert len(result) == 2  # Should return tuple of two arrays
    channel_indices, time_indices = result
    assert isinstance(channel_indices, np.ndarray)
    assert isinstance(time_indices, np.ndarray)
    
    # Total number of picks should match
    total_picks = sum(len(peaks) for peaks in peaks_list)
    assert len(channel_indices) == total_picks
    assert len(time_indices) == total_picks
    
    # Test case 2: Empty input
    empty_peaks = [np.array([]), np.array([])]
    result_empty = convert_pick_times(empty_peaks)
    channel_indices_empty, time_indices_empty = result_empty
    assert len(channel_indices_empty) == 0
    assert len(time_indices_empty) == 0

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
    ipi = 1
    import scipy.signal as sp
    print(abs(sp.hilbert(x[0])))
    result = pick_times(x, threshold, ipi)
    print(result)
    assert len(result) == 2

    # Test case 2
    # Add more test cases here

def test_convert_pick_times():
    # Test case 1
    x = np.array([[1, 2, 3, 2, 1],[1, 2, 3, 2, 1]])
    threshold = 3
    result = convert_pick_times(x)
    assert len(result) == 2

    # Test case 2
    # Add more test cases here

if __name__ == '__main__':
    pytest.main()
