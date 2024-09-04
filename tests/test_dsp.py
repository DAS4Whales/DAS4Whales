import numpy as np
import pytest
from das4whales.dsp import *

# Test transformations
#TODO: fix those tests
# def test_get_fx():
#     trace = np.array([1, 2, 3, 4, 5])
#     nfft = 4
#     fx = get_fx(trace, nfft)
#     assert np.allclose(fx, np.array([1, 2, 3, 4]))

# def test_get_spectrogram():
#     waveform = np.array([1, 2, 3, 4, 5])
#     fs = 10
#     nfft = 4
#     overlap_pct = 0.5
#     spectrogram = get_spectrogram(waveform, fs, nfft, overlap_pct)
#     assert spectrogram.shape == (3, 3)

def test_fk_filter_design():
    trace_shape = (10, 10)
    selected_channels = [0, 1, 2]
    dx = 1
    fs = 100
    cs_min = 1400
    cp_min = 1450
    cp_max = 3400
    cs_max = 3500
    fk_filter_matrix = fk_filter_design(trace_shape, selected_channels, dx, fs, cs_min, cp_min, cp_max, cs_max)
    assert fk_filter_matrix.shape == (10, 10)

def test_hybrid_filter_design():
    trace_shape = (10, 10)
    selected_channels = [0, 1, 2]
    dx = 1
    fs = 100
    cs_min = 1400
    cp_min = 1450
    fmin = 15
    fmax = 25
    hybrid_filter_matrix = hybrid_filter_design(trace_shape, selected_channels, dx, fs, cs_min, cp_min, fmin, fmax)
    assert hybrid_filter_matrix.shape == (10, 10)

def test_hybrid_ninf_filter_design():
    trace_shape = (10, 10)
    selected_channels = [0, 1, 2]
    dx = 1
    fs = 100
    cs_min = 1400
    cp_min = 1450
    cp_max = 3400
    cs_max = 3500
    fmin = 15
    fmax = 25
    hybrid_ninf_filter_matrix = hybrid_ninf_filter_design(trace_shape, selected_channels, dx, fs, cs_min, cp_min, cp_max, cs_max, fmin, fmax)
    assert hybrid_ninf_filter_matrix.shape == (10, 10)

def test_hybrid_gs_filter_design():
    trace_shape = (10, 10)
    selected_channels = [0, 1, 2]
    dx = 1
    fs = 100
    cs_min = 1400
    cp_min = 1450
    fmin = 15
    fmax = 25
    hybrid_gs_filter_matrix = hybrid_gs_filter_design(trace_shape, selected_channels, dx, fs, cs_min, cp_min, fmin, fmax)
    assert hybrid_gs_filter_matrix.shape == (10, 10)

def test_hybrid_ninf_gs_filter_design():
    trace_shape = (10, 10)
    selected_channels = [0, 1, 2]
    dx = 1
    fs = 100
    fk_params = {
        'c_min': 1400,
        'c_max': 3500,
        'fmin': 15,
        'fmax': 25
    }
    hybrid_ninf_gs_filter_matrix = hybrid_ninf_gs_filter_design(trace_shape, selected_channels, dx, fs, fk_params)
    assert hybrid_ninf_gs_filter_matrix.shape == (10, 10)

def test_taper_data():
    trace = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=float)
    tapered_trace = taper_data(trace)
    assert np.allclose(tapered_trace, np.array([[0, 2, 3, 4, 0], [0, 2, 3, 4, 0]]))

# def test_fk_filter_filt():
#     trace = np.array([1, 2, 3, 4, 5], dtype=float)
#     trace = np.tile(trace, 5)
#     print(trace)
#     fk_filter_matrix = np.fft.fftshift(np.fft.fft2(np.eye(5)))
#     filtered_trace = fk_filter_filt(trace, fk_filter_matrix)
#     assert np.allclose(filtered_trace, trace)

# def test_fk_filter_sparsefilt():
#     trace = np.array([1, 2, 3, 4, 5])
#     fk_filter_matrix = np.eye(5)
#     filtered_trace = fk_filter_sparsefilt(trace, fk_filter_matrix)
#     assert np.allclose(filtered_trace, trace)

def test_butterworth_filter():
    filterspec = [4, 1000, 'lp']
    fs = 10000
    b, a = butterworth_filter(filterspec, fs)
    assert len(b) == 6
    assert len(a) == 6

def test_instant_freq():
    channel = np.array([1, 2, 3, 4, 5])
    fs = 10
    inst_freq = instant_freq(channel, fs)
    assert len(inst_freq) == len(channel) - 1

# def test_bp_filt():
#     data = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=float)
#     fs = 5
#     fmin = 1
#     fmax = 2
#     filtered_data = bp_filt(data, fs, fmin, fmax)
#     assert np.shape(filtered_data) == np.shape(data)

def test_fk_filt():
    data = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=float)
    tint = 1
    fs = 100
    xint = 1
    dx = 1
    c_min = 10
    c_max = 3000
    filtered_data = fk_filt(data, tint, fs, xint, dx, c_min, c_max)
    assert np.shape(filtered_data) == np.shape(data)

def test_snr_tr_array():
    trace = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=float)
    snr = snr_tr_array(trace)
    print(snr)
    assert np.allclose(snr, np.array([[ 2.89413593,  4.69411612,  6.73126528,  9.51687805, 11.44487452],
                                      [ 2.89413593,  4.69411612,  6.73126528,  9.51687805, 11.44487452]]))


if __name__ == "__main__":
    pytest.main()