import numpy as np
import pytest

from das4whales.dsp import (
    # fk_filter_filt,
    # fk_filter_sparsefilt,
    bp_filt,
    butterworth_filter,
    fk_filt,
    fk_filter_design,
    get_fx,
    get_spectrogram,
    hybrid_filter_design,
    hybrid_gs_filter_design,
    hybrid_ninf_filter_design,
    hybrid_ninf_gs_filter_design,
    instant_freq,
    resample,
    snr_tr_array,
    taper_data,
)


def test_resample():
    """Test the resample function."""
    # Test case 1: Downsampling
    tr = np.random.randn(2, 1000)  # 2 channels, 1000 samples
    fs = 1000
    desired_fs = 500

    tr_resampled, fs_new, tx_new = resample(tr, fs, desired_fs)

    assert fs_new == desired_fs
    assert tr_resampled.shape[0] == tr.shape[0]  # Same number of channels
    assert tr_resampled.shape[1] == 500  # Half the samples
    assert len(tx_new) == tr_resampled.shape[1]

    # Test case 2: Upsampling
    tr = np.random.randn(3, 500)
    fs = 500
    desired_fs = 1000

    tr_resampled, fs_new, tx_new = resample(tr, fs, desired_fs)

    assert fs_new == desired_fs
    assert tr_resampled.shape[0] == tr.shape[0]
    assert tr_resampled.shape[1] == 1000  # Double the samples
    assert len(tx_new) == tr_resampled.shape[1]


def test_get_fx():
    """Test the get_fx function."""
    # Test case 1: Basic functionality
    trace = np.random.randn(3, 512)
    nfft = 512

    fx = get_fx(trace, nfft)

    assert fx.shape[0] == trace.shape[0]  # Same number of channels
    assert fx.shape[1] == nfft  # FFT length
    assert np.all(fx >= 0)  # Should be magnitude spectrum

    # Test case 2: Different FFT length
    nfft = 256
    fx = get_fx(trace, nfft)
    assert fx.shape[1] == nfft


def test_get_spectrogram():
    """Test the get_spectrogram function."""
    # Test case 1: Basic functionality
    waveform = np.random.randn(1000)
    fs = 1000.0
    nfft = 128
    overlap_pct = 0.8

    Sxx, t, f = get_spectrogram(waveform, fs, nfft, overlap_pct)

    assert len(f) == nfft // 2 + 1  # Frequency bins
    assert len(t) > 0  # Time bins
    assert Sxx.shape == (len(f), len(t))
    assert np.all(f >= 0)  # Frequencies should be positive

    # Test case 2: Different parameters
    nfft = 64
    overlap_pct = 0.5
    Sxx2, t2, f2 = get_spectrogram(waveform, fs, nfft, overlap_pct)

    assert len(f2) == nfft // 2 + 1
    assert Sxx2.shape == (len(f2), len(t2))


def test_fk_filter_design():
    trace_shape = (10, 10)
    selected_channels = [0, 1, 2]
    dx = 1
    fs = 100
    cs_min = 1400
    cp_min = 1450
    cp_max = 3400
    cs_max = 3500
    fk_filter_matrix = fk_filter_design(
        trace_shape, selected_channels, dx, fs, cs_min, cp_min, cp_max, cs_max
    )
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
    hybrid_filter_matrix = hybrid_filter_design(
        trace_shape, selected_channels, dx, fs, cs_min, cp_min, fmin, fmax
    )
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
    hybrid_ninf_filter_matrix = hybrid_ninf_filter_design(
        trace_shape,
        selected_channels,
        dx,
        fs,
        cs_min,
        cp_min,
        cp_max,
        cs_max,
        fmin,
        fmax,
    )
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
    hybrid_gs_filter_matrix = hybrid_gs_filter_design(
        trace_shape, selected_channels, dx, fs, cs_min, cp_min, fmin, fmax
    )
    assert hybrid_gs_filter_matrix.shape == (10, 10)


def test_hybrid_ninf_gs_filter_design():
    trace_shape = (10, 10)
    selected_channels = [0, 1, 2]
    dx = 1
    fs = 100
    fk_params = {"c_min": 1400, "c_max": 3500, "fmin": 15, "fmax": 25}
    hybrid_ninf_gs_filter_matrix = hybrid_ninf_gs_filter_design(
        trace_shape, selected_channels, dx, fs, fk_params
    )
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
    filterspec = [4, 1000, "lp"]
    fs = 10000
    b, a = butterworth_filter(filterspec, fs)
    assert len(b) == 6
    assert len(a) == 6


def test_instant_freq():
    channel = np.array([1, 2, 3, 4, 5])
    fs = 10
    inst_freq = instant_freq(channel, fs)
    assert len(inst_freq) == len(channel) - 1


def test_bp_filt():
    # Regression: the order-8 Butterworth was built in (b, a) transfer-function
    # form, which is numerically unstable for narrow bands (poles land outside
    # the unit circle) and silently destroyed the data for fs >= ~500 Hz.
    fs = 1000
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 4000))
    filtered_data = bp_filt(data, fs, 15, 25)
    assert np.shape(filtered_data) == np.shape(data)
    assert np.all(np.isfinite(filtered_data))
    assert np.max(np.abs(filtered_data)) <= np.max(np.abs(data))

    # The filter still passes in-band content and rejects out-of-band content
    t = np.arange(0, 10, 1 / fs)
    tone_in = np.sin(2 * np.pi * 20 * t)[np.newaxis, :]
    tone_out = np.sin(2 * np.pi * 60 * t)[np.newaxis, :]
    assert np.max(np.abs(bp_filt(tone_in, fs, 15, 25))) > 0.9
    # ignore the filtfilt edge transients at the abrupt tone onset/offset
    assert np.max(np.abs(bp_filt(tone_out, fs, 15, 25)[:, fs:-fs])) < 0.01


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
    assert np.allclose(
        snr,
        np.array(
            [
                [2.89413593, 4.69411612, 6.73126528, 9.51687805, 11.44487452],
                [2.89413593, 4.69411612, 6.73126528, 9.51687805, 11.44487452],
            ]
        ),
    )


if __name__ == "__main__":
    pytest.main()
