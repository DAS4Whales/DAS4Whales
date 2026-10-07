"""Filter-stability suite: every filter-design function in ``das4whales.dsp``
over a grid of DAS sampling rates and the frequency bands the repo uses.

Background: an order-8 Butterworth bandpass built in (b, a) transfer-function
form is ill-conditioned for narrow bands at high fs (poles land on/outside the
unit circle). Wilmund found this in ``bp_filt`` (PR #38) and in
``hybrid_ninf_filter_design`` (issue #39, item 1). This suite checks, for each
design function and each (fs, band):

* the output is finite,
* the gain is bounded (never above unity),
* the passband is actually passed (gain ~ 1 in the core of the band),
* out-of-band content is attenuated.
"""

import numpy as np
import pytest
import scipy.signal as sp

from das4whales.dsp import (
    bp_filt,
    butterworth_filter,
    fk_filter_design,
    hybrid_filter_design,
    hybrid_gs_filter_design,
    hybrid_ninf_filter_design,
    hybrid_ninf_gs_filter_design,
    resample,
)

# Sampling rates: 200 Hz is the OOI RCA example data; 500-2000 Hz are common
# interrogator settings (the OOI campaign also recorded at up to 1 kHz).
FS_GRID = [200, 500, 1000, 2000]
# Bands used in the repo's scripts/docs: 14-30 Hz (scripts/main_*.py),
# 15-25 Hz (function defaults), 10-30 Hz (tutorial / Example.py),
# 32-48 Hz (noise band in scripts/main_timepicking.py).
BANDS = [(14.0, 30.0), (15.0, 25.0), (10.0, 30.0), (32.0, 48.0)]

# f-k grid: 128 channels at 2 m spacing, 4 s of data (0.25 Hz resolution)
NNX, DX, CHANNELS = 128, 2.0, [0, 128, 1]


def _dense(m):
    return m.todense() if hasattr(m, "todense") else np.asarray(m)


def _freq_axis(nns, fs):
    return np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))


def _core(lo, hi):
    """central half of the passband, away from the transition bands"""
    q = 0.25 * (hi - lo)
    return lo + q, hi - q


def _assert_bounded(x, y):
    """no blow-up: output power <= input power, and no peak far above the input's"""
    assert np.std(y) <= 1.01 * np.std(x)
    assert np.max(np.abs(y)) <= 1.5 * np.max(np.abs(x))


def _tone_gain(filt, fs, f, dur=8.0):
    """steady-state amplitude gain of ``filt`` for a unit sine at ``f`` Hz"""
    t = np.arange(int(dur * fs)) / fs
    x = np.sin(2 * np.pi * f * t)[np.newaxis, :]
    y = filt(x)
    return np.max(np.abs(y[:, 2 * fs : -2 * fs]))  # skip edge transients


# --------------------------------------------------------------------------
# Time-domain IIR filters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fs", FS_GRID)
@pytest.mark.parametrize("band", BANDS, ids=lambda b: f"{b[0]:g}-{b[1]:g}Hz")
def test_bp_filt_stability(fs, band):
    lo, hi = band
    rng = np.random.default_rng(20261007)
    x = rng.standard_normal((2, 8 * fs))
    y = bp_filt(x, fs, lo, hi)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    _assert_bounded(x, y)

    filt = lambda d: bp_filt(d, fs, lo, hi)  # noqa: E731
    g_pass = _tone_gain(filt, fs, 0.5 * (lo + hi))
    assert 0.9 <= g_pass <= 1.01
    assert _tone_gain(filt, fs, 2 * hi) < 1e-3
    assert _tone_gain(filt, fs, 0.5 * lo) < 1e-3


@pytest.mark.parametrize("fs", FS_GRID)
@pytest.mark.parametrize(
    "spec",
    [
        [5, [10, 30], "bp"],  # Example.py / tutorial
        [2, 5, "hp"],  # Example.py / tutorial
        [8, [14, 30], "bp"],  # same design as bp_filt, 14-30 Hz
        [8, [15, 25], "bp"],
    ],
    ids=lambda s: f"order{s[0]}-{s[2]}-{s[1]}",
)
def test_butterworth_filter_stability(fs, spec):
    sos = butterworth_filter(spec, fs)
    assert np.all(np.isfinite(sos))
    # every pole strictly inside the unit circle
    poles = np.concatenate([np.roots(s[3:]) for s in sos])
    assert np.max(np.abs(poles)) < 1.0

    rng = np.random.default_rng(20261007)
    x = rng.standard_normal((2, 8 * fs))
    y = sp.sosfiltfilt(sos, x, axis=1)
    assert np.all(np.isfinite(y))
    _assert_bounded(x, y)

    filt = lambda d: sp.sosfiltfilt(sos, d, axis=1)  # noqa: E731
    if spec[2] == "bp":
        lo, hi = spec[1]
        assert 0.9 <= _tone_gain(filt, fs, 0.5 * (lo + hi)) <= 1.01
        assert _tone_gain(filt, fs, 3 * hi) < 1e-3
    else:  # high-pass
        assert 0.9 <= _tone_gain(filt, fs, 20.0) <= 1.01
        assert _tone_gain(filt, fs, 0.5) < 1e-3


@pytest.mark.parametrize("fs", [500, 1000, 2000])
def test_resample_antialias_stability(fs):
    # resample() low-passes with an order-8 Butterworth (sos) before decimating
    rng = np.random.default_rng(20261007)
    x = rng.standard_normal((2, 4 * fs))
    y, fs_new, _ = resample(x, fs, 200)
    assert fs_new == 200
    assert np.all(np.isfinite(y))
    _assert_bounded(x, y)

    t = np.arange(4 * fs) / fs
    tone = np.sin(2 * np.pi * 20.0 * t)[np.newaxis, :]
    y_tone, _, _ = resample(tone, fs, 200)
    assert 0.9 <= np.max(np.abs(y_tone[:, 200:-200])) <= 1.05


# --------------------------------------------------------------------------
# f-k filter designs (frequency-domain masks)
# --------------------------------------------------------------------------


def _design(name, fs, nns, lo, hi):
    shape = (NNX, nns)
    if name == "fk_filter_design":
        return fk_filter_design(shape, CHANNELS, DX, fs)
    if name == "hybrid_filter_design":
        return hybrid_filter_design(shape, CHANNELS, DX, fs, fmin=lo, fmax=hi)
    if name == "hybrid_ninf_filter_design":
        return hybrid_ninf_filter_design(shape, CHANNELS, DX, fs, fmin=lo, fmax=hi)
    if name == "hybrid_gs_filter_design":
        return hybrid_gs_filter_design(shape, CHANNELS, DX, fs, fmin=lo, fmax=hi)
    if name == "hybrid_ninf_gs_filter_design":
        params = {"c_min": 1400.0, "c_max": 3300.0, "fmin": lo, "fmax": hi}
        return hybrid_ninf_gs_filter_design(shape, CHANNELS, DX, fs, params)
    raise ValueError(name)


FK_DESIGNS = [
    "fk_filter_design",
    "hybrid_filter_design",
    "hybrid_ninf_filter_design",
    "hybrid_gs_filter_design",
    "hybrid_ninf_gs_filter_design",
]


@pytest.mark.parametrize("fs", FS_GRID)
@pytest.mark.parametrize("band", BANDS[:3], ids=lambda b: f"{b[0]:g}-{b[1]:g}Hz")
@pytest.mark.parametrize("name", FK_DESIGNS)
def test_fk_designs_finite_and_bounded(name, fs, band):
    nns = 4 * fs
    m = _dense(_design(name, fs, nns, *band))
    assert m.shape == (NNX, nns)
    assert np.all(np.isfinite(m))
    assert m.min() >= 0.0
    assert m.max() <= 1.0 + 1e-4  # a filter mask must never amplify


@pytest.mark.parametrize("fs", FS_GRID)
@pytest.mark.parametrize("band", BANDS[:3], ids=lambda b: f"{b[0]:g}-{b[1]:g}Hz")
def test_hybrid_filter_design_passband(fs, band):
    lo, hi = band
    nns = 4 * fs
    freq = _freq_axis(nns, fs)
    env = _dense(_design("hybrid_filter_design", fs, nns, lo, hi)).max(axis=0)
    c0, c1 = _core(lo, hi)
    assert np.allclose(env[(freq >= c0) & (freq <= c1)], 1.0)
    assert np.all(env[np.abs(freq) >= 2 * hi] == 0.0)


@pytest.mark.parametrize("fs", FS_GRID)
@pytest.mark.parametrize("band", BANDS[:3], ids=lambda b: f"{b[0]:g}-{b[1]:g}Hz")
def test_hybrid_ninf_filter_design_frequency_response(fs, band):
    """The frequency envelope of the f-k mask (max over k) must match the
    |H|^2 of the same order-8 Butterworth evaluated stably (second-order
    sections): ~1 in the passband, attenuated outside it, at every fs."""
    lo, hi = band
    nns = 4 * fs
    freq = _freq_axis(nns, fs)
    env = _dense(_design("hybrid_ninf_filter_design", fs, nns, lo, hi)).max(axis=0)

    sos = sp.butter(8, [lo / (fs / 2), hi / (fs / 2)], "bp", output="sos")
    _, h = sp.sosfreqz(sos, worN=freq[freq >= 0], fs=fs)
    ref = np.abs(h) ** 2

    c0, c1 = _core(lo, hi)
    core = (freq >= c0) & (freq <= c1)
    assert np.all(env[core] >= 0.9), f"passband gain min {env[core].min():.3g}"
    assert np.all(env[core] <= 1.0 + 1e-4), f"passband gain max {env[core].max():.3g}"
    np.testing.assert_allclose(env[core], ref[core[freq >= 0]], atol=1e-4)
    stop = np.abs(freq) >= 2 * hi
    assert np.all(env[stop] < 1e-3)


@pytest.mark.xfail(
    strict=True,
    reason="issue #39 item 1 (Wilmund): odd sample counts give a response one "
    "bin short; left for a separate design decision",
)
def test_hybrid_ninf_filter_design_odd_length():
    m = hybrid_ninf_filter_design((NNX, 801), CHANNELS, DX, 200, fmin=14, fmax=30)
    assert m.shape == (NNX, 801)
