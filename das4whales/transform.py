import numpy as np


def get_fx(trace, nfft):
    fx = (abs(np.fft.fftshift(np.fft.fft(trace, nfft), axes=1)))
    fx /= nfft
    fx *= 10 ** 9
    return fx
