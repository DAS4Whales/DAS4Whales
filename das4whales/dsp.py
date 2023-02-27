import numpy as np
import scipy.signal as sp


# Transformations
def get_fx(trace, nfft):
    """
    Apply a fast Fourier transform (fft) to each channel of the strain data matrix

    Inputs:
    - trace, a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    - nfft, the fft size in sample

    Outputs:
    - trace, a [channel x freq. sample] nparray containing the strain data in the spatio-spectral domain


    """

    fx = 2*(abs(np.fft.fftshift(np.fft.fft(trace, nfft), axes=1)))
    fx /= nfft
    fx *= 10 ** 9
    return fx


# Filters
def fk_filtering(trace, selected_channels, dx, fs, c_min=1450, c_max=3000):
    """
    Designs and apply f-k filtering to DAS strain data
    Keeps by default data with propagation speed [1450-3000] m/s

    Inputs:
    - trace, a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    - selected_channels, a list of the selected channels number  [start, end, step]
    - dx, the channel spacing (m)
    - fs, the sampling frequency (Hz)
    - c_min and c_max: the selected sound speeds for the f-k "bandpass" filtering

    Outputs:
    - trace, a [channel x time sample] nparray containing the f-k-filtered strain data in the spatio-temporal domain

    """
    #Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx

    # Get the dimensions of the trace data
    nnx = trace.shape[0]
    nns = trace.shape[1]

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=selected_channels[2] * dx))

    # Calculate the frequency-wavenumber spectrum
    fk_trace = np.fft.fftshift(np.fft.fft2(trace))

    # Supress/hide the warning
    np.seterr(invalid='ignore')

    # Create the filter
    fk_binary_filter_matrix = np.ndarray(shape=fk_trace.shape, dtype=float, order='F')
    for i in range(len(knum)):
        if abs(knum[i]) < 0.005:
            fk_binary_filter_matrix[i, :] = np.zeros(shape=[len(freq)], dtype=float, order='F')
        else:
            binary_line = np.ones(shape=[len(freq)], dtype=float, order='F')
            line = abs(freq / knum[i])
            binary_line[line > c_max] = 0
            binary_line[line < c_min] = 0
            fk_binary_filter_matrix[i, :] = binary_line

    # Apply the filter
    fk_filtered_trace = fk_trace * fk_binary_filter_matrix

    # Back to the t-x domain
    trace = np.fft.ifft2(np.fft.ifftshift(fk_filtered_trace))

    return trace.real


def butterworth_filter(filterspec, fs):
    """
        Designs and a butterworth filter see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

        Apply as.
        trace_filtered = sp.sosfiltfilt(filter_sos, trace_original, axis=1)

    """
    filter_order, filter_critical_freq, filter_type_str = filterspec
    # Build a filter of the desired type
    wn = np.array(filter_critical_freq) / (fs / 2)  # convert to angular frequency

    filter_sos = sp.butter(filter_order, wn, btype=filter_type_str, output='sos')

    return filter_sos
