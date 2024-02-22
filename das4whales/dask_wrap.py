# File that wrap up functions in dsp.py in a dask way
import dask
import das4whales as dw


def fk_filt(data, fk_filter_matrix, tapering):

    kwargs = {'fk_filter_matrix':fk_filter_matrix, 'tapering':tapering}
    data_gx = data.map_blocks(dw.dsp.fk_filter_filt, kwargs=kwargs, template=data)

    return data_gx



# fk_filter = dw.dsp.hybrid_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, 
#                                     cs_min=1350, cp_min=1450, fmin=17, fmax=25, display_filter=False)

