# File that wrap up functions in dsp.py in a dask way
import dask
import das4whales as dw

def fk_filt(data,tint,fs,xint,dx,c_min,c_max):

    kwargs = {'tint':tint, 'fs':fs, 'xint':xint, 'dx':dx, 'c_min':c_min, 'c_max':c_max}
    data_gx = data.map_blocks(dw.dsp.hybrid_filter_design, kwargs=kwargs, template=data)

    return data_gx