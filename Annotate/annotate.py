# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# +
import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.xarray
import hvplot.pandas
import panel as pn
from holoviews import streams
from scipy.interpolate import make_interp_spline, UnivariateSpline


hv.extension('bokeh')
pn.extension()
# -

batch_number = 1
output_dir = os.path.join('denoised_data', f'Batch_{batch_number}')
north_files = sorted(glob.glob(os.path.join("..", output_dir, "Denoised_SNR_North_*.nc")))
south_files = sorted(glob.glob(os.path.join("..", output_dir, "Denoised_SNR_South_*.nc")))

id = 0 # 0 to 11 
print(f"Loading {north_files[id]}\n        {south_files[id]}")

# +
# ---------- Load and Downsample ----------
n_denoised = xr.open_dataset(north_files[id])
s_denoised = xr.open_dataset(south_files[id])

n_ds = n_denoised.isel(dist=slice(None, None, 10), time=slice(None, None, 10))
s_ds = s_denoised.isel(dist=slice(None, None, 10), time=slice(None, None, 10))

nhf_snr, nlf_snr = n_ds['SNR_hf'], n_ds['SNR_lf']
shf_snr, slf_snr = s_ds['SNR_hf'], s_ds['SNR_lf']

time_vals = s_ds['time'].values
utc_time_vals = pd.to_datetime(s_ds['utc_time'].values)

# +
# ---------- Output files ----------
utc_begin    = n_ds.attrs['fileBeginTimeUTC']
output_north = f'Batch{batch_number}/annotated_calls_north_{utc_begin}.csv'
output_south = f'Batch{batch_number}/annotated_calls_south_{utc_begin}.csv'

n_filename = os.path.basename(north_files[id])
s_filename = os.path.basename(south_files[id])

# ---------- Helpers ----------
def get_nearest_utc(clicked_time):
    idx = np.abs(time_vals - clicked_time).argmin()
    return str(utc_time_vals[idx])

# initialize call IDs
annotation_data = []

if os.path.exists(output_north) and os.path.exists(output_south):
    north_ann = pd.read_csv(output_north)
    south_ann = pd.read_csv(output_south)
    last_id = max(north_ann['call_id'].max(), south_ann['call_id'].max())
    if pd.isna(last_id):
        call_id = [1]
    else:
        call_id = [last_id + 1]

else:
    call_id = [1]

segment_id = [1]

# create a little “reload” stream we can fire whenever we save
Reload = streams.Stream.define('Reload')  
reload_stream = Reload()

def make_existing_dmap(cable, freq):
    """
    Returns a DynamicMap that, on every reload_stream.event(),
    re-reads the CSV for `cable`/`freq` and emits a Points overlay.
    """
    def _read_and_filter():
        fname = output_north if cable=='North' else output_south
        if not os.path.exists(fname):
            return hv.Points([])   # nothing yet
        df = pd.read_csv(fname)
        df = df[(df.cable==cable)&(df.call_type==freq)]
        return hv.Points(df, kdims=['time','dist'], vdims=list(df.columns))\
                 .opts(size=6, color='yellow', marker='o')
    return hv.DynamicMap(lambda **kwargs: _read_and_filter(),
                         streams=[reload_stream])

def create_annotator(snr_data, title, cable, freq):
    # base SNR heatmap
    base = snr_data.hvplot(
        x='time', y='dist', z=snr_data.name, title=title,
        xlabel='Time (s)', ylabel='Distance (km)',
        colormap='turbo', colorbar=False,  # Using a non-perceptually uniform colormap is actually better here because it highlights faint parts of the signal and noise. Inferno and viridis are good to see mostly the High SNR parts.
        clim=(snr_data.min().item(), snr_data.max().item()),
        width=600, height=500
    ).opts(
        xlim=(float(snr_data.time.values.min()), float(snr_data.time.values.max())),
        ylim=(float(snr_data.dist.values.min()), float(snr_data.dist.values.max())),
        fontscale=1.5,
    )

    # existing annotations overlay
    existing_dmap = make_existing_dmap(cable, freq)

    # new‐point tap stream
    tap = streams.Tap(source=base)
    def on_tap(x, y):
        if x is None or y is None:
            return hv.Points([])
        utc = get_nearest_utc(x)
        pt = {
            'utc_time': utc,
            'time':    x,
            'dist':    y,
            'call_type': freq,
            'cable':     cable,
            'call_id':   call_id[0],
            'segment_id':segment_id[0],
            'snr': snr_data.sel(time=x, dist=y, method='nearest').values.item(),
            'filename': n_filename if cable=='North' else s_filename
        }
        annotation_data.append(pt)
        df = pd.DataFrame(annotation_data)
        filt = df[(df.cable==cable)&(df.call_type==freq)]
        return hv.Points(filt, kdims=['time','dist'], vdims=list(filt.columns))\
                 .opts(size=6, color='magenta', tools=['hover'], marker='o')

    newdmap = hv.DynamicMap(on_tap, streams=[tap])
    return base * existing_dmap * newdmap

# ---------- Widgets ----------
nextseg = pn.widgets.Button(name='➡️ Next Segment', button_type='success')
savebtn = pn.widgets.Button(name='💾 Save Call',    button_type='primary')
resetbtn= pn.widgets.Button(name='🔁 Reset Call',   button_type='warning')
clearbtn= pn.widgets.Button(name='❌ Clear All',    button_type='danger')
message= pn.pane.Markdown('', width=300, styles={'font-size':'18px','font-weight':'bold'})

def next_segment(event):
    segment_id[0] += 1
    message.object = f"➡️ Segment ID now {segment_id[0]}"

nextseg.on_click(next_segment)

def save_current_call(event):
    if not annotation_data:
        message.object = "No annotations to save."
        return

    df = pd.DataFrame(annotation_data)
    df = df[['call_id','segment_id','utc_time','time','dist','call_type','cable','snr','filename']]

    def _append(df_, fname):
        hdr = not os.path.exists(fname)
        df_.to_csv(fname, mode='a', index=False, header=hdr)

    north = df[df.cable=='North']
    south = df[df.cable=='South']
    if not north.empty: _append(north, output_north)
    if not south.empty: _append(south, output_south)

    # clear & bump IDs
    message.object = f"✅ Call {call_id[0]} saved with {len(annotation_data)} points."
    call_id[0] += 1
    segment_id[0] = 1
    annotation_data.clear()

    # *this* is the magic line that makes all existing‐overlays refresh:
    reload_stream.event()

savebtn.on_click(save_current_call)

def reset_current_call(event):
    message.object = "🔁 Annotations cleared (not saved)."
    segment_id[0] = 1
    annotation_data.clear()

resetbtn.on_click(reset_current_call)

def clear_annotations(event):
    for f in (output_north, output_south):
        pd.DataFrame(
            columns=['call_id','segment_id','utc_time','time','dist',
                     'call_type','cable','snr','filename']
        ).to_csv(f, index=False)
    message.object = "❌ All annotations cleared."
    annotation_data.clear()
    call_id[0] = 1
    reload_stream.event()      # wipe the overlays too

clearbtn.on_click(clear_annotations)

# ---------- Layout & Serve ----------
annotated_nhf = create_annotator(nhf_snr, "North HF", cable='North', freq='HF')
annotated_nlf = create_annotator(nlf_snr, "North LF", cable='North', freq='LF')
annotated_shf = create_annotator(shf_snr, "South HF", cable='South', freq='HF')
annotated_slf = create_annotator(slf_snr, "South LF", cable='South', freq='LF')

layout = pn.Row(
    pn.Column(annotated_nhf, annotated_shf),
    pn.Column(annotated_nlf, annotated_slf),
    pn.Column(nextseg, savebtn, resetbtn, clearbtn, message)
)

layout.servable()

