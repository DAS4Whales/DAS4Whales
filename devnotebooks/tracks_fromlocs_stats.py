# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import das4whales as dw
import pickle
import pandas as pd
import os
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
import datetime
from pathlib import Path
plt.rcParams['font.size'] = 20
import scipy.spatial as spa
from IPython.display import HTML
import datetime
import logging

# +
# Data paths
# csv_path    = 'all_localizations_with_coords.csv'   # your combined CSV
csv_path   = '../data/4d_localizations_with_coords.csv'
north_csv   = '../data/north_DAS_multicoord.csv'
south_csv   = '../data/south_DAS_multicoord.csv'
bathy_file  = '../data/GMRT_OOI_RCA_Cables.grd'
out_dir     = '../figs/4dtracks/'
os.makedirs(out_dir, exist_ok=True)

# Load data
df_all   = pd.read_csv(csv_path, parse_dates=['utc'])
df_north = pd.read_csv(north_csv)
df_south = pd.read_csv(south_csv)

bathy, xlon, ylat = dw.map.load_bathymetry(bathy_file)

# Set up logging to file
log_file = '../figs/4dtracks/animation_debug.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(log_file)]
)
logger = logging.getLogger(__name__)

# Fix inverted call_type and sensor for pair-located calls
mask = df_all['call_type'].str.contains('pair', case=False, na=False)
num_inverted = mask.sum()
if num_inverted > 0:
    logger.info(f"Found {num_inverted} pair-located calls with inverted sensor/call_type. Swapping...")
    df_all.loc[mask, ['call_type', 'sensor']] = df_all.loc[mask, ['sensor', 'call_type']].values
    df_all.to_csv(csv_path, index=False)
    logger.info(f"Corrected CSV saved to {csv_path}")
else:
    logger.info("No inverted pair-located calls found.")


# -

def generate_track_animation(df: pd.DataFrame, out_dir: str, bathy: np.ndarray, xlon: np.ndarray, ylat: np.ndarray, 
                            interval: int = 100, window_minutes: int = 60, overlap_minutes: int = 30,
                            deltax_threshold: float = None, spatial_proximity_m: float = None, 
                            time_proximity_s: int = None, min_time_spacing_s: float = None):
    """
    Generate an animation of whale tracks from localization data using sliding time windows.
    Each frame covers `window_minutes` with an overlap of `overlap_minutes` (e.g., 60 min windows, 30 min step).
    HF calls use a plasma colormap; LF calls use a viridis colormap.
    
    Optional spatio-temporal filtering:
      - deltax_threshold: remove calls with `deltax` larger than this (meters).
      - spatial_proximity_m: remove calls without spatial neighbors within this distance (meters).
      - time_proximity_s: require spatial neighbors within this time window (seconds).
      - min_time_spacing_s: remove detections closer than this in time (seconds).
    """
    if overlap_minutes >= window_minutes:
        raise ValueError("overlap_minutes must be smaller than window_minutes")

    # Prepare output folder
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if bathy is not None:
        utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
        utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])
        extent = [utm_xf - utm_x0, 0, 0, utm_yf - utm_y0]
    else:
        # if no bathy available, compute extents from data
        minx, maxx = df['x_local'].min(), df['x_local'].max()
        miny, maxy = df['y_local'].min(), df['y_local'].max()
        extent = [maxx/1000.0, minx/1000.0, miny/1000.0, maxy/1000.0]

    # Precompute kilometers conversions and prepare global arrays for proximity searches
    df_all = df.copy()
    df_all['x_km'] = df_all['x_local'] / 1000.0
    df_all['y_km'] = df_all['y_local'] / 1000.0

    # Sort and cache arrays for time-based neighbour searches
    df_all = df_all.sort_values('utc').reset_index(drop=True)
    all_times = df_all['utc'].values.astype('datetime64[s]')
    all_x = df_all['x_local'].to_numpy()
    all_y = df_all['y_local'].to_numpy()

    df_north['x_km'] = df_north['x'] / 1000.0
    df_north['y_km'] = df_north['y'] / 1000.0
    df_south['x_km'] = df_south['x'] / 1000.0
    df_south['y_km'] = df_south['y'] / 1000.0

    # --- Prepare figure and static background ---

    # Compute bathymetry shading if available
    if bathy is not None:
        try:
            import cmocean.cm as cmo
        except Exception:
            cmo = None
        if 'cmo' in locals() and cmo is not None:
            colors_undersea = cmo.deep_r(np.linspace(0, 1, 256))
            colors_land = np.array([[0.5, 0.5, 0.5, 1]])
            all_colors = np.vstack((colors_undersea, colors_land))
            custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)
            ls = LightSource(azdeg=350, altdeg=45)
            rgb = ls.shade(bathy,
                            cmap=custom_cmap,
                            vert_exag=0.1,
                            blend_mode='overlay',
                            vmin=bathy.min(), vmax=0)
        else:
            rgb = None

    fig, ax = plt.subplots(figsize=(12,8), layout='constrained')

    # Plot bathymetry as background
    if bathy is not None and 'rgb' in locals() and rgb is not None:
        ax.imshow(rgb, extent=[e/1000.0 for e in extent], aspect='equal')

    # Plot cables locations
    north_cplot, = ax.plot(df_north.x_km, df_north.y_km, c='red', label='North cable')
    south_cplot, = ax.plot(df_south.x_km, df_south.y_km, c='orange', label='South cable')

    # Add legend and empty markers for HF and LF
    hf_marker = plt.scatter([], [], c='gold', marker='o', edgecolor='k', s=50, label='HF calls')
    lf_marker = plt.scatter([], [], c='magenta', marker='d', edgecolor='k', s=50, label='LF calls')
    ax.legend(handles=[north_cplot, south_cplot, hf_marker, lf_marker], loc='upper right', fontsize='small')

    # Plot some bathymetry contours
    levels = [-1500, -1000, -600, -250, -80]
    if bathy is not None and 'bathy' in locals() and bathy is not None:
        try:
            cnt = ax.contour(bathy, levels=levels,
                            colors='k', linestyles='--',
                            extent=[e/1000.0 for e in extent], alpha=0.6)
            ax.clabel(cnt, fmt='%d m', inline=True)
        except Exception:
            pass

    # norm = mcolors.Normalize(vmin=0, vmax=window_minutes)
    # cbar_hf = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'),
    #                     ax=ax, pad=0.015, aspect=30, fraction=0.017)
    # cbar_hf.set_label('HF time into window [minutes]')
    # cbar_lf = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
    #                     ax=ax, pad=0.06, aspect=30, fraction=0.017)
    # cbar_lf.set_label('LF time into window [minutes]')

    # Init function
    def init():
        ax.set_xlim(extent[0]/1000.0, extent[1]/1000.0)
        ax.set_ylim(extent[2]/1000.0, extent[3]/1000.0)
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_title('')
        return []
    
    # precompute overlapping time windows
    start = df_all['utc'].min()
    end = df_all['utc'].max()
    step_minutes = max(1, window_minutes - overlap_minutes)
    windows = []
    t0 = start
    while t0 < end:
        t1 = t0 + pd.Timedelta(minutes=window_minutes)
        windows.append((t0, t1))
        t0 = t0 + pd.Timedelta(minutes=step_minutes)
    
    # Animation update function
    def update(frame):
        # Clear previous scatter plots to prevent accumulation
        for artist in ax.collections[:]:
            artist.remove()
        
        window_start, window_end = windows[frame]
        mask = (df_all['utc'] >= window_start) & (df_all['utc'] < window_end)
        df_window = df_all[mask].copy()
        if df_window.empty:
            # ax.set_title(f"No calls between {window_start.strftime('%Y-%m-%d %H:%M')} and {window_end.strftime('%Y-%m-%d %H:%M UTC')}")
            return []
        
        logger.info(f"\nFrame {frame} ({window_start.strftime('%Y-%m-%d %H:%M')} to {window_end.strftime('%Y-%m-%d %H:%M')})")
        logger.info(f"  Initial calls in window: {len(df_window)}")

        # Apply deltax threshold filter if requested
        if deltax_threshold is not None and 'deltax' in df_window.columns:
            df_window = df_window[df_window['deltax'] <= deltax_threshold].copy()
            logger.info(f"  After deltax filter: {len(df_window)}")

        # Apply spatial proximity filter first
        if spatial_proximity_m is not None and len(df_window) > 1:
            try:
                dist_matrix = spa.distance_matrix(df_window[['x_local', 'y_local']].to_numpy(), df_window[['x_local', 'y_local']].to_numpy())
                has_spatial_neighbor_mask = np.any((dist_matrix <= spatial_proximity_m) & (dist_matrix > 0), axis=1)
                df_window = df_window[has_spatial_neighbor_mask].copy()
                logger.info(f"  After spatial proximity filter: {len(df_window)}")
            except Exception:
                pass

        # Apply temporal validation on spatial neighbors
        if len(df_window) > 0 and (time_proximity_s is not None) and (spatial_proximity_m is not None):
            keep_mask = np.zeros(len(df_window), dtype=bool)
            for ii, (_, row) in enumerate(df_window.iterrows()):
                t = np.datetime64(row['utc']).astype('datetime64[s]')
                left = int(np.searchsorted(all_times, t - np.timedelta64(int(time_proximity_s), 's')))
                right = int(np.searchsorted(all_times, t + np.timedelta64(int(time_proximity_s), 's'), side='right'))
                if right <= left:
                    continue
                dx = all_x[left:right] - row['x_local']
                dy = all_y[left:right] - row['y_local']
                d2 = dx*dx + dy*dy
                if np.any((d2 <= (spatial_proximity_m**2)) & (d2 > 0)):
                    keep_mask[ii] = True
            df_window = df_window[keep_mask].copy()
            logger.info(f"  After temporal validation: {len(df_window)}")

        # Apply minimum time spacing
        if len(df_window) > 0 and (min_time_spacing_s is not None):
            df_window = df_window.sort_values('utc').reset_index(drop=True)
            keep_indices = [0] if len(df_window) > 0 else []
            for i in range(1, len(df_window)):
                time_diff = (df_window.iloc[i]['utc'] - df_window.iloc[keep_indices[-1]]['utc']).total_seconds()
                if time_diff >= min_time_spacing_s:
                    keep_indices.append(i)
            df_window = df_window.iloc[keep_indices].copy()
            logger.info(f"  After time spacing filter: {len(df_window)}")

        if df_window.empty:
            ax.set_title(f"Tracks {window_start.strftime('%Y-%m-%d %H:%M')} to {window_end.strftime('%Y-%m-%d %H:%M UTC')}")
            return []

        times_in_window = (df_window['utc'] - window_start).dt.total_seconds() / 60.0
        df_window = df_window.assign(minutes_into_window=times_in_window)

        df_hf = df_window[df_window['call_type'].str.contains('hf', case=False, na=False)]
        df_lf = df_window[df_window['call_type'].str.contains('lf', case=False, na=False)]

        artists = []
        if not df_lf.empty:
            sc_lf = ax.scatter(df_lf['x_km'], df_lf['y_km'],
                               c='magenta',   #df_lf['minutes_into_window'],
                                              #cmap='viridis',
                            #    norm=norm,
                               s=50,
                               marker='d',
                               edgecolor='k',
                               alpha=df_lf['minutes_into_window']/window_minutes)   
            artists.append(sc_lf)

        if not df_hf.empty:
            sc_hf = ax.scatter(df_hf['x_km'], df_hf['y_km'],
                               c='gold',    #df_hf['minutes_into_window'],
                                            #cmap='plasma',
                            #    norm=norm,
                               s=50,
                               marker='o',
                               edgecolor='k',
                               alpha=df_hf['minutes_into_window']/window_minutes)   
            artists.append(sc_hf)

        if artists:
            ax.set_title(f"Tracks {window_start.strftime('%Y-%m-%d %H:%M')} to {window_end.strftime('%Y-%m-%d %H:%M UTC')}")
        return tuple(artists)

    anim = FuncAnimation(
        fig, update, frames=len(windows),
        init_func=init, blit=False, interval=interval
    )

    out_path = Path(out_dir) / f"4d_{window_minutes}win_{overlap_minutes}overlap_tracks.mp4"
    writer = FFMpegWriter(fps=8, bitrate=1800)
    anim.save(out_path, writer=writer, dpi=300, savefig_kwargs={'transparent': True, 'bbox_inches': 'tight', 'pad_inches': 0})
    plt.close(fig)


# +
# Animation without filtering (unfiltered)
# generate_track_animation(df_all, out_dir, bathy=bathy, xlon=xlon, ylat=ylat, interval=250,
#                          window_minutes=60, overlap_minutes=40)
# -

# Animation with spatio-temporal filtering (optional)
generate_track_animation(df_all, out_dir, bathy=bathy, xlon=xlon, ylat=ylat, interval=250,
                         window_minutes=90, overlap_minutes=88,
                         deltax_threshold=80, spatial_proximity_m=250, time_proximity_s=70, min_time_spacing_s=5)
