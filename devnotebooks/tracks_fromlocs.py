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
import matplotlib.animation as animation
from matplotlib.colors import LightSource
import matplotlib as mpl
import datetime
from pathlib import Path
plt.rcParams['font.size'] = 20
import scipy.spatial as spa
from IPython.display import HTML
import datetime

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


# -

def generate_hourly_plots(csv_path: str, north_csv: str, south_csv: str, bathy_file: str, out_dir: str, window_minutes: int = 60, 
                          deltax_threshold: float = None, spatial_proximity_m: float = 250.0, time_proximity_s: int = None,
                          min_time_spacing_s: float = None):
    """Generate track figures for consecutive time windows (default 60 minutes) from a CSV of localizations.
    Additional filtering options:
      - deltax_threshold: if provided, remove calls with `deltax` larger than this (meters).
      - spatial_proximity_m: remove calls that do not have at least one spatial neighbor within this distance (meters).
      - time_proximity_s: if provided, require that spatial neighbors fall within this time window (seconds).
        This ensures temporal coherence with the expected inter-pulse interval.
      - min_time_spacing_s: if provided, remove detections that are too close in time (seconds).
        Useful for eliminating rapid false positive bursts. Applied after spatial/temporal filtering."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(csv_path, parse_dates=['utc'])
    df_north = pd.read_csv(north_csv)
    df_south = pd.read_csv(south_csv)

    bathy, xlon, ylat = None, None, None
    if bathy_file:
        bathy, xlon, ylat = dw.map.load_bathymetry(bathy_file)

    if bathy is not None:
        utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
        utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])
        extent = [utm_xf - utm_x0, 0, 0, utm_yf - utm_y0]
    else:
        # if no bathy available, compute extents from data
        minx, maxx = df_all['x_local'].min(), df_all['x_local'].max()
        miny, maxy = df_all['y_local'].min(), df_all['y_local'].max()
        extent = [maxx/1000.0, minx/1000.0, miny/1000.0, maxy/1000.0]

    # Precompute kilometers conversions and prepare global arrays for proximity searches
    df_all = df_all.copy()
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

    # compute time windows
    start = df_all['utc'].min()
    end = df_all['utc'].max()
    total_minutes = int((end - start).total_seconds() / 60.0)
    n_windows = max(1, (total_minutes // window_minutes) + 1)

    for w in range(n_windows):
        t0 = start + datetime.timedelta(minutes=w*window_minutes)
        t1 = t0 + datetime.timedelta(minutes=window_minutes)
        df_win = df_all[(df_all['utc'] >= t0) & (df_all['utc'] < t1)]
        if df_win.empty:
            continue

        # Apply deltax threshold filter if requested
        if deltax_threshold is not None and 'deltax' in df_win.columns:
            df_win = df_win[df_win['deltax'] <= deltax_threshold].copy()

        # Apply spatial proximity filter first
        # Remove isolated localizations that have no neighbor within spatial threshold
        if spatial_proximity_m is not None and len(df_win) > 1:
            try:
                dist_matrix = spa.distance_matrix(df_win[['x_local', 'y_local']].to_numpy(), df_win[['x_local', 'y_local']].to_numpy())
                has_spatial_neighbor_mask = np.any((dist_matrix <= spatial_proximity_m) & (dist_matrix > 0), axis=1)
                df_win = df_win[has_spatial_neighbor_mask].copy()
            except Exception:
                pass

        # Apply temporal validation on spatial neighbors
        # If time_proximity_s is provided, require that spatial neighbors fall within this time window
        if len(df_win) > 0 and (time_proximity_s is not None) and (spatial_proximity_m is not None):
            keep_mask = np.zeros(len(df_win), dtype=bool)
            for ii, (_, row) in enumerate(df_win.iterrows()):
                t = np.datetime64(row['utc']).astype('datetime64[s]')
                left = int(np.searchsorted(all_times, t - np.timedelta64(int(time_proximity_s), 's')))
                right = int(np.searchsorted(all_times, t + np.timedelta64(int(time_proximity_s), 's'), side='right'))
                if right <= left:
                    continue
                # Find spatial neighbors within time window
                dx = all_x[left:right] - row['x_local']
                dy = all_y[left:right] - row['y_local']
                d2 = dx*dx + dy*dy
                if np.any((d2 <= (spatial_proximity_m**2)) & (d2 > 0)):
                    keep_mask[ii] = True
            df_win = df_win[keep_mask].copy()

        # Apply minimum time spacing to cluster tracks and eliminate rapid false positive bursts
        if len(df_win) > 0 and (min_time_spacing_s is not None):
            df_win = df_win.sort_values('utc').reset_index(drop=True)
            keep_indices = [0] if len(df_win) > 0 else []
            for i in range(1, len(df_win)):
                time_diff = (df_win.iloc[i]['utc'] - df_win.iloc[keep_indices[-1]]['utc']).total_seconds()
                if time_diff >= min_time_spacing_s:
                    keep_indices.append(i)
            df_win = df_win.iloc[keep_indices].copy()

        # skip empty windows after filtering
        if df_win.empty:
            continue

        # Compute bathy shading if available
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

        # Normalize time colormap based on window
        norm = mcolors.Normalize(vmin=0, vmax=window_minutes)

        median_rms = np.median(df_win['wrms']) if 'wrms' in df_win.columns else np.nan
        median_deltax = np.median(df_win['deltax']) if 'deltax' in df_win.columns else np.nan

        fig, ax = plt.subplots(figsize=(12,8), constrained_layout=True)
        if bathy is not None and 'rgb' in locals() and rgb is not None:
            ax.imshow(rgb, extent=[e/1000.0 for e in extent], aspect='equal')

        # cables
        ax.plot(df_north.x_km, df_north.y_km, c='red', label='North cable')
        ax.plot(df_south.x_km, df_south.y_km, c='orange', label='South cable')

        markers = {
            'hf-pair': '*',
            'lf-pair': 'P',
            'north-hf': '+', 
            'north-lf': '2',
            'south-hf': 'x',
            'south-lf': '1'
        }

        # create minutes into window for coloring
        df_win = df_win.copy()
        df_win['minutes_into_window'] = (df_win['utc'] - t0).dt.total_seconds() / 60.0

        groups = df_win.groupby(['sensor','call_type'])
        for (sensor, call), grp in groups:
            lbl = f"{sensor}-{call}"
            marker = markers.get(lbl)
            ax.scatter(grp.x_km, grp.y_km,
                      c=grp.minutes_into_window,
                      cmap='plasma',
                      norm=norm,
                      marker=marker if marker is not None else 'o',
                      s=100,
                      edgecolors='k',
                      label=lbl)

        levels = [-1500, -1000, -600, -250, -80]
        if bathy is not None and 'bathy' in locals() and bathy is not None:
            try:
                cnt = ax.contour(bathy, levels=levels,
                                colors='k', linestyles='--',
                                extent=[e/1000.0 for e in extent], alpha=0.6)
                ax.clabel(cnt, fmt='%d m', inline=True)
            except Exception:
                pass

        stats_text = f"Number of calls: {len(df_win)}\nMedian $\\eta_{{RMS}}$: {median_rms:.2f}s\nMedian $\\delta$x: {median_deltax:.2f}m" if not np.isnan(median_rms) and not np.isnan(median_deltax) else "Statistics not available"
        ax.text(0.35, 0.14, stats_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='white', alpha=0.8))

        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'),
                           ax=ax, pad=0.015, aspect=30, fraction=0.017)
        cbar.set_label('Time into window [minutes]')

        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(linestyle='--', alpha=0.6, color='gray')
        ax.set_title(f"UTC window: {t0.strftime('%Y-%m-%d %H:%M')} to {t1.strftime('%Y-%m-%d %H:%M')}")

        out_path = Path(out_dir) / f"tracks_{t0.strftime('%Y%m%d_%H%M')}.pdf"
        fig.savefig(out_path, format='pdf', bbox_inches='tight', transparent=True)
        plt.close(fig)


generate_hourly_plots(csv_path, north_csv, south_csv, bathy_file, out_dir, window_minutes=60,
                      deltax_threshold=80, spatial_proximity_m=250, time_proximity_s=70, min_time_spacing_s=5)
