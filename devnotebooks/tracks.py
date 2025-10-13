# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
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
from datetime import datetime, timedelta

# %%
with open('../out/batch1_baseline/association_2021-11-04_02:00:02.pkl', 'rb') as f:
    # Load the association object
    association = pickle.load(f)

# Explore the keys
print(association.keys())
print(association['assoc_pair'].keys())
print(association['assoc'].keys())
print(len(association['assoc']['north']['hf']))
print(np.shape(association['assoc']['north']['hf'][0]))
print(association['metadata']['north'].keys())

# Load the metadata
c0 = 1480 # m/s
fs = association['metadata']['north']['fs']
dx = association['metadata']['north']['dx']
n_selected_channel_m = association['metadata']['north']['selected_channels_m']
n_selected_channels = association['metadata']['north']['selected_channels']
s_selected_channel_m = association['metadata']['south']['selected_channels_m']
s_selected_channels = association['metadata']['south']['selected_channels']
nnx = association['metadata']['north']['data_shape'][0]
snx = association['metadata']['south']['data_shape'][0]

utc_str = association['metadata']['south']['fileBeginTimeUTC']

# %% [markdown]
# ## Statistics on time picks

# %%
# Directory containing pickle files
pkl_dir = '../out/batch1_baseline/'

# Initialize list to hold all pick counts
pick_counts = []

# Iterate through pickle files
for pkl in sorted(Path(pkl_dir).glob('association_*.pkl')):
    with open(pkl, 'rb') as f:
        # Load the association object
        assoc = pickle.load(f)
        
        # Extract all pick counts
        counts = []
        
        # From paired associations
        for region in ['north', 'south']:
            for freq in ['hf', 'lf']:
                data = assoc['assoc_pair'][region][freq]
                if isinstance(data, list) and len(data) > 0 :
                    for pair in data:
                        counts.append(len(pair[0]))
        
        # From individual associations
        for region in ['north', 'south']:
            for freq in ['hf', 'lf']:
                data = assoc['assoc'][region][freq]
                if isinstance(data, list) and len(data) > 0:
                    for pair in data:
                        counts.append(len(pair[0]))
        
        # Add all counts from this file to our collection
        pick_counts.extend(counts)

# Create histogram visualization
plt.figure(figsize=(12, 7))

# Display quartiles
q1 = np.percentile(pick_counts, 25)
q2 = np.median(pick_counts)
q3 = np.percentile(pick_counts, 75)
plt.axvline(q1, color='r', linestyle='--', label='Q1')
plt.axvline(q2, color='g', linestyle='--', label='Median')
plt.axvline(q3, color='b', linestyle='--', label='Q3')

# Create histogram
n, bins, patches = plt.hist(pick_counts, bins=100, alpha=0.7, color='steelblue', edgecolor='black')

# Add labels and title
plt.xlabel('Number of Channels (picks)', fontsize=14)
plt.ylabel('Occurence', fontsize=14)
plt.title('Distribution of Time Picks Count', fontsize=16)

# Add grid for better readability
plt.grid(axis='y', alpha=0.75, linestyle='--')

# Add statistics as text
stats_text = f"Total samples: {len(pick_counts)}\n"
stats_text += f"Mean: {np.mean(pick_counts):.2f}\n"
stats_text += f"Median: {np.median(pick_counts):.2f}\n"
stats_text += f"Min: {np.min(pick_counts)}\n"
stats_text += f"Max: {np.max(pick_counts)}\n"
stats_text += f"Q1: {q1:.2f}\n"
stats_text += f"Q3: {q3:.2f}"

# Position the text in the upper right corner
plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
             ha='right', va='top', fontsize=11)
# Add legend
plt.legend(loc='upper left', fontsize=12)

plt.tight_layout()


# Print summary statistics
print(f"Total number of samples: {len(pick_counts)}")
print(f"Mean number of picks: {np.mean(pick_counts):.2f}")
print(f"Median number of picks: {np.median(pick_counts):.2f}")
print(f"Range: {np.min(pick_counts)} to {np.max(pick_counts)}")

# %% [markdown]
# ## Cables Geometry

# %%
# Import the cable location
df_north = pd.read_csv('../data/north_DAS_multicoord.csv')
df_south = pd.read_csv('../data/south_DAS_multicoord.csv')


# Extract the part of the dataframe used for the time picking process
idx_shift0 = int(n_selected_channels[0] - df_north["chan_idx"].iloc[0]) # Shift between the cable locations (starting at the beach) and the channel locations
idx_shiftn = int(n_selected_channels[1] - df_north["chan_idx"].iloc[-1])

df_north_used = df_north.iloc[idx_shift0:idx_shiftn:n_selected_channels[2]][:nnx]

idx_shift0 = int(s_selected_channels[0] - df_south["chan_idx"].iloc[0]) # Shift between the cable locations (starting at the beach) and the channel locations
idx_shiftn = int(s_selected_channels[1] - df_south["chan_idx"].iloc[-1])

df_south_used = df_south.iloc[idx_shift0:idx_shiftn:s_selected_channels[2]][:snx]

# Import the bathymetry data
bathy, xlon, ylat = dw.map.load_bathymetry('../data/GMRT_OOI_RCA_Cables.grd')
print(f'Origin of the corrdinates. Latitude = {ylat[0]}, Longitude = {xlon[-1]}')

utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])

# Change the reference point to the last point
x0, y0 = utm_xf - utm_x0, utm_y0 - utm_y0
xf, yf = utm_xf - utm_xf, utm_yf - utm_y0
print(xf, yf)
# # Create vectors of coordinates
utm_x = np.linspace(utm_x0, utm_xf, len(xlon))
utm_y = np.linspace(utm_y0, utm_yf, len(ylat))
x = np.linspace(x0, xf, len(xlon))
y = np.linspace(y0, yf, len(ylat))

# Plot the map 
# dw.map.plot_cables2D(df_north, df_south, bathy, xlon, ylat)
# dw.map.plot_cables2D_m(df_north, df_south, bathy, x, y)

# %%
# Cable geometry (make it correspond to x,y,z = cable_pos[:, 0], cable_pos[:, 1], cable_pos[:, 2])
n_cable_pos = np.zeros((len(df_north_used), 3))
s_cable_pos = np.zeros((len(df_south_used), 3))

n_cable_pos[:, 0] = df_north_used['x']
n_cable_pos[:, 1] = df_north_used['y']
n_cable_pos[:, 2] = df_north_used['depth']

s_cable_pos[:, 0] = df_south_used['x']
s_cable_pos[:, 1] = df_south_used['y']
s_cable_pos[:, 2] = df_south_used['depth']

bicable_pos = (n_cable_pos, s_cable_pos)

# # Localize the associated calls 

# # Paired calls
# hf_pair_loc = dw.loc.loc_picks_bicable_list(nhf_pairs, shf_pairs, bicable_pos, c0, fs)
# lf_pair_loc = dw.loc.loc_picks_bicable_list(nlf_pairs, slf_pairs, bicable_pos, c0, fs)

# # North calls
# n_hf_loc = dw.loc.loc_from_picks(nhf_assoc, n_cable_pos, c0, fs)
# n_lf_loc = dw.loc.loc_from_picks(nlf_assoc, n_cable_pos, c0, fs)
# # South calls
# s_hf_loc = dw.loc.loc_from_picks(shf_assoc, s_cable_pos, c0, fs)
# s_lf_loc = dw.loc.loc_from_picks(slf_assoc, s_cable_pos, c0, fs)


# %%
# def local_to_utm(localizations, utm_xf, utm_y0):
#     loc_utm = []
#     for loc in localizations:
#         loc_utm.append([utm_xf - loc[0], loc[1] + utm_y0, loc[2]])
#     return loc_utm

# def batch_utm_to_latlon(loc_utm):
#     loc_latlon = []
#     for loc in loc_utm:
#         lon, lat = dw.map.utm_to_latlon(loc[0], loc[1])
#         loc_latlon.append([lon, lat, loc[2]])
#     return loc_latlon

# hf_pair_loc_utm = local_to_utm(hf_pair_loc, utm_xf, utm_y0)
# lf_pair_loc_utm = local_to_utm(lf_pair_loc, utm_xf, utm_y0)
# hf_pair_loc_latlon = batch_utm_to_latlon(hf_pair_loc_utm)
# lf_pair_loc_latlon = batch_utm_to_latlon(lf_pair_loc_utm)

# n_hf_loc_utm = local_to_utm(n_hf_loc, utm_xf, utm_y0)
# n_lf_loc_utm = local_to_utm(n_lf_loc, utm_xf, utm_y0)
# s_hf_loc_utm = local_to_utm(s_hf_loc, utm_xf, utm_y0)
# s_lf_loc_utm = local_to_utm(s_lf_loc, utm_xf, utm_y0)

# # Convert to latlon
# n_hf_loc_latlon = batch_utm_to_latlon(n_hf_loc_utm)
# n_lf_loc_latlon = batch_utm_to_latlon(n_lf_loc_utm)
# s_hf_loc_latlon = batch_utm_to_latlon(s_hf_loc_utm)
# s_lf_loc_latlon = batch_utm_to_latlon(s_lf_loc_utm)

# %%

# utc = datetime.datetime.strptime(utc_str, "%Y-%m-%d_%H:%M:%S")
# # Create the DataFrame with predefined columns
# df_loc = pd.DataFrame(columns=['utc', 'sensor', 'call_type', 'x', 'y', 'z', 'lat', 'lon'])
# rows = []
# # Add localization results
# for i, loc in enumerate(hf_pair_loc):
#     line = {
#         'utc': utc + datetime.timedelta(seconds=loc[3]),
#         'sensor': 'pair',
#         'call_type': 'hf',
#         'x': loc[0],
#         'y': loc[1],
#         'z': loc[2],
#         'lat': hf_pair_loc_latlon[i][1],
#         'lon': hf_pair_loc_latlon[i][0]
#     }
#     rows.append(line)

# df_loc = pd.DataFrame(rows, columns=['utc', 'sensor', 'call_type', 'x', 'y', 'z', 'lat', 'lon'])
# # Optionally save to CSV
# df_loc.to_csv('localization_results.csv', index=False)

# %%
# # Create two list of coordinates, for ponts every 10 km along the cables, the spatial resolution is 2m 
# opticald_n = []
# opticald_s = []

# disp_step = 10000 # [m]
# dx_ch = 2.0419 # [m]
# idx_step = int(disp_step / dx_ch)

# for i in range(int(idx_step-df_north["chan_idx"].iloc[0]), len(df_north), int(10000/2)):
#     opticald_n.append((df_north['x'][i], df_north['y'][i]))

# for i in range(int(idx_step-df_north["chan_idx"].iloc[0]), len(df_south), int(10000/2)):
#     opticald_s.append((df_south['x'][i], df_south['y'][i]))
# # Plot the grid points on the map
# import cmocean.cm as cmo
# colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
# colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

# # Combine the color maps
# all_colors = np.vstack((colors_undersea, colors_land))
# custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

# extent = [x[0], x[-1], y[0], y[-1]]

# # Set the light source
# ls = LightSource(azdeg=350, altdeg=45)

# # Plot the location of the apex
# plt.figure(figsize=(14, 7))
# ax = plt.gca()
# # Plot the bathymetry relief in background
# rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
# plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
# # Plot the cable location in 2D
# ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable')
# ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable')
# # ax.plot(cable_pos[j_hf_call[i]][:,0], cable_pos[j_hf_call[i]][:,1], 'tab:green', label='used_cable')

# # Add dashed contours at selected depths with annotations
# depth_levels = [-1500, -1000, -600, -250, -80]

# contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
# ax.clabel(contour_dashed, fmt='%d m', inline=True)

# # Plot points along the cable every 10 km in terms of optical distance
# for i, point in enumerate(opticald_n, start=1):
#     ax.plot(point[0], point[1], '.', color='k')
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 8), ha='center', fontsize=12)

# for i, point in enumerate(opticald_s, start=1):
#     ax.plot(point[0], point[1], '.', color='k')
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=12)

# for i, loc in enumerate(hf_pair_loc):
#     if i == 0:
#         ax.plot(loc[0], loc[1], '>', c='tab:red', lw=4, label='Paired call - HF')
#     ax.plot(loc[0], loc[1], '>', c='tab:red', lw=4)

# for i, loc in enumerate(lf_pair_loc):
#     if i == 0:
#         ax.plot(loc[0], loc[1], 'o', c='tab:blue', lw=4, label='Paired call - LF')
#     ax.plot(loc[0], loc[1], 'o', c='tab:blue', lw=4)

# for i, loc in enumerate(n_hf_loc):
#     # Put label only for the first point
#     if i == 0:
#         ax.plot(loc[0], loc[1], '^', c='tab:orange', label='Localized call - north - HF', lw=4)
#     else:
#         ax.plot(loc[0], loc[1], '^', c='tab:orange', lw=4)

# for i, loc in enumerate(s_hf_loc):
#     # Put label only for the first point
#     if i == 0:
#         ax.plot(loc[0], loc[1], 'v', c='tab:green', label='Localized call - south - HF', lw=4)
#     else:
#         ax.plot(loc[0], loc[1], 'v', c='tab:green', lw=4)
# for i, loc in enumerate(n_lf_loc):
#     # Put label only for the first point
#     if i == 0:
#         ax.plot(loc[0], loc[1], '2', c='tab:green', label='Localized call - north - LF', lw=4)
#     else:
#         ax.plot(loc[0], loc[1], '2', c='tab:green', lw=4)
# for i, loc in enumerate(s_lf_loc):
#     # Put label only for the first point
#     if i == 0:
#         ax.plot(loc[0], loc[1], '1', c='tab:orange', label='Localized call - south - LF', lw=4)
#     else:
#         ax.plot(loc[0], loc[1], '1', c='tab:orange', lw=4)

# # Use a proxy artist for the color bar
# im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
# # Calculate width of image over height
# im_ratio = bathy.shape[1] / bathy.shape[0]
# plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
# im.remove()
# # Set the labels
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# # plt.xlim(40000, 34000)
# # plt.ylim(15000, 25000)
# plt.legend(loc='upper left')
# plt.grid(linestyle='--', alpha=0.6, color='k')
# plt.tight_layout()
# plt.show()

# %%
# # Create two list of coordinates, for ponts every 10 km along the cables, the spatial resolution is 2m 
# opticald_n = []
# opticald_s = []

# disp_step = 10000 # [m]
# dx_ch = 2.0419 # [m]
# idx_step = int(disp_step / dx_ch)
# for i in range(int(10000/2-df_north["chan_idx"].iloc[0]), len(df_north), int(10000/2)):
#     opticald_n.append((df_north['lon'][i], df_north['lat'][i]))

# for i in range(int(10000/2-df_south["chan_idx"].iloc[0]), len(df_south), int(10000/2)):
#     opticald_s.append((df_south['lon'][i], df_south['lat'][i]))

# colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
# colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

# # Combine the color maps
# all_colors = np.vstack((colors_undersea, colors_land))
# custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

# # Set extent of the plot
# extent = [xlon[0], xlon[-1], ylat[0], ylat[-1]]

# # Set the light source
# ls = LightSource(azdeg=350, altdeg=45)

# plt.figure(figsize=(14,50))
# ax = plt.gca()

# # Plot the bathymetry relief in background
# rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
# plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower' , vmin=np.min(bathy), vmax=0)

# # Plot the cable location in 2D
# ax.plot(df_north['lon'], df_north['lat'], 'tab:red', label='North cable', lw=2.5)
# ax.plot(df_south['lon'], df_south['lat'], 'tab:orange', label='South cable', lw=2.5)

# # Add dashed contours at selected depths with annotations
# depth_levels = [-500, -400, -300, -200, -100]

# contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
# ax.clabel(contour_dashed, fmt='%d m', inline=True)

# # Plot points along the cable every 10 km in terms of optical distance
# for i, point in enumerate(opticald_n, start=1):
#     # Plot the points
#     ax.plot(point[0], point[1], '.', color='k')
#     # Annotate the points with the distance
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 8), ha='center', fontsize=12)

# for i, point in enumerate(opticald_s, start=1):
#     ax.plot(point[0], point[1], '.', color='k')
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=12)

# # Plot the localized calls
# for loc in n_hf_loc_latlon:
#     ax.plot(loc[0], loc[1], 'o', label='Localized call')

# # Use a proxy artist for the color bar
# im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
# im_ratio = bathy.shape[1] / bathy.shape[0]
# plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0145)

# im.remove()

# # Set the labels
# plt.xlabel('Longitude [°]')
# plt.ylabel('Latitude [°]')
# plt.legend(loc='upper left')
# plt.xlim()

# # Dashed grid lines
# plt.grid(linestyle='--', alpha=0.6, color='k')
# # plt.tight_layout()
# plt.show()

# %%
import pickle
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import das4whales as dw  # your toolbox

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
C0 = 1480  # sound speed (m/s)


# ──────────────────────────────────────────────────────────────────────────────
# I/O & metadata utilities
# ──────────────────────────────────────────────────────────────────────────────
def load_association(pkl_path: Path) -> dict:
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def get_cable_pos(df_path: str, side_meta: dict) -> np.ndarray:
    df = pd.read_csv(df_path)
    sel_start, sel_end, sel_step = side_meta['selected_channels']
    chan_idx = df["chan_idx"]
    idx0 = int(sel_start - chan_idx.iloc[0])
    idxn = int(sel_end   - chan_idx.iloc[-1])
    n_samp = side_meta['data_shape'][0]
    df_used = df.iloc[idx0:idxn:sel_step][:n_samp]
    return df_used[['x','y','depth']].to_numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate transforms
# ──────────────────────────────────────────────────────────────────────────────
def local_to_utm(localizations: np.ndarray, utm_xf: float, utm_y0: float) -> np.ndarray:
    return np.array([[utm_xf - x, utm_y0 + y, z, t] for x, y, z, t in localizations])


def batch_utm_to_latlon(loc_utm: np.ndarray) -> np.ndarray:
    out = []
    for x_utm, y_utm, z, t in loc_utm:
        lon, lat = dw.map.utm_to_latlon(x_utm, y_utm)
        out.append([lon, lat, z, t])
    return np.array(out)


def convert_coords(localizations: np.ndarray, utm_xf: float, utm_y0: float):
    loc_utm = local_to_utm(localizations, utm_xf, utm_y0)
    loc_latlon = batch_utm_to_latlon(loc_utm)
    return loc_utm, loc_latlon


# ──────────────────────────────────────────────────────────────────────────────
# Localization wrapper
# ──────────────────────────────────────────────────────────────────────────────
def localize_calls(assoc: dict, fs: float, north_pos: np.ndarray, south_pos: np.ndarray, utm_xf: float, utm_y0: float) -> dict:
    p_n_hf = assoc['assoc_pair']['north']['hf']
    p_s_hf = assoc['assoc_pair']['south']['hf']
    p_n_lf = assoc['assoc_pair']['north']['lf']
    p_s_lf = assoc['assoc_pair']['south']['lf']
    n_hf   = assoc['assoc']['north']['hf']
    n_lf   = assoc['assoc']['north']['lf']
    s_hf   = assoc['assoc']['south']['hf']
    s_lf   = assoc['assoc']['south']['lf']

    hf_pair_loc = dw.loc.loc_picks_bicable_list(p_n_hf, p_s_hf, (north_pos, south_pos), C0, fs)
    lf_pair_loc = dw.loc.loc_picks_bicable_list(p_n_lf, p_s_lf, (north_pos, south_pos), C0, fs)
    n_hf_loc = dw.loc.loc_from_picks(n_hf, north_pos, C0, fs)
    n_lf_loc = dw.loc.loc_from_picks(n_lf, north_pos, C0, fs)
    s_hf_loc = dw.loc.loc_from_picks(s_hf, south_pos, C0, fs)
    s_lf_loc = dw.loc.loc_from_picks(s_lf, south_pos, C0, fs)

    def wrap(locs):
        loc_utm, loc_latlon = convert_coords(locs, utm_xf, utm_y0)
        return {'local': locs, 'utm': loc_utm, 'latlon': loc_latlon}

    return {
        'hf_pair': wrap(hf_pair_loc),
        'lf_pair': wrap(lf_pair_loc),
        'north_hf': wrap(n_hf_loc),
        'north_lf': wrap(n_lf_loc),
        'south_hf': wrap(s_hf_loc),
        'south_lf': wrap(s_lf_loc),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Row conversion
# ──────────────────────────────────────────────────────────────────────────────
def locs_to_rows(local_arr: np.ndarray, utm_arr: np.ndarray, latlon_arr: np.ndarray, sensor: str, call_type: str, utc0: datetime.datetime) -> list:
    rows = []
    for (x_loc,y_loc,z_loc,t), (x_utm,y_utm,_,_), (lon,lat,_,_) in zip(local_arr, utm_arr, latlon_arr):
        rows.append({
            'utc':       utc0 + datetime.timedelta(seconds=int(t)),
            'sensor':    sensor,
            'call_type': call_type,
            'x_local':   x_loc, 'y_local': y_loc, 'z_local': z_loc,
            'x_utm':     x_utm, 'y_utm':   y_utm, 'z_utm':   z_loc,
            'lat':       lat,   'lon':     lon
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# File processing
# ──────────────────────────────────────────────────────────────────────────────
def process_one_file(pkl_path: Path, north_csv: str, south_csv: str, utm_xf: float, utm_y0: float) -> list:
    assoc = load_association(pkl_path)
    meta_n = assoc['metadata']['north']
    meta_s = assoc['metadata']['south']
    fs = meta_n['fs']
    utc0 = datetime.datetime.strptime(meta_s['fileBeginTimeUTC'], "%Y-%m-%d_%H:%M:%S")

    north_pos = get_cable_pos(north_csv, meta_n)
    south_pos = get_cable_pos(south_csv, meta_s)

    loc_dict = localize_calls(assoc, fs, north_pos, south_pos, utm_xf, utm_y0)
    rows = []
    for key, data in loc_dict.items():
        sensor, call = key.split('_', 1)
        rows.extend(locs_to_rows(data['local'], data['utm'], data['latlon'], sensor, call, utc0))
    return rows


def process_all(pkl_dir: str, north_csv: str, south_csv: str, utm_xf: float, utm_y0: float) -> pd.DataFrame:
    all_rows = []
    for pkl in sorted(Path(pkl_dir).glob('association_*.pkl')):
        print(f"Processing {pkl.name}...")
        all_rows.extend(process_one_file(pkl, north_csv, south_csv, utm_xf, utm_y0))
    return pd.DataFrame(all_rows, columns=[
        'utc','sensor','call_type',
        'x_local','y_local','z_local',
        'x_utm','y_utm','z_utm',
        'lat','lon'
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Main execution (uncomment & run when ready)
# ──────────────────────────────────────────────────────────────────────────────
bathy, xlon, ylat = dw.map.load_bathymetry('../data/GMRT_OOI_RCA_Cables.grd')
utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])

df_all = process_all(
    pkl_dir    = '../out/batch1_baseline/',
    north_csv  = '../data/north_DAS_multicoord.csv',
    south_csv  = '../data/south_DAS_multicoord.csv',
    utm_xf     = utm_xf - utm_x0,
    utm_y0     = utm_y0
)
df_all.to_csv('batch1_localizations_with_coords.csv', index=False)


# %%
# Data paths
# _______________________________________
# csv_path    = 'all_localizations_with_coords.csv'   # your combined CSV
csv_path   = 'batch1_localizations_with_coords.csv'
north_csv   = '../data/north_DAS_multicoord.csv'
south_csv   = '../data/south_DAS_multicoord.csv'
bathy_file  = '../data/GMRT_OOI_RCA_Cables.grd'


# Load data
# _______________________________________
df_all   = pd.read_csv(csv_path, parse_dates=['utc'])
df_north = pd.read_csv(north_csv)
df_south = pd.read_csv(south_csv)

bathy, xlon, ylat = dw.map.load_bathymetry(bathy_file)


# %%
# Distance filtering
# _______________________________________

dist_matrix = spa.distance_matrix(df_all[['x_local', 'y_local']].to_numpy(), df_all[['x_local', 'y_local']].to_numpy())

print("Distance matrix shape:", dist_matrix.shape)
print("Distance matrix max value:", np.max(dist_matrix))
print("Distance matrix min value:", np.min(dist_matrix))
print("Distance matrix mean value:", np.mean(dist_matrix))

# Plot the delaunay triangulation before filtering
tri = spa.Delaunay(df_all[['x_local', 'y_local']].to_numpy())
_ = spa.delaunay_plot_2d(tri)
plt.show()

proximity_threshold = 500  # in meters - filter out points that are more than this distance from any other point
has_neighbor_mask = np.any((dist_matrix <= proximity_threshold) & (dist_matrix > 0), axis=1)

df_filtered = df_all[has_neighbor_mask].copy()
# Print the number of localizations before and after filtering
print(f"Number of localizations before filtering: {len(df_all)}")
print(f"Number of localizations after filtering: {len(df_filtered)}")
# df_filtered = df_all 

# Plot the delaunay triangulation after filtering
tri = spa.Delaunay(df_filtered[['x_local', 'y_local']].to_numpy())
_ = spa.delaunay_plot_2d(tri)
plt.show()

# %%
plt.rcParams['font.size'] = 24
plt.rcParams['lines.linewidth'] = 2

# %%
# Compute UTM extents
utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])
extent = [utm_xf - utm_x0, 0, 0, utm_yf - utm_y0]
xmax, xmin, ymin, ymax = extent

# Filter out bad localizations that are outside the bathymetry
df_valid = df_all[
    (df_all['x_local'] >= xmin) & (df_all['x_local'] <= xmax) &
    (df_all['y_local'] >= ymin) & (df_all['y_local'] <= ymax)
]

df_valid = df_filtered

# Precompute "minutes since start" instead of seconds
t0 = df_valid['utc'].min()
df_valid = df_valid.copy()
df_valid['minutes'] = (df_valid['utc'] - t0).dt.total_seconds() / 60.0  # Convert to minutes

# Group by sensor & call_type
groups = df_valid.groupby(['sensor','call_type'])


# Convert meters to kilometers for coordinates
# _______________________________________
# Convert coordinates to kilometers
df_valid['x_km'] = df_valid['x_local'] / 1000.0
df_valid['y_km'] = df_valid['y_local'] / 1000.0
df_north['x_km'] = df_north['x'] / 1000.0
df_north['y_km'] = df_north['y'] / 1000.0
df_south['x_km'] = df_south['x'] / 1000.0
df_south['y_km'] = df_south['y'] / 1000.0

# Convert extent to kilometers
extent_km = [e / 1000.0 for e in extent]


# Bathymetry shading
# ______________________________________________`

import cmocean.cm as cmo
colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

# Combine the color maps
all_colors = np.vstack((colors_undersea, colors_land))
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)
ls = LightSource(azdeg=350, altdeg=45)
rgb = ls.shade(bathy,
               cmap=custom_cmap,
               vert_exag=0.1,
               blend_mode='overlay',
               vmin=bathy.min(), vmax=0)

# Normalize time colormap based on minutes
norm = mcolors.Normalize(vmin=df_valid['minutes'].min(),
                         vmax=df_valid['minutes'].max())


# Plot
# ______________________________________
fig, ax = plt.subplots(figsize=(12,8), constrained_layout=True)
ax.imshow(rgb, extent=extent_km, aspect='equal')  # Use kilometer extent

# cables
ax.plot(df_north.x_km, df_north.y_km, c='red', label='North cable')
ax.plot(df_south.x_km, df_south.y_km, c='orange', label='South cable')

# scatter each group with markers
markers = {
    'hf-pair': '*',
    'lf-pair': 'P',
    'north-hf': '+', 
    'north-lf': '2',
    'south-hf': 'x',
    'south-lf': '1'
}

for (sensor, call), grp in groups:
    lbl = f"{sensor}-{call}"
    marker = markers.get(lbl)
    ax.scatter(grp.x_km, grp.y_km,  # Use kilometer coordinates
              c=grp.minutes,  # Use minutes for coloring
              cmap='plasma',
              norm=norm,
              marker=marker,
              s=80,
              label=lbl)

# contours (adjust extent for km)
levels = [-1500, -1000, -600, -250, -80]
cnt = ax.contour(bathy, levels=levels,
                colors='k', linestyles='--',
                extent=extent_km, alpha=0.6)  # Use kilometer extent
ax.clabel(cnt, fmt='%d m', inline=True)

# Colorbar management
# _____________________________________

# colorbar for global view
# cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'),
#                    ax=ax, pad=0.015, aspect=30, fraction=0.0165)

# colorbar for zoom 2
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'),
                   ax=ax, pad=0.015, aspect=30, fraction=0.017)

cbar.set_label('Time [minutes]')  # Change label to minutes

# Set labels with kilometers
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.legend(loc='upper right', fontsize='small')
ax.grid(linestyle='--', alpha=0.6, color='gray')
ax.set_title(f'UTC: {df_valid['utc'].min().strftime("%Y-%m-%d %H:%M")}')

# plt.tight_layout()

plt.savefig('localization_kilometers.pdf', format='pdf', bbox_inches='tight', transparent=True)

# Zoom 1
# plt.xlim(62, 50)
# plt.ylim(20, 36)

# Zoom figure 8b
# plt.ylim(28, 30)
# plt.xlim(58, 56)
# plt.savefig('localization_zoom1.pdf', format='pdf', bbox_inches='tight', transparent=True)

# Zoom 2
# plt.xlim(39.5, 35.5)
# plt.ylim(19, 22.6)
# plt.xlim(41, 33)
# plt.ylim(16, 24.5)

# Zoom figure 8c
# plt.xlim(37.5, 36.5)
# plt.ylim(22, 23)
# plt.savefig('localization_zoom2.pdf', format='pdf', bbox_inches='tight', transparent=True)

# Zoom 3
# plt.xlim(83.5, 81)
# plt.ylim(11.9, 14.5)  
# plt.savefig('localization_zoom3.pdf', format='pdf', bbox_inches='tight', transparent=True)

# Zoom figure 8d
# plt.xlim(86.5, 82.5)
# plt.ylim(12, 14.5)  
# plt.savefig('Figure8d.pdf', format='pdf', bbox_inches='tight', transparent=True)

# Zoom 4
# plt.xlim(80, 70)
# plt.ylim(25, 30)
# plt.savefig('localization_zoom4.pdf', format='pdf', bbox_inches='tight', transparent=True)

# Zoom figure 8e
# plt.xlim(80, 70)
# plt.ylim(25, 30)
# plt.savefig('Figure8e.pdf', format='pdf', bbox_inches='tight', transparent=True)

# Zoom 5
# plt.xlim(89.5, 79)
# plt.ylim(10.5, 16)
# plt.savefig('localization_zoom5.pdf', format='pdf', bbox_inches='tight', transparent=True)

# Zoom batch4_gabor
# plt.xlim(50, 40)
# plt.ylim(5, 15)
# plt.savefig('localization_batch4_gabor.png', format='png', bbox_inches='tight', transparent=True)
plt.show()

