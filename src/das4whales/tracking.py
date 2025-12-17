"""
tracking.py - Enhanced tracking module with uncertainty quantification for the das4whales package.

This module provides enhanced functions for processing localization results with uncertainty
and creating comprehensive tracking datasets.

Author: Quentin Goestchel, Léa Bouffaut  
Date: 2025-01-11
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from das4whales.loc import (
    LocalizationResult, 
    localization_results_to_dict,
    loc_picks_bicable_list,
    loc_from_picks
)


def process_association_enhanced(pkl_path: Path, north_csv: str, south_csv: str, 
                               utm_xf: float, utm_y0: float, c0: float = 1480) -> Dict[str, Any]:
    """Process a single association file with enhanced uncertainty quantification.
    
    Parameters
    ----------
    pkl_path : Path
        Path to the association pickle file
    north_csv : str
        Path to north cable CSV file
    south_csv : str
        Path to south cable CSV file
    utm_xf : float
        UTM X reference for coordinate transformation
    utm_y0 : float
        UTM Y reference for coordinate transformation
    c0 : float, optional
        Sound speed in m/s (default=1480)
        
    Returns
    -------
    dict
        Dictionary containing processed localization results with uncertainties
    """
    import pickle
    
    # Load association data
    with open(pkl_path, 'rb') as f:
        assoc = pickle.load(f)
    
    # Extract metadata
    meta_n = assoc['metadata']['north']
    meta_s = assoc['metadata']['south']
    fs = meta_n['fs']
    utc0 = datetime.datetime.strptime(meta_s['fileBeginTimeUTC'], "%Y-%m-%d_%H:%M:%S")
    
    # Get cable positions
    north_pos = get_cable_pos(north_csv, meta_n)
    south_pos = get_cable_pos(south_csv, meta_s)
    
    # Extract association data
    p_n_hf = assoc['assoc_pair']['north']['hf']
    p_s_hf = assoc['assoc_pair']['south']['hf']
    p_n_lf = assoc['assoc_pair']['north']['lf']
    p_s_lf = assoc['assoc_pair']['south']['lf']
    n_hf   = assoc['assoc']['north']['hf']
    n_lf   = assoc['assoc']['north']['lf']
    s_hf   = assoc['assoc']['south']['hf']
    s_lf   = assoc['assoc']['south']['lf']

    # Perform localizations with uncertainty
    results = {}
    
    if p_n_hf and p_s_hf:
        results['hf_pair'] = loc_picks_bicable_list(p_n_hf, p_s_hf, (north_pos, south_pos), c0, fs, return_uncertainty=True)
    else:
        results['hf_pair'] = []
        
    if p_n_lf and p_s_lf:
        results['lf_pair'] = loc_picks_bicable_list(p_n_lf, p_s_lf, (north_pos, south_pos), c0, fs, return_uncertainty=True)
    else:
        results['lf_pair'] = []
        
    if n_hf:
        results['north_hf'] = loc_from_picks(n_hf, north_pos, c0, fs, return_uncertainty=True)
    else:
        results['north_hf'] = []
        
    if n_lf:
        results['north_lf'] = loc_from_picks(n_lf, north_pos, c0, fs, return_uncertainty=True)
    else:
        results['north_lf'] = []
        
    if s_hf:
        results['south_hf'] = loc_from_picks(s_hf, south_pos, c0, fs, return_uncertainty=True)
    else:
        results['south_hf'] = []
        
    if s_lf:
        results['south_lf'] = loc_from_picks(s_lf, south_pos, c0, fs, return_uncertainty=True)  
    else:
        results['south_lf'] = []
    
    # Convert to coordinate systems
    processed_results = {}
    for key, loc_results in results.items():
        if loc_results:  # Only process if we have results
            processed_results[key] = convert_coordinates_enhanced(loc_results, utm_xf, utm_y0)
        else:
            processed_results[key] = {'local': [], 'utm': [], 'latlon': [], 'results': []}
    
    return {
        'file': pkl_path.name,
        'utc_start': utc0,
        'results': processed_results,
        'metadata': {
            'fs': fs,
            'c0': c0,
            'utm_ref': (utm_xf, utm_y0)
        }
    }


def get_cable_pos(df_path: str, side_meta: dict) -> np.ndarray:
    """Extract cable positions from CSV file based on metadata."""
    df = pd.read_csv(df_path)
    sel_start, sel_end, sel_step = side_meta['selected_channels']
    chan_idx = df["chan_idx"]
    idx0 = int(sel_start - chan_idx.iloc[0])
    idxn = int(sel_end - chan_idx.iloc[-1])
    n_samp = side_meta['data_shape'][0]
    df_used = df.iloc[idx0:idxn:sel_step][:n_samp]
    return df_used[['x','y','depth']].to_numpy()


def convert_coordinates_enhanced(localization_results: List[LocalizationResult], 
                               utm_xf: float, utm_y0: float) -> Dict[str, List]:
    """Convert localization results to different coordinate systems.
    
    Parameters
    ----------
    localization_results : list of LocalizationResult
        List of localization results with uncertainties
    utm_xf : float
        UTM X reference
    utm_y0 : float  
        UTM Y reference
        
    Returns
    -------
    dict
        Dictionary with local, utm, latlon coordinates and full results
    """
    import das4whales as dw
    
    local_coords = []
    utm_coords = []
    latlon_coords = []
    
    for result in localization_results:
        pos = result.position
        
        # Local coordinates (original)
        local_coords.append([pos[0], pos[1], pos[2], pos[3]])
        
        # Convert to UTM
        x_utm = utm_xf - pos[0]
        y_utm = utm_y0 + pos[1]
        utm_coords.append([x_utm, y_utm, pos[2], pos[3]])
        
        # Convert to lat/lon
        lon, lat = dw.map.utm_to_latlon(x_utm, y_utm)
        latlon_coords.append([lon, lat, pos[2], pos[3]])
    
    return {
        'local': np.array(local_coords) if local_coords else np.array([]),
        'utm': np.array(utm_coords) if utm_coords else np.array([]),
        'latlon': np.array(latlon_coords) if latlon_coords else np.array([]),
        'results': localization_results
    }


def create_comprehensive_dataframe(processed_files: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a comprehensive DataFrame from processed localization files.
    
    Parameters
    ----------
    processed_files : list of dict
        List of processed file results from process_association_enhanced
        
    Returns
    -------
    pd.DataFrame
        Comprehensive DataFrame with all localization results and uncertainties
    """
    all_rows = []
    
    for file_data in processed_files:
        utc_start = file_data['utc_start']
        filename = file_data['file']
        
        for sensor_call, data in file_data['results'].items():
            if data['results']:  # Only process if we have results
                sensor, call = sensor_call.split('_', 1) if '_' in sensor_call else (sensor_call, '')
                
                # Convert results to dictionary format
                rows = localization_results_to_dict(
                    data['results'], 
                    utc_start=utc_start,
                    sensor=sensor, 
                    call_type=call
                )
                
                # Add coordinate information and metadata
                for i, row in enumerate(rows):
                    if i < len(data['utm']):
                        row.update({
                            'x_utm': data['utm'][i][0],
                            'y_utm': data['utm'][i][1], 
                            'z_utm': data['utm'][i][2],
                            'lon': data['latlon'][i][0],
                            'lat': data['latlon'][i][1],
                            'filename': filename
                        })
                    
                    all_rows.append(row)
    
    return pd.DataFrame(all_rows)


def add_spatial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add spatial quality metrics to the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with localization results
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional spatial metrics
    """
    df = df.copy()
    
    # Add quality metrics
    df['horizontal_uncertainty'] = np.sqrt(df['unc_x']**2 + df['unc_y']**2)
    df['positional_quality'] = np.where(
        df['ellipse_area'] < 1e6,  # < 1 km²
        'high',
        np.where(df['ellipse_area'] < 1e7, 'medium', 'low')  # < 10 km²
    )
    
    # Add coordinate transformations in km
    df['x_km'] = df['x'] / 1000
    df['y_km'] = df['y'] / 1000
    df['x_utm_km'] = df['x_utm'] / 1000  
    df['y_utm_km'] = df['y_utm'] / 1000
    
    # Add time metrics
    if 'utc' in df.columns:
        df['minutes'] = (df['utc'] - df['utc'].min()).dt.total_seconds() / 60
        df['hours'] = df['minutes'] / 60
    
    return df


def filter_by_quality(df: pd.DataFrame, 
                     max_rms: float = 0.5,
                     max_ellipse_area: float = 1e7,
                     min_picks: int = 5) -> pd.DataFrame:
    """Filter localization results by quality metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with localization results
    max_rms : float, optional
        Maximum RMS threshold (default=0.5 seconds)
    max_ellipse_area : float, optional  
        Maximum ellipse area threshold (default=10 km²)
    min_picks : int, optional
        Minimum number of picks (default=5)
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    mask = (
        (df['rms'] <= max_rms) & 
        (df['ellipse_area'] <= max_ellipse_area) &
        (df['n_picks'] >= min_picks)
    )
    
    return df[mask].copy()


def plot_uncertainty_ellipses(df: pd.DataFrame, ax=None, scale_factor=1000, **kwargs):
    """Plot uncertainty ellipses on a map.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with localization results including ellipse parameters
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (default=None creates new figure)
    scale_factor : float, optional
        Scale factor for coordinates (default=1000 for km)
    **kwargs
        Additional keyword arguments for ellipse plotting
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object with plotted ellipses
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Default ellipse parameters
    ellipse_kwargs = {
        'alpha': 0.3,
        'facecolor': 'red',
        'edgecolor': 'darkred',
        'linewidth': 1
    }
    ellipse_kwargs.update(kwargs)
    
    for _, row in df.iterrows():
        if not (np.isnan(row['ellipse_semi_major']) or np.isnan(row['ellipse_semi_minor'])):
            ellipse = Ellipse(
                xy=(row['x'] / scale_factor, row['y'] / scale_factor),
                width=2 * row['ellipse_semi_major'] / scale_factor,
                height=2 * row['ellipse_semi_minor'] / scale_factor,
                angle=np.degrees(row['ellipse_rotation']),
                **ellipse_kwargs
            )
            ax.add_patch(ellipse)
    
    return ax
