"""
Create a map visualization of the enhanced localization results with uncertainty ellipses.
This follows the same style as tracks.py but adds error ellipses to show uncertainties.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.colors import LightSource
import das4whales as dw
import cmocean.cm as cmo

def plot_localization_map_with_uncertainty():
    """Create a map showing localizations with uncertainty"""
    
    # Load test results
    df = pd.read_csv('localization_results_with_uncertainty.csv')

    # Load coordinate system (same as tracks.py)
    bathy, xlon, ylat = dw.map.load_bathymetry('data/GMRT_OOI_RCA_Cables.grd')
    utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
    utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])
    
    x0, y0 = utm_xf - utm_x0, utm_y0 - utm_y0
    xf, yf = utm_xf - utm_xf, utm_yf - utm_y0
    x = np.linspace(x0, xf, len(xlon))
    y = np.linspace(y0, yf, len(ylat))
    
    # Load cable data
    df_north = pd.read_csv('data/north_DAS_multicoord.csv')
    df_south = pd.read_csv('data/south_DAS_multicoord.csv')
    
    # Set up colors (same as tracks.py)
    colors_undersea = cmo.deep_r(np.linspace(0, 1, 256))
    colors_land = np.array([[0.5, 0.5, 0.5, 1]])
    all_colors = np.vstack((colors_undersea, colors_land))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)
    
    # Set up light source
    ls = LightSource(azdeg=350, altdeg=45)
    
    # Create extent in km (like tracks.py)
    extent_km = [x[0]/1000, x[-1]/1000, y[0]/1000, y[-1]/1000]
    
    # Create the map
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot bathymetry relief
    rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', 
                   vmin=np.min(bathy), vmax=0)
    ax.imshow(rgb, extent=extent_km, aspect='equal', origin='lower')
    
    # Plot cables
    ax.plot(df_north['x']/1000, df_north['y']/1000, 'tab:red', linewidth=2, label='North cable')
    ax.plot(df_south['x']/1000, df_south['y']/1000, 'tab:orange', linewidth=2, label='South cable')
    
    # Add bathymetry contours
    depth_levels = [-1500, -1000, -600, -250, -80]
    contour = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', 
                        extent=extent_km, alpha=0.6)
    ax.clabel(contour, fmt='%d m', inline=True)
    
    # Plot localizations with uncertainty ellipses
    markers = {'hf': '*', 'north': '^'}
    colors = {'pair': 'red', 'hf': 'blue'}
    
    # First plot uncertainty ellipses (behind the points)
    for _, row in df.iterrows():
        if not (np.isnan(row['ellipse_semi_major']) or np.isnan(row['ellipse_semi_minor'])):
            # Create ellipse
            ellipse = Ellipse(
                xy=(row['x_km'], row['y_km']),
                width=2 * row['ellipse_semi_major'] / 1000,  # Convert to km
                height=2 * row['ellipse_semi_minor'] / 1000,
                angle=np.degrees(row['ellipse_rotation']),
                facecolor='lightblue' if row['sensor'] == 'hf' else 'lightgreen',
                edgecolor='darkblue' if row['sensor'] == 'hf' else 'darkgreen',
                alpha=0.4,
                linewidth=1
            )
            ax.add_patch(ellipse)
    
    # Then plot the localization points
    for sensor in df['sensor'].unique():
        for call_type in df['call_type'].unique():
            mask = (df['sensor'] == sensor) & (df['call_type'] == call_type)
            if mask.any():
                subset = df[mask]
                marker = markers.get(sensor, 'o')
                color = colors.get(call_type, 'gray')
                
                # Color by time (minutes from start)
                times_min = (pd.to_datetime(subset['utc']) - pd.to_datetime(subset['utc']).min()).dt.total_seconds() / 60
                
                scatter = ax.scatter(subset['x_km'], subset['y_km'],
                          marker=marker, c=times_min, cmap='plasma',
                          s=120, alpha=0.9, edgecolors='black', linewidths=1,
                          label=f'{sensor}-{call_type}')
    
    # Add colorbar for time
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30, fraction=0.0195)
    cbar.set_label('Time [minutes]')
    
    # Add quality indicators as text annotations
    for _, row in df.iterrows():
        # Only annotate high-quality localizations
        if row['rms'] < 0.25 and row['n_picks'] > 5000:
            ax.annotate(f"RMS: {row['rms']:.3f}s\nUnc: {row['horizontal_uncertainty']:.1f}m",
                       xy=(row['x_km'], row['y_km']), 
                       xytext=(15, 15), textcoords='offset points',
                       fontsize=20, alpha=0.8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_title('Localizations with uncertainties\n'
                f'File: association_2021-11-04_02:00:02.pkl', fontsize=20)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=20)
    
    # Add grid
    ax.grid(linestyle='--', alpha=0.3, color='k')
    
    # Add text box with statistics
    stats_text = f"Total localizations: {len(df)}\n"
    stats_text += f"Mean RMS: {df['rms'].mean():.3f}s\n"
    stats_text += f"Mean weighted RMS: {df['weighted_rms'].mean():.3f}s\n"
    stats_text += f"Mean horizontal uncertainty: {df['horizontal_uncertainty'].mean():.1f}m\n"
    stats_text += f"High quality (RMS<0.3s): {len(df[df['rms']<0.3])}/{len(df)}"
    
    ax.text(0.02, 0.80, stats_text, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('whale_localizations_with_uncertainty_ellipses.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('whale_localizations_with_uncertainty_ellipses.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("Map saved as:")
    print("- whale_localizations_with_uncertainty_ellipses.png")
    print("- whale_localizations_with_uncertainty_ellipses.pdf")
    
    plt.show()
    
    return fig, ax

def create_zoomed_views():
    """Create zoomed views of different localization clusters"""

    df = pd.read_csv('localization_results_with_uncertainty.csv')

    # Group localizations by spatial clusters
    # Cluster 1: Around x=37km, y=22km (bicable near-field)
    cluster1 = df[(df['x_km'] > 35) & (df['x_km'] < 40) & (df['y_km'] > 20) & (df['y_km'] < 25)]
    
    # Cluster 2: Around x=83km, y=12km (bicable far-field) 
    cluster2 = df[(df['x_km'] > 80) & (df['x_km'] < 90) & (df['y_km'] > 10) & (df['y_km'] < 15)]
    
    # Cluster 3: Around x=57-72km, y=23-28km (north cable)
    cluster3 = df[(df['x_km'] > 55) & (df['x_km'] < 75) & (df['y_km'] > 22) & (df['y_km'] < 30)]
    
    clusters = [
        (cluster1, "Near-field Bicable Cluster", (35, 40, 20, 25)),
        (cluster2, "Far-field Bicable Cluster", (80, 90, 10, 15)), 
        (cluster3, "North Cable Cluster", (55, 75, 22, 30))
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (cluster_df, title, (xmin, xmax, ymin, ymax)) in enumerate(clusters):
        if len(cluster_df) == 0:
            axes[i].text(0.5, 0.5, 'No data in this region', 
                        transform=axes[i].transAxes, ha='center')
            axes[i].set_title(title)
            continue
            
        ax = axes[i]
        
        # Plot ellipses
        for _, row in cluster_df.iterrows():
            if not (np.isnan(row['ellipse_semi_major']) or np.isnan(row['ellipse_semi_minor'])):
                ellipse = Ellipse(
                    xy=(row['x_km'], row['y_km']),
                    width=2 * row['ellipse_semi_major'] / 1000,
                    height=2 * row['ellipse_semi_minor'] / 1000,
                    angle=np.degrees(row['ellipse_rotation']),
                    facecolor='lightcoral',
                    edgecolor='darkred',
                    alpha=0.5,
                    linewidth=2
                )
                ax.add_patch(ellipse)
        
        # Plot points
        markers = {'hf': '*', 'north': '^'}
        colors = {'pair': 'red', 'hf': 'blue'}
        
        for _, row in cluster_df.iterrows():
            marker = markers.get(row['sensor'], 'o')
            color = colors.get(row['call_type'], 'gray')
            ax.scatter(row['x_km'], row['y_km'], marker=marker, c=color, 
                      s=150, edgecolors='black', linewidths=1)
            
            # Add quality annotation
            ax.annotate(f"RMS: {row['rms']:.3f}s", 
                       xy=(row['x_km'], row['y_km']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=20, alpha=0.9)
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_title(f'{title}\n({len(cluster_df)} localizations)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('localization_clusters_detailed.png', dpi=300, bbox_inches='tight')
    print("Detailed cluster view saved as: localization_clusters_detailed.png")
    plt.show()

def print_input_data_summary():
    """Print summary of the input data used in the test"""

    df = pd.read_csv('localization_results_with_uncertainty.csv')

    print("=== INPUT DATA SUMMARY ===")
    print(f"Source file: association_2021-11-04_02:00:02.pkl")
    print(f"Time period: {df['utc'].min()} to {df['utc'].max()}")
    print(f"Duration: {(pd.to_datetime(df['utc'].max()) - pd.to_datetime(df['utc'].min())).total_seconds():.1f} seconds")
    print()
    
    print("Localization breakdown:")
    for sensor in df['sensor'].unique():
        sensor_data = df[df['sensor'] == sensor]
        print(f"  {sensor.upper()}:")
        for call_type in sensor_data['call_type'].unique():
            count = len(sensor_data[sensor_data['call_type'] == call_type])
            print(f"    {call_type}: {count} localizations")
        print(f"    Total picks range: {sensor_data['n_picks'].min()}-{sensor_data['n_picks'].max()}")
    print()
    
    print("Spatial distribution:")
    print(f"  X range: {df['x_km'].min():.1f} - {df['x_km'].max():.1f} km")
    print(f"  Y range: {df['y_km'].min():.1f} - {df['y_km'].max():.1f} km")
    print(f"  Area covered: ~{(df['x_km'].max()-df['x_km'].min()) * (df['y_km'].max()-df['y_km'].min()):.0f} km²")

if __name__ == "__main__":
    print_input_data_summary()
    print("\n" + "="*50 + "\n")
    
    # Create main map with uncertainty ellipses
    plot_localization_map_with_uncertainty()
    
    print("\n" + "="*50 + "\n")
    
    # Create detailed cluster views
    create_zoomed_views()
