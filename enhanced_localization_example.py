"""
Enhanced localization example with uncertainty quantification.

This script demonstrates how to use the new uncertainty features in the das4whales
localization module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import das4whales as dw
from das4whales.tracking import (
    process_association_enhanced,
    create_comprehensive_dataframe,
    add_spatial_metrics,
    filter_by_quality,
    plot_uncertainty_ellipses
)

# Configuration
pkl_dir = Path('out/batch1_baseline/')
north_csv = 'data/north_DAS_multicoord.csv'
south_csv = 'data/south_DAS_multicoord.csv'

# UTM reference (adjust based on your data)
# These should match your existing coordinate system
utm_xf = 123456.0  # Replace with actual UTM reference
utm_y0 = 789012.0  # Replace with actual UTM reference

def main():
    """Main processing function."""
    
    print("Processing association files with enhanced uncertainty quantification...")
    
    # Process all association files
    processed_files = []
    pkl_files = sorted(pkl_dir.glob('association_*.pkl'))
    print(f"Found {len(pkl_files)} association files to process.")

    for pkl_file in pkl_files:  # Process all files
        print(f"Processing {pkl_file.name}...")
        try:
            result = process_association_enhanced(pkl_file, north_csv, south_csv, utm_xf, utm_y0)
            processed_files.append(result)
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {e}")
        
        if not processed_files:
            print("No files processed successfully!")
            return
    
    # Create comprehensive DataFrame
    print("Creating comprehensive DataFrame...")
    df = create_comprehensive_dataframe(processed_files)
    
    # Add spatial metrics
    df = add_spatial_metrics(df)
    
    print(f"Total localizations: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Display some statistics
    print("\n=== Localization Statistics ===")
    print(f"RMS statistics:")
    print(f"  Mean: {df['rms'].mean():.4f} s")
    print(f"  Median: {df['rms'].median():.4f} s")
    print(f"  95th percentile: {df['rms'].quantile(0.95):.4f} s")
    
    print(f"\nWeighted RMS statistics:")
    print(f"  Mean: {df['weighted_rms'].mean():.4f} s")
    print(f"  Median: {df['weighted_rms'].median():.4f} s")
    print(f"  95th percentile: {df['weighted_rms'].quantile(0.95):.4f} s")
    
    print(f"\nHorizontal uncertainty statistics:")
    print(f"  Mean: {df['horizontal_uncertainty'].mean():.1f} m")
    print(f"  Median: {df['horizontal_uncertainty'].median():.1f} m")
    print(f"  95th percentile: {df['horizontal_uncertainty'].quantile(0.95):.1f} m")
    
    print(f"\nEllipse area statistics:")
    print(f"  Mean: {df['ellipse_area'].mean()/1e6:.2f} km²")
    print(f"  Median: {df['ellipse_area'].median()/1e6:.2f} km²")
    print(f"  95th percentile: {df['ellipse_area'].quantile(0.95)/1e6:.2f} km²")
    
    # Quality filtering
    df_filtered = filter_by_quality(df, max_rms=0.3, max_ellipse_area=5e6, min_picks=8)
    print(f"\nAfter quality filtering: {len(df_filtered)} localizations ({len(df_filtered)/len(df)*100:.1f}%)")
    
    # Save comprehensive results
    output_file = 'localization_results_with_uncertainty.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Create visualization
    create_uncertainty_visualization(df, df_filtered)


def create_uncertainty_visualization(df_all, df_filtered):
    """Create visualization showing localization results with uncertainty."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: RMS vs Number of picks
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df_all['n_picks'], df_all['rms'], 
                         c=df_all['ellipse_area']/1e6, cmap='viridis', alpha=0.6)
    ax1.set_xlabel('Number of Picks')
    ax1.set_ylabel('RMS (s)')
    ax1.set_title('RMS vs Number of Picks')
    plt.colorbar(scatter, ax=ax1, label='Ellipse Area (km²)')
    
    # Plot 2: Weighted vs Unweighted RMS
    ax2 = axes[0, 1]
    ax2.scatter(df_all['rms'], df_all['weighted_rms'], alpha=0.6)
    ax2.plot([0, df_all['rms'].max()], [0, df_all['rms'].max()], 'r--', alpha=0.8)
    ax2.set_xlabel('Unweighted RMS (s)')
    ax2.set_ylabel('Weighted RMS (s)')
    ax2.set_title('Weighted vs Unweighted RMS')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty ellipse size distribution
    ax3 = axes[1, 0]
    ax3.hist(df_all['ellipse_area']/1e6, bins=50, alpha=0.7, label='All')
    ax3.hist(df_filtered['ellipse_area']/1e6, bins=50, alpha=0.7, label='Filtered')
    ax3.set_xlabel('Ellipse Area (km²)')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Ellipse Area Distribution')
    ax3.legend()
    ax3.set_yscale('log')
    
    # Plot 4: Spatial distribution with uncertainty
    ax4 = axes[1, 1]
    # Plot points sized by uncertainty
    sizes = 50 * (1 + df_filtered['horizontal_uncertainty'] / df_filtered['horizontal_uncertainty'].max())
    scatter = ax4.scatter(df_filtered['x_km'], df_filtered['y_km'], 
                         s=sizes, c=df_filtered['rms'], 
                         cmap='plasma', alpha=0.6)
    ax4.set_xlabel('X (km)')
    ax4.set_ylabel('Y (km)')
    ax4.set_title('Spatial Distribution (size=uncertainty, color=RMS)')
    plt.colorbar(scatter, ax=ax4, label='RMS (s)')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('localization_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_map_with_uncertainties(df):
    """Create a map showing localizations with uncertainty ellipses."""
    
    # This function would integrate with your existing mapping code
    # Here's a template that you can adapt:
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot bathymetry (adapt to your bathymetry loading code)
    # ax.imshow(bathy, extent=extent, ...)
    
    # Plot cable positions (adapt to your cable data)
    # ax.plot(cable_x, cable_y, 'r-', label='Cable')
    
    # Plot localizations by type
    sensors = df['sensor'].unique()
    call_types = df['call_type'].unique()
    
    markers = {'pair': '*', 'north': '^', 'south': 'v'}
    colors = {'hf': 'red', 'lf': 'blue'}
    
    for sensor in sensors:
        for call_type in call_types:
            mask = (df['sensor'] == sensor) & (df['call_type'] == call_type)
            if mask.any():
                subset = df[mask]
                ax.scatter(subset['x_km'], subset['y_km'],
                          marker=markers.get(sensor, 'o'),
                          c=colors.get(call_type, 'gray'),
                          s=100, alpha=0.8,
                          label=f'{sensor}-{call_type}')
    
    # Plot uncertainty ellipses for high-quality localizations
    high_quality = df[df['positional_quality'] == 'high']
    plot_uncertainty_ellipses(high_quality, ax=ax, alpha=0.2)
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Whale Localizations with Uncertainty Ellipses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('whale_localizations_with_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
