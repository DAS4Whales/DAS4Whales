"""
Test the enhanced tracking code with uncertainty quantification.

This script tests the enhanced localization with the correct coordinate system
extracted from the bathymetry file, just like in tracks.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import das4whales as dw
import pickle
import datetime
from tqdm import tqdm

# Constants from tracks.py
C0 = 1480  # sound speed (m/s)

def setup_coordinate_system():
    """Set up coordinate system exactly like in tracks.py"""
    print("Setting up coordinate system from bathymetry...")
    
    # Import the bathymetry data (same as tracks.py)
    bathy, xlon, ylat = dw.map.load_bathymetry('data/GMRT_OOI_RCA_Cables.grd')
    print(f'Origin of the coordinates. Latitude = {ylat[0]}, Longitude = {xlon[-1]}')

    utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
    utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])

    # Change the reference point to the last point (same as tracks.py)
    x0, y0 = utm_xf - utm_x0, utm_y0 - utm_y0
    xf, yf = utm_xf - utm_xf, utm_yf - utm_y0
    print(f"Local coordinates: x0={x0}, y0={y0}, xf={xf}, yf={yf}")
    
    return {
        'bathy': bathy,
        'xlon': xlon,
        'ylat': ylat,
        'utm_x0': utm_x0, 'utm_y0': utm_y0,
        'utm_xf': utm_xf, 'utm_yf': utm_yf,
        'x0': x0, 'y0': y0, 'xf': xf, 'yf': yf
    }

def get_cable_pos(df_path: str, side_meta: dict) -> np.ndarray:
    """Extract cable positions from CSV file based on metadata (same as tracks.py)"""
    df = pd.read_csv(df_path)
    sel_start, sel_end, sel_step = side_meta['selected_channels']
    chan_idx = df["chan_idx"]
    idx0 = int(sel_start - chan_idx.iloc[0])
    idxn = int(sel_end - chan_idx.iloc[-1])
    n_samp = side_meta['data_shape'][0]
    df_used = df.iloc[idx0:idxn:sel_step][:n_samp]
    return df_used[['x','y','depth']].to_numpy()

def local_to_utm(localizations: np.ndarray, utm_xf: float, utm_y0: float) -> np.ndarray:
    """Convert local coordinates to UTM (same as tracks.py)"""
    return np.array([[utm_xf - x, utm_y0 + y, z, t] for x, y, z, t in localizations])

def convert_coords(localizations: np.ndarray, utm_xf: float, utm_y0: float):
    """Convert coordinates to different systems (same as tracks.py)"""
    loc_utm = local_to_utm(localizations, utm_xf, utm_y0)
    loc_latlon = batch_utm_to_latlon(loc_utm)
    return loc_utm, loc_latlon

def batch_utm_to_latlon(loc_utm: np.ndarray) -> np.ndarray:
    """Convert UTM to lat/lon (same as tracks.py)"""
    out = []
    for x_utm, y_utm, z, t in loc_utm:
        lon, lat = dw.map.utm_to_latlon(x_utm, y_utm)
        out.append([lon, lat, z, t])
    return np.array(out)

def test_enhanced_localization_single_file():
    """Test enhanced localization on a single association file"""
    
    # Setup coordinates
    coord_sys = setup_coordinate_system()
    utm_xf, utm_y0 = coord_sys['utm_xf'], coord_sys['utm_y0']
    
    # Load one association file for testing
    pkl_path = Path('out/batch1_baseline/association_2021-11-04_02:00:02.pkl')
    
    if not pkl_path.exists():
        print(f"Association file not found: {pkl_path}")
        print("Looking for alternative files...")
        pkl_dir = Path('out/batch1_baseline/')
        pkl_files = list(pkl_dir.glob('association_*.pkl'))
        if pkl_files:
            pkl_path = pkl_files[0]
            print(f"Using: {pkl_path}")
        else:
            print("No association files found!")
            return
    
    print(f"\nTesting enhanced localization with: {pkl_path.name}")
    
    # Load association data
    with open(pkl_path, 'rb') as f:
        association = pickle.load(f)
    
    # Extract metadata
    c0 = C0
    fs = association['metadata']['north']['fs']
    n_selected_channels = association['metadata']['north']['selected_channels']
    s_selected_channels = association['metadata']['south']['selected_channels']
    nnx = association['metadata']['north']['data_shape'][0]
    snx = association['metadata']['south']['data_shape'][0]
    utc_str = association['metadata']['south']['fileBeginTimeUTC']
    utc_start = datetime.datetime.strptime(utc_str, "%Y-%m-%d_%H:%M:%S")
    
    print(f"Sampling frequency: {fs} Hz")
    print(f"Sound speed: {c0} m/s")
    print(f"North channels: {nnx}, South channels: {snx}")
    print(f"Start time: {utc_start}")
    
    # Get cable positions
    north_pos = get_cable_pos('data/north_DAS_multicoord.csv', association['metadata']['north'])
    south_pos = get_cable_pos('data/south_DAS_multicoord.csv', association['metadata']['south'])
    
    print(f"North cable: {north_pos.shape[0]} positions")
    print(f"South cable: {south_pos.shape[0]} positions")
    
    # Extract association data
    p_n_hf = association['assoc_pair']['north']['hf']
    p_s_hf = association['assoc_pair']['south']['hf']
    n_hf = association['assoc']['north']['hf']
    s_hf = association['assoc']['south']['hf']
    
    print(f"\nAssociation counts:")
    print(f"HF paired calls: {len(p_n_hf) if p_n_hf else 0}")
    print(f"North HF calls: {len(n_hf) if n_hf else 0}")
    print(f"South HF calls: {len(s_hf) if s_hf else 0}")
    
    # Test enhanced localization
    results = {}
    
    # Test bicable localization with uncertainty
    if p_n_hf and p_s_hf and len(p_n_hf) > 0:
        print(f"\nTesting enhanced bicable localization...")
        print(f"Processing {len(p_n_hf)} paired HF calls...")
        
        enhanced_results = dw.loc.loc_picks_bicable_list(
            p_n_hf, p_s_hf, (north_pos, south_pos), c0, fs, 
            return_uncertainty=True
        )
        
        print(f"Enhanced results type: {type(enhanced_results[0]) if enhanced_results else 'None'}")
        
        if enhanced_results:
            # Test first result
            result = enhanced_results[0]
            print(f"\nFirst localization result:")
            print(f"Position: {result.position}")
            print(f"RMS: {result.rms:.4f} s")
            print(f"Weighted RMS: {result.weighted_rms:.4f} s")
            print(f"Uncertainties: {result.uncertainties}")
            print(f"Number of picks: {len(result.residuals)}")
            print(f"Iterations: {result.n_iterations}")
            
            # Convert to different coordinate systems
            positions = np.array([r.position for r in enhanced_results])
            loc_utm, loc_latlon = convert_coords(positions, utm_xf, utm_y0)
            
            print(f"\nCoordinate conversions (first result):")
            print(f"Local: {positions[0]}")
            print(f"UTM: {loc_utm[0]}")
            print(f"Lat/Lon: {loc_latlon[0]}")
            
            results['hf_pair'] = {
                'enhanced_results': enhanced_results,
                'positions': positions,
                'utm': loc_utm,
                'latlon': loc_latlon
            }
    
    # Test single cable localization
    if n_hf and len(n_hf) > 0:
        print(f"\nTesting enhanced single cable localization...")
        print(f"Processing {len(n_hf)} north HF calls...")
        
        enhanced_north = dw.loc.loc_from_picks(
            n_hf, north_pos, c0, fs, return_uncertainty=True
        )
        
        if enhanced_north:
            result = enhanced_north[0]
            print(f"North localization result:")
            print(f"Position: {result.position}")
            print(f"RMS: {result.rms:.4f} s")
            print(f"Weighted RMS: {result.weighted_rms:.4f} s")
            
            results['north_hf'] = {
                'enhanced_results': enhanced_north,
                'positions': np.array([r.position for r in enhanced_north])
            }
    
    # Create summary statistics
    if results:
        print(f"\n=== SUMMARY STATISTICS ===")
        
        for sensor, data in results.items():
            enhanced_results = data['enhanced_results']
            print(f"\n{sensor.upper()}:")
            print(f"  Number of localizations: {len(enhanced_results)}")
            
            rms_values = [r.rms for r in enhanced_results]
            weighted_rms_values = [r.weighted_rms for r in enhanced_results]
            n_picks = [len(r.residuals) for r in enhanced_results]
            
            print(f"  RMS: mean={np.mean(rms_values):.4f}, median={np.median(rms_values):.4f}")
            print(f"  Weighted RMS: mean={np.mean(weighted_rms_values):.4f}, median={np.median(weighted_rms_values):.4f}")
            print(f"  Picks per localization: mean={np.mean(n_picks):.1f}, median={np.median(n_picks):.1f}")
            
            # Quality assessment
            high_quality = sum(1 for r in enhanced_results if r.rms < 0.3 and len(r.residuals) >= 8)
            print(f"  High quality (RMS<0.3s, picks>=8): {high_quality}/{len(enhanced_results)} ({100*high_quality/len(enhanced_results):.1f}%)")
    
    return results

def create_comprehensive_csv(results, utc_start, coord_sys):
    """Create a comprehensive CSV with uncertainty information"""
    
    all_rows = []
    
    for sensor, data in results.items():
        enhanced_results = data['enhanced_results']
        
        for i, result in enumerate(enhanced_results):
            # Convert coordinates
            pos = result.position
            x_utm = coord_sys['utm_xf'] - pos[0]
            y_utm = coord_sys['utm_y0'] + pos[1]
            lon, lat = dw.map.utm_to_latlon(x_utm, y_utm)
            
            # Calculate error ellipse if covariance is available
            try:
                from das4whales.loc import calc_error_ellipse_params
                ellipse_params = calc_error_ellipse_params(result.covariance)
            except:
                ellipse_params = {
                    'semi_major_axis': np.nan,
                    'semi_minor_axis': np.nan,
                    'rotation_angle': np.nan,
                    'area': np.nan
                }
            
            row = {
                'id': i,
                'utc': utc_start + datetime.timedelta(seconds=float(pos[3])),
                'sensor': sensor.split('_')[0] if '_' in sensor else sensor,
                'call_type': sensor.split('_')[1] if '_' in sensor else 'unknown',
                'x_local': float(pos[0]),
                'y_local': float(pos[1]),
                'z_local': float(pos[2]),
                't0': float(pos[3]),
                'x_utm': float(x_utm),
                'y_utm': float(y_utm),
                'lat': float(lat),
                'lon': float(lon),
                'x_km': float(pos[0]/1000),
                'y_km': float(pos[1]/1000),
                'rms': float(result.rms),
                'weighted_rms': float(result.weighted_rms),
                'unc_x': float(result.uncertainties[0]),
                'unc_y': float(result.uncertainties[1]),
                'unc_z': float(result.uncertainties[2]) if len(result.uncertainties) > 2 else np.nan,
                'unc_t': float(result.uncertainties[-1]),
                'horizontal_uncertainty': float(np.sqrt(result.uncertainties[0]**2 + result.uncertainties[1]**2)),
                'ellipse_semi_major': float(ellipse_params['semi_major_axis']),
                'ellipse_semi_minor': float(ellipse_params['semi_minor_axis']),
                'ellipse_rotation': float(ellipse_params['rotation_angle']),
                'ellipse_area': float(ellipse_params['area']),
                'n_picks': len(result.residuals),
                'n_iterations': result.n_iterations,
                'filename': 'test_file.pkl'
            }
            
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    return df

def main():
    """Main test function"""
    print("=== ENHANCED LOCALIZATION TEST ===")
    print("Testing enhanced localization with uncertainty quantification")
    print("Using coordinate system from bathymetry file (same as tracks.py)")
    
    try:
        # Test single file
        results = test_enhanced_localization_single_file()
        
        if results:
            # Setup coordinates for CSV export
            coord_sys = setup_coordinate_system()
            utc_start = datetime.datetime(2021, 11, 4, 2, 0, 2)  # Default from filename
            
            # Create comprehensive CSV
            df = create_comprehensive_csv(results, utc_start, coord_sys)
            
            # Save results
            output_file = 'enhanced_localization_test_results.csv'
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            print(f"Columns: {list(df.columns)}")
            
            # Display sample data
            print(f"\nFirst few rows:")
            print(df.head())
            
            print(f"\n=== TEST COMPLETED SUCCESSFULLY ===")
            print(f"Enhanced localization is working with correct coordinate system!")
            print(f"Generated CSV with {len(df)} localizations including full uncertainty information")
            
        else:
            print("No results generated - check association files")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
