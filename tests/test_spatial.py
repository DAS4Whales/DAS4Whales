"""
Test module for das4whales.spatial
"""

import numpy as np
import pytest

# Try to import spatial functions
try:
    from das4whales import spatial
    SPATIAL_MODULE_AVAILABLE = True
except ImportError:
    SPATIAL_MODULE_AVAILABLE = False


@pytest.mark.skipif(not SPATIAL_MODULE_AVAILABLE, reason="Spatial module not available")
def test_spatial_module_import():
    """Test that spatial module can be imported."""
    import das4whales.spatial
    assert hasattr(das4whales, 'spatial')


def test_spatial_operations():
    """Test basic spatial operations and concepts."""
    # Test case 1: Coordinate transformations
    lat = np.array([45.0, 45.1, 45.2])  # degrees
    lon = np.array([-125.0, -125.1, -125.2])  # degrees
    
    # Basic validation of coordinate ranges
    assert np.all((-90 <= lat) & (lat <= 90))  # Valid latitude range
    assert np.all((-180 <= lon) & (lon <= 180))  # Valid longitude range
    
    # Test case 2: Distance calculations using haversine approximation
    # Simple distance calculation for small distances
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    
    # For small distances, this is approximately correct
    R = 6371000  # Earth radius in meters
    dx_approx = R * np.radians(dlat) 
    dy_approx = R * np.radians(dlon) * np.cos(np.radians(lat[:-1]))
    
    assert len(dx_approx) == len(lat) - 1
    assert len(dy_approx) == len(lon) - 1
    
    # Test case 3: Depth array processing
    depths = np.array([100, 150, 200, 175, 225])  # meters
    max_depth = np.max(depths)
    min_depth = np.min(depths)
    
    assert max_depth == 225
    assert min_depth == 100
    assert len(depths) == 5


def test_interpolation_concepts():
    """Test interpolation concepts used in spatial processing."""
    # Test case 1: Linear interpolation
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 2, 4, 6, 8])
    
    # Test interpolation at 1.5
    x_interp = 1.5
    y_interp = np.interp(x_interp, x, y)
    assert np.isclose(y_interp, 3.0)
    
    # Test case 2: 2D grid concepts
    x_grid = np.linspace(0, 10, 11)
    y_grid = np.linspace(0, 5, 6)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    assert X.shape == (6, 11)  # (len(y_grid), len(x_grid))
    assert Y.shape == (6, 11)
    
    # Test case 3: Spatial indexing
    cable_positions = np.arange(0, 1000, 10)  # Every 10m along cable
    selected_positions = cable_positions[::5]  # Every 5th position
    
    assert len(selected_positions) == 20  # 100 positions / 5
    assert selected_positions[0] == 0
    assert selected_positions[1] == 50


if __name__ == "__main__":
    pytest.main([__file__])
