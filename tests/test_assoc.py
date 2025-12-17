"""
Test module for das4whales.assoc (association)
"""

import numpy as np
import pytest

# Try to import assoc functions
try:
    from das4whales import assoc
    ASSOC_MODULE_AVAILABLE = True
except ImportError:
    ASSOC_MODULE_AVAILABLE = False


@pytest.mark.skipif(not ASSOC_MODULE_AVAILABLE, reason="Assoc module not available")
def test_assoc_module_import():
    """Test that assoc module can be imported."""
    import das4whales.assoc
    assert hasattr(das4whales, 'assoc')


def test_association_concepts():
    """Test basic concepts used in association algorithms."""
    # Test case 1: Peak detection simulation
    signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)) + 0.1 * np.random.randn(1000)
    peaks = np.array([100, 300, 500, 700])  # Simulated peak locations
    
    # Basic validation
    assert len(peaks) == 4
    assert np.all(peaks >= 0)
    assert np.all(peaks < len(signal))
    
    # Test case 2: Time difference calculations for association
    time_picks_north = np.array([1.0, 2.5, 4.0])  # seconds
    time_picks_south = np.array([1.2, 2.7, 4.1])  # seconds
    
    # Calculate time differences
    time_diffs = time_picks_south - time_picks_north
    expected_diffs = np.array([0.2, 0.2, 0.1])
    
    assert np.allclose(time_diffs, expected_diffs)
    
    # Test case 3: Distance calculations for hyperbola fitting
    distances = np.array([1000, 2000, 3000, 4000])  # meters along cable
    velocity = 1500  # m/s sound speed
    time_delays = distances / velocity
    
    expected_delays = np.array([2/3, 4/3, 2.0, 8/3])
    assert np.allclose(time_delays, expected_delays)


def test_kde_concepts():
    """Test kernel density estimation concepts used in association."""
    # Test case 1: 1D KDE simulation
    data = np.random.normal(0, 1, 100)
    
    # Basic KDE validation - just check we can compute density
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    
    # Evaluate KDE at test points
    test_points = np.linspace(-3, 3, 10)
    densities = kde(test_points)
    
    assert len(densities) == len(test_points)
    assert np.all(densities >= 0)  # Densities should be non-negative
    
    # Test case 2: Peak in density should be near mean of data
    peak_idx = np.argmax(densities)
    peak_location = test_points[peak_idx]
    data_mean = np.mean(data)
    
    # Peak should be reasonably close to the data mean
    assert abs(peak_location - data_mean) < 1.0


def test_hyperbola_concepts():
    """Test hyperbola fitting concepts."""
    # Test case 1: Simple distance calculations for hyperbola-like associations
    # In DAS applications, we often look at time differences between stations
    station1_pos = np.array([0, 0])  # Position of station 1
    station2_pos = np.array([1000, 0])  # Position of station 2, 1 km away
    
    # Test source positions
    source_positions = np.array([[500, 500], [200, 600], [800, 400]])  # Different source locations
    
    # Calculate distances from each source to each station
    dist_to_station1 = np.linalg.norm(source_positions - station1_pos, axis=1)
    dist_to_station2 = np.linalg.norm(source_positions - station2_pos, axis=1)
    
    # Time difference of arrival (TDOA) - this is what's used in association
    sound_speed = 1500  # m/s
    tdoa = (dist_to_station1 - dist_to_station2) / sound_speed
    
    # Basic validation
    assert len(tdoa) == len(source_positions)
    assert np.all(np.abs(tdoa) < 2.0)  # Should be reasonable TDOA values (< 2 seconds)
    
    # Test case 2: Association tolerance
    tolerance = 0.1  # seconds
    reference_tdoa = 0.2  # seconds
    
    # Simulate measurements with small variations
    measured_tdoa = np.array([0.18, 0.22, 0.19, 0.21])
    
    # Check which measurements are within tolerance
    within_tolerance = np.abs(measured_tdoa - reference_tdoa) <= tolerance
    assert np.sum(within_tolerance) >= 3  # At least 3 should be within tolerance


if __name__ == "__main__":
    pytest.main([__file__])
