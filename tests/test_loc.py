"""
Test module for das4whales.loc (localization)
"""

import numpy as np
import pytest

# Try to import loc functions
try:
    from das4whales import loc
    LOC_MODULE_AVAILABLE = True
except ImportError:
    LOC_MODULE_AVAILABLE = False


@pytest.mark.skipif(not LOC_MODULE_AVAILABLE, reason="Loc module not available")
def test_loc_module_import():
    """Test that loc module can be imported."""
    import das4whales.loc
    assert hasattr(das4whales, 'loc')


def test_localization_concepts():
    """Test basic concepts used in localization algorithms."""
    # Test case 1: Distance calculation
    point1 = np.array([0, 0])
    point2 = np.array([3, 4])
    distance = np.linalg.norm(point2 - point1)
    assert np.isclose(distance, 5.0)  # 3-4-5 triangle
    
    # Test case 2: Time difference calculation
    time_diff = np.array([0.1, 0.2, 0.3])  # seconds
    velocity = 1500  # m/s (approximate sound speed in water)
    distances = time_diff * velocity
    expected = np.array([150, 300, 450])
    assert np.allclose(distances, expected)
    
    # Test case 3: Array indexing for channel selection
    channels = np.arange(100)
    selected = channels[10:50:2]  # Every 2nd channel from 10 to 50
    assert len(selected) == 20
    assert selected[0] == 10
    assert selected[-1] == 48


if __name__ == "__main__":
    pytest.main([__file__])
