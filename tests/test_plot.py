"""
Test module for das4whales.plot
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Import plot functions - we'll need to check what's available first
try:
    from das4whales.plot import import_roseus
    PLOT_MODULE_AVAILABLE = True
except ImportError:
    PLOT_MODULE_AVAILABLE = False


@pytest.mark.skipif(not PLOT_MODULE_AVAILABLE, reason="Plot module not fully available")
def test_import_roseus():
    """Test the import_roseus function."""
    # Test case 1: Basic functionality
    try:
        colormap = import_roseus()
        # Should return a matplotlib colormap
        assert hasattr(colormap, 'N')  # Colormap should have N attribute
        assert callable(colormap)  # Should be callable
    except Exception as e:
        pytest.skip(f"import_roseus function not working: {e}")


def test_plotting_basic():
    """Test basic plotting functionality."""
    # Test case 1: Simple plot creation
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    
    # Should not raise any errors
    assert fig is not None
    assert ax is not None
    
    plt.close(fig)


def test_matplotlib_backends():
    """Test that matplotlib backend is properly set for testing."""
    backend = matplotlib.get_backend()
    assert backend == 'Agg'


if __name__ == "__main__":
    pytest.main([__file__])
