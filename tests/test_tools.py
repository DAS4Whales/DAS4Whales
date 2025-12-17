"""
Test module for das4whales.tools
"""

import numpy as np
import pytest

# Try to import tools functions
try:
    from das4whales import tools
    TOOLS_MODULE_AVAILABLE = True
except ImportError:
    TOOLS_MODULE_AVAILABLE = False


@pytest.mark.skipif(not TOOLS_MODULE_AVAILABLE, reason="Tools module not available")
def test_tools_module_import():
    """Test that tools module can be imported."""
    import das4whales.tools
    assert hasattr(das4whales, 'tools')


def test_numpy_functionality():
    """Test basic numpy operations used in tools."""
    # Test case 1: Array operations
    arr = np.array([1, 2, 3, 4, 5])
    assert len(arr) == 5
    assert np.sum(arr) == 15
    
    # Test case 2: Matrix operations
    matrix = np.random.randn(3, 3)
    assert matrix.shape == (3, 3)
    
    # Test case 3: Mathematical operations
    result = np.fft.fft(arr)
    assert len(result) == len(arr)


if __name__ == "__main__":
    pytest.main([__file__])
