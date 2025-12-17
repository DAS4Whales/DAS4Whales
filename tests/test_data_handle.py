import numpy as np
import pytest
import tempfile
import os
from unittest.mock import patch, mock_open, MagicMock
from das4whales.data_handle import (
    hello_world_das_package, 
    get_acquisition_parameters, 
    get_metadata_optasense, 
    raw2strain, 
    load_das_data, 
    dl_file,
    calc_dist_to_xidx,
    get_selected_channels,
    extract_timestamp,
    generate_file_list
)

def test_hello_world_das_package(capfd):
    # Test case 1: Check if the function returns the expected output
    hello_world_das_package()
    # Capture the standard output
    out, _ = capfd.readouterr()
    # Assert the printed message
    assert out.strip() == "Yepee! You now have access to all the functionalities of the das4whale python package!"


def test_get_acquisition_parameters():
    # Test case 1: Check if the function returns the expected output
    filepath = "/path/to/file"
    interrogator = "test"
    with pytest.raises(ValueError) as e:
        get_acquisition_parameters(filepath, interrogator)
    assert str(e.value) == "Interrogator name incorrect"
    
    # Test case 2: Test with valid interrogator but invalid file
    with pytest.raises(FileNotFoundError):
        get_acquisition_parameters("/nonexistent/file", "optasense")


def test_get_metadata_optasense():
    with pytest.raises(FileNotFoundError) as e:
        filepath = "/path/to/nonexistent/file"
        get_metadata_optasense(filepath)
    assert str(e.value) == f'File {filepath} not found'

def test_raw2strain():
    # Test case 1: Check if the function returns the expected output
    trace = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=float)
    metadata = {"scale_factor": 1000}
    result = raw2strain(trace, metadata)
    assert result.shape == trace.shape
    
    # Test case 2: Test that mean is removed and scaling is applied
    trace_copy = trace.copy()
    result = raw2strain(trace_copy, metadata)
    # Check that mean along axis 1 is approximately zero
    assert np.allclose(np.mean(result, axis=1), 0, atol=1e-10)
    
    # Test case 3: Test with different scale factor
    metadata_different = {"scale_factor": 0.5}
    result_different = raw2strain(trace.copy(), metadata_different)
    assert result_different.shape == trace.shape

def test_load_das_data():
    # Test case 1: Check if the function returns the expected output
    filename = "/path/to/file"
    selected_channels = [1, 2, 3]
    metadata = {"sample_rate": 1000}
    with pytest.raises(FileNotFoundError) as e:
        load_das_data(filename, selected_channels, metadata)
    assert str(e.value) == f'File {filename} not found'


def test_calc_dist_to_xidx():
    """Test the calc_dist_to_xidx function."""
    # Test case 1: Basic functionality
    x = 100.0
    selected_channels_m = [0.0, 500.0, 1000.0]
    selected_channels = [0, 50, 100] 
    dx = 10.0
    result = calc_dist_to_xidx(x, selected_channels_m, selected_channels, dx)
    expected = int((x - selected_channels_m[0]) / (dx * selected_channels[2]))
    assert result == expected
    
    # Test case 2: Different values
    x = 250.0
    result = calc_dist_to_xidx(x, selected_channels_m, selected_channels, dx)
    expected = int((250.0 - 0.0) / (10.0 * 100))
    assert result == expected


def test_get_selected_channels():
    """Test the get_selected_channels function."""
    # Test case 1: Basic functionality
    selected_channels_m = [0.0, 500.0, 10.0]
    dx = 10.0
    result = get_selected_channels(selected_channels_m, dx)
    expected = [0, 50, 1]  # Each divided by dx and converted to int
    assert result == expected
    
    # Test case 2: Different values
    selected_channels_m = [100.0, 1000.0, 50.0]
    dx = 25.0
    result = get_selected_channels(selected_channels_m, dx)
    expected = [4, 40, 2]
    assert result == expected


def test_extract_timestamp():
    """Test the extract_timestamp function."""
    # Test case 1: Valid filename with timestamp
    filename = "South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-04T020014Z.h5"
    result = extract_timestamp(filename)
    assert result is not None
    assert result.year == 2021
    assert result.month == 11
    assert result.day == 4
    assert result.hour == 2
    assert result.minute == 0
    assert result.second == 14
    
    # Test case 2: Invalid filename without timestamp
    filename_invalid = "invalid_filename.h5"
    result = extract_timestamp(filename_invalid)
    assert result is None
    
    # Test case 3: Another valid timestamp format
    filename2 = "North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5"
    result2 = extract_timestamp(filename2)
    assert result2 is not None
    assert result2.minute == 0
    assert result2.second == 2


def test_generate_file_list():
    """Test the generate_file_list function."""
    # Test case 1: Basic functionality
    base_url = "https://example.com/data/"
    start_file = "test_2021-11-04T020000Z.h5"
    duration = 120  # 2 minutes
    
    result = generate_file_list(base_url, start_file, duration)
    
    # Should return at least 3 files (0, 60, 120 seconds)
    assert len(result) >= 3
    assert result[0] == base_url + start_file
    
    # Test case 2: Invalid filename should raise ValueError
    with pytest.raises(ValueError):
        generate_file_list(base_url, "invalid_filename.h5", 60)

if __name__ == "__main__":
    pytest.main()
