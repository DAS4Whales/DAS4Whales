import numpy as np
import pytest
from das4whales.data_handle import hello_world_das_package, get_acquisition_parameters, get_metadata_optasense, raw2strain, load_das_data, dl_file

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

def test_load_das_data():
    # Test case 1: Check if the function returns the expected output
    filename = "/path/to/file"
    selected_channels = [1, 2, 3]
    metadata = {"sample_rate": 1000}
    with pytest.raises(FileNotFoundError) as e:
        load_das_data(filename, selected_channels, metadata)
    assert str(e.value) == f'File {filename} not found'

if __name__ == "__main__":
    pytest.main()
