import re
import datetime
from urllib.parse import urljoin

def extract_timestamp(filename):
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{6})Z', filename) # Specific to the format of the filenames for OOI-2021
    if match:
        return datetime.datetime.strptime(match.group(1), '%Y-%m-%dT%H%M%S').replace(tzinfo=datetime.timezone.utc)
    return None

def generate_sliding_windows(base_url, start_file, window_start, duration, window_size, overlap):
    # Extract the actual timestamp from the start file
    file_time = extract_timestamp(start_file)
    if file_time is None:
        raise ValueError(f"Could not extract timestamp from filename: {start_file}")
    
    step = window_size - overlap  # Compute step size
    selected_windows = []
    
    # Generate existing file names that will be used for the sliding windows
    all_files = []
    current_time = file_time
    end_time = file_time + datetime.timedelta(seconds=duration)
    
    while current_time < end_time:
        filename = re.sub(r'(\d{4}-\d{2}-\d{2}T\d{6})Z', current_time.strftime('%Y-%m-%dT%H%M%S') + 'Z', start_file)
        all_files.append({
            'time': current_time,
            'url': urljoin(base_url, filename)
        })
        current_time += datetime.timedelta(seconds=60)  # Move to next file (60s increment)
    
    # Now create the sliding windows that overlap the actual files
    window_start_time = window_start
    end_window_start = file_time + datetime.timedelta(seconds=duration - window_size)
    
    while window_start_time <= end_window_start:
        window_end_time = window_start_time + datetime.timedelta(seconds=window_size)
        window_files = []
        
        # Get the file that falls within the current window
        for file_info in all_files:
            # Files that start within the window
            if window_start_time <= file_info['time'] < window_end_time:
                window_files.append(file_info['url'])
            # Files that end within the window
            elif file_info['time'] + datetime.timedelta(seconds=60) > window_start_time and file_info['time'] + datetime.timedelta(seconds=60) <= window_end_time:
                window_files.append(file_info['url'])
        
        selected_windows.append({
            'timestamp': window_start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'files': window_files
        })
        
        window_start_time += datetime.timedelta(seconds=step)
    
    return selected_windows

def combine_north_south_windows(windows_north, windows_south, 
                               channel_range_north, channel_range_south):
    combined_results = []
    
    # Use the minimum of start timestamps for window alignment
    min_windows = min(len(windows_north), len(windows_south))
    
    for i in range(min_windows):
        north_window = windows_north[i]
        south_window = windows_south[i]
        
        # Use the window timestamp
        window_timestamp = north_window['timestamp']
        
        # Start the line with timestamp
        combined_line = [window_timestamp]
        
        # Add all north files first
        north_files = north_window['files']
        combined_line.extend(north_files)
        
        # Add north channel range parameters
        combined_line.extend([
            str(channel_range_north[0]),
            str(channel_range_north[1]),
            str(channel_range_north[2])
        ])
        
        # Add all south files
        south_files = south_window['files']
        combined_line.extend(south_files)
        
        # Add south channel range parameters
        combined_line.extend([
            str(channel_range_south[0]),
            str(channel_range_south[1]),
            str(channel_range_south[2])
        ])
        
        combined_results.append(combined_line)
    
    return combined_results

# Parameters
base_url_south = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/SouthCable/TransmitFiber/South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-01T16_09_15-0700/'
start_file_south = 'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-04T015914Z.h5'
base_url_north = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'
start_file_north = 'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5'
duration = 600  # 10 minutes
channel_range_north = [12000, 66000, 5]
channel_range_south = [12000, 95000, 5]
window_size = 60  # 60-second windows (changed from 180)
overlap = 30    # 30-second overlap (changed from 60)

# Extract actual start times from filenames
north_start_time = extract_timestamp(start_file_north)
south_start_time = extract_timestamp(start_file_south)

if north_start_time is None or south_start_time is None:
    raise ValueError("Could not extract timestamps from filenames")

# Generate sliding windows for each cable separately using their actual start times
windows_north = generate_sliding_windows(
    base_url_north, start_file_north, north_start_time, duration, 
    window_size, overlap
)

windows_south = generate_sliding_windows(
    base_url_south, start_file_south, north_start_time, duration, 
    window_size, overlap
)

# Combine the north and south windows
combined_windows = combine_north_south_windows(
    windows_north, windows_south, 
    channel_range_north, channel_range_south,
)

# Print each combined window
for window in combined_windows:
    print(" ".join(window))