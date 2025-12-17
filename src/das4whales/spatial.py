"""
spatial.py - Spatial calculations for DAS data analysis in DAS4Whales.


Authors: Léa Bouffaut
Date: 2025
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np

def to_rad(degree: float) -> float:
    """
    Convert angle from degrees to radians.

    Parameters
    ----------
    degree : float
        The angle value in degrees to be converted.

    Returns
    -------
    radian : float
        The corresponding angle value in radians.
    """
    return degree * np.pi / 180

def degree_to_km_at_latitude(latitude: float) -> float:
    """
    Convert degrees of longitude to kilometers at a specific latitude.

    Parameters
    ----------
    latitude : float
        The latitude in degrees where the conversion is calculated.

    Returns
    -------
    distance_km : float
        The distance in kilometers corresponding to one degree of longitude at the specified latitude.
    """
    return 111.32 * np.cos(np.radians(latitude))

def km_to_degree_at_latitude(km, latitude):
    """
    Convert distance in kilometers to degrees of longitude at a specific latitude.

    Parameters
    ----------
    km : float
        The distance in kilometers to convert.
    latitude : float
        The latitude in degrees where the conversion is calculated.

    Returns
    -------
    degrees : float
        The equivalent distance in degrees of longitude at the specified latitude.
    """
    return km / (111.32 * np.cos(np.radians(latitude)))

def calc_dist_lat_lon(position_1, position_2):
    """
    Calculate the distance between two sets of latitude and longitude positions using the Haversine formula.

    Parameters
    ----------
    position_1 : dict
        A dictionary containing latitude and longitude values for the first position.
        Keys:
        - 'lat': float, the latitude (in degrees).
        - 'lon': float, the longitude (in degrees).

    position_2 : dict
        A dictionary containing latitude and longitude values for the second position.
        It may contain either single values (float) or lists/numpy arrays for multiple positions.
        Keys:
        - 'lat': float, list, or numpy.ndarray, the latitude(s) (in degrees).
        - 'lon': float, list, or numpy.ndarray, the longitude(s) (in degrees).

    Returns
    -------
    distance : numpy.ndarray or float
        The distance(s) between the positions in meters:
        - If `position_2['lat']` and `position_2['lon']` are lists or numpy arrays,
          returns a numpy array of distances for each corresponding position.
        - Otherwise, returns a single float distance in meters for the two positions.
    """
    R = 6373.0  # Approximate radius of the Earth in kilometers

    # Check if position_2 contains lists or single values
    if isinstance(position_2['lat'], (list, np.ndarray)):
        arc = np.zeros(len(position_2['lat']))

        for dd in range(len(position_2['lat'])):
            dlon = to_rad(position_2['lon'][dd]) - to_rad(position_1['lon'])
            dlat = to_rad(position_2['lat'][dd]) - to_rad(position_1['lat'])
            a = (np.sin(dlat / 2)) ** 2 + np.cos(to_rad(position_1['lat'])) * np.cos(
                to_rad(position_2['lat'][dd])) * (
                    np.sin(dlon / 2)) ** 2
            arc[dd] = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * arc * 1000  # Convert to meters
    else:
        # If position_2 contains single values
        dlon = to_rad(position_2['lon']) - to_rad(position_1['lon'])
        dlat = to_rad(position_2['lat']) - to_rad(position_1['lat'])
        a = (np.sin(dlat / 2)) ** 2 + np.cos(to_rad(position_1['lat'])) * np.cos(to_rad(position_2['lat'])) * (
            np.sin(dlon / 2)) ** 2
        arc = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * arc * 1000  # Convert to meters

    return distance

def calc_das_section_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two geographic points on the Earth's surface.
    This is used to calculate the orientation (bearing) of a section of the DAS cable.

    Parameters
    ----------
    lat1 : float
        The latitude of the first point in degrees.
    lon1 : float
        The longitude of the first point in degrees.
    lat2 : float
        The latitude of the second point in degrees.
    lon2 : float
        The longitude of the second point in degrees.

    Returns
    -------
    das_bearing : float
        The bearing in degrees from the first point to the second point, relative to true north.
        The bearing is normalized to a range of 0 to 360 degrees.

    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate the difference in longitude
    delta_lon = lon2_rad - lon1_rad

    # Calculate the bearing
    x = np.sin(delta_lon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - (np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon))
    initial_bearing = np.arctan2(x, y)

    # Convert from radians to degrees
    initial_bearing_deg = np.degrees(initial_bearing)

    # Normalize the bearing to 0 - 360 degrees
    das_bearing = (initial_bearing_deg + 360) % 360

    return das_bearing

def calc_cumulative_dist(position):
    """
     Calculate the cumulative distance along a path defined by latitude and longitude coordinates
     using the Haversine formula.

     Parameters
     ----------
     position : pandas.DataFrame
         A DataFrame containing at least 'lat' and 'lon' columns, representing latitude and longitude
         coordinates respectively. Typically, this data is loaded from a CSV file with position data.

     Returns
     -------
     dist : numpy.ndarray
         An array of cumulative distances (in meters), where each element represents the total distance
         traveled up to that point along the path.
     """

    # Extract latitude, longitude, and depth
    lon = position['lat'].values

    # Get cumulative distance
    dist = np.zeros_like(lon)
    for ind in range(1, len(lon)):
        first_position = {key: value[ind - 1] for key, value in position.items()}
        second_position = {key: value[ind] for key, value in position.items()}
        local_dist = calc_dist_lat_lon(first_position, second_position)
        dist[ind] = dist[ind - 1] + local_dist

    return dist

def calc_source_position_lat_lon(lat_ref, lon_ref, distance_m, bearing, side):
    """
    Calculate the latitude and longitude of a point that lies perpendicular to a given bearing
    at a specified distance, relative to a reference point.

    Parameters
    ----------
    lat_ref : float
        Latitude of the reference point in degrees.
    lon_ref : float
        Longitude of the reference point in degrees.
    distance_m : float
        Distance to the perpendicular point in meters.
    bearing : float
        The bearing of the line in degrees (relative to true north).
    side : str
        Indicates the side of the bearing where the perpendicular point lies ('right' or 'left').

    Returns
    -------
    lat_lon : tuple
        A tuple containing:
        - lat (float): The latitude of the perpendicular point in degrees.
        - lon (float): The longitude of the perpendicular point in degrees.
    """
    R = 6371000
    # Convert the reference latitude, longitude, and bearing to radians
    lat1 = np.radians(lat_ref)
    lon1 = np.radians(lon_ref)

    # Adjust the bearing by 90 degrees depending on the side (right or left)
    if side == 'right':
        bearing_perp = (bearing + 90) % 360
    elif side == 'left':
        bearing_perp = (bearing - 90) % 360
    elif side == 'either':
        bearing_perp = (bearing + 90) % 360
    else:
        raise ValueError(f"Side must be either 'right' or 'left' - here {side}")

    # Convert the perpendicular bearing to radians
    theta = np.radians(bearing_perp)

    # Convert the distance to radians
    d_by_R = distance_m / R

    # Calculate the new latitude using the Haversine formula
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d_by_R) + np.cos(lat1) * np.sin(d_by_R) * np.cos(theta))

    # Calculate the new longitude
    lon2 = lon1 + np.arctan2(np.sin(theta) * np.sin(d_by_R) * np.cos(lat1),
                             np.cos(d_by_R) - np.sin(lat1) * np.sin(lat2))

    # Convert the results from radians back to degrees
    lat2_deg = np.degrees(lat2)
    lon2_deg = np.degrees(lon2)

    return lat2_deg, lon2_deg


