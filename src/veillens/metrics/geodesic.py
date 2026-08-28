#!/usr/bin/python
#
# Distance Calculation Utilities for Geolocation Evaluation
#
# This module provides accurate distance calculations between geographic coordinates
# and related utilities for geolocation error analysis.

from __future__ import absolute_import, division, print_function

import math
from typing import Dict, List, Tuple


class DistanceCalculator:
    """
    Utility class for calculating distances between geographic coordinates
    and performing geolocation error analysis
    """

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    def __init__(self):
        pass

    def haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate the great-circle distance between two points on Earth using the Haversine formula
        
        Args:
            coord1: (latitude, longitude) of first point in decimal degrees
            coord2: (latitude, longitude) of second point in decimal degrees
        
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)

        c = 2 * math.asin(math.sqrt(a))

        # Distance in kilometers
        distance = self.EARTH_RADIUS_KM * c

        return distance

    def calculate_error_statistics(self, errors: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive error statistics
        
        Args:
            errors: List of error distances in kilometers
        
        Returns:
            Dictionary with error statistics
        """
        if not errors:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'min': 0.0,
                'max': 0.0,
                'q25': 0.0,
                'q75': 0.0,
                'rmse': 0.0
            }

        errors_sorted = sorted(errors)
        n = len(errors)

        # Basic statistics
        mean_error = sum(errors) / n
        median_error = errors_sorted[n // 2] if n % 2 == 1 else (errors_sorted[n // 2 - 1] + errors_sorted[n // 2]) / 2
        min_error = min(errors)
        max_error = max(errors)

        # Standard deviation
        variance = sum((x - mean_error) ** 2 for x in errors) / n
        std_dev = math.sqrt(variance)

        # Quartiles
        q25_idx = n // 4
        q75_idx = 3 * n // 4
        q25 = errors_sorted[q25_idx]
        q75 = errors_sorted[q75_idx]

        # Root Mean Square Error
        rmse = math.sqrt(sum(x ** 2 for x in errors) / n)

        return {
            'count': n,
            'mean': mean_error,
            'median': median_error,
            'std_dev': std_dev,
            'min': min_error,
            'max': max_error,
            'q25': q25,
            'q75': q75,
            'rmse': rmse
        }


