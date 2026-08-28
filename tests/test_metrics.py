"""The geolocation error metric.

Every geolocation number in the paper is a great-circle distance between a ground-truth
coordinate and an attacker's guess, and the iteration gate stops an image once that distance
crosses `experiment.val_dist`. So the metric is load-bearing twice over, and it is worth pinning
against distances that can be checked against an atlas rather than against the implementation.
"""

import pytest

from veillens.metrics.geodesic import DistanceCalculator


@pytest.fixture(scope="module")
def calc():
    return DistanceCalculator()


# (name, coord_a, coord_b, published great-circle distance in km)
KNOWN_PAIRS = [
    ("new york -> los angeles", (40.7128, -74.0060), (34.0522, -118.2437), 3936),
    ("london -> paris", (51.5074, -0.1278), (48.8566, 2.3522), 344),
    ("sydney -> tokyo", (-33.8688, 151.2093), (35.6762, 139.6503), 7823),
]


@pytest.mark.parametrize("name,a,b,expected_km", KNOWN_PAIRS, ids=[p[0] for p in KNOWN_PAIRS])
def test_haversine_matches_published_distance(calc, name, a, b, expected_km):
    # 1% covers the difference between a spherical earth and the published ellipsoidal figures.
    assert calc.haversine_distance(a, b) == pytest.approx(expected_km, rel=0.01)


def test_identical_coordinates_are_zero(calc):
    berlin = (52.5200, 13.4050)
    assert calc.haversine_distance(berlin, berlin) == pytest.approx(0.0, abs=1e-9)


def test_distance_is_symmetric(calc):
    a, b = (40.7128, -74.0060), (34.0522, -118.2437)
    assert calc.haversine_distance(a, b) == pytest.approx(calc.haversine_distance(b, a))


def test_antimeridian_is_not_measured_the_long_way_round(calc):
    """A guess just across the date line is close, not half a planet away."""
    west, east = (0.0, 179.9), (0.0, -179.9)
    assert calc.haversine_distance(west, east) < 30


def test_error_statistics_summarise_a_run(calc):
    stats = calc.calculate_error_statistics([1.0, 2.0, 3.0, 4.0])
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(4.0)


def test_error_statistics_tolerate_an_empty_run(calc):
    """Every image can fail to yield a comparable prediction; that must not raise."""
    stats = calc.calculate_error_statistics([])
    assert isinstance(stats, dict)
