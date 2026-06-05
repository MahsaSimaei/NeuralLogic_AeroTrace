"""Unit tests for TrafficEngineer metrics calculations."""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_STUBS = [
    "cv2", "ultralytics", "supervision", "tqdm",
    "plotly", "plotly.graph_objects", "plotly.subplots",
    "matplotlib", "matplotlib.pyplot",
    "streamlit",
]
for mod in _STUBS:
    sys.modules.setdefault(mod, MagicMock())

sv_mock = sys.modules["supervision"]
sv_mock.Point = MagicMock(return_value=MagicMock())
sv_mock.ByteTrack = MagicMock()
sv_mock.Detections = MagicMock()

cv2_mock = sys.modules["cv2"]
cv2_mock.getPerspectiveTransform = MagicMock(return_value=np.eye(3, dtype=np.float32))

sys.path.insert(0, str(Path(__file__).parent.parent / "BLOCKCHAIN"))

import main  # noqa: E402


def make_engineer(width: int = 1920, height: int = 1080) -> main.TrafficEngineer:
    return main.TrafficEngineer(image_width_px=width, image_height_px=height)


class TestGSDCalibration:
    def test_gsd_is_positive(self):
        assert make_engineer().gsd > 0

    def test_gsd_formula(self):
        h = main.CONFIG.ALTURA_VUELO_M
        fov_rad = math.radians(main.CONFIG.FOV_HORIZONTAL_GRADOS)
        expected_gsd = (2 * h * math.tan(fov_rad / 2)) / 1920
        assert math.isclose(make_engineer().gsd, expected_gsd, rel_tol=1e-6)

    def test_pixel_to_meters_zero(self):
        assert make_engineer().pixel_to_meters(0) == 0.0

    def test_pixel_to_meters_scaling(self):
        eng = make_engineer()
        assert math.isclose(eng.pixel_to_meters(100), 100 * eng.gsd)

    def test_area_conversion(self):
        eng = make_engineer()
        assert math.isclose(eng.area_px_to_m2(100), 100 * (eng.gsd ** 2))


class TestDensityCalculation:
    def test_zero_area_returns_zero(self):
        assert make_engineer().calculate_density(10, 0) == 0.0

    def test_density_unit(self):
        assert math.isclose(make_engineer().calculate_density(10, 10_000), 1_000.0)

    def test_zero_vehicles(self):
        assert make_engineer().calculate_density(0, 5_000) == 0.0

    def test_density_proportional_to_count(self):
        eng = make_engineer()
        d1 = eng.calculate_density(5, 10_000)
        d2 = eng.calculate_density(10, 10_000)
        assert math.isclose(d2, 2 * d1)


class TestLevelOfService:
    @pytest.mark.parametrize(
        "density,expected_los",
        [
            (0, "A"), (13, "A"),
            (14, "B"), (21, "B"),
            (22, "C"), (31, "C"),
            (32, "D"), (44, "D"),
            (45, "E"), (66, "E"),
            (67, "F"), (200, "F"),
        ],
    )
    def test_los_thresholds(self, density, expected_los):
        los, _ = make_engineer().calculate_level_of_service(density)
        assert los == expected_los

    def test_los_returns_description(self):
        _, description = make_engineer().calculate_level_of_service(0)
        assert isinstance(description, str) and len(description) > 0

    def test_extreme_density_is_f(self):
        los, _ = make_engineer().calculate_level_of_service(1_000_000)
        assert los == "F"


class TestOccupancyRate:
    def test_zero_total_area(self):
        assert make_engineer().calculate_occupancy_rate(100, 0) == 0.0

    def test_full_occupancy(self):
        assert math.isclose(make_engineer().calculate_occupancy_rate(50, 50), 100.0)

    def test_half_occupancy(self):
        assert math.isclose(make_engineer().calculate_occupancy_rate(25, 50), 50.0)

    def test_occupancy_bounded(self):
        rate = make_engineer().calculate_occupancy_rate(10, 100)
        assert 0 <= rate <= 100
