"""Unit tests for DatasetLoader utilities."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

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

cv2_mock = sys.modules["cv2"]
cv2_mock.getPerspectiveTransform = MagicMock(return_value=np.eye(3, dtype=np.float32))

sys.path.insert(0, str(Path(__file__).parent.parent / "BLOCKCHAIN"))

import main  # noqa: E402


class TestSmartSort:
    def test_sorts_by_number(self):
        files = ["frame_3.jpg", "frame_1.jpg", "frame_2.jpg"]
        assert main.DatasetLoader.smart_sort_files(files) == ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"]

    def test_zero_padded_names(self):
        files = ["0003.jpg", "0001.jpg", "0002.jpg"]
        assert main.DatasetLoader.smart_sort_files(files) == ["0001.jpg", "0002.jpg", "0003.jpg"]

    def test_empty_list(self):
        assert main.DatasetLoader.smart_sort_files([]) == []

    def test_single_file(self):
        assert main.DatasetLoader.smart_sort_files(["only.jpg"]) == ["only.jpg"]

    def test_files_without_numbers_stable(self):
        assert len(main.DatasetLoader.smart_sort_files(["alpha.jpg", "beta.jpg"])) == 2

    def test_mixed_prefix_format(self):
        files = ["img_0010.png", "img_0002.png", "img_0100.png"]
        assert main.DatasetLoader.smart_sort_files(files) == ["img_0002.png", "img_0010.png", "img_0100.png"]


class TestDatasetLoaderInit:
    def test_init_without_csv(self):
        assert main.DatasetLoader(scenes_csv_path=None).scenes_df is None

    def test_list_available_scenes_no_csv(self):
        assert main.DatasetLoader().list_available_scenes() == []

    def test_get_scene_info_no_csv(self):
        info = main.DatasetLoader().get_scene_info("seq001")
        assert info["name"] == "seq001"
        assert info["type"] == "Unknown"
