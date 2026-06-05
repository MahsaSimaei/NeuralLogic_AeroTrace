"""Unit tests for BSVEvidenceRegistry."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "BLOCKCHAIN"))

from bsv_blockchain import (  # noqa: E402
    AnalysisEvidence,
    BlockchainTransaction,
    BSVEvidenceRegistry,
    ImageEvidence,
)


@pytest.fixture()
def registry(tmp_path):
    return BSVEvidenceRegistry(scene_id="test_scene", output_dir=str(tmp_path))


class TestHashing:
    def test_hash_bytes(self, registry):
        h = registry._hash(b"hello")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_hash_str(self, registry):
        h = registry._hash("hello")
        assert len(h) == 64

    def test_hash_dict(self, registry):
        h = registry._hash({"a": 1})
        assert len(h) == 64

    def test_hash_ndarray(self, registry):
        arr = np.zeros((4, 4), dtype=np.uint8)
        h = registry._hash(arr)
        assert len(h) == 64

    def test_hash_deterministic(self, registry):
        assert registry._hash("test") == registry._hash("test")

    def test_hash_different_inputs(self, registry):
        assert registry._hash("a") != registry._hash("b")


class TestImageEvidence:
    def test_register_image_returns_evidence(self, registry):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ev = registry.register_image(frame)
        assert isinstance(ev, ImageEvidence)
        assert ev.scene_id == "test_scene"
        assert len(ev.image_hash) == 64

    def test_register_image_appended(self, registry):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        registry.register_image(frame)
        registry.register_image(frame)
        assert len(registry.image_evidences) == 2

    def test_different_frames_different_hashes(self, registry):
        f1 = np.zeros((10, 10, 3), dtype=np.uint8)
        f2 = np.ones((10, 10, 3), dtype=np.uint8)
        ev1 = registry.register_image(f1)
        ev2 = registry.register_image(f2)
        assert ev1.image_hash != ev2.image_hash


class TestAnalysisEvidence:
    def test_register_analysis_returns_evidence(self, registry):
        ev = registry.register_analysis(
            metrics={"density": 12.5},
            total_vehicles=10,
            incident_count=1,
            processing_time_sec=2.5,
        )
        assert isinstance(ev, AnalysisEvidence)
        assert ev.total_vehicles == 10
        assert ev.incident_count == 1

    def test_analysis_hash_is_deterministic(self, registry):
        metrics = {"flow": 100, "density": 20}
        ev1 = registry.register_analysis(metrics, 5, 0)
        ev2 = registry.register_analysis(metrics, 5, 0)
        assert ev1.metrics_hash == ev2.metrics_hash


class TestBlockchainChaining:
    def _make_evidences(self, registry):
        img_ev = registry.register_image(np.zeros((10, 10, 3), dtype=np.uint8))
        ana_ev = registry.register_analysis({"x": 1}, 3, 0)
        return img_ev, ana_ev

    def test_create_transaction_returns_tx(self, registry):
        img_ev, ana_ev = self._make_evidences(registry)
        tx = registry.create_blockchain_transaction(img_ev, ana_ev)
        assert isinstance(tx, BlockchainTransaction)

    def test_first_tx_has_no_previous(self, registry):
        img_ev, ana_ev = self._make_evidences(registry)
        tx = registry.create_blockchain_transaction(img_ev, ana_ev)
        assert tx.previous_transaction_id is None

    def test_second_tx_references_first(self, registry):
        img_ev, ana_ev = self._make_evidences(registry)
        tx1 = registry.create_blockchain_transaction(img_ev, ana_ev)
        tx2 = registry.create_blockchain_transaction(img_ev, ana_ev)
        assert tx2.previous_transaction_id == tx1.transaction_id

    def test_etl_chain_grows(self, registry):
        img_ev, ana_ev = self._make_evidences(registry)
        registry.create_blockchain_transaction(img_ev, ana_ev)
        registry.create_blockchain_transaction(img_ev, ana_ev)
        assert len(registry.etl_chain) == 2


class TestExportEvidenceLog:
    def test_export_creates_file(self, registry, tmp_path):
        img_ev = registry.register_image(np.zeros((10, 10, 3), dtype=np.uint8))
        ana_ev = registry.register_analysis({"d": 1}, 2, 0)
        registry.create_blockchain_transaction(img_ev, ana_ev)
        path = registry.export_evidence_log()
        assert Path(path).exists()

    def test_export_valid_json(self, registry, tmp_path):
        img_ev = registry.register_image(np.zeros((10, 10, 3), dtype=np.uint8))
        ana_ev = registry.register_analysis({}, 0, 0)
        registry.create_blockchain_transaction(img_ev, ana_ev)
        path = registry.export_evidence_log()
        with open(path) as f:
            data = json.load(f)
        assert data["scene_id"] == "test_scene"

    def test_export_statistics_field(self, registry, tmp_path):
        img_ev = registry.register_image(np.zeros((10, 10, 3), dtype=np.uint8))
        ana_ev = registry.register_analysis({}, 0, 0)
        registry.create_blockchain_transaction(img_ev, ana_ev)
        path = registry.export_evidence_log()
        with open(path) as f:
            data = json.load(f)
        assert "statistics" in data
        assert data["statistics"]["total_transactions"] == 1
