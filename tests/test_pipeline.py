"""Integration tests for the feature pipeline."""

import numpy as np
import pytest
from features.cicflowmeter import (
    CIC_FEATURE_NAMES,
    extract_features_from_openflow,
    flow_record_from_openflow,
)
from features.entropy import ENTROPY_FEATURE_NAMES


class TestOpenFlowBridge:
    @pytest.fixture
    def sample_stat(self):
        return {
            "packet_count": 1000,
            "byte_count": 64000,
            "duration_sec": 1,
            "duration_nsec": 0,
            "match": {
                "ipv4_src": "10.0.1.100",
                "ipv4_dst": "10.0.0.1",
                "tp_dst": 53,
                "tp_src": 12345,
                "ip_proto": 17,
            },
        }

    def test_returns_all_cic_features(self, sample_stat):
        feats = extract_features_from_openflow(sample_stat)
        assert len(feats) == len(CIC_FEATURE_NAMES)
        for name in CIC_FEATURE_NAMES:
            assert name in feats

    def test_correct_bytes_per_second(self, sample_stat):
        feats = extract_features_from_openflow(sample_stat)
        assert feats["Flow_Bytes_s"] == pytest.approx(64000.0, rel=1e-3)

    def test_correct_destination_port(self, sample_stat):
        feats = extract_features_from_openflow(sample_stat)
        assert feats["Destination_Port"] == 53.0

    def test_zero_duration_handled(self):
        stat = {"packet_count": 100, "byte_count": 6400,
                 "duration_sec": 0, "duration_nsec": 0, "match": {}}
        feats = extract_features_from_openflow(stat)
        assert feats["Flow_Bytes_s"] == 0.0  # Guard against div/0

    def test_flow_record_extraction(self, sample_stat):
        record = flow_record_from_openflow(sample_stat)
        assert record["src_ip"] == "10.0.1.100"
        assert record["dst_ip"] == "10.0.0.1"
        assert record["dst_port"] == 53
        assert record["protocol"] == 17
        assert record["pkt_len_mean"] > 0


class TestFeatureVector:
    def test_full_vector_88_dim(self):
        """Combining CIC (80) + entropy (8) should give 88-dim vector."""
        assert len(CIC_FEATURE_NAMES) == 80
        assert len(ENTROPY_FEATURE_NAMES) == 8
        total = len(CIC_FEATURE_NAMES) + len(ENTROPY_FEATURE_NAMES)
        assert total == 88

    def test_no_duplicate_feature_names(self):
        all_features = CIC_FEATURE_NAMES + ENTROPY_FEATURE_NAMES
        assert len(all_features) == len(set(all_features)), "Duplicate feature names found"
