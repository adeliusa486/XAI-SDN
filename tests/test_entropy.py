"""Unit tests for features/entropy.py"""

import math
import pytest
import numpy as np
from features.entropy import (
    shannon_entropy,
    EntropyFeatureExtractor,
    compute_entropy_features_offline,
    ENTROPY_FEATURE_NAMES,
    DEFAULT_WINDOW_SIZE,
)


class TestShannonEntropy:
    def test_uniform_distribution_max_entropy(self):
        """Equal probabilities yield maximum entropy = log2(n)."""
        n = 8
        values = list(range(n))
        h = shannon_entropy(values)
        assert math.isclose(h, math.log2(n), abs_tol=1e-9)

    def test_single_value_zero_entropy(self):
        """All same values → H = 0."""
        assert shannon_entropy(["a", "a", "a"]) == 0.0

    def test_two_equal_values_one_bit(self):
        """Two equally likely values → H = 1 bit."""
        h = shannon_entropy(["x", "y"])
        assert math.isclose(h, 1.0, abs_tol=1e-9)

    def test_empty_sequence_returns_zero(self):
        assert shannon_entropy([]) == 0.0

    def test_single_element_zero_entropy(self):
        assert shannon_entropy([42]) == 0.0

    def test_non_negative(self):
        for _ in range(20):
            vals = list(np.random.randint(0, 5, size=100))
            assert shannon_entropy(vals) >= 0.0

    def test_mixed_types(self):
        h = shannon_entropy(["10.0.0.1", "10.0.0.2", "10.0.0.1"])
        assert h > 0.0


class TestEntropyFeatureExtractor:
    @pytest.fixture
    def sample_records(self):
        return [
            {
                "src_ip": f"10.0.0.{i % 5}",
                "dst_ip": "10.0.0.1",
                "dst_port": 53,
                "protocol": 17,
                "pkt_len_mean": 64.0,
                "iat_mean": 1000.0,
                "tcp_flags": 0,
                "ttl": 64,
            }
            for i in range(20)
        ]

    def test_init_defaults(self):
        e = EntropyFeatureExtractor()
        assert e.window_size == DEFAULT_WINDOW_SIZE
        assert len(e) == 0

    def test_window_size_enforced(self):
        e = EntropyFeatureExtractor(window_size=5)
        records = [
            {
                "src_ip": f"10.0.0.{i}",
                "dst_ip": "10.0.0.1",
                "dst_port": 80,
                "protocol": 6,
                "pkt_len_mean": 100.0,
                "iat_mean": 500.0,
                "tcp_flags": 2,
                "ttl": 64,
            }
            for i in range(10)
        ]
        for r in records:
            e.update_and_compute(r)
        assert len(e) == 5

    def test_returns_all_entropy_features(self, sample_records):
        e = EntropyFeatureExtractor(window_size=10)
        for r in sample_records[:5]:
            feats = e.update_and_compute(r)
        assert set(feats.keys()) == set(ENTROPY_FEATURE_NAMES)

    def test_single_dst_ip_zero_entropy(self):
        e = EntropyFeatureExtractor(window_size=10)
        for i in range(10):
            feats = e.update_and_compute(
                {
                    "src_ip": f"10.0.0.{i}",
                    "dst_ip": "192.168.1.1",  # constant
                    "dst_port": 53,
                    "protocol": 17,
                    "pkt_len_mean": 64.0,
                    "iat_mean": 1000.0,
                    "tcp_flags": 0,
                    "ttl": 64,
                }
            )
        assert feats["H_dst_ip"] == 0.0

    def test_compute_as_array_shape(self, sample_records):
        e = EntropyFeatureExtractor(window_size=10)
        for r in sample_records:
            e.update_and_compute(r)
        arr = e.compute_as_array()
        assert arr.shape == (8,)
        assert arr.dtype == np.float64

    def test_reset_clears_window(self, sample_records):
        e = EntropyFeatureExtractor(window_size=10)
        for r in sample_records[:5]:
            e.update_and_compute(r)
        assert len(e) == 5
        e.reset()
        assert len(e) == 0

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError):
            EntropyFeatureExtractor(window_size=0)

    def test_empty_window_returns_zeros(self):
        e = EntropyFeatureExtractor(window_size=10)
        feats = e.compute_from_window([])
        assert all(v == 0.0 for v in feats.values())

    def test_ddos_udp_has_low_entropy(self):
        """UDP flood: all flows from same source → low H_src_ip."""
        e = EntropyFeatureExtractor(window_size=50)
        for _ in range(50):
            feats = e.update_and_compute(
                {
                    "src_ip": "10.0.1.100",  # single attacker
                    "dst_ip": "10.0.0.1",
                    "dst_port": 53,
                    "protocol": 17,
                    "pkt_len_mean": 64.0,
                    "iat_mean": 100.0,
                    "tcp_flags": 0,
                    "ttl": 64,
                }
            )
        assert feats["H_src_ip"] == 0.0
        assert feats["H_dst_ip"] == 0.0
        assert feats["H_dst_port"] == 0.0

    def test_benign_has_higher_entropy_than_ddos(self):
        e_benign = EntropyFeatureExtractor(window_size=50)
        e_ddos = EntropyFeatureExtractor(window_size=50)

        for i in range(50):
            e_benign.update_and_compute(
                {
                    "src_ip": f"10.0.0.{i % 20}",
                    "dst_ip": f"10.0.1.{i % 10}",
                    "dst_port": [80, 443, 22, 53, 8080][i % 5],
                    "protocol": [6, 17][i % 2],
                    "pkt_len_mean": 200.0 + i,
                    "iat_mean": 1000.0,
                    "tcp_flags": i % 8,
                    "ttl": 64,
                }
            )
            e_ddos.update_and_compute(
                {
                    "src_ip": "10.0.1.100",
                    "dst_ip": "10.0.0.1",
                    "dst_port": 53,
                    "protocol": 17,
                    "pkt_len_mean": 64.0,
                    "iat_mean": 100.0,
                    "tcp_flags": 0,
                    "ttl": 64,
                }
            )

        benign_feats = e_benign.compute_from_window(e_benign.window)
        ddos_feats = e_ddos.compute_from_window(e_ddos.window)

        assert benign_feats["H_src_ip"] > ddos_feats["H_src_ip"]
        assert benign_feats["H_dst_port"] > ddos_feats["H_dst_port"]


class TestOfflineBatchComputation:
    def test_output_shape(self):
        records = [
            {
                "src_ip": f"10.0.0.{i%5}",
                "dst_ip": "10.0.0.1",
                "dst_port": 80,
                "protocol": 6,
                "pkt_len_mean": 100.0,
                "iat_mean": 500.0,
                "tcp_flags": 2,
                "ttl": 64,
            }
            for i in range(100)
        ]
        result = compute_entropy_features_offline(records, window_size=10)
        assert result.shape == (100, 8)

    def test_non_negative_values(self):
        records = [
            {
                "src_ip": "1.1.1.1",
                "dst_ip": "2.2.2.2",
                "dst_port": 443,
                "protocol": 6,
                "pkt_len_mean": 1500.0,
                "iat_mean": 2000.0,
                "tcp_flags": 16,
                "ttl": 128,
            }
            for _ in range(30)
        ]
        result = compute_entropy_features_offline(records, window_size=5)
        assert (result >= 0).all()

    def test_window_warmup_period(self):
        """First few rows should have lower entropy (smaller window fill)."""
        records = [
            {
                "src_ip": f"10.0.0.{i}",
                "dst_ip": "10.0.0.1",
                "dst_port": 80,
                "protocol": 6,
                "pkt_len_mean": 100.0,
                "iat_mean": 500.0,
                "tcp_flags": 0,
                "ttl": 64,
            }
            for i in range(100)
        ]
        result = compute_entropy_features_offline(records, window_size=50)
        # Row 0 has only 1 record → H=0; row 49+ has 50 diverse records → H>0
        assert result[0, 0] == 0.0  # H_src_ip for first flow = 0 (1 unique value)
        assert result[99, 0] > 0.0  # H_src_ip for last flow should be positive
