"""
XAI-SDN Feature Engineering Package.

Modules:
    cicflowmeter: CICFlowMeter-compatible statistical feature extraction.
    entropy:      Shannon entropy feature computation over sliding windows.
    pipeline:     Full offline (batch) and online (streaming) feature pipelines.
"""
from features.cicflowmeter import CICFlowMeterExtractor
from features.entropy import EntropyFeatureExtractor, shannon_entropy
from features.pipeline import FeaturePipeline

__all__ = [
    "CICFlowMeterExtractor",
    "EntropyFeatureExtractor",
    "shannon_entropy",
    "FeaturePipeline",
]
