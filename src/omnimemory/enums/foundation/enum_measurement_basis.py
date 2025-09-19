"""
Measurement basis enumeration for confidence scoring.
"""

from enum import Enum


class EnumMeasurementBasis(str, Enum):
    """Basis types for confidence measurements in success metrics."""

    DATA_QUALITY = "data_quality"
    ALGORITHM_CERTAINTY = "algorithm_certainty"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    SAMPLE_SIZE_ADEQUACY = "sample_size_adequacy"
    HISTORICAL_PERFORMANCE = "historical_performance"
    CROSS_VALIDATION = "cross_validation"
    EXPERT_ASSESSMENT = "expert_assessment"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    ERROR_RATE_ANALYSIS = "error_rate_analysis"
    CONSISTENCY_METRICS = "consistency_metrics"
    RELIABILITY_INDICATORS = "reliability_indicators"
    VALIDATION_SCORES = "validation_scores"
    PEER_REVIEW = "peer_review"
    AUTOMATED_TESTING = "automated_testing"
    REAL_WORLD_VALIDATION = "real_world_validation"
