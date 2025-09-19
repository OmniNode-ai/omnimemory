"""
Calculation method enumeration for confidence scoring.
"""

from enum import Enum


class EnumCalculationMethod(str, Enum):
    """Methods used for calculating confidence scores in success metrics."""

    STATISTICAL = "statistical"
    HEURISTIC = "heuristic"
    ML_BASED = "ml_based"
    BAYESIAN = "bayesian"
    FREQUENTIST = "frequentist"
    MONTE_CARLO = "monte_carlo"
    BOOTSTRAP = "bootstrap"
    CROSS_ENTROPY = "cross_entropy"
    WEIGHTED_AVERAGE = "weighted_average"
    ENSEMBLE_METHOD = "ensemble_method"
    FUZZY_LOGIC = "fuzzy_logic"
    NEURAL_NETWORK = "neural_network"
    REGRESSION_BASED = "regression_based"
    RULE_BASED = "rule_based"
    HYBRID_APPROACH = "hybrid_approach"
