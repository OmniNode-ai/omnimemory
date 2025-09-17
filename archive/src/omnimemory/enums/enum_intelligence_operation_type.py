"""
Enum for intelligence operation types following ONEX standards.
"""

from enum import Enum


class EnumIntelligenceOperationType(str, Enum):
    """Types of intelligence operations in the ONEX memory system."""

    SEMANTIC_ANALYSIS = "semantic_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    CONTENT_CLASSIFICATION = "content_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENTITY_EXTRACTION = "entity_extraction"
    TOPIC_MODELING = "topic_modeling"
    SIMILARITY_ANALYSIS = "similarity_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"
    RECOMMENDATION_GENERATION = "recommendation_generation"
