# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Enum for semantic/NLP entity types following ONEX standards.

This enum is domain-specific to OmniMemory's semantic analysis capabilities
and intentionally separate from omnibase_core's EnumEntityType which handles
code/infrastructure entities (NODE, FUNCTION, etc.).

Standard NER taxonomy aligned with spaCy, Stanza, and common NLP frameworks.
"""

from enum import Enum


class EnumSemanticEntityType(str, Enum):
    """Types of named entities extracted from natural language content.

    Aligned with standard NER taxonomies for interoperability with common
    NLP frameworks (spaCy, Stanza, Hugging Face, etc.).

    Example:
        >>> entity_type = EnumSemanticEntityType.PERSON
        >>> entity_type.value
        'person'
    """

    # Core entity types
    PERSON = "person"
    """A named person (e.g., 'Albert Einstein', 'Jane Doe')."""

    ORGANIZATION = "organization"
    """Companies, agencies, institutions (e.g., 'Google', 'United Nations')."""

    LOCATION = "location"
    """Physical locations, countries, cities (e.g., 'Paris', 'Mount Everest')."""

    # Temporal entities
    DATE = "date"
    """Absolute or relative dates (e.g., 'January 1st', 'next Tuesday')."""

    TIME = "time"
    """Times of day (e.g., '3:00 PM', 'midnight')."""

    # Numeric entities
    MONEY = "money"
    """Monetary values (e.g., '$100', '50 euros')."""

    PERCENT = "percent"
    """Percentage values (e.g., '25%', 'fifty percent')."""

    QUANTITY = "quantity"
    """Measurements and amounts (e.g., '100 kg', '5 miles')."""

    CARDINAL = "cardinal"
    """Numerals not covered by other types (e.g., 'three', '1000')."""

    ORDINAL = "ordinal"
    """Ordinal numbers (e.g., 'first', '3rd')."""

    # Named entities
    PRODUCT = "product"
    """Products, objects, vehicles (e.g., 'iPhone', 'Boeing 747')."""

    EVENT = "event"
    """Named events (e.g., 'World War II', 'Olympics 2024')."""

    WORK_OF_ART = "work_of_art"
    """Titles of books, songs, etc. (e.g., 'Hamlet', 'Mona Lisa')."""

    LAW = "law"
    """Named laws, regulations (e.g., 'GDPR', 'First Amendment')."""

    LANGUAGE = "language"
    """Named languages (e.g., 'English', 'Mandarin')."""

    # Group entities
    NORP = "norp"
    """Nationalities, religious, or political groups (e.g., 'American', 'Buddhist')."""

    # Geopolitical entities
    GPE = "gpe"
    """Geopolitical entities - countries, cities, states (e.g., 'France', 'California')."""

    FACILITY = "facility"
    """Buildings, airports, highways (e.g., 'JFK Airport', 'Golden Gate Bridge')."""

    # Catch-all
    MISC = "misc"
    """Miscellaneous entities not fitting other categories."""

    UNKNOWN = "unknown"
    """Entity type could not be determined."""

    @classmethod
    def is_temporal(cls, entity_type: "EnumSemanticEntityType") -> bool:
        """Check if entity type is temporal (date/time related)."""
        return entity_type in {cls.DATE, cls.TIME}

    @classmethod
    def is_numeric(cls, entity_type: "EnumSemanticEntityType") -> bool:
        """Check if entity type is numeric."""
        return entity_type in {
            cls.MONEY,
            cls.PERCENT,
            cls.QUANTITY,
            cls.CARDINAL,
            cls.ORDINAL,
        }

    @classmethod
    def is_named(cls, entity_type: "EnumSemanticEntityType") -> bool:
        """Check if entity type is a proper named entity."""
        return entity_type in {
            cls.PERSON,
            cls.ORGANIZATION,
            cls.LOCATION,
            cls.GPE,
            cls.FACILITY,
            cls.PRODUCT,
            cls.EVENT,
            cls.WORK_OF_ART,
            cls.LAW,
        }
