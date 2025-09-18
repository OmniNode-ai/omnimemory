"""
Typed Collections for ONEX Foundation Architecture

This module provides utility functions for converting between legacy types
and strongly typed Pydantic models. All actual models have been split into
separate files following the one-model-per-file ONEX standard.

All models follow ONEX standards with:
- Strong typing with zero Any types
- Comprehensive Field descriptions
- Validation and serialization support
- Monadic composition patterns
"""

from typing import Dict, List, Union

# Import the split models
from .model_configuration import ModelConfiguration
from .model_event_collection import ModelEventCollection
from .model_metadata import ModelMetadata
from .model_result_collection import ModelResultCollection
from .model_string_list import ModelStringList
from .model_structured_data import ModelStructuredData

# === UTILITY FUNCTIONS ===


def convert_dict_to_metadata(
    data: Dict[str, Union[str, int, float, bool]]
) -> ModelMetadata:
    """Convert a dictionary to ModelMetadata."""
    return ModelMetadata.from_dict(data)


def convert_list_to_string_list(data: List[str]) -> ModelStringList:
    """Convert a list of strings to ModelStringList."""
    return ModelStringList(values=data)


def convert_list_of_dicts_to_structured_data(
    data: List[Dict[str, Union[str, int, float, bool]]]
) -> ModelResultCollection:
    """Convert a list of dictionaries to structured result collection."""
    collection = ModelResultCollection()

    for i, item in enumerate(data):
        # Convert dict to structured data
        structured_data = ModelStructuredData()
        for key, value in item.items():
            structured_data.set_field_value(key, str(value))

        collection.add_result(
            id=str(i),
            status="success",
            message=f"Converted item {i}",
            data=structured_data,
        )

    return collection
