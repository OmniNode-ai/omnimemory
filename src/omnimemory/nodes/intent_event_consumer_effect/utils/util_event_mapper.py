"""Event mapper utility for intent event consumer.

Maps incoming intent-classified events to storage requests.
"""

from omnibase_core.enums.intelligence import EnumIntentCategory
from omnibase_core.models.intelligence import ModelIntentClassificationOutput

from omnimemory.models.events import ModelIntentClassifiedEvent
from omnimemory.nodes.intent_storage_effect.models import ModelIntentStorageRequest


def map_event_to_storage_request(
    event: ModelIntentClassifiedEvent,
) -> ModelIntentStorageRequest:
    """Map incoming classified event to storage request.

    Args:
        event: The incoming intent-classified event from omniintelligence.

    Returns:
        A storage request ready for HandlerIntentStorageAdapter.
    """
    # Convert event intent_category string to EnumIntentCategory
    try:
        intent_category = EnumIntentCategory(event.intent_category)
    except ValueError:
        intent_category = EnumIntentCategory.UNKNOWN

    return ModelIntentStorageRequest(
        operation="store",
        session_id=event.session_id,
        intent_data=ModelIntentClassificationOutput(
            success=True,
            intent_category=intent_category,
            confidence=event.confidence,
            keywords=event.keywords,
        ),
        correlation_id=event.correlation_id,
    )
