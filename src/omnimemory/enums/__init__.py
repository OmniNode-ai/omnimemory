# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
ONEX-compliant enums for omnimemory system.

All enums are centralized here for better maintainability and ONEX compliance.
"""

from .crawl import EnumContextSourceType, EnumCrawlerType, EnumDetectedDocType
from .enum_attribution_signal_type import EnumAttributionSignalType
from .enum_context_item_type import EnumContextItemType
from .enum_entity_extraction_mode import EnumEntityExtractionMode
from .enum_error_code import EnumOmniMemoryErrorCode
from .enum_intelligence_operation_type import EnumIntelligenceOperationType
from .enum_lifecycle_state import EnumLifecycleState
from .enum_memory_operation_type import EnumMemoryOperationType
from .enum_memory_storage_type import EnumMemoryStorageType
from .enum_migration_status import (
    EnumFileProcessingStatus,
    EnumMigrationPriority,
    EnumMigrationStatus,
)
from .enum_priority_level import EnumPriorityLevel
from .enum_promotion_tier import EnumPromotionTier
from .enum_semantic_entity_type import EnumSemanticEntityType
from .enum_subscription_status import EnumSubscriptionStatus
from .enum_trust_level import EnumDecayFunction, EnumTrustLevel

__all__ = [
    "EnumAttributionSignalType",
    "EnumContextItemType",
    "EnumContextSourceType",
    "EnumCrawlerType",
    "EnumDecayFunction",
    "EnumDetectedDocType",
    "EnumEntityExtractionMode",
    "EnumFileProcessingStatus",
    "EnumIntelligenceOperationType",
    "EnumLifecycleState",
    "EnumMemoryOperationType",
    "EnumMemoryStorageType",
    "EnumMigrationPriority",
    "EnumMigrationStatus",
    "EnumOmniMemoryErrorCode",
    "EnumPriorityLevel",
    "EnumPromotionTier",
    "EnumSemanticEntityType",
    "EnumSubscriptionStatus",
    "EnumTrustLevel",
]
