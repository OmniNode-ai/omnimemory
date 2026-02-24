# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for document ingestion domain models.

Tests cover:
- All new models are frozen (reject mutation)
- Round-trip serialization (model_dump -> model_validate)
- Enum values are stable and match design doc strings
- ModelDocumentChangedEvent includes previous-version fields
- ModelContextItemStats bootstrap fields default correctly
- ModelPromotionDecision tier_changed semantics

Ticket: OMN-2426
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType
from omnimemory.enums.crawl.enum_detected_doc_type import EnumDetectedDocType
from omnimemory.enums.enum_attribution_signal_type import EnumAttributionSignalType
from omnimemory.enums.enum_promotion_tier import EnumPromotionTier
from omnimemory.models.config.model_doc_source_config import ModelDocSourceConfig
from omnimemory.models.config.model_promotion_threshold_set import (
    DEFAULT_PROMOTION_THRESHOLDS,
    ModelPromotionThresholdSet,
)
from omnimemory.models.crawl.model_document_changed_event import (
    ModelDocumentChangedEvent,
)
from omnimemory.models.crawl.model_document_discovered_event import (
    ModelDocumentDiscoveredEvent,
)
from omnimemory.models.crawl.model_document_removed_event import (
    ModelDocumentRemovedEvent,
)
from omnimemory.models.events.model_crawl_tick_command import ModelCrawlTickCommand
from omnimemory.models.scoring.model_context_item_stats import ModelContextItemStats
from omnimemory.models.scoring.model_context_policy_config import (
    ModelContextPolicyConfig,
)
from omnimemory.models.scoring.model_promotion_decision import ModelPromotionDecision

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

_FINGERPRINT = "a" * 64
_PREV_FINGERPRINT = "b" * 64


# ---------------------------------------------------------------------------
# Enum stability
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnumValues:
    """Verify enum string values match the design doc."""

    def test_crawler_type_values(self) -> None:
        assert EnumCrawlerType.FILESYSTEM == "filesystem"
        assert EnumCrawlerType.GIT_REPO == "git_repo"
        assert EnumCrawlerType.LINEAR == "linear"
        assert EnumCrawlerType.WATCHDOG == "watchdog"

    def test_detected_doc_type_values(self) -> None:
        assert EnumDetectedDocType.CLAUDE_MD == "claude_md"
        assert EnumDetectedDocType.DESIGN_DOC == "design_doc"
        assert EnumDetectedDocType.ARCHITECTURE_DOC == "architecture_doc"
        assert EnumDetectedDocType.PLAN == "plan"
        assert EnumDetectedDocType.HANDOFF == "handoff"
        assert EnumDetectedDocType.README == "readme"
        assert EnumDetectedDocType.TICKET == "ticket"
        assert EnumDetectedDocType.LINEAR_DOCUMENT == "linear_document"
        assert EnumDetectedDocType.DEEP_DIVE == "deep_dive"
        assert EnumDetectedDocType.UNKNOWN_MD == "unknown_md"

    def test_attribution_signal_type_v0_signals_unchanged(self) -> None:
        """Hook-derived signals must retain their v0 string values."""
        assert EnumAttributionSignalType.FILE_TOUCHED_MATCH == "file_touched_match"
        assert EnumAttributionSignalType.RULE_ID_CITED == "rule_id_cited"
        assert (
            EnumAttributionSignalType.FAILURE_SIGNATURE_MATCH
            == "failure_signature_match"
        )
        assert EnumAttributionSignalType.FAILURE_RESOLVED == "failure_resolved"
        assert EnumAttributionSignalType.DIFF_HUNK_MATCH == "diff_hunk_match"
        assert (
            EnumAttributionSignalType.GATE_DELTA_IMPROVEMENT == "gate_delta_improvement"
        )
        assert (
            EnumAttributionSignalType.NEGATIVE_CONTRADICTION == "negative_contradiction"
        )
        assert EnumAttributionSignalType.DUPLICATE_MATCH == "duplicate_match"

    def test_attribution_signal_type_new_doc_signals(self) -> None:
        """New document-specific signals (OMN-2426)."""
        assert EnumAttributionSignalType.RULE_FOLLOWED == "rule_followed"
        assert EnumAttributionSignalType.STANDARD_CITED == "standard_cited"
        assert EnumAttributionSignalType.PATTERN_VIOLATED == "pattern_violated"
        assert EnumAttributionSignalType.DOC_SECTION_MATCHED == "doc_section_matched"

    def test_promotion_tier_values(self) -> None:
        assert EnumPromotionTier.QUARANTINE == "quarantine"
        assert EnumPromotionTier.VALIDATED == "validated"
        assert EnumPromotionTier.SHARED == "shared"
        assert EnumPromotionTier.BLACKLISTED == "blacklisted"

    def test_context_source_type_values(self) -> None:
        assert EnumContextSourceType.STATIC_STANDARDS == "static_standards"
        assert EnumContextSourceType.REPO_DERIVED == "repo_derived"
        assert EnumContextSourceType.MEMORY_HOOK == "memory_hook"
        assert EnumContextSourceType.LINEAR_TICKET == "linear_ticket"
        assert EnumContextSourceType.MEMORY_PATTERN == "memory_pattern"


# ---------------------------------------------------------------------------
# ModelCrawlTickCommand
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelCrawlTickCommand:
    def _make(self) -> ModelCrawlTickCommand:
        return ModelCrawlTickCommand(
            crawl_type=EnumCrawlerType.FILESYSTEM,
            crawl_scope="omninode/omnimemory",
            correlation_id=uuid4(),
            triggered_at_utc=datetime(2026, 2, 20, 0, 0, 0, tzinfo=timezone.utc),
            trigger_source="scheduled",
        )

    def test_construction_succeeds(self) -> None:
        cmd = self._make()
        assert cmd.crawl_type == EnumCrawlerType.FILESYSTEM
        assert cmd.event_type == "CrawlTickRequested"

    def test_is_frozen(self) -> None:
        cmd = self._make()
        with pytest.raises(ValidationError):
            cmd.crawl_scope = "modified"  # type: ignore[misc]

    def test_round_trip(self) -> None:
        cmd = self._make()
        data = cmd.model_dump()
        restored = ModelCrawlTickCommand.model_validate(data)
        assert restored == cmd

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelCrawlTickCommand.model_validate(
                {
                    "crawl_type": "filesystem",
                    "crawl_scope": "omninode/omnimemory",
                    "correlation_id": str(uuid4()),
                    "triggered_at_utc": "2026-02-20T00:00:00Z",
                    "trigger_source": "scheduled",
                    "unknown_field": "oops",
                }
            )


# ---------------------------------------------------------------------------
# ModelDocumentDiscoveredEvent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelDocumentDiscoveredEvent:
    def _make(self) -> ModelDocumentDiscoveredEvent:
        return ModelDocumentDiscoveredEvent(
            correlation_id=uuid4(),
            emitted_at_utc=datetime(2026, 2, 20, 0, 0, 0, tzinfo=timezone.utc),
            crawler_type=EnumCrawlerType.FILESYSTEM,
            crawl_scope="omninode/omnimemory",
            trigger_source="scheduled",
            source_ref="/Volumes/PRO-G40/Code/omnimemory2/CLAUDE.md",
            source_type=EnumContextSourceType.STATIC_STANDARDS,
            content_fingerprint=_FINGERPRINT,
            content_blob_ref="blob://abc123",
            token_estimate=512,
            scope_ref="omninode/omnimemory",
            detected_doc_type=EnumDetectedDocType.CLAUDE_MD,
            priority_hint=85,
        )

    def test_construction_succeeds(self) -> None:
        evt = self._make()
        assert evt.event_type == "DocumentDiscovered"
        assert evt.schema_version == "v1"

    def test_is_frozen(self) -> None:
        evt = self._make()
        with pytest.raises(ValidationError):
            evt.source_ref = "modified"  # type: ignore[misc]

    def test_round_trip(self) -> None:
        evt = self._make()
        data = evt.model_dump()
        restored = ModelDocumentDiscoveredEvent.model_validate(data)
        assert restored == evt

    def test_priority_hint_upper_bound_validated(self) -> None:
        evt = self._make()
        with pytest.raises(ValidationError):
            ModelDocumentDiscoveredEvent.model_validate(
                {**evt.model_dump(), "priority_hint": 101}
            )


# ---------------------------------------------------------------------------
# ModelDocumentChangedEvent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelDocumentChangedEvent:
    def _make(self) -> ModelDocumentChangedEvent:
        return ModelDocumentChangedEvent(
            correlation_id=uuid4(),
            emitted_at_utc=datetime(2026, 2, 20, 0, 0, 0, tzinfo=timezone.utc),
            crawler_type=EnumCrawlerType.GIT_REPO,
            crawl_scope="omninode/omniintelligence",
            trigger_source="git_hook",
            source_ref="/Volumes/PRO-G40/Code/omniintelligence/CLAUDE.md",
            source_type=EnumContextSourceType.STATIC_STANDARDS,
            source_version="abc123sha",
            content_fingerprint=_FINGERPRINT,
            content_blob_ref="blob://new",
            token_estimate=600,
            scope_ref="omninode/omniintelligence",
            detected_doc_type=EnumDetectedDocType.CLAUDE_MD,
            priority_hint=85,
            previous_content_fingerprint=_PREV_FINGERPRINT,
            previous_source_version="old123sha",
        )

    def test_construction_succeeds(self) -> None:
        evt = self._make()
        assert evt.event_type == "DocumentChanged"
        assert evt.previous_content_fingerprint == _PREV_FINGERPRINT

    def test_is_frozen(self) -> None:
        evt = self._make()
        with pytest.raises(ValidationError):
            evt.source_ref = "modified"  # type: ignore[misc]

    def test_round_trip(self) -> None:
        evt = self._make()
        assert ModelDocumentChangedEvent.model_validate(evt.model_dump()) == evt

    def test_previous_source_version_can_be_none(self) -> None:
        evt = self._make()
        modified = ModelDocumentChangedEvent.model_validate(
            {**evt.model_dump(), "previous_source_version": None}
        )
        assert modified.previous_source_version is None


# ---------------------------------------------------------------------------
# ModelDocumentRemovedEvent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelDocumentRemovedEvent:
    def _make(self) -> ModelDocumentRemovedEvent:
        return ModelDocumentRemovedEvent(
            correlation_id=uuid4(),
            emitted_at_utc=datetime(2026, 2, 20, 0, 0, 0, tzinfo=timezone.utc),
            crawler_type=EnumCrawlerType.LINEAR,
            crawl_scope="omninode/omnimemory",
            trigger_source="scheduled",
            source_ref="linear://OMN-1234",
            source_type=EnumContextSourceType.LINEAR_TICKET,
            scope_ref="omninode/shared",
            last_known_content_fingerprint=_FINGERPRINT,
        )

    def test_construction_succeeds(self) -> None:
        evt = self._make()
        assert evt.event_type == "DocumentRemoved"

    def test_is_frozen(self) -> None:
        evt = self._make()
        with pytest.raises(ValidationError):
            evt.source_ref = "modified"  # type: ignore[misc]

    def test_round_trip(self) -> None:
        evt = self._make()
        assert ModelDocumentRemovedEvent.model_validate(evt.model_dump()) == evt


# ---------------------------------------------------------------------------
# ModelDocSourceConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelDocSourceConfig:
    def test_defaults_match_design_doc(self) -> None:
        cfg = ModelDocSourceConfig()
        assert cfg.doc_token_budget_fraction_default == 0.30
        assert cfg.max_doc_items == 8
        assert cfg.doc_min_similarity == 0.65
        assert cfg.allow_bootstrap_validated is True
        assert cfg.allow_unscored_static_standards is True

    def test_intent_overrides_present(self) -> None:
        cfg = ModelDocSourceConfig()
        assert cfg.doc_token_budget_fraction_overrides["architecture"] == 0.40
        assert cfg.doc_token_budget_fraction_overrides["debugging"] == 0.20

    def test_is_frozen(self) -> None:
        cfg = ModelDocSourceConfig()
        with pytest.raises(ValidationError):
            cfg.max_doc_items = 99  # type: ignore[misc]

    def test_round_trip(self) -> None:
        cfg = ModelDocSourceConfig(doc_min_similarity=0.70, max_doc_items=5)
        assert ModelDocSourceConfig.model_validate(cfg.model_dump()) == cfg


# ---------------------------------------------------------------------------
# ModelPromotionThresholdSet
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelPromotionThresholdSet:
    def test_static_standards_has_no_quarantine_requirement(self) -> None:
        ts = DEFAULT_PROMOTION_THRESHOLDS[EnumContextSourceType.STATIC_STANDARDS]
        assert ts.quarantine_to_validated_runs is None

    def test_repo_derived_quarantine_threshold(self) -> None:
        ts = DEFAULT_PROMOTION_THRESHOLDS[EnumContextSourceType.REPO_DERIVED]
        assert ts.quarantine_to_validated_runs == 5

    def test_memory_hook_uses_v0_thresholds(self) -> None:
        ts = DEFAULT_PROMOTION_THRESHOLDS[EnumContextSourceType.MEMORY_HOOK]
        assert ts.quarantine_to_validated_runs == 10
        assert ts.validated_to_shared_runs == 30
        assert ts.validated_to_shared_used_rate == 0.25

    def test_is_frozen(self) -> None:
        ts = ModelPromotionThresholdSet(
            source_type=EnumContextSourceType.REPO_DERIVED,
            quarantine_to_validated_runs=5,
            validated_to_shared_runs=20,
            validated_to_shared_used_rate=0.15,
        )
        with pytest.raises(ValidationError):
            ts.validated_to_shared_runs = 99  # type: ignore[misc]

    def test_round_trip(self) -> None:
        ts = DEFAULT_PROMOTION_THRESHOLDS[EnumContextSourceType.REPO_DERIVED]
        assert ModelPromotionThresholdSet.model_validate(ts.model_dump()) == ts


# ---------------------------------------------------------------------------
# ModelContextPolicyConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelContextPolicyConfig:
    def test_doc_source_config_defaults_to_none(self) -> None:
        """Hook-only mode: doc_source_config must default to None."""
        cfg = ModelContextPolicyConfig()
        assert cfg.doc_source_config is None

    def test_doc_source_config_can_be_set(self) -> None:
        doc_cfg = ModelDocSourceConfig()
        cfg = ModelContextPolicyConfig(doc_source_config=doc_cfg)
        assert cfg.doc_source_config == doc_cfg

    def test_is_frozen(self) -> None:
        cfg = ModelContextPolicyConfig()
        with pytest.raises(ValidationError):
            cfg.max_total_items = 99  # type: ignore[misc]

    def test_round_trip_without_doc_config(self) -> None:
        cfg = ModelContextPolicyConfig()
        assert ModelContextPolicyConfig.model_validate(cfg.model_dump()) == cfg

    def test_round_trip_with_doc_config(self) -> None:
        cfg = ModelContextPolicyConfig(doc_source_config=ModelDocSourceConfig())
        assert ModelContextPolicyConfig.model_validate(cfg.model_dump()) == cfg


# ---------------------------------------------------------------------------
# ModelContextItemStats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelContextItemStats:
    def test_defaults_for_new_item(self) -> None:
        stats = ModelContextItemStats(context_item_id="item-001")
        assert stats.scored_runs == 0
        assert stats.used_runs == 0
        assert stats.citation_count == 0
        assert stats.bootstrap_confidence is None
        assert stats.bootstrap_runs_remaining is None
        assert stats.bootstrap_cleared is False
        assert stats.source_type is None
        assert stats.doc_version_hash is None

    def test_bootstrap_fields_set_correctly(self) -> None:
        stats = ModelContextItemStats(
            context_item_id="item-002",
            source_type=EnumContextSourceType.STATIC_STANDARDS,
            bootstrap_confidence=0.85,
            bootstrap_runs_remaining=5,
        )
        assert stats.bootstrap_confidence == 0.85
        assert stats.bootstrap_runs_remaining == 5
        assert stats.bootstrap_cleared is False

    def test_is_frozen(self) -> None:
        stats = ModelContextItemStats(context_item_id="item-003")
        with pytest.raises(ValidationError):
            stats.scored_runs = 99  # type: ignore[misc]

    def test_round_trip(self) -> None:
        stats = ModelContextItemStats(
            context_item_id="item-004",
            scored_runs=10,
            used_runs=7,
            positive_signals=3,
            citation_count=2,
            source_type=EnumContextSourceType.REPO_DERIVED,
            doc_version_hash="def456",
            bootstrap_cleared=True,
        )
        assert ModelContextItemStats.model_validate(stats.model_dump()) == stats


# ---------------------------------------------------------------------------
# ModelPromotionDecision
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelPromotionDecision:
    def test_tier_changed_true_on_transition(self) -> None:
        decision = ModelPromotionDecision(
            context_item_id="item-001",
            evaluated_at_utc=datetime(2026, 2, 20, 0, 0, 0, tzinfo=timezone.utc),
            tier_before=EnumPromotionTier.QUARANTINE,
            tier_after=EnumPromotionTier.VALIDATED,
            tier_changed=True,
            reason="Reached quarantine_to_validated_runs threshold.",
            source_type=EnumContextSourceType.REPO_DERIVED,
            threshold_set_used="repo_derived_v1",
        )
        assert decision.tier_changed is True
        assert decision.tier_after == EnumPromotionTier.VALIDATED

    def test_tier_changed_false_on_no_transition(self) -> None:
        decision = ModelPromotionDecision(
            context_item_id="item-002",
            evaluated_at_utc=datetime(2026, 2, 20, 0, 0, 0, tzinfo=timezone.utc),
            tier_before=EnumPromotionTier.VALIDATED,
            tier_after=EnumPromotionTier.VALIDATED,
            tier_changed=False,
            reason="Thresholds not yet met.",
        )
        assert decision.tier_changed is False

    def test_bootstrap_cleared_field(self) -> None:
        decision = ModelPromotionDecision(
            context_item_id="item-003",
            evaluated_at_utc=datetime(2026, 2, 20, 0, 0, 0, tzinfo=timezone.utc),
            tier_before=EnumPromotionTier.VALIDATED,
            tier_after=EnumPromotionTier.VALIDATED,
            tier_changed=False,
            reason="Bootstrap cleared; transitioned to earned VALIDATED.",
            bootstrap_cleared=True,
        )
        assert decision.bootstrap_cleared is True

    def test_is_frozen(self) -> None:
        decision = ModelPromotionDecision(
            context_item_id="item-004",
            evaluated_at_utc=datetime(2026, 2, 20, 0, 0, 0, tzinfo=timezone.utc),
            tier_before=EnumPromotionTier.QUARANTINE,
            tier_after=EnumPromotionTier.VALIDATED,
            tier_changed=True,
            reason="Test.",
        )
        with pytest.raises(ValidationError):
            decision.tier_after = EnumPromotionTier.SHARED  # type: ignore[misc]

    def test_round_trip(self) -> None:
        decision = ModelPromotionDecision(
            context_item_id="item-005",
            evaluated_at_utc=datetime(2026, 2, 20, 0, 0, 0, tzinfo=timezone.utc),
            tier_before=EnumPromotionTier.VALIDATED,
            tier_after=EnumPromotionTier.SHARED,
            tier_changed=True,
            reason="Promoted to SHARED.",
            source_type=EnumContextSourceType.STATIC_STANDARDS,
            threshold_set_used="static_standards_v1",
            bootstrap_cleared=False,
        )
        assert ModelPromotionDecision.model_validate(decision.model_dump()) == decision
