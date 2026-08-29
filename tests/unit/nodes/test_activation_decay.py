# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""OMN-16765 step 4 — activation decay.

Guards on the decay function itself. The *comparative* measurement — whether
the ``created_at``-only path costs ranking quality against one that also reads
``last_accessed_at`` — needs timestamps in the fixture corpus and lands
separately; these tests establish the function is correct before anything
measures with it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_decay import (
    DECAY_LAMBDA_PER_DAY,
    STALE_ACTIVATION_THRESHOLD,
    activation_decay,
    freshest_timestamp,
    is_stale,
)

NOW = datetime(2026, 8, 29, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# freshest_timestamp
# =============================================================================


@pytest.mark.unit
def test_falls_back_to_created_at_when_no_access_time() -> None:
    """The snapshot path: ModelMemorySnapshot carries created_at only."""
    created = NOW - timedelta(days=10)
    assert freshest_timestamp(created, None) == created


@pytest.mark.unit
def test_prefers_the_later_of_the_two() -> None:
    created = NOW - timedelta(days=30)
    accessed = NOW - timedelta(days=2)
    assert freshest_timestamp(created, accessed) == accessed


@pytest.mark.unit
def test_ignores_an_access_time_older_than_creation() -> None:
    """Takes the max rather than preferring last_accessed_at when non-null.

    A backfill that stamped access times independently, or clock skew, can
    produce last_accessed_at < created_at. Preferring the field whenever it is
    present would let that silently age every affected record.
    """
    created = NOW - timedelta(days=2)
    accessed = NOW - timedelta(days=30)
    assert freshest_timestamp(created, accessed) == created


# =============================================================================
# activation_decay
# =============================================================================


@pytest.mark.unit
def test_freshly_touched_record_is_undecayed() -> None:
    assert activation_decay(NOW, now=NOW) == pytest.approx(1.0)


@pytest.mark.unit
def test_matches_the_documented_curve() -> None:
    """Pins the actual curve: 0.900 at one week, 0.657 at four weeks.

    Asserted as literal expected values rather than by recomputing the formula,
    so a change to the constant fails here instead of silently agreeing with
    itself.

    Note the four-week figure. ``handler_agent_learning_retrieval.py:21`` and
    the architecture plan's §2.3 both describe this decay as "~60% at 4 weeks"
    / "a 4-week-old learning scores 0.6x". That is wrong by about six points —
    ``exp(-0.015 * 28) = 0.657``, and 0.600 is not reached until day 34. Writing
    this test against the *documentation* rather than the formula is what
    surfaced it. The constant is fine; only its description is off.
    """
    assert activation_decay(NOW - timedelta(weeks=1), now=NOW) == pytest.approx(
        0.9003, abs=0.0005
    )
    assert activation_decay(NOW - timedelta(weeks=4), now=NOW) == pytest.approx(
        0.6570, abs=0.0005
    )
    # And the value the docs attribute to four weeks is really a ~34-day figure.
    assert activation_decay(NOW - timedelta(days=34), now=NOW) == pytest.approx(
        0.600, abs=0.001
    )


@pytest.mark.unit
def test_decays_monotonically_with_age() -> None:
    scores = [
        activation_decay(NOW - timedelta(days=d), now=NOW) for d in (0, 1, 7, 30, 365)
    ]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 for s in scores)


@pytest.mark.unit
def test_access_keeps_an_old_record_fresh() -> None:
    """The whole point of the enhancement, stated as a property.

    An old document that is still being read must outscore an equally old one
    that is not.
    """
    created = NOW - timedelta(days=60)
    still_read = activation_decay(created, NOW - timedelta(days=1), now=NOW)
    untouched = activation_decay(created, None, now=NOW)
    assert still_read > untouched


@pytest.mark.unit
def test_future_timestamps_do_not_amplify() -> None:
    """Clock skew must not become a ranking advantage."""
    assert activation_decay(NOW + timedelta(days=5), now=NOW) == pytest.approx(1.0)


@pytest.mark.unit
def test_uses_the_same_constant_as_the_agent_learning_path() -> None:
    """Two decay implementations that disagree would rank the same document
    differently depending on which path served it."""
    from omnimemory.nodes.node_agent_learning_retrieval_effect.handlers.handler_agent_learning_retrieval import (
        compute_freshness_score,
    )

    created = NOW - timedelta(days=17)
    assert activation_decay(created, None, now=NOW) == pytest.approx(
        compute_freshness_score(created, NOW)
    )
    assert DECAY_LAMBDA_PER_DAY == 0.015


# =============================================================================
# is_stale
# =============================================================================


@pytest.mark.unit
def test_stale_threshold_boundary() -> None:
    assert is_stale(STALE_ACTIVATION_THRESHOLD - 0.001)
    assert not is_stale(STALE_ACTIVATION_THRESHOLD)
    assert not is_stale(1.0)


@pytest.mark.unit
def test_a_record_becomes_stale_at_the_expected_age() -> None:
    """0.3 corresponds to ~80 days untouched. Recorded as a test so the
    threshold's practical meaning is visible rather than implied."""
    assert not is_stale(activation_decay(NOW - timedelta(days=75), now=NOW))
    assert is_stale(activation_decay(NOW - timedelta(days=85), now=NOW))
