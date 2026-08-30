# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Behavioral tests for :class:`PIIDetector` (OMN-17236).

``omnimemory.adapters.adapter_pii_detector.PIIDetector`` sits on a live
persistence path with two callers:

* ``node_intent_storage_effect/models/model_intent_storage_request.py`` --
  a ``@model_validator`` that **raises** when ``detect_pii`` reports PII in
  ``user_context``. A false positive here is a hard request refusal.
* ``node_intent_storage_effect/adapters/adapter_intent_storage.py`` --
  the redact-before-persist arm.

Until this module landed it had **zero behavioral tests**.

Case source
-----------
Many positive/negative cases here are ported from the (dead) core engine's
suite at ``omnibase_core/tests/unit/models/workflow/
test_model_workflow_state_snapshot_pii.py``. Nothing in ``omnibase_core`` is
modified or removed by this lane -- it is used purely as a case source.

Two conventions worth stating once:

* ``sensitivity_level`` names are **inverted** relative to intuition:
  ``"low"`` selects the *strictest* confidence floor (0.95) and therefore
  detects the *fewest* types, ``"high"`` selects the most permissive floor
  (0.60). ``is_content_safe()`` always uses ``"high"``.
* ``ModelPIIMatch.masked_value`` is the pattern's ``mask_template``
  verbatim -- it is not derived per match, so a mask never preserves any
  part of the matched value.
"""

from __future__ import annotations

import re
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnimemory.adapters.adapter_pii_detector import (
    ModelPIIDetectorConfig,
    PIIDetector,
    PIIType,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def detector() -> PIIDetector:
    """A detector on stock configuration."""
    return PIIDetector()


def _types(detector: PIIDetector, content: str, level: str = "medium") -> set[PIIType]:
    return detector.detect_pii(content, sensitivity_level=level).pii_types_detected


# ---------------------------------------------------------------------------
# EMAIL
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmailDetection:
    """EMAIL true positives and the mask that replaces them."""

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("john.doe@example.com", id="simple"),
            pytest.param("john+tag@example.com", id="plus-sign"),
            pytest.param("user@mail.company.co.uk", id="subdomain"),
            pytest.param("UPPER.CASE@Example.COM", id="mixed-case"),
            pytest.param("a_b-c%d@sub.domain.io", id="punctuated-local-part"),
        ],
    )
    def test_detects_email(self, detector: PIIDetector, content: str) -> None:
        assert PIIType.EMAIL in _types(detector, content)

    def test_email_is_replaced_by_mask_template(self, detector: PIIDetector) -> None:
        result = detector.detect_pii("john.doe@example.com")
        assert result.sanitized_content == "***@***.***"
        assert result.matches[0].masked_value == "***@***.***"

    def test_detects_multiple_emails_in_one_string(self, detector: PIIDetector) -> None:
        result = detector.detect_pii("Contact john@test.com or jane@test.com")
        assert len(result.matches) == 2
        assert result.sanitized_content == "Contact ***@***.*** or ***@***.***"

    def test_match_offsets_bound_the_email(self, detector: PIIDetector) -> None:
        content = "reach me at john@test.com please"
        match = detector.detect_pii(content).matches[0]
        assert content[match.start_index : match.end_index] == "john@test.com"

    def test_preserves_non_email_strings(self, detector: PIIDetector) -> None:
        content = "John Doe is active"
        result = detector.detect_pii(content)
        assert not result.has_pii
        assert result.sanitized_content == content


# ---------------------------------------------------------------------------
# PHONE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhoneDetection:
    """PHONE true positives across the formats the core engine redacted."""

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("555-123-4567", id="dashes"),
            pytest.param("555.123.4567", id="dots"),
            pytest.param("555 123 4567", id="spaces"),
            pytest.param("(555) 123-4567", id="parentheses"),
            pytest.param("+1-555-123-4567", id="country-code"),
            pytest.param("+1 555 123 4567", id="country-code-spaces"),
            pytest.param("5551234567", id="plain-ten-digit"),
        ],
    )
    def test_detects_phone(self, detector: PIIDetector, content: str) -> None:
        assert PIIType.PHONE in _types(detector, content)

    def test_phone_is_replaced_by_mask_template(self, detector: PIIDetector) -> None:
        assert detector.detect_pii("555-123-4567").sanitized_content == "***-***-****"

    def test_detects_phone_embedded_in_prose(self, detector: PIIDetector) -> None:
        result = detector.detect_pii("Call me at 555-123-4567 tomorrow.")
        assert result.sanitized_content == "Call me at ***-***-**** tomorrow."


@pytest.mark.unit
class TestPhoneBoundaryGuard:
    """OMN-17236 D1: PHONE must not match a digit run inside a longer token.

    Before the fix the PHONE pattern carried no lookaround, so any ten
    consecutive digits matched -- including digits *inside* a credit card
    number, a UUID, or a numeric order id. Because PHONE is registered
    before CREDIT_CARD in the pattern table and both carry the same 0.90
    confidence, the overlap de-duplicator kept PHONE and dropped
    CREDIT_CARD, so a Visa number sanitized to ``***-***-****111111`` --
    the trailing six digits of the card survived into stored content.
    """

    @pytest.mark.parametrize(
        ("content", "leaked_tail"),
        [
            pytest.param("4111111111111111", "111111", id="visa"),
            pytest.param("5500000000000004", "000004", id="mastercard"),
            pytest.param("378282246310005", "10005", id="amex"),
        ],
    )
    def test_card_digits_do_not_survive_as_phone_remainder(
        self, detector: PIIDetector, content: str, leaked_tail: str
    ) -> None:
        result = detector.detect_pii(content)
        assert PIIType.PHONE not in result.pii_types_detected
        assert PIIType.CREDIT_CARD in result.pii_types_detected
        assert leaked_tail not in result.sanitized_content
        assert result.sanitized_content == "****-****-****-****"

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("550e8400-e29b-41d4-a716-446655440000", id="uuid-lower"),
            pytest.param("550E8400-E29B-41D4-A716-446655440000", id="uuid-upper"),
            pytest.param("order 12345678901234", id="fourteen-digit-order-id"),
            pytest.param("1234567890123456789012", id="twenty-two-digit-run"),
            pytest.param("build 20260830123456", id="numeric-build-stamp"),
        ],
    )
    def test_long_digit_runs_are_not_phones(
        self, detector: PIIDetector, content: str
    ) -> None:
        result = detector.detect_pii(content)
        assert PIIType.PHONE not in result.pii_types_detected
        assert result.sanitized_content == content

    def test_uuid_survives_detection_unchanged(self, detector: PIIDetector) -> None:
        """A UUID is an identifier, not PII -- it must round-trip intact."""
        content = f"session {uuid4()} started"
        result = detector.detect_pii(content)
        assert not result.has_pii
        assert result.sanitized_content == content


# ---------------------------------------------------------------------------
# SSN
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSSNDetection:
    """SSN true positives, including the space-separated form."""

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("123-45-6789", id="dashes"),
            pytest.param("123 45 6789", id="spaces"),
            pytest.param("123456789", id="plain-nine-digit"),
        ],
    )
    def test_detects_ssn(self, detector: PIIDetector, content: str) -> None:
        assert PIIType.SSN in _types(detector, content)

    def test_dashed_ssn_uses_dashed_mask(self, detector: PIIDetector) -> None:
        assert detector.detect_pii("123-45-6789").sanitized_content == "***-**-****"

    def test_undashed_ssn_uses_flat_mask(self, detector: PIIDetector) -> None:
        assert detector.detect_pii("123456789").sanitized_content == "*********"

    def test_detects_ssn_in_prose(self, detector: PIIDetector) -> None:
        result = detector.detect_pii("SSN is 123-45-6789 on file")
        assert result.sanitized_content == "SSN is ***-**-**** on file"


@pytest.mark.unit
class TestSSNValidityExclusions:
    """OMN-17236 D2: the documented SSN exclusions must apply to both forms.

    ``_build_ssn_validation_pattern`` excludes area 000/666/900-999, group
    00 and serial 0000 -- but before the fix those exclusions were wired
    only into the *undashed* nine-digit pattern. The dashed pattern was a
    bare ``\\b\\d{3}-\\d{2}-\\d{4}\\b``, so structurally-invalid values
    (including the ``000-00-0000`` placeholder) were reported as SSNs and
    the request-model validator refused the request.
    """

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("000-45-6789", id="area-000"),
            pytest.param("666-45-6789", id="area-666"),
            pytest.param("900-45-6789", id="area-900"),
            pytest.param("999-45-6789", id="area-999"),
            pytest.param("123-00-6789", id="group-00"),
            pytest.param("123-45-0000", id="serial-0000"),
            pytest.param("000-00-0000", id="all-zero-placeholder"),
        ],
    )
    def test_structurally_invalid_ssn_is_not_detected(
        self, detector: PIIDetector, content: str
    ) -> None:
        result = detector.detect_pii(content)
        assert PIIType.SSN not in result.pii_types_detected
        assert result.sanitized_content == content

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("000456789", id="area-000-undashed"),
            pytest.param("666456789", id="area-666-undashed"),
            pytest.param("900456789", id="area-900-undashed"),
            pytest.param("123006789", id="group-00-undashed"),
            pytest.param("123450000", id="serial-0000-undashed"),
        ],
    )
    def test_structurally_invalid_undashed_ssn_is_not_detected(
        self, detector: PIIDetector, content: str
    ) -> None:
        assert PIIType.SSN not in _types(detector, content)

    def test_valid_ssn_still_detected_after_exclusions(
        self, detector: PIIDetector
    ) -> None:
        assert PIIType.SSN in _types(detector, "123-45-6789")


# ---------------------------------------------------------------------------
# CREDIT_CARD
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreditCardDetection:
    """CREDIT_CARD true positives for every issuer the pattern claims."""

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("4111111111111111", id="visa"),
            pytest.param("4012888888881881", id="visa-alt"),
            pytest.param("5500000000000004", id="mastercard"),
            pytest.param("5105105105105100", id="mastercard-alt"),
            pytest.param("378282246310005", id="amex-37"),
            pytest.param("341111111111111", id="amex-34"),
        ],
    )
    def test_detects_card(self, detector: PIIDetector, content: str) -> None:
        assert PIIType.CREDIT_CARD in _types(detector, content)

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("6011111111111117", id="discover-6011"),
            pytest.param("6511111111111119", id="discover-65"),
            pytest.param("6441111111111112", id="discover-644"),
            pytest.param("6491111111111116", id="discover-649"),
        ],
    )
    def test_detects_discover_card(self, detector: PIIDetector, content: str) -> None:
        """OMN-17236 D3: Discover was absent from the CREDIT_CARD pattern.

        Before the D1 boundary guard, a Discover number was (incorrectly)
        caught by PHONE, which masked most of it by accident. Adding the
        guard would have left Discover numbers entirely unredacted on the
        persistence path, so Discover is added to the pattern here.
        """
        assert PIIType.CREDIT_CARD in _types(detector, content)

    def test_card_is_replaced_by_mask_template(self, detector: PIIDetector) -> None:
        result = detector.detect_pii("card 4111111111111111 on file")
        assert result.sanitized_content == "card ****-****-****-**** on file"

    def test_separated_card_digits_are_not_matched(self, detector: PIIDetector) -> None:
        """Known limitation: the pattern requires unseparated digits.

        ``4111-1111-1111-1111`` matches no CREDIT_CARD pattern. Pinned so a
        future change to the pattern is a deliberate, visible decision.
        """
        assert PIIType.CREDIT_CARD not in _types(detector, "4111-1111-1111-1111")


# ---------------------------------------------------------------------------
# IP_ADDRESS
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIPAddressDetection:
    """IPv4 and full-form IPv6."""

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("192.168.1.1", id="ipv4-private"),  # onex-allow-internal-ip
            pytest.param("10.0.0.1", id="ipv4-rfc1918"),  # onex-allow-internal-ip
            pytest.param("8.8.8.8", id="ipv4-public"),
        ],
    )
    def test_detects_ipv4(self, detector: PIIDetector, content: str) -> None:
        assert PIIType.IP_ADDRESS in _types(detector, content)

    def test_ipv4_is_replaced_by_mask_template(self, detector: PIIDetector) -> None:
        result = detector.detect_pii(
            "Request from 10.0.0.1 failed"  # onex-allow-internal-ip
        )
        assert result.sanitized_content == "Request from ***.***.***.*** failed"

    def test_detects_full_form_ipv6(self, detector: PIIDetector) -> None:
        content = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
        result = detector.detect_pii(content)
        assert PIIType.IP_ADDRESS in result.pii_types_detected
        assert result.sanitized_content == "****:****:****:****"

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("::1", id="ipv6-loopback"),
            pytest.param("fe80::1", id="ipv6-link-local"),
            pytest.param("2001:db8::8a2e:370:7334", id="ipv6-compressed"),
        ],
    )
    def test_compressed_ipv6_is_not_detected(
        self, detector: PIIDetector, content: str
    ) -> None:
        """Known limitation, documented in the pattern comment.

        The IPv6 pattern is full-form only. Compressed forms pass through
        unredacted. Pinned so the gap stays visible.
        """
        assert PIIType.IP_ADDRESS not in _types(detector, content)

    def test_four_segment_version_string_is_a_known_false_positive(
        self, detector: PIIDetector
    ) -> None:
        """Residual, NOT fixed here: dotted-quad shape wins over semantics.

        ``1.2.3.4`` is a valid four-part version string and a valid IPv4
        address; the pattern cannot tell them apart, so a four-segment
        version in ``user_context`` is redacted (and, on the request-model
        path, refused). Octet-range validation would not help. Pinned as
        current behavior rather than silently tolerated.
        """
        assert PIIType.IP_ADDRESS in _types(detector, "1.2.3.4")

    def test_three_segment_semver_is_not_matched(self, detector: PIIDetector) -> None:
        assert PIIType.IP_ADDRESS not in _types(detector, "version 1.2.3")


# ---------------------------------------------------------------------------
# API_KEY / PASSWORD_HASH
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAPIKeyDetection:
    """Credential shapes and the fixed redaction strings they collapse to."""

    @pytest.mark.parametrize(
        ("content", "expected_mask"),
        [
            pytest.param(
                'api_key: "abcdef1234567890abcdef"',
                "api_key=***REDACTED***",
                id="generic-api-key",
            ),
            pytest.param(
                'token = "abcdefghijklmnopqrstuvwxyz"',
                "token=***REDACTED***",
                id="generic-token",
            ),
            pytest.param("sk-" + "a" * 40, "sk-***REDACTED***", id="openai-secret-key"),
            pytest.param("ghp_" + "b" * 36, "ghp_***REDACTED***", id="github-pat"),
            pytest.param("AIza" + "c" * 35, "AIza***REDACTED***", id="google-api-key"),
            pytest.param("AWSABCDEFGHIJKLMNOP", "AWS***REDACTED***", id="aws-key-id"),
        ],
    )
    def test_detects_and_masks_api_key(
        self, detector: PIIDetector, content: str, expected_mask: str
    ) -> None:
        result = detector.detect_pii(content)
        assert PIIType.API_KEY in result.pii_types_detected
        assert expected_mask in result.sanitized_content

    def test_api_key_mask_leaves_no_secret_material(
        self, detector: PIIDetector
    ) -> None:
        secret = "d" * 40
        result = detector.detect_pii("sk-" + secret)
        assert secret not in result.sanitized_content

    def test_detects_password_field(self, detector: PIIDetector) -> None:
        result = detector.detect_pii('password: "abcdefghij0123456789"')
        assert PIIType.PASSWORD_HASH in result.pii_types_detected
        assert "password=***REDACTED***" in result.sanitized_content

    def test_short_token_below_length_floor_is_not_matched(
        self, detector: PIIDetector
    ) -> None:
        """The generic token pattern requires 20+ characters."""
        assert PIIType.API_KEY not in _types(detector, 'token = "short"')


# ---------------------------------------------------------------------------
# True negatives -- shapes that must never redact
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNonPIIShapesArePreserved:
    """Identifier shapes that appear constantly in ONEX content.

    Every one of these reaching ``detect_pii`` on the request-model path
    would raise and refuse an otherwise-valid storage request, so a false
    positive here is an availability defect, not just noise.
    """

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("550e8400-e29b-41d4-a716-446655440000", id="uuid"),
            pytest.param("550E8400-E29B-41D4-A716-446655440000", id="uuid-uppercase"),
            pytest.param("da39a3ee5e6b4b0d3255bfef95601890afd80709", id="git-sha1"),
            pytest.param(
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                id="sha256",
            ),
            pytest.param("751b5cb1c", id="git-short-sha"),
            pytest.param("OMN-17236", id="ticket-ref"),
            pytest.param("See OMN-17209 for details", id="ticket-ref-in-prose"),
            pytest.param("2026-08-30T12:34:56Z", id="iso-timestamp"),
            pytest.param(
                "2026-08-30T12:34:56.123456+00:00", id="iso-timestamp-microseconds"
            ),
            pytest.param("2026-08-30", id="iso-date"),
            pytest.param("version 1.2.3", id="semver"),
            pytest.param("localhost:8085", id="host-port"),
            pytest.param("The quick brown fox jumps over the lazy dog", id="prose"),
            pytest.param("John Doe", id="person-name-not-implemented"),
            pytest.param("", id="empty-string"),
        ],
    )
    def test_shape_is_not_redacted(self, detector: PIIDetector, content: str) -> None:
        result = detector.detect_pii(content)
        assert not result.has_pii
        assert result.matches == []
        assert result.sanitized_content == content

    def test_mixed_document_redacts_only_the_pii(self, detector: PIIDetector) -> None:
        content = (
            "OMN-17236 at 2026-08-30T12:34:56Z: commit 751b5cb1c, "
            "run 550e8400-e29b-41d4-a716-446655440000, "
            "reported by john.doe@example.com"
        )
        result = detector.detect_pii(content)
        assert result.pii_types_detected == {PIIType.EMAIL}
        assert "OMN-17236" in result.sanitized_content
        assert "2026-08-30T12:34:56Z" in result.sanitized_content
        assert "751b5cb1c" in result.sanitized_content
        assert "550e8400-e29b-41d4-a716-446655440000" in result.sanitized_content
        assert "john.doe@example.com" not in result.sanitized_content


# ---------------------------------------------------------------------------
# Result-model invariants, dedup, masking
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectionResultInvariants:
    """Shape guarantees callers rely on."""

    def test_clean_content_returns_content_unchanged(
        self, detector: PIIDetector
    ) -> None:
        content = "nothing sensitive here"
        result = detector.detect_pii(content)
        assert result.has_pii is False
        assert result.pii_types_detected == set()
        assert result.sanitized_content == content

    def test_has_pii_tracks_matches(self, detector: PIIDetector) -> None:
        assert detector.detect_pii("a@b.com").has_pii is True

    def test_scan_duration_is_recorded(self, detector: PIIDetector) -> None:
        assert detector.detect_pii("a@b.com").scan_duration_ms >= 0.0

    def test_matches_are_sorted_by_position(self, detector: PIIDetector) -> None:
        result = detector.detect_pii("a@b.com then 555-123-4567 then c@d.com")
        starts = [m.start_index for m in result.matches]
        assert starts == sorted(starts)

    def test_overlapping_matches_are_deduplicated(self, detector: PIIDetector) -> None:
        """One credit card yields exactly one match, not one per pattern."""
        result = detector.detect_pii("4111111111111111")
        assert len(result.matches) == 1

    def test_masked_value_is_the_template_not_a_derivation(
        self, detector: PIIDetector
    ) -> None:
        """Masks are fixed strings; they never echo part of the match."""
        result = detector.detect_pii("alice@example.com")
        match = result.matches[0]
        assert match.masked_value == "***@***.***"
        assert match.value == "alice@example.com"
        assert "alice" not in match.masked_value

    def test_sanitized_content_contains_no_original_pii(
        self, detector: PIIDetector
    ) -> None:
        content = "email alice@example.com phone 555-123-4567 ssn 123-45-6789"
        result = detector.detect_pii(content)
        for match in result.matches:
            assert match.value not in result.sanitized_content

    def test_detect_pii_does_not_mutate_input(self, detector: PIIDetector) -> None:
        content = "alice@example.com"
        detector.detect_pii(content)
        assert content == "alice@example.com"


# ---------------------------------------------------------------------------
# Sensitivity levels
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSensitivityLevels:
    """``sensitivity_level`` selects a confidence floor -- inverted naming."""

    def test_low_is_the_strictest_floor(self, detector: PIIDetector) -> None:
        """``"low"`` uses 0.95, so 0.90-confidence PHONE is skipped."""
        assert PIIType.PHONE not in _types(detector, "555-123-4567", level="low")
        assert PIIType.EMAIL in _types(detector, "a@b.com", level="low")

    def test_medium_admits_medium_confidence_patterns(
        self, detector: PIIDetector
    ) -> None:
        assert PIIType.PHONE in _types(detector, "555-123-4567", level="medium")

    def test_high_is_the_most_permissive_floor(self, detector: PIIDetector) -> None:
        assert PIIType.PHONE in _types(detector, "555-123-4567", level="high")

    def test_unknown_level_falls_back_to_medium_floor(
        self, detector: PIIDetector
    ) -> None:
        content = "555-123-4567"
        assert _types(detector, content, level="nonsense") == _types(
            detector, content, level="medium"
        )


# ---------------------------------------------------------------------------
# is_content_safe
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsContentSafe:
    """The detect arm used as a boolean gate."""

    def test_clean_content_is_safe(self, detector: PIIDetector) -> None:
        assert detector.is_content_safe("no pii here at all") is True

    def test_empty_content_is_safe(self, detector: PIIDetector) -> None:
        assert detector.is_content_safe("") is True

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("alice@example.com", id="email"),
            pytest.param("555-123-4567", id="phone"),
            pytest.param("123-45-6789", id="ssn"),
            pytest.param("4111111111111111", id="credit-card"),
            pytest.param("8.8.8.8", id="ipv4"),
            pytest.param("sk-" + "e" * 40, id="api-key"),
        ],
    )
    def test_content_with_pii_is_unsafe(
        self, detector: PIIDetector, content: str
    ) -> None:
        assert detector.is_content_safe(content) is False

    def test_max_pii_count_allows_a_budget(self, detector: PIIDetector) -> None:
        content = "a@b.com and c@d.com"
        assert detector.is_content_safe(content, max_pii_count=0) is False
        assert detector.is_content_safe(content, max_pii_count=1) is False
        assert detector.is_content_safe(content, max_pii_count=2) is True

    def test_uses_the_permissive_high_floor(self, detector: PIIDetector) -> None:
        """A 0.90-confidence PHONE is caught even though "low" would skip it."""
        assert detector.is_content_safe("555-123-4567") is False

    def test_identifier_shapes_are_safe(self, detector: PIIDetector) -> None:
        assert detector.is_content_safe(f"run {uuid4()} OMN-17236") is True


# ---------------------------------------------------------------------------
# Config-driven behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigDrivenBehavior:
    """``ModelPIIDetectorConfig`` -- and nothing else -- tunes the detector."""

    def test_raising_a_pattern_confidence_floor_disables_that_pattern(self) -> None:
        """Confidence config is the on/off switch for a pattern family."""
        stock = PIIDetector()
        assert PIIType.PHONE in _types(stock, "555-123-4567")

        muted = PIIDetector(ModelPIIDetectorConfig(medium_confidence=0.50))
        assert PIIType.PHONE not in _types(muted, "555-123-4567")

    def test_lowering_the_floor_re_enables_a_pattern(self) -> None:
        loud = PIIDetector(
            ModelPIIDetectorConfig(medium_confidence=0.90, reduced_confidence=0.10)
        )
        assert PIIType.PHONE in _types(loud, "555-123-4567", level="medium")

    def test_email_confidence_comes_from_config(self) -> None:
        detector = PIIDetector(ModelPIIDetectorConfig(medium_high_confidence=0.80))
        match = detector.detect_pii("a@b.com").matches[0]
        assert match.confidence == pytest.approx(0.80)

    def test_max_text_length_is_enforced(self) -> None:
        detector = PIIDetector(ModelPIIDetectorConfig(max_text_length=1000))
        with pytest.raises(ValueError, match="exceeds max"):
            detector.detect_pii("x" * 1001)

    def test_content_at_the_length_limit_is_accepted(self) -> None:
        detector = PIIDetector(ModelPIIDetectorConfig(max_text_length=1000))
        assert detector.detect_pii("x" * 1000).has_pii is False

    def test_max_matches_per_type_caps_matches(self) -> None:
        detector = PIIDetector(ModelPIIDetectorConfig(max_matches_per_type=2))
        content = " ".join(f"user{i}@example.com" for i in range(6))
        result = detector.detect_pii(content)
        assert len(result.matches) == 2
        assert "user5@example.com" in result.sanitized_content

    def test_default_config_is_used_when_none_supplied(self) -> None:
        assert PIIDetector().config == ModelPIIDetectorConfig()

    def test_supplied_config_is_retained(self) -> None:
        config = ModelPIIDetectorConfig(max_matches_per_type=7)
        assert PIIDetector(config).config.max_matches_per_type == 7

    def test_config_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            ModelPIIDetectorConfig(sensitivity="paranoid")  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "env_name",
        [
            "PII_DETECTION_ENABLED",
            "OMNIMEMORY_PII_DETECTION_ENABLED",
            "OMNIMEMORY_PII_SENSITIVITY",
            "OMNIMEMORY_PII_MAX_TEXT_LENGTH",
            "PII_SENSITIVITY_LEVEL",
        ],
    )
    def test_environment_variables_do_not_change_behavior(
        self, monkeypatch: pytest.MonkeyPatch, env_name: str
    ) -> None:
        """Detection is config-driven, never env-driven (OMN-16544)."""
        content = "alice@example.com and 555-123-4567"
        baseline = PIIDetector().detect_pii(content).sanitized_content

        monkeypatch.setenv(env_name, "false")
        assert PIIDetector().detect_pii(content).sanitized_content == baseline
        monkeypatch.setenv(env_name, "0")
        assert PIIDetector().detect_pii(content).sanitized_content == baseline

    def test_detector_module_reads_no_environment(self) -> None:
        """No ``os.environ`` / ``getenv`` access in the detector source."""
        import omnimemory.adapters.adapter_pii_detector as module

        with open(module.__file__, encoding="utf-8") as handle:
            source = handle.read()
        assert "os.environ" not in source
        assert "getenv" not in source


# ---------------------------------------------------------------------------
# Pattern table structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPatternTable:
    """Compiled-pattern caching and the documented not-implemented types."""

    def test_every_pattern_compiles(self, detector: PIIDetector) -> None:
        for pii_type, configs in detector._patterns.items():
            compiled = detector._compiled_patterns[pii_type]
            assert len(compiled) == len(configs)
            for config, pattern in zip(configs, compiled, strict=True):
                assert pattern.pattern == config.pattern

    def test_patterns_are_compiled_case_insensitively(
        self, detector: PIIDetector
    ) -> None:
        for patterns in detector._compiled_patterns.values():
            for pattern in patterns:
                assert pattern.flags & re.IGNORECASE

    def test_patterns_are_compiled_once_at_construction(
        self, detector: PIIDetector
    ) -> None:
        before = detector._compiled_patterns
        detector.detect_pii("alice@example.com")
        detector.detect_pii("555-123-4567")
        assert detector._compiled_patterns is before

    @pytest.mark.parametrize(
        "pii_type",
        [PIIType.URL, PIIType.PERSON_NAME, PIIType.ADDRESS],
    )
    def test_documented_unimplemented_types_have_no_patterns(
        self, detector: PIIDetector, pii_type: PIIType
    ) -> None:
        """TODO(OMN-5762) types are declared but carry no pattern."""
        assert pii_type not in detector._patterns

    @pytest.mark.parametrize(
        "pii_type",
        [
            PIIType.EMAIL,
            PIIType.PHONE,
            PIIType.SSN,
            PIIType.CREDIT_CARD,
            PIIType.IP_ADDRESS,
            PIIType.API_KEY,
            PIIType.PASSWORD_HASH,
        ],
    )
    def test_implemented_types_have_patterns(
        self, detector: PIIDetector, pii_type: PIIType
    ) -> None:
        assert detector._patterns[pii_type]

    def test_person_name_corpus_is_loaded_but_unused(
        self, detector: PIIDetector
    ) -> None:
        """The name set exists; no pattern consumes it (TODO(OMN-5762))."""
        assert "john" in detector._common_names
        assert PIIType.PERSON_NAME not in _types(detector, "john smith")
