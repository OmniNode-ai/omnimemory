# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Connection strings carrying credentials are redacted whatever their scheme."""

from __future__ import annotations

import pytest

from omnimemory.adapters.adapter_error_sanitizer import (
    SanitizationLevel,
    sanitize_error,
)


@pytest.mark.unit
class TestConnectionStringRedaction:
    @pytest.mark.parametrize(
        ("scheme", "password"),
        [
            ("postgresql", "s3cret"),
            ("mysql", "s3cret"),
            ("mongodb", "s3cret"),
            ("redis", "s3cret"),
            ("bolt", "s3cret"),
            ("postgresql", "pa/ss"),
        ],
    )
    def test_credentialed_connection_string_is_redacted(
        self, scheme: str, password: str
    ) -> None:
        connection_string = "{}://{}:{}@{}/app".format(
            scheme, "svc", password, "db.internal"
        )

        message = sanitize_error(
            Exception(f"connect failed for {connection_string}"),
            SanitizationLevel.STANDARD,
        )

        assert password not in message
        assert "[REDACTED]" in message

    def test_url_without_credentials_is_left_alone(self) -> None:
        message = sanitize_error(
            Exception("fetch failed for https://example.com/a/b"),
            SanitizationLevel.STANDARD,
        )

        assert "https://example.com/a/b" in message
