# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerKreuzbergParse.

Tests cover the kreuzberg parse handler's core behaviors:
    - Hard limit: documents exceeding max_doc_bytes emit parse-failed without
      calling kreuzberg.
    - Successful inline parse: short extracted text stored directly in the event.
    - Successful file-pointer parse: long extracted text stored on disk.
    - Timeout: KreuzbergTimeoutError maps to error_code=timeout.
    - Idempotency: second call with same fingerprint skips kreuzberg HTTP.

All tests mock filesystem I/O and the kreuzberg client to avoid external dependencies.

Related Tickets:
    - OMN-2733: Adopt kreuzberg as document parsing handler for omnimemory

Usage:
    pytest tests/unit/nodes/test_kreuzberg_parse_effect/ -m unit -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnimemory.models.crawl.model_document_discovered_event import (
    ModelDocumentDiscoveredEvent,
)
from omnimemory.nodes.kreuzberg_parse_effect.clients.client_kreuzberg import (
    KreuzbergExtractResult,
    KreuzbergTimeoutError,
)
from omnimemory.nodes.kreuzberg_parse_effect.handler_kreuzberg_parse import (
    HandlerKreuzbergParse,
)
from omnimemory.nodes.kreuzberg_parse_effect.models.model_kreuzberg_parse_config import (
    ModelKreuzbergParseConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAKE_FINGERPRINT = "a" * 64
_FAKE_CONTENT_BLOB_REF = "sha256:" + "a" * 64

# Module paths for patching — patch where names are looked up, not where defined
_HANDLER_MOD = "omnimemory.nodes.kreuzberg_parse_effect.handler_kreuzberg_parse"
_CLIENT_MOD = "omnimemory.nodes.kreuzberg_parse_effect.clients.client_kreuzberg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_discovered_event(
    source_ref: str,
    content_fingerprint: str = _FAKE_FINGERPRINT,
) -> ModelDocumentDiscoveredEvent:
    from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType
    from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType
    from omnimemory.enums.crawl.enum_detected_doc_type import EnumDetectedDocType

    return ModelDocumentDiscoveredEvent(
        correlation_id=uuid4(),
        emitted_at_utc=datetime.now(tz=timezone.utc),
        crawler_type=EnumCrawlerType.FILESYSTEM,
        crawl_scope="omninode/omnimemory",
        trigger_source="scheduled",
        source_ref=source_ref,
        source_type=EnumContextSourceType.REPO_DERIVED,
        source_version=None,
        content_fingerprint=content_fingerprint,
        content_blob_ref=_FAKE_CONTENT_BLOB_REF,
        token_estimate=100,
        scope_ref="omninode/omnimemory",
        detected_doc_type=EnumDetectedDocType.UNKNOWN_MD,
        tags=[],
        priority_hint=50,
    )


def _make_config(
    text_store_path: str,
    max_doc_bytes: int = 50_000_000,
    inline_text_max_chars: int = 4096,
) -> ModelKreuzbergParseConfig:
    return ModelKreuzbergParseConfig(
        kreuzberg_url="http://localhost:8090",
        text_store_path=text_store_path,
        parser_version="1.0.0",
        max_doc_bytes=max_doc_bytes,
        timeout_ms=30_000,
        inline_text_max_chars=inline_text_max_chars,
    )


async def _run_handler(
    event: ModelDocumentDiscoveredEvent,
    config: ModelKreuzbergParseConfig,
    file_bytes: bytes,
    extract_result: KreuzbergExtractResult | Exception,
    cached: tuple[str, str] | None = None,
) -> list[tuple[str, dict[str, object]]]:
    """Run handler with all I/O patched. Returns list of (topic, payload) tuples.

    Patches:
        - Path.read_bytes → returns file_bytes
        - read_cached_text (in handler module) → returns cached
        - write_cached_text (in handler module) → no-op
        - call_kreuzberg_extract (in handler module) → returns/raises extract_result
    """
    published: list[tuple[str, dict[str, object]]] = []

    async def _cb(topic: str, payload: dict[str, object]) -> None:
        published.append((topic, payload))

    handler = HandlerKreuzbergParse(config=config)

    if isinstance(extract_result, Exception):
        mock_extract = AsyncMock(side_effect=extract_result)
    else:
        mock_extract = AsyncMock(return_value=extract_result)

    with (
        patch("pathlib.Path.read_bytes", return_value=file_bytes),
        patch(f"{_HANDLER_MOD}.read_cached_text", return_value=cached),
        patch(f"{_HANDLER_MOD}.write_cached_text"),
        patch(f"{_HANDLER_MOD}.call_kreuzberg_extract", mock_extract),
    ):
        await handler.process_event(
            event=event,
            env_prefix="dev",
            publish_callback=_cb,
        )

    return published


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_too_large_file_emits_parse_failed(tmp_path: Path) -> None:
    """Documents exceeding max_doc_bytes emit parse-failed without calling kreuzberg."""
    config = _make_config(
        text_store_path=str(tmp_path / "texts"),
        max_doc_bytes=10,
    )
    source_ref = str(tmp_path / "big_doc.md")
    event = _make_discovered_event(source_ref=source_ref)
    handler = HandlerKreuzbergParse(config=config)

    published: list[tuple[str, dict[str, object]]] = []

    async def _cb(topic: str, payload: dict[str, object]) -> None:
        published.append((topic, payload))

    mock_extract = AsyncMock()

    with (
        patch("pathlib.Path.read_bytes", return_value=b"x" * 11),  # 11 > max=10
        patch(f"{_HANDLER_MOD}.call_kreuzberg_extract", mock_extract),
        patch(f"{_HANDLER_MOD}.read_cached_text", return_value=None),
    ):
        await handler.process_event(
            event=event,
            env_prefix="dev",
            publish_callback=_cb,
        )

    # kreuzberg must NOT be called
    mock_extract.assert_not_called()

    assert len(published) == 1
    topic, payload = published[0]
    assert "parse-failed" in topic
    assert payload["error_code"] == "too_large"
    assert payload["source_url"] == source_ref


@pytest.mark.unit
@pytest.mark.asyncio
async def test_successful_parse_inline_text(tmp_path: Path) -> None:
    """Short extracted text (<4096 chars) is stored inline in the event."""
    config = _make_config(
        text_store_path=str(tmp_path / "texts"),
        inline_text_max_chars=4096,
    )
    source_ref = str(tmp_path / "small_doc.md")
    event = _make_discovered_event(source_ref=source_ref)

    published = await _run_handler(
        event=event,
        config=config,
        file_bytes=b"# Hello\n",
        extract_result=KreuzbergExtractResult(extracted_text="hello world"),
        cached=None,
    )

    assert len(published) == 1
    topic, payload = published[0]
    assert "document-indexed" in topic
    assert payload["extracted_text_ref"] == "hello world"
    assert payload["parse_status"] == "ok"
    assert payload["parser_version"] == "1.0.0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_successful_parse_file_pointer(tmp_path: Path) -> None:
    """Long extracted text (>= inline_text_max_chars) is referenced via file://."""
    config = _make_config(
        text_store_path=str(tmp_path / "texts"),
        inline_text_max_chars=10,  # very small threshold
    )
    source_ref = str(tmp_path / "large_doc.md")
    event = _make_discovered_event(source_ref=source_ref)

    long_text = "x" * 20  # 20 chars > inline_text_max_chars=10
    published = await _run_handler(
        event=event,
        config=config,
        file_bytes=b"content bytes",
        extract_result=KreuzbergExtractResult(extracted_text=long_text),
        cached=None,
    )

    assert len(published) == 1
    topic, payload = published[0]
    assert "document-indexed" in topic
    extracted_ref = str(payload["extracted_text_ref"])
    assert extracted_ref.startswith("file://"), (
        f"Expected file:// pointer, got: {extracted_ref!r}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_timeout_emits_parse_failed(tmp_path: Path) -> None:
    """KreuzbergTimeoutError from the client maps to error_code=timeout."""
    config = _make_config(text_store_path=str(tmp_path / "texts"))
    source_ref = str(tmp_path / "timeout_doc.md")
    event = _make_discovered_event(source_ref=source_ref)

    published = await _run_handler(
        event=event,
        config=config,
        file_bytes=b"file content",
        extract_result=KreuzbergTimeoutError("timed out"),
        cached=None,
    )

    assert len(published) == 1
    topic, payload = published[0]
    assert "parse-failed" in topic
    assert payload["error_code"] == "timeout"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_idempotent_second_call_no_reparse(tmp_path: Path) -> None:
    """Second call with same fingerprint re-emits indexed event without calling kreuzberg."""
    config = _make_config(text_store_path=str(tmp_path / "texts"))
    source_ref = str(tmp_path / "idempotent_doc.md")
    fingerprint = _FAKE_FINGERPRINT
    event = _make_discovered_event(
        source_ref=source_ref,
        content_fingerprint=fingerprint,
    )
    handler = HandlerKreuzbergParse(config=config)

    published: list[tuple[str, dict[str, object]]] = []

    async def _cb(topic: str, payload: dict[str, object]) -> None:
        published.append((topic, payload))

    cached_text = "previously extracted text"
    mock_extract = AsyncMock()

    with (
        patch("pathlib.Path.read_bytes", return_value=b"file content"),
        patch(
            f"{_HANDLER_MOD}.read_cached_text",
            return_value=(fingerprint, cached_text),
        ),
        patch(f"{_HANDLER_MOD}.write_cached_text"),
        patch(f"{_HANDLER_MOD}.call_kreuzberg_extract", mock_extract),
    ):
        # First call
        await handler.process_event(
            event=event,
            env_prefix="dev",
            publish_callback=_cb,
        )
        # Second call — same event
        await handler.process_event(
            event=event,
            env_prefix="dev",
            publish_callback=_cb,
        )

    # kreuzberg must NOT be called since cache always hits
    mock_extract.assert_not_called()

    # Both calls should emit an indexed event
    assert len(published) == 2
    for topic, payload in published:
        assert "document-indexed" in topic
        assert payload["extracted_text_ref"] == cached_text
