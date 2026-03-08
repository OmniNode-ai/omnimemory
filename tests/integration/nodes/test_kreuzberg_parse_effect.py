# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerKreuzbergParse.

These tests exercise the handler end-to-end using mocked kreuzberg client
responses (via unittest.mock). They do NOT require a running Docker container.

Test Categories:
    - kreuzberg_extract: Successful extraction for PDF, DOCX, HTML, Markdown
    - parse_failure: HTTP 422 error maps to document-parse-failed event
    - idempotent_crawl: Second call with matching fingerprint skips re-parse

Related Tickets:
    - OMN-2733: Adopt kreuzberg as document parsing handler for omnimemory

Usage:
    pytest tests/integration/nodes/test_kreuzberg_parse_effect.py \\
        -k "kreuzberg_extract or parse_failure or idempotent_crawl" -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnimemory.models.crawl.model_document_discovered_event import (
    ModelDocumentDiscoveredEvent,
)
from omnimemory.nodes.node_kreuzberg_parse_effect.clients.client_kreuzberg import (
    KreuzbergExtractionError,
    KreuzbergExtractResult,
)
from omnimemory.nodes.node_kreuzberg_parse_effect.handler_kreuzberg_parse import (
    HandlerKreuzbergParse,
)
from omnimemory.nodes.node_kreuzberg_parse_effect.models.model_kreuzberg_parse_config import (
    ModelKreuzbergParseConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAKE_FINGERPRINT = "b" * 64
_FAKE_CONTENT_BLOB_REF = "sha256:" + "b" * 64
_HANDLER_MOD = "omnimemory.nodes.node_kreuzberg_parse_effect.handler_kreuzberg_parse"


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def _make_event(
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
        token_estimate=42,
        scope_ref="omninode/omnimemory",
        detected_doc_type=EnumDetectedDocType.UNKNOWN_MD,
        tags=[],
        priority_hint=50,
    )


def _make_config(text_store_path: str) -> ModelKreuzbergParseConfig:
    return ModelKreuzbergParseConfig(
        kreuzberg_url="http://localhost:8090",
        text_store_path=text_store_path,
        parser_version="1.0.0",
        max_doc_bytes=50_000_000,
        timeout_ms=30_000,
        inline_text_max_chars=4096,
    )


async def _run_handler(
    event: ModelDocumentDiscoveredEvent,
    config: ModelKreuzbergParseConfig,
    extract_result: KreuzbergExtractResult | Exception,
    file_bytes: bytes = b"fake file content",
    cached: tuple[str, str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Run the handler with mocked filesystem I/O and kreuzberg client.

    Patches:
        - Path.read_bytes → returns file_bytes
        - read_cached_text (in handler module) → returns cached
        - write_cached_text (in handler module) → no-op
        - call_kreuzberg_extract (in handler module) → returns/raises extract_result

    Returns published events as list of (topic, payload) tuples.
    """
    published: list[tuple[str, dict[str, Any]]] = []

    async def _cb(topic: str, payload: dict[str, Any]) -> None:
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
# kreuzberg_extract tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kreuzberg_extract_pdf(tmp_path: Path) -> None:
    """Successful kreuzberg extraction for a PDF file."""
    config = _make_config(str(tmp_path / "texts"))
    event = _make_event(source_ref=str(tmp_path / "report.pdf"))
    extracted = "PDF extracted text content"

    published = await _run_handler(
        event,
        config,
        KreuzbergExtractResult(extracted_text=extracted),
        file_bytes=b"%PDF-1.4 fake",
    )

    assert len(published) == 1
    topic, payload = published[0]
    assert "document-indexed" in topic
    assert payload["extracted_text_ref"] == extracted
    assert payload["source_url"] == str(tmp_path / "report.pdf")
    assert payload["parse_status"] == "ok"
    assert payload["parser_version"] == "1.0.0"
    assert "document_id" in payload


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kreuzberg_extract_docx(tmp_path: Path) -> None:
    """Successful kreuzberg extraction for a DOCX file."""
    config = _make_config(str(tmp_path / "texts"))
    event = _make_event(source_ref=str(tmp_path / "document.docx"))
    extracted = "DOCX paragraph one. Paragraph two."

    published = await _run_handler(
        event,
        config,
        KreuzbergExtractResult(extracted_text=extracted),
        file_bytes=b"PK\x03\x04",  # ZIP/DOCX magic bytes
    )

    assert len(published) == 1
    topic, payload = published[0]
    assert "document-indexed" in topic
    assert payload["extracted_text_ref"] == extracted


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kreuzberg_extract_html(tmp_path: Path) -> None:
    """Successful kreuzberg extraction for an HTML file."""
    config = _make_config(str(tmp_path / "texts"))
    event = _make_event(source_ref=str(tmp_path / "page.html"))
    extracted = "Page title. Body content."

    published = await _run_handler(
        event,
        config,
        KreuzbergExtractResult(extracted_text=extracted),
        file_bytes=b"<html><body><p>Page title. Body content.</p></body></html>",
    )

    assert len(published) == 1
    topic, payload = published[0]
    assert "document-indexed" in topic
    assert payload["extracted_text_ref"] == extracted


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kreuzberg_extract_md(tmp_path: Path) -> None:
    """Successful kreuzberg extraction for a Markdown file."""
    config = _make_config(str(tmp_path / "texts"))
    event = _make_event(source_ref=str(tmp_path / "README.md"))
    extracted = "# README\n\nThis is the readme."

    published = await _run_handler(
        event,
        config,
        KreuzbergExtractResult(extracted_text=extracted),
        file_bytes=b"# README\n\nThis is the readme.",
    )

    assert len(published) == 1
    topic, payload = published[0]
    assert "document-indexed" in topic
    assert payload["extracted_text_ref"] == extracted


# ---------------------------------------------------------------------------
# parse_failure tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parse_failure_corrupted_pdf(tmp_path: Path) -> None:
    """kreuzberg HTTP 422 maps to document-parse-failed with error_code=parse_error."""
    config = _make_config(str(tmp_path / "texts"))
    event = _make_event(source_ref=str(tmp_path / "corrupted.pdf"))

    published = await _run_handler(
        event,
        config,
        KreuzbergExtractionError(status_code=422, detail="Unprocessable Entity"),
        file_bytes=b"%PDF-corrupted-not-valid",
    )

    assert len(published) == 1
    topic, payload = published[0]
    assert "parse-failed" in topic
    assert payload["error_code"] == "parse_error"
    assert payload["source_url"] == str(tmp_path / "corrupted.pdf")
    assert "Unprocessable Entity" in str(payload["error_detail"])


# ---------------------------------------------------------------------------
# idempotent_crawl tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_idempotent_crawl_no_duplicate_events(tmp_path: Path) -> None:
    """Second call with identical event + fingerprint emits indexed event per call (no re-parse)."""
    config = _make_config(str(tmp_path / "texts"))
    fingerprint = _FAKE_FINGERPRINT
    event = _make_event(
        source_ref=str(tmp_path / "stable.md"),
        content_fingerprint=fingerprint,
    )

    published: list[tuple[str, dict[str, Any]]] = []

    async def _cb(topic: str, payload: dict[str, Any]) -> None:
        published.append((topic, payload))

    handler = HandlerKreuzbergParse(config=config)
    cached_text = "stable document content"
    mock_extract = AsyncMock()

    with (
        patch("pathlib.Path.read_bytes", return_value=b"stable content bytes"),
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
        # Second call — same event, same fingerprint
        await handler.process_event(
            event=event,
            env_prefix="dev",
            publish_callback=_cb,
        )

    # kreuzberg must NOT be called for either invocation
    mock_extract.assert_not_called()

    # Both calls emit an indexed event (idempotent at parse level, not at emit level)
    assert len(published) == 2, (
        f"Expected 2 indexed events (one per call), got {len(published)}"
    )
    for topic, payload in published:
        assert "document-indexed" in topic
        assert payload["extracted_text_ref"] == cached_text
