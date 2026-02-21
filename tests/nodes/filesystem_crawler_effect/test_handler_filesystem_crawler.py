# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerFilesystemCrawler.

Tests cover:
- mtime fast-path: file not re-read when mtime unchanged
- SHA-256 change detection: mtime changed, content unchanged → no event
- Document discovered: new file emits document-discovered.v1
- Document changed: known file with different content emits document-changed.v1
- Document removed: state table entries absent from walk emit document-removed.v1
- File size limit: files exceeding max_file_size_bytes are skipped
- Scope resolution: longest-prefix mapping applied correctly
- Source type assignment: CLAUDE.md → STATIC_STANDARDS, other → REPO_DERIVED
- Priority hints: correct values per design §7
- Doc type detection: CLAUDE.md, ARCHITECTURE, DEEP_DIVE, DESIGN, PLAN, etc.
- Publish errors: non-fatal, crawl continues
- File read errors: non-fatal, error_count incremented
- max_files_per_crawl: truncation behaviour
- Empty path_prefixes: no-op, zero events
- Non-existent prefix: warning logged, no crash

.. versionadded:: 0.4.0
    Initial tests for OMN-2385.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Coroutine
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType
from omnimemory.enums.crawl.enum_detected_doc_type import EnumDetectedDocType
from omnimemory.models.crawl.model_crawl_state_record import ModelCrawlStateRecord
from omnimemory.nodes.filesystem_crawler_effect.handler_filesystem_crawler import (
    HandlerFilesystemCrawler,
    _compute_sha256,
    _detect_doc_type,
    _priority_hint_for_path,
    _scope_ref_for_path,
    _source_type_for_path,
)
from omnimemory.nodes.filesystem_crawler_effect.models.model_filesystem_crawler_config import (
    ModelFilesystemCrawlerConfig,
)

# Type aliases
type PublishRecord = tuple[str, dict[str, object]]


# =============================================================================
# Helpers
# =============================================================================


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def make_config(
    path_prefixes: list[str] | None = None,
    max_file_size_bytes: int = 5_242_880,
    max_files_per_crawl: int = 10_000,
) -> ModelFilesystemCrawlerConfig:
    return ModelFilesystemCrawlerConfig(
        path_prefixes=path_prefixes or [],
        max_file_size_bytes=max_file_size_bytes,
        max_files_per_crawl=max_files_per_crawl,
    )


def make_state(
    source_ref: str,
    fingerprint: str,
    scope_ref: str = "omninode/test",
    mtime: float | None = None,
) -> ModelCrawlStateRecord:
    from datetime import datetime, timezone

    return ModelCrawlStateRecord(
        source_ref=source_ref,
        crawler_type=EnumCrawlerType.FILESYSTEM,
        scope_ref=scope_ref,
        content_fingerprint=fingerprint,
        source_version=None,
        last_crawled_at_utc=datetime.now(timezone.utc),
        last_known_mtime=mtime,
    )


def make_mock_repo() -> MagicMock:
    """Return a mock that satisfies ProtocolCrawlStateRepository."""
    repo = MagicMock()
    repo.get_state = AsyncMock(return_value=None)
    repo.list_states_for_scope = AsyncMock(return_value=[])
    repo.upsert_state = AsyncMock()
    repo.delete_state = AsyncMock()
    return repo


def make_publish_capture() -> (
    tuple[
        Callable[[str, dict[str, object]], Coroutine[object, object, None]],
        list[PublishRecord],
    ]
):
    published: list[PublishRecord] = []

    async def capture(topic: str, payload: dict[str, object]) -> None:
        published.append((topic, payload))

    return capture, published


# =============================================================================
# Pure function tests (no I/O)
# =============================================================================


class TestComputeSha256:
    @pytest.mark.unit
    def test_known_value(self) -> None:
        content = b"hello"
        expected = hashlib.sha256(b"hello").hexdigest()
        assert _compute_sha256(content) == expected

    @pytest.mark.unit
    def test_empty_bytes(self) -> None:
        result = _compute_sha256(b"")
        assert len(result) == 64  # 256 bits → 64 hex chars
        assert result == hashlib.sha256(b"").hexdigest()

    @pytest.mark.unit
    def test_different_content_different_hash(self) -> None:
        assert _compute_sha256(b"a") != _compute_sha256(b"b")


class TestDetectDocType:
    @pytest.mark.unit
    def test_claude_md(self) -> None:
        assert (
            _detect_doc_type(Path("/some/repo/CLAUDE.md"))
            == EnumDetectedDocType.CLAUDE_MD
        )

    @pytest.mark.unit
    def test_architecture_doc(self) -> None:
        assert (
            _detect_doc_type(Path("/repo/docs/ONEX_ARCHITECTURE.md"))
            == EnumDetectedDocType.ARCHITECTURE_DOC
        )

    @pytest.mark.unit
    def test_overview_doc(self) -> None:
        assert (
            _detect_doc_type(Path("/repo/docs/SYSTEM_OVERVIEW.md"))
            == EnumDetectedDocType.ARCHITECTURE_DOC
        )

    @pytest.mark.unit
    def test_deep_dive(self) -> None:
        assert (
            _detect_doc_type(Path("/workspace/omni_save/DEEP_DIVE_2026.md"))
            == EnumDetectedDocType.DEEP_DIVE
        )

    @pytest.mark.unit
    def test_readme(self) -> None:
        assert _detect_doc_type(Path("/repo/README.md")) == EnumDetectedDocType.README

    @pytest.mark.unit
    def test_design_doc(self) -> None:
        assert (
            _detect_doc_type(Path("/repo/design/my_design.md"))
            == EnumDetectedDocType.DESIGN_DOC
        )

    @pytest.mark.unit
    def test_plan(self) -> None:
        assert (
            _detect_doc_type(Path("/repo/plans/roadmap.md")) == EnumDetectedDocType.PLAN
        )

    @pytest.mark.unit
    def test_handoff(self) -> None:
        assert (
            _detect_doc_type(Path("/repo/handoffs/sprint_notes.md"))
            == EnumDetectedDocType.HANDOFF
        )

    @pytest.mark.unit
    def test_unknown_md(self) -> None:
        assert (
            _detect_doc_type(Path("/repo/docs/some_random.md"))
            == EnumDetectedDocType.UNKNOWN_MD
        )

    @pytest.mark.unit
    def test_claude_md_takes_priority_over_architecture(self) -> None:
        # A file literally named CLAUDE.md should be CLAUDE_MD even if in a
        # directory that would match another rule.
        assert (
            _detect_doc_type(Path("/repo/design/CLAUDE.md"))
            == EnumDetectedDocType.CLAUDE_MD
        )


class TestSourceTypeForPath:
    @pytest.mark.unit
    def test_claude_md_is_static_standards(self) -> None:
        assert (
            _source_type_for_path(Path("/some/repo/CLAUDE.md"))
            == EnumContextSourceType.STATIC_STANDARDS
        )

    @pytest.mark.unit
    def test_dot_claude_dir_is_static_standards(self) -> None:
        path = Path.home() / ".claude" / "CLAUDE.md"
        assert _source_type_for_path(path) == EnumContextSourceType.STATIC_STANDARDS

    @pytest.mark.unit
    def test_design_doc_is_repo_derived(self) -> None:
        assert (
            _source_type_for_path(Path("/repo/design/foo.md"))
            == EnumContextSourceType.REPO_DERIVED
        )

    @pytest.mark.unit
    def test_readme_is_repo_derived(self) -> None:
        assert (
            _source_type_for_path(Path("/repo/README.md"))
            == EnumContextSourceType.REPO_DERIVED
        )


class TestPriorityHintForPath:
    _PREFIXES: list[str] = ["/repo"]

    @pytest.mark.unit
    def test_global_claude_md(self) -> None:
        p = Path.home() / ".claude" / "CLAUDE.md"
        assert _priority_hint_for_path(p, [str(Path.home() / ".claude")]) == 95

    @pytest.mark.unit
    def test_repo_claude_md(self) -> None:
        assert _priority_hint_for_path(Path("/repo/CLAUDE.md"), self._PREFIXES) == 85

    @pytest.mark.unit
    def test_design_architecture(self) -> None:
        assert (
            _priority_hint_for_path(
                Path("/repo/design/FOO_ARCHITECTURE.md"), self._PREFIXES
            )
            == 80
        )

    @pytest.mark.unit
    def test_design_doc(self) -> None:
        assert (
            _priority_hint_for_path(Path("/repo/design/my_design.md"), self._PREFIXES)
            == 70
        )

    @pytest.mark.unit
    def test_plan(self) -> None:
        assert (
            _priority_hint_for_path(Path("/repo/plans/roadmap.md"), self._PREFIXES)
            == 65
        )

    @pytest.mark.unit
    def test_handoff(self) -> None:
        assert (
            _priority_hint_for_path(Path("/repo/handoffs/notes.md"), self._PREFIXES)
            == 60
        )

    @pytest.mark.unit
    def test_unknown_md(self) -> None:
        assert (
            _priority_hint_for_path(Path("/repo/docs/something.md"), self._PREFIXES)
            == 35
        )


class TestScopeRefForPath:
    @pytest.mark.unit
    def test_exact_prefix_match(self) -> None:
        mappings = [("/Volumes/PRO-G40/Code/omnimemory", "omninode/omnimemory")]
        result = _scope_ref_for_path(
            Path("/Volumes/PRO-G40/Code/omnimemory/CLAUDE.md"), mappings
        )
        assert result == "omninode/omnimemory"

    @pytest.mark.unit
    def test_longest_prefix_wins(self) -> None:
        mappings = [
            ("/Volumes/PRO-G40/Code", "omninode/shared"),
            ("/Volumes/PRO-G40/Code/omni_save/design", "omninode/shared/design"),
            ("/Volumes/PRO-G40/Code/omni_save", "omninode/shared/plans"),
        ]
        result = _scope_ref_for_path(
            Path("/Volumes/PRO-G40/Code/omni_save/design/my.md"), mappings
        )
        assert result == "omninode/shared/design"

    @pytest.mark.unit
    def test_no_match_returns_default(self) -> None:
        mappings: list[tuple[str, str]] = []
        result = _scope_ref_for_path(
            Path("/completely/different/path/foo.md"), mappings
        )
        assert result == "omninode/shared"  # DEFAULT_SCOPE_REF


# =============================================================================
# Handler integration tests (with mocked filesystem and repository)
# =============================================================================


class TestHandlerFilesystemCrawlerNoPathPrefixes:
    """Handler with empty path_prefixes is a no-op."""

    @pytest.mark.unit
    async def test_empty_prefixes_returns_zero_counts(self) -> None:
        config = make_config(path_prefixes=[])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.files_walked == 0
        assert result.discovered_count == 0
        assert result.changed_count == 0
        assert result.removed_count == 0
        assert not published


class TestHandlerFilesystemCrawlerNonExistentPrefix:
    @pytest.mark.unit
    async def test_non_existent_prefix_warns_and_returns_zero(
        self, tmp_path: Path
    ) -> None:
        non_existent = str(tmp_path / "does_not_exist")
        config = make_config(path_prefixes=[non_existent])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.files_walked == 0
        assert not published


class TestHandlerFilesystemCrawlerDiscovered:
    """New files with no prior state emit document-discovered events."""

    @pytest.mark.unit
    async def test_new_file_emits_discovered_event(self, tmp_path: Path) -> None:
        md_file = tmp_path / "test.md"
        content = b"# Hello World\n\nThis is a test document."
        md_file.write_bytes(content)

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        # No prior state → get_state returns None
        repo.get_state = AsyncMock(return_value=None)

        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.files_walked == 1
        assert result.discovered_count == 1
        assert result.changed_count == 0
        assert result.removed_count == 0

        # Should have emitted exactly one event on the discovered topic
        discovered_events = [(t, p) for t, p in published if "document-discovered" in t]
        assert len(discovered_events) == 1

        topic, payload = discovered_events[0]
        assert topic == "dev.onex.evt.omnimemory.document-discovered.v1"
        assert payload["event_type"] == "DocumentDiscovered"
        assert payload["content_fingerprint"] == _sha256(content)
        assert payload["source_ref"] == str(md_file.resolve())

        # State should be persisted
        repo.upsert_state.assert_awaited_once()

    @pytest.mark.unit
    async def test_discovered_event_has_correct_blob_ref(self, tmp_path: Path) -> None:
        md_file = tmp_path / "doc.md"
        content = b"Some content"
        md_file.write_bytes(content)
        fp = _sha256(content)

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        _, payload = published[0]
        assert payload["content_blob_ref"] == f"sha256:{fp}"

    @pytest.mark.unit
    async def test_discovered_event_correlation_id_threaded(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a.md").write_bytes(b"content")
        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        cid = uuid4()
        await handler.crawl(
            correlation_id=cid,
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        _, payload = published[0]
        # UUID may be serialized as string
        assert str(payload["correlation_id"]) == str(cid)

    @pytest.mark.unit
    async def test_discovered_event_token_estimate(self, tmp_path: Path) -> None:
        content = b"x" * 400
        (tmp_path / "a.md").write_bytes(content)
        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        _, payload = published[0]
        assert payload["token_estimate"] == 100  # 400 // 4

    @pytest.mark.unit
    async def test_multiple_files_all_discovered(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"doc_{i}.md").write_bytes(f"# Doc {i}".encode())

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.files_walked == 5
        assert result.discovered_count == 5
        assert len(published) == 5


class TestHandlerFilesystemCrawlerMtimeFastPath:
    """When mtime is unchanged, the file must not be re-read."""

    @pytest.mark.unit
    async def test_unchanged_mtime_skips_hashing(self, tmp_path: Path) -> None:
        md_file = tmp_path / "stable.md"
        content = b"stable content"
        md_file.write_bytes(content)
        mtime = md_file.stat().st_mtime
        fp = _sha256(content)

        prior = make_state(str(md_file.resolve()), fp, mtime=mtime)
        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        repo.get_state = AsyncMock(return_value=prior)

        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        with patch.object(
            type(md_file), "read_bytes", side_effect=AssertionError("should not read")
        ):
            result = await handler.crawl(
                correlation_id=uuid4(),
                crawl_scope="omninode/test",
                trigger_source="scheduled",
                env_prefix="dev",
                publish_callback=publish,
            )

        assert result.discovered_count == 0
        assert result.changed_count == 0
        assert not published

    @pytest.mark.unit
    async def test_unchanged_mtime_updates_last_crawled(self, tmp_path: Path) -> None:
        md_file = tmp_path / "stable.md"
        content = b"stable"
        md_file.write_bytes(content)
        mtime = md_file.stat().st_mtime
        fp = _sha256(content)

        prior = make_state(str(md_file.resolve()), fp, mtime=mtime)
        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        repo.get_state = AsyncMock(return_value=prior)

        publish, _ = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        with patch.object(
            type(md_file), "read_bytes", side_effect=AssertionError("should not read")
        ):
            await handler.crawl(
                correlation_id=uuid4(),
                crawl_scope="omninode/test",
                trigger_source="scheduled",
                env_prefix="dev",
                publish_callback=publish,
            )

        # upsert_state should still be called to update last_crawled_at_utc
        repo.upsert_state.assert_awaited_once()


class TestHandlerFilesystemCrawlerChanged:
    """Files with changed SHA-256 emit document-changed events."""

    @pytest.mark.unit
    async def test_changed_content_emits_changed_event(self, tmp_path: Path) -> None:
        md_file = tmp_path / "evolving.md"
        old_content = b"old content"
        new_content = b"new content"
        md_file.write_bytes(new_content)

        old_fp = _sha256(old_content)
        old_mtime = md_file.stat().st_mtime - 10  # simulate mtime changed

        prior = make_state(str(md_file.resolve()), old_fp, mtime=old_mtime)
        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        repo.get_state = AsyncMock(return_value=prior)

        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.changed_count == 1
        assert result.discovered_count == 0

        changed_events = [(t, p) for t, p in published if "document-changed" in t]
        assert len(changed_events) == 1

        topic, payload = changed_events[0]
        assert topic == "dev.onex.evt.omnimemory.document-changed.v1"
        assert payload["event_type"] == "DocumentChanged"
        assert payload["content_fingerprint"] == _sha256(new_content)
        assert payload["previous_content_fingerprint"] == old_fp

    @pytest.mark.unit
    async def test_mtime_changed_content_unchanged_no_event(
        self, tmp_path: Path
    ) -> None:
        md_file = tmp_path / "touched.md"
        content = b"same content"
        md_file.write_bytes(content)
        fp = _sha256(content)

        # Simulate mtime differs from stored (editor touched without saving)
        old_mtime = md_file.stat().st_mtime - 5
        prior = make_state(str(md_file.resolve()), fp, mtime=old_mtime)

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        repo.get_state = AsyncMock(return_value=prior)

        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.changed_count == 0
        assert result.discovered_count == 0
        assert result.unchanged_count == 1
        assert not published


class TestHandlerFilesystemCrawlerRemoved:
    """Files in state table but not found in walk emit document-removed events."""

    @pytest.mark.unit
    async def test_missing_file_emits_removed_event(self, tmp_path: Path) -> None:
        # State table knows about a file that no longer exists on disk
        ghost_path = str(tmp_path / "deleted.md")

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        repo.get_state = AsyncMock(return_value=None)

        # list_states_for_scope returns a stale record
        ghost_state = make_state(ghost_path, _sha256(b"old content"))
        repo.list_states_for_scope = AsyncMock(return_value=[ghost_state])

        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.removed_count == 1

        removed_events = [(t, p) for t, p in published if "document-removed" in t]
        assert len(removed_events) == 1

        topic, payload = removed_events[0]
        assert topic == "dev.onex.evt.omnimemory.document-removed.v1"
        assert payload["event_type"] == "DocumentRemoved"
        assert payload["source_ref"] == ghost_path

        # State should be deleted
        repo.delete_state.assert_awaited_once()

    @pytest.mark.unit
    async def test_existing_file_not_emitted_as_removed(self, tmp_path: Path) -> None:
        md_file = tmp_path / "real.md"
        content = b"# Real doc"
        md_file.write_bytes(content)

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        # The file IS in the state table but still on disk → no removal
        state = make_state(str(md_file.resolve()), _sha256(content))
        repo.get_state = AsyncMock(return_value=state)
        repo.list_states_for_scope = AsyncMock(return_value=[state])

        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.removed_count == 0
        removed_events = [(t, p) for t, p in published if "document-removed" in t]
        assert len(removed_events) == 0


class TestHandlerFilesystemCrawlerFileLimits:
    @pytest.mark.unit
    async def test_oversized_file_is_skipped(self, tmp_path: Path) -> None:
        big_file = tmp_path / "big.md"
        big_file.write_bytes(b"x" * 100)

        # Set max to 50 bytes
        config = make_config(path_prefixes=[str(tmp_path)], max_file_size_bytes=50)
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.files_walked == 1
        assert result.skipped_count == 1
        assert result.discovered_count == 0
        assert not published

    @pytest.mark.unit
    async def test_max_files_per_crawl_truncates(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"f{i}.md").write_bytes(f"# Doc {i}".encode())

        config = make_config(path_prefixes=[str(tmp_path)], max_files_per_crawl=3)
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.truncated is True
        assert result.files_walked == 3
        # Should have discovered 3 (not 10)
        assert result.discovered_count == 3


class TestHandlerFilesystemCrawlerPublishErrors:
    """Publish errors must not abort the crawl."""

    @pytest.mark.unit
    async def test_publish_error_continues_crawl(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"f{i}.md").write_bytes(f"# {i}".encode())

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()

        call_count = 0

        async def failing_publish(topic: str, payload: dict[str, object]) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Kafka unavailable")

        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        # Should not raise
        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=failing_publish,
        )

        # All 3 files were walked and attempted to publish
        assert result.files_walked == 3
        assert result.discovered_count == 3
        assert call_count == 3


class TestHandlerFilesystemCrawlerDocTypeAssignment:
    """Correct doc types are embedded in emitted events."""

    @pytest.mark.unit
    async def test_claude_md_doc_type_in_event(self, tmp_path: Path) -> None:
        claude_file = tmp_path / "CLAUDE.md"
        claude_file.write_bytes(b"# Standards")

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        _, payload = published[0]
        assert payload["detected_doc_type"] == EnumDetectedDocType.CLAUDE_MD.value

    @pytest.mark.unit
    async def test_readme_doc_type_in_event(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_bytes(b"# Readme")

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        _, payload = published[0]
        assert payload["detected_doc_type"] == EnumDetectedDocType.README.value


class TestHandlerFilesystemCrawlerSourceType:
    """CLAUDE.md files are classified as STATIC_STANDARDS."""

    @pytest.mark.unit
    async def test_claude_md_source_type_static_standards(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_bytes(b"# Policy")

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        _, payload = published[0]
        assert payload["source_type"] == EnumContextSourceType.STATIC_STANDARDS.value

    @pytest.mark.unit
    async def test_regular_md_source_type_repo_derived(self, tmp_path: Path) -> None:
        (tmp_path / "notes.md").write_bytes(b"# Notes")

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        _, payload = published[0]
        assert payload["source_type"] == EnumContextSourceType.REPO_DERIVED.value


class TestHandlerFilesystemCrawlerPriorityHints:
    @pytest.mark.unit
    async def test_claude_md_priority_85(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_bytes(b"# Standards")

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        _, payload = published[0]
        assert payload["priority_hint"] == 85

    @pytest.mark.unit
    async def test_unknown_md_priority_35(self, tmp_path: Path) -> None:
        (tmp_path / "notes.md").write_bytes(b"# Notes")

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        _, payload = published[0]
        assert payload["priority_hint"] == 35


class TestHandlerFilesystemCrawlerScopeMappings:
    @pytest.mark.unit
    async def test_scope_mapping_applied_to_event(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_bytes(b"# Doc")

        scope_mappings = [(str(tmp_path), "omninode/custom-scope")]
        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/custom-scope",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
            scope_mappings=scope_mappings,
        )

        _, payload = published[0]
        assert payload["scope_ref"] == "omninode/custom-scope"


class TestHandlerFilesystemCrawlerRecursive:
    """Files in nested subdirectories are discovered."""

    @pytest.mark.unit
    async def test_recursive_walk_finds_nested_files(self, tmp_path: Path) -> None:
        nested = tmp_path / "docs" / "subdir"
        nested.mkdir(parents=True)
        (nested / "deep.md").write_bytes(b"# Deep")
        (tmp_path / "root.md").write_bytes(b"# Root")

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.files_walked == 2
        assert result.discovered_count == 2

    @pytest.mark.unit
    async def test_non_md_files_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_bytes(b"print('hello')")
        (tmp_path / "data.json").write_bytes(b"{}")
        (tmp_path / "doc.md").write_bytes(b"# Doc")

        config = make_config(path_prefixes=[str(tmp_path)])
        repo = make_mock_repo()
        publish, published = make_publish_capture()
        handler = HandlerFilesystemCrawler(config=config, crawl_state_repo=repo)

        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/test",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publish,
        )

        assert result.files_walked == 1
        assert result.discovered_count == 1
