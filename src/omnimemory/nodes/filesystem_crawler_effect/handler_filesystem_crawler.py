# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""FilesystemCrawler handler: walks path prefixes and emits document lifecycle events.

Architecture:
    Document Ingestion Pipeline design (§5 Crawl State and Change Detection).

    Change Detection Strategy (two-stage):
        1. mtime fast-path: if stat.st_mtime is unchanged vs stored state, skip
           the file without re-reading or re-hashing it (most common case).
        2. If mtime changed: compute SHA-256(content). If the hash is also
           unchanged, update last_crawled_at_utc only (mtime bumped by an
           editor without content change). If the hash differs, emit
           document.changed.v1.
        3. Not in state table: emit document.discovered.v1.
        4. Records in state table but not found in walk: emit document.removed.v1.

    Scope Assignment:
        Path-to-scope mapping uses longest-prefix matching against the
        scope_mappings list passed to crawl(). The longest matching prefix
        wins. Paths with no match use DEFAULT_SCOPE_REF.

    Source Type Assignment:
        static_standards: ~/.claude/ prefixes and **/CLAUDE.md
        repo_derived: everything else

    Priority Hints:
        Computed by _priority_hint_for_path() following the table in design §7.

    Blob Storage:
        Content is addressed by 'sha256:<fingerprint>'. In this implementation
        blobs are not persisted — only the fingerprint is computed and the
        blob_ref is constructed as the content-addressed key. Blob persistence
        is the responsibility of downstream processors (DocumentFetchEffect).

Example::

    from omnimemory.nodes.filesystem_crawler_effect.handler_filesystem_crawler import (
        HandlerFilesystemCrawler,
    )

    async def example(state_repo, publisher, config):
        handler = HandlerFilesystemCrawler(
            config=config,
            crawl_state_repo=state_repo,
        )
        result = await handler.crawl(
            correlation_id=uuid4(),
            crawl_scope="omninode/omnimemory",
            trigger_source="scheduled",
            env_prefix="dev",
            publish_callback=publisher,
        )
        print(f"discovered={result.discovered_count}")

.. versionadded:: 0.4.0
    Initial implementation for OMN-2385.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType
from omnimemory.enums.crawl.enum_detected_doc_type import EnumDetectedDocType
from omnimemory.models.crawl.model_crawl_state_record import ModelCrawlStateRecord
from omnimemory.models.crawl.model_document_changed_event import (
    ModelDocumentChangedEvent,
)
from omnimemory.models.crawl.model_document_discovered_event import (
    ModelDocumentDiscoveredEvent,
)
from omnimemory.models.crawl.model_document_indexed_event import (
    ModelDocumentIndexedEvent,
)
from omnimemory.models.crawl.model_document_removed_event import (
    ModelDocumentRemovedEvent,
)
from omnimemory.nodes.filesystem_crawler_effect.models.model_filesystem_crawl_result import (
    ModelFilesystemCrawlResult,
)

if TYPE_CHECKING:
    from omnimemory.models.crawl.types import TriggerSource
    from omnimemory.nodes.filesystem_crawler_effect.models.model_filesystem_crawler_config import (
        ModelFilesystemCrawlerConfig,
    )
    from omnimemory.protocols.protocol_crawl_state_repository import (
        ProtocolCrawlStateRepository,
    )

__all__ = ["HandlerFilesystemCrawler"]

logger = logging.getLogger(__name__)

HANDLER_ID_FILESYSTEM_CRAWLER: str = "filesystem-crawler"

# Fallback scope used when no prefix matches
DEFAULT_SCOPE_REF: str = "omninode/shared"

# Mapping of path prefix patterns to EnumContextSourceType
# Evaluated in order; first match wins.
_STATIC_STANDARDS_PREFIXES: tuple[str, ...] = (str(Path("~/.claude").expanduser()),)

# Intermediate directory names excluded when heuristically extracting repo names from paths
_SKIP_DIRS: frozenset[str] = frozenset({"src", "docs", "design", "plans", "handoffs"})


def _compute_sha256(content: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of content."""
    return hashlib.sha256(content).hexdigest()


def _detect_doc_type(path: Path) -> EnumDetectedDocType:
    """Classify the document type based on path heuristics.

    Rules are evaluated in priority order per the design spec §8.
    First match wins; no fallthrough.

    Args:
        path: Absolute path to the .md file.

    Returns:
        The detected document type enum value.
    """
    name = path.name
    name_upper = name.upper()
    parts_lower = [p.lower() for p in path.parts]

    # Exact-case match: CLAUDE.md is always capitalised by convention.
    if name == "CLAUDE.md":
        return EnumDetectedDocType.CLAUDE_MD

    if (
        name_upper == "DEEP_DIVE.MD"
        or name_upper.startswith("DEEP_DIVE_")
        or name_upper.endswith("_DEEP_DIVE.MD")
        # Files with DEEP_DIVE mid-name (e.g. MY_DEEP_DIVE_SUMMARY.MD) fall through to UNKNOWN_MD by design.
    ):
        return EnumDetectedDocType.DEEP_DIVE

    # DEEP_DIVE is checked first; files matching both patterns (e.g.
    # ARCHITECTURE_DEEP_DIVE.md) are classified as DEEP_DIVE by design.
    if name_upper.endswith(".MD") and (
        "ARCHITECTURE" in name_upper or "OVERVIEW" in name_upper
    ):
        return EnumDetectedDocType.ARCHITECTURE_DOC

    # Case-insensitive: README.md, readme.md, Readme.md are all valid across repos.
    if name.upper() == "README.MD":
        return EnumDetectedDocType.README

    if "design" in parts_lower:
        return EnumDetectedDocType.DESIGN_DOC

    if "plans" in parts_lower or "plan" in parts_lower:
        return EnumDetectedDocType.PLAN

    if "handoffs" in parts_lower or "handoff" in parts_lower:
        return EnumDetectedDocType.HANDOFF

    return EnumDetectedDocType.UNKNOWN_MD


def _source_type_for_path(path: Path) -> EnumContextSourceType:
    """Determine the context source type based on the file path.

    CLAUDE.md files and files under ~/.claude/ are authoritative policy
    documents (STATIC_STANDARDS). Everything else is REPO_DERIVED.

    Args:
        path: Absolute path to the .md file.

    Returns:
        The context source type enum value.
    """
    for prefix in _STATIC_STANDARDS_PREFIXES:
        if path.is_relative_to(Path(prefix)):
            return EnumContextSourceType.STATIC_STANDARDS

    if path.name == "CLAUDE.md":
        return EnumContextSourceType.STATIC_STANDARDS

    return EnumContextSourceType.REPO_DERIVED


def _priority_hint_for_path(path: Path, path_prefixes: list[str]) -> int:
    """Compute the initial priority hint (0-100) for a document.

    Follows the priority hint table in design doc §7.
    The scoring system adjusts these values over time.

    Args:
        path: Absolute path to the .md file.
        path_prefixes: Configured crawl path prefixes (as strings) used to
            determine whether a README is at the repo root level.

    Returns:
        An integer in [0, 100].
    """
    name_upper = path.name.upper()
    parts_lower = [p.lower() for p in path.parts]

    # ~/.claude/CLAUDE.md — global standards
    for prefix in _STATIC_STANDARDS_PREFIXES:
        if path.is_relative_to(Path(prefix)) and path.name == "CLAUDE.md":
            return 95

    # Any CLAUDE.md in a repo
    if path.name == "CLAUDE.md":
        return 85

    # Architecture or overview documents in a design/ directory
    if "design" in parts_lower and (
        "ARCHITECTURE" in name_upper or "OVERVIEW" in name_upper
    ):
        return 80

    # Design docs generally
    if "design" in parts_lower:
        return 70

    # Plans
    if "plans" in parts_lower or "plan" in parts_lower:
        return 65

    # Handoffs
    if "handoffs" in parts_lower or "handoff" in parts_lower:
        return 60

    # README at repo root: parent directory is one of the configured crawl prefixes
    if path.name.upper() == "README.MD" and str(path.parent) in path_prefixes:
        return 55

    # Deep dive reports
    if (
        name_upper == "DEEP_DIVE.MD"
        or name_upper.startswith("DEEP_DIVE_")
        or name_upper.endswith("_DEEP_DIVE.MD")
    ):
        return 45

    return 35


def _scope_ref_for_path(
    path: Path,
    scope_mappings: list[tuple[str, str]],
) -> str:
    """Resolve scope_ref using longest-prefix matching.

    Args:
        path: Absolute path to the file.
        scope_mappings: List of (path_prefix, scope_ref) pairs evaluated
            for longest-prefix match. Longest-prefix match wins.

    Returns:
        The matched scope_ref, or DEFAULT_SCOPE_REF if no prefix matches.
    """
    best_prefix_len = -1
    best_scope = DEFAULT_SCOPE_REF

    for prefix, scope in scope_mappings:
        prefix_path = Path(prefix)
        prefix_parts_len = len(prefix_path.parts)
        if path.is_relative_to(prefix_path) and prefix_parts_len > best_prefix_len:
            best_prefix_len = prefix_parts_len
            best_scope = scope

    return best_scope


def _extract_tags(path: Path, doc_type: EnumDetectedDocType) -> list[str]:
    """Build a minimal tag list for a discovered document.

    Tags enable FILE_TOUCHED_MATCH attribution signal computation in the
    tier system. Full tag extraction (code fence languages, service names,
    etc.) is delegated to ChunkClassifierCompute in Stream B.

    Args:
        path: Absolute path to the .md file.
        doc_type: Detected document type.

    Returns:
        A list of tag strings. Tags are structurally distinct (different
        prefixes), so duplicates cannot arise with the current logic.
    """
    tags: list[str] = []

    # File path tag for FILE_TOUCHED_MATCH attribution
    tags.append(str(path))

    # Document type tag
    tags.append(f"doctype:{doc_type.value}")

    # Repository name heuristic: last path component before .md file
    # that is not a common intermediate directory
    for part in reversed(path.parts[:-1]):
        if part not in _SKIP_DIRS and not part.startswith("."):
            tags.append(f"repo:{part}")
            break

    return tags


class HandlerFilesystemCrawler:
    """Walks configured path prefixes for .md files and emits crawl events.

    Implements the FilesystemCrawler described in the OmniMemory Document
    Ingestion Pipeline design (§5). Change detection uses a two-stage
    mtime -> SHA-256 strategy to avoid unnecessary reads.

    Attributes:
        _config: Crawler configuration including path prefixes and topics.
        _crawl_state_repo: Repository for loading and storing crawl state.
    """

    def __init__(
        self,
        config: ModelFilesystemCrawlerConfig,
        crawl_state_repo: ProtocolCrawlStateRepository,
    ) -> None:
        """Initialize the handler.

        Args:
            config: Crawler configuration.
            crawl_state_repo: Repository for crawl state persistence.
        """
        self._config = config
        self._crawl_state_repo = crawl_state_repo

        logger.info(
            "HandlerFilesystemCrawler initialized",
            extra={
                "handler": HANDLER_ID_FILESYSTEM_CRAWLER,
                "path_prefixes": config.path_prefixes,
                "max_file_size_bytes": config.max_file_size_bytes,
            },
        )

    async def crawl(
        self,
        correlation_id: UUID,
        crawl_scope: str,
        trigger_source: TriggerSource,
        env_prefix: str,
        publish_callback: Callable[
            [str, dict[str, object]], Coroutine[object, object, None]
        ],
        scope_mappings: list[tuple[str, str]] | None = None,
    ) -> ModelFilesystemCrawlResult:
        """Execute a full filesystem crawl and emit lifecycle events.

        Steps:
            1. Walk each path prefix for .md files.
            2. For each file: load prior state, apply mtime fast-path,
               compute SHA-256 if needed, emit discovered/changed event.
            3. After walk: compare state table to walked set and emit
               removed events for missing files.
            4. Persist updated state for all processed files.

        Args:
            correlation_id: Threaded from the originating crawl tick command.
            crawl_scope: Scope string from the crawl tick command.
            trigger_source: What triggered this crawl run.
            env_prefix: Deployment environment prefix (e.g. "dev").
            publish_callback: Async callback for publishing events.
                Signature: (full_topic, message_dict) -> Coroutine[..., None].
            scope_mappings: Optional list of (path_prefix, scope_ref) pairs
                for scope resolution. If None, DEFAULT_SCOPE_REF is used
                for all files.

        Returns:
            Summary statistics for this crawl run.
        """
        resolved_mappings = scope_mappings or []
        now_utc = datetime.now(timezone.utc)

        files_walked = 0
        discovered_count = 0
        changed_count = 0
        unchanged_count = 0
        skipped_count = 0
        mtime_skipped_count = 0
        indexed_count = 0
        error_count = 0
        truncated = False

        # Track absolute paths seen during this walk (for removal detection)
        walked_paths: set[str] = set()
        # Track all distinct scope_refs assigned during the walk so that
        # _detect_and_emit_removals can query every scope that may contain
        # stale records -- including non-default scopes from scope_mappings.
        scope_refs_seen: set[str] = set()

        # Validate that all path_prefixes are absolute before starting the walk.
        # Relative paths would resolve against the process CWD, producing
        # inconsistent source_ref values across environments.
        for prefix_str in self._config.path_prefixes:
            if not Path(prefix_str).is_absolute():
                raise ValueError(f"path_prefix must be absolute: {prefix_str!r}")

        for prefix_str in self._config.path_prefixes:
            prefix_path = Path(prefix_str)
            if not await asyncio.to_thread(prefix_path.exists):
                logger.warning(
                    "Path prefix does not exist, skipping",
                    extra={
                        "handler": HANDLER_ID_FILESYSTEM_CRAWLER,
                        "prefix": prefix_str,
                        "correlation_id": str(correlation_id),
                    },
                )
                continue

            file_glob = self._config.file_glob

            # Default-argument capture prevents the classic Python
            # closure-over-loop-variable bug: without _p=prefix_path and
            # _g=file_glob, all iterations would share the last value of
            # prefix_path/file_glob by the time _rglob_prefix is invoked.
            def _rglob_prefix(
                _p: Path = prefix_path, _g: str = file_glob
            ) -> list[Path]:
                # Collect matching paths one-by-one so that a PermissionError
                # or other OSError on an individual subdirectory does not abort
                # the entire prefix walk.
                results: list[Path] = []
                try:
                    for entry in _p.rglob(_g):
                        results.append(entry)
                except OSError as exc:
                    logger.warning(
                        "OSError during rglob, partial results returned",
                        extra={
                            "handler": HANDLER_ID_FILESYSTEM_CRAWLER,
                            "prefix": str(_p),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                return results

            resolved_prefix_path = await asyncio.to_thread(prefix_path.resolve)

            for md_path in await asyncio.to_thread(_rglob_prefix):
                if files_walked >= self._config.max_files_per_crawl:
                    truncated = True
                    logger.warning(
                        "max_files_per_crawl reached, crawl truncated",
                        extra={
                            "handler": HANDLER_ID_FILESYSTEM_CRAWLER,
                            "limit": self._config.max_files_per_crawl,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    break

                resolved_path = await asyncio.to_thread(md_path.resolve)

                # Symlink escape guard: reject any path that resolves outside
                # the configured prefix (e.g. a symlink pointing to /etc/passwd).
                if not resolved_path.is_relative_to(resolved_prefix_path):
                    logger.warning(
                        "Resolved path escapes crawl prefix (possible symlink"
                        " traversal), skipping",
                        extra={
                            "handler": HANDLER_ID_FILESYSTEM_CRAWLER,
                            "path": str(md_path),
                            "resolved_path": str(resolved_path),
                            "prefix": str(resolved_prefix_path),
                            "correlation_id": str(correlation_id),
                        },
                    )
                    skipped_count += 1
                    continue

                files_walked += 1

                abs_path_str = str(resolved_path)

                try:
                    # stat() is called on md_path (not resolved_path) so the
                    # logged path matches the rglob result. Path.stat() follows
                    # symlinks by default, so for a valid in-bounds symlink the
                    # stat is against the target.
                    stat = await asyncio.to_thread(md_path.stat)
                except OSError as exc:
                    logger.warning(
                        "Could not stat file, skipping",
                        extra={
                            "handler": HANDLER_ID_FILESYSTEM_CRAWLER,
                            "path": abs_path_str,
                            "error": str(exc),
                            "correlation_id": str(correlation_id),
                        },
                    )
                    error_count += 1
                    continue

                if stat.st_size > self._config.max_file_size_bytes:
                    logger.warning(
                        "File exceeds max_file_size_bytes, skipping",
                        extra={
                            "handler": HANDLER_ID_FILESYSTEM_CRAWLER,
                            "path": abs_path_str,
                            "size_bytes": stat.st_size,
                            "max_bytes": self._config.max_file_size_bytes,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    skipped_count += 1
                    continue

                # Only mark as walked after a successful stat and size check —
                # a file deleted between rglob and stat, or a file that exceeds
                # max_file_size_bytes, must not be treated as "seen". Size-exceeded
                # files are treated as non-existent so that previously-indexed
                # records for them trigger removal events on the next crawl.
                walked_paths.add(abs_path_str)

                scope_ref = _scope_ref_for_path(resolved_path, resolved_mappings)
                # Note: scope_refs_seen only contains scopes with at least one
                # successfully stat'd and size-checked file. If all files in a
                # scope fail stat checks, that scope never enters scope_refs_seen
                # and stale state records for it will not be queried for removal
                # in _detect_and_emit_removals. This is a known limitation.
                scope_refs_seen.add(scope_ref)

                prior_state = await self._crawl_state_repo.get_state(
                    source_ref=abs_path_str,
                    crawler_type=EnumCrawlerType.FILESYSTEM,
                    scope_ref=scope_ref,
                )

                # mtime fast-path: skip if unchanged
                current_mtime = stat.st_mtime
                if (
                    prior_state is not None
                    and prior_state.last_known_mtime is not None
                    and prior_state.last_known_mtime == current_mtime
                ):
                    # File has not been modified at the OS level; skip.
                    # Update last_crawled_at_utc to reflect we checked it.
                    updated_state = ModelCrawlStateRecord(
                        source_ref=abs_path_str,
                        crawler_type=EnumCrawlerType.FILESYSTEM,
                        scope_ref=scope_ref,
                        content_fingerprint=prior_state.content_fingerprint,
                        source_version=prior_state.source_version,
                        last_crawled_at_utc=now_utc,
                        last_changed_at_utc=prior_state.last_changed_at_utc,
                        last_known_mtime=current_mtime,
                    )
                    await self._crawl_state_repo.upsert_state(updated_state)
                    mtime_skipped_count += 1
                    continue

                # mtime changed (or no prior state) -- read and hash content
                content = await _read_file_async(md_path)
                if content is None:
                    error_count += 1
                    continue

                fingerprint = _compute_sha256(content)
                blob_ref = f"sha256:{fingerprint}"
                # errors='replace' substitutes U+FFFD for invalid byte sequences,
                # which may over-estimate token count for binary files accidentally
                # matched by the glob (e.g. a .md symlink pointing to a binary).
                # This is intentional: the estimate is approximate and non-fatal.
                token_estimate = len(content.decode("utf-8", errors="replace")) // 4
                doc_type = _detect_doc_type(resolved_path)
                source_type = _source_type_for_path(resolved_path)
                priority = _priority_hint_for_path(
                    resolved_path, self._config.path_prefixes
                )
                tags = _extract_tags(resolved_path, doc_type)

                if prior_state is None:
                    # New document
                    discovered_event = ModelDocumentDiscoveredEvent(
                        correlation_id=correlation_id,
                        emitted_at_utc=now_utc,
                        crawler_type=EnumCrawlerType.FILESYSTEM,
                        crawl_scope=crawl_scope,
                        trigger_source=trigger_source,
                        source_ref=abs_path_str,
                        source_type=source_type,
                        source_version=None,
                        content_fingerprint=fingerprint,
                        content_blob_ref=blob_ref,
                        token_estimate=token_estimate,
                        scope_ref=scope_ref,
                        detected_doc_type=doc_type,
                        tags=tags,
                        priority_hint=priority,
                    )
                    await _publish_event(
                        publish_callback,
                        f"{env_prefix}.{self._config.publish_topic_discovered}",
                        discovered_event.model_dump(mode="json"),
                        correlation_id,
                    )
                    discovered_count += 1

                    indexed_event = ModelDocumentIndexedEvent(
                        correlation_id=correlation_id,
                        emitted_at_utc=now_utc,
                        crawler_type=EnumCrawlerType.FILESYSTEM,
                        crawl_scope=crawl_scope,
                        trigger_source=trigger_source,
                        source_ref=abs_path_str,
                        source_type=source_type,
                        content_fingerprint=fingerprint,
                        scope_ref=scope_ref,
                    )
                    await _publish_event(
                        publish_callback,
                        f"{env_prefix}.{self._config.publish_topic_indexed}",
                        indexed_event.model_dump(mode="json"),
                        correlation_id,
                    )
                    indexed_count += 1

                    new_state = ModelCrawlStateRecord(
                        source_ref=abs_path_str,
                        crawler_type=EnumCrawlerType.FILESYSTEM,
                        scope_ref=scope_ref,
                        content_fingerprint=fingerprint,
                        source_version=None,
                        last_crawled_at_utc=now_utc,
                        last_changed_at_utc=now_utc,
                        last_known_mtime=current_mtime,
                    )
                    await self._crawl_state_repo.upsert_state(new_state)

                elif prior_state.content_fingerprint == fingerprint:
                    # mtime bumped but content unchanged -- update crawl time + mtime
                    unchanged_count += 1
                    updated_state = ModelCrawlStateRecord(
                        source_ref=abs_path_str,
                        crawler_type=EnumCrawlerType.FILESYSTEM,
                        scope_ref=scope_ref,
                        content_fingerprint=fingerprint,
                        source_version=prior_state.source_version,
                        last_crawled_at_utc=now_utc,
                        last_changed_at_utc=prior_state.last_changed_at_utc,
                        last_known_mtime=current_mtime,
                    )
                    await self._crawl_state_repo.upsert_state(updated_state)

                else:
                    # Content changed
                    changed_event = ModelDocumentChangedEvent(
                        correlation_id=correlation_id,
                        emitted_at_utc=now_utc,
                        crawler_type=EnumCrawlerType.FILESYSTEM,
                        crawl_scope=crawl_scope,
                        trigger_source=trigger_source,
                        source_ref=abs_path_str,
                        source_type=source_type,
                        source_version=None,
                        content_fingerprint=fingerprint,
                        content_blob_ref=blob_ref,
                        token_estimate=token_estimate,
                        scope_ref=scope_ref,
                        detected_doc_type=doc_type,
                        tags=tags,
                        priority_hint=priority,
                        previous_content_fingerprint=prior_state.content_fingerprint,
                        previous_source_version=prior_state.source_version,
                    )
                    await _publish_event(
                        publish_callback,
                        f"{env_prefix}.{self._config.publish_topic_changed}",
                        changed_event.model_dump(mode="json"),
                        correlation_id,
                    )
                    changed_count += 1

                    indexed_event = ModelDocumentIndexedEvent(
                        correlation_id=correlation_id,
                        emitted_at_utc=now_utc,
                        crawler_type=EnumCrawlerType.FILESYSTEM,
                        crawl_scope=crawl_scope,
                        trigger_source=trigger_source,
                        source_ref=abs_path_str,
                        source_type=source_type,
                        content_fingerprint=fingerprint,
                        scope_ref=scope_ref,
                    )
                    await _publish_event(
                        publish_callback,
                        f"{env_prefix}.{self._config.publish_topic_indexed}",
                        indexed_event.model_dump(mode="json"),
                        correlation_id,
                    )
                    indexed_count += 1

                    updated_state = ModelCrawlStateRecord(
                        source_ref=abs_path_str,
                        crawler_type=EnumCrawlerType.FILESYSTEM,
                        scope_ref=scope_ref,
                        content_fingerprint=fingerprint,
                        source_version=None,
                        last_crawled_at_utc=now_utc,
                        last_changed_at_utc=now_utc,
                        last_known_mtime=current_mtime,
                    )
                    await self._crawl_state_repo.upsert_state(updated_state)

            if truncated:
                break

        # Detect removals: state table entries not found in the walk.
        # Skip when truncated — walked_paths is a subset of disk, so removal
        # detection would emit spurious events for unvisited files.
        if truncated:
            logger.warning(
                "crawl truncated at max_files_per_crawl; skipping removal"
                " detection to avoid spurious removals",
                extra={
                    "handler": HANDLER_ID_FILESYSTEM_CRAWLER,
                    "max_files_per_crawl": self._config.max_files_per_crawl,
                    "correlation_id": str(correlation_id),
                },
            )
            removed_count = 0
        else:
            removed_count = await self._detect_and_emit_removals(
                walked_paths=walked_paths,
                scope_refs_seen=scope_refs_seen,
                correlation_id=correlation_id,
                emitted_at_utc=now_utc,
                crawl_scope=crawl_scope,
                trigger_source=trigger_source,
                env_prefix=env_prefix,
                publish_callback=publish_callback,
                resolved_mappings=resolved_mappings,
            )

        result = ModelFilesystemCrawlResult(
            files_walked=files_walked,
            discovered_count=discovered_count,
            changed_count=changed_count,
            unchanged_count=unchanged_count,
            skipped_count=skipped_count,
            mtime_skipped_count=mtime_skipped_count,
            indexed_count=indexed_count,
            removed_count=removed_count,
            error_count=error_count,
            truncated=truncated,
        )

        logger.info(
            "Filesystem crawl complete",
            extra={
                "handler": HANDLER_ID_FILESYSTEM_CRAWLER,
                "correlation_id": str(correlation_id),
                "files_walked": files_walked,
                "discovered": discovered_count,
                "changed": changed_count,
                "unchanged": unchanged_count,
                "skipped": skipped_count,
                "indexed": indexed_count,
                "removed": removed_count,
                "errors": error_count,
                "truncated": truncated,
            },
        )

        return result

    async def _detect_and_emit_removals(
        self,
        walked_paths: set[str],
        scope_refs_seen: set[str],
        correlation_id: UUID,
        emitted_at_utc: datetime,
        crawl_scope: str,
        trigger_source: TriggerSource,
        env_prefix: str,
        publish_callback: Callable[
            [str, dict[str, object]], Coroutine[object, object, None]
        ],
        resolved_mappings: list[tuple[str, str]],
    ) -> int:
        """Emit document-removed events for state records no longer on disk.

        Loads all state records for each scope encountered during the walk and
        emits removed events for any record whose source_ref was not visited.
        Using scope_refs_seen (collected per file during the walk) ensures that
        files assigned non-default scopes via scope_mappings are also checked.

        When scope_refs_seen is non-empty, we also include DEFAULT_SCOPE_REF
        if any configured prefix has no explicit scope_ref mapping (i.e. would
        resolve to DEFAULT_SCOPE_REF). This prevents a gap where prefixes that
        produce zero files during a crawl are never queried for removals because
        they never contributed a scope_ref to scope_refs_seen.

        Args:
            walked_paths: Set of absolute path strings visited this run.
            scope_refs_seen: Set of all scope_ref values assigned to files
                during the walk, collected as each file is processed.
            correlation_id: Correlation ID for event envelope.
            emitted_at_utc: UTC datetime when the crawl run started.
            crawl_scope: Scope string from the originating crawl tick command.
            trigger_source: What triggered the crawl run.
            env_prefix: Environment prefix for topic construction.
            publish_callback: Async publish callback.
            resolved_mappings: Scope mappings for source_type resolution.

        Returns:
            Number of document-removed events emitted.
        """
        removed_count = 0

        # Use the scope_refs seen during the walk: this covers both the
        # default scope and any non-default scopes assigned via scope_mappings.
        # Fall back to prefix-level scopes if the walk produced no scope refs
        # (e.g. all prefixes were missing / no files found).
        if scope_refs_seen:
            affected_scopes = set(scope_refs_seen)
            # Also include DEFAULT_SCOPE_REF if any configured prefix has no
            # explicit scope_ref mapping. Prefixes without explicit mappings
            # resolve to DEFAULT_SCOPE_REF; if they produced zero files this
            # run they never added DEFAULT_SCOPE_REF to scope_refs_seen, so
            # stale records under DEFAULT_SCOPE_REF would be missed.
            for prefix_str in self._config.path_prefixes:
                prefix_path = Path(prefix_str)
                if (
                    _scope_ref_for_path(prefix_path, resolved_mappings)
                    == DEFAULT_SCOPE_REF
                ):
                    affected_scopes.add(DEFAULT_SCOPE_REF)
                    break
        else:
            affected_scopes = set()
            for prefix_str in self._config.path_prefixes:
                prefix_path = Path(prefix_str)
                scope = _scope_ref_for_path(prefix_path, resolved_mappings)
                affected_scopes.add(scope)

        for scope_ref in affected_scopes:
            known_records = await self._crawl_state_repo.list_states_for_scope(
                crawler_type=EnumCrawlerType.FILESYSTEM,
                scope_ref=scope_ref,
            )

            for record in known_records:
                if record.source_ref not in walked_paths:
                    source_type = _source_type_for_path(Path(record.source_ref))
                    removed_event = ModelDocumentRemovedEvent(
                        correlation_id=correlation_id,
                        emitted_at_utc=emitted_at_utc,
                        crawler_type=EnumCrawlerType.FILESYSTEM,
                        crawl_scope=crawl_scope,
                        trigger_source=trigger_source,
                        source_ref=record.source_ref,
                        source_type=source_type,
                        scope_ref=scope_ref,
                        last_known_content_fingerprint=record.content_fingerprint,
                        last_known_source_version=record.source_version,
                    )
                    await _publish_event(
                        publish_callback,
                        f"{env_prefix}.{self._config.publish_topic_removed}",
                        removed_event.model_dump(mode="json"),
                        correlation_id,
                    )
                    await self._crawl_state_repo.delete_state(
                        source_ref=record.source_ref,
                        crawler_type=EnumCrawlerType.FILESYSTEM,
                        scope_ref=scope_ref,
                    )
                    removed_count += 1

        return removed_count


async def _read_file_async(path: Path) -> bytes | None:
    """Read a file's raw bytes asynchronously, returning None on error.

    Uses asyncio.to_thread() to offload the blocking file read to a thread
    pool, ensuring the event loop is not blocked for the duration of the I/O.

    Args:
        path: Path to read.

    Returns:
        Raw file bytes, or None if the read fails.
    """
    try:
        return await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        logger.warning(
            "Failed to read file content",
            extra={"path": str(path), "error": str(exc)},
        )
        return None


async def _publish_event(
    publish_callback: Callable[
        [str, dict[str, object]], Coroutine[object, object, None]
    ],
    topic: str,
    payload: dict[str, object],
    correlation_id: UUID,
) -> None:
    """Await the async publish callback, swallowing publish errors non-fatally.

    Crawl progress must not be blocked by a single publish failure.
    Errors are logged for observability.

    Args:
        publish_callback: Caller-supplied async publish function.
        topic: Full topic name (env prefix already applied).
        payload: Serialized event dict.
        correlation_id: Used in error log for traceability.
    """
    try:
        await publish_callback(topic, payload)
    except Exception as exc:
        logger.error(
            "Failed to publish crawl event",
            extra={
                "topic": topic,
                "correlation_id": str(correlation_id),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
