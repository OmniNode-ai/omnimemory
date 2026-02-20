"""
Crawler type enumeration for the document ingestion pipeline.

Identifies the subsystem responsible for discovering and emitting
document events. Used as a discriminator in crawl state tracking,
Kafka topic routing, and per-source debounce configuration.

Design doc: DESIGN_OMNIMEMORY_DOCUMENT_INGESTION_PIPELINE.md §4
Ticket: OMN-2426
"""

from enum import Enum


class EnumCrawlerType(str, Enum):
    """Identifies the crawler subsystem that produced a crawl event.

    Values are persisted in the ``omnimemory_crawl_state`` table
    (``crawler_type`` column) so must remain stable once deployed.
    """

    FILESYSTEM = "filesystem"
    """Walks configured directory trees on the local filesystem."""

    GIT_REPO = "git_repo"
    """Tracks changes via ``git log`` / ``git diff`` for versioned repos."""

    LINEAR = "linear"
    """Fetches issues and documents from the Linear MCP API."""

    WATCHDOG = "watchdog"
    """Receives real-time FSEvents/inotify notifications for critical files."""
