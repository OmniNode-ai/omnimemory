#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""ONEX HTTP Import Boundary Enforcement.

Ensures that direct HTTP client imports (httpx, requests, urllib3) are only
allowed in the adapter layer. This enforces the contract boundary principle
that all external HTTP calls must go through the adapter layer.

Allowed locations for HTTP imports:
- src/omnimemory/handlers/adapters/ (the contract boundary)
- src/omnimemory/nodes/*/clients/ (legacy clients - being migrated)
- tests/ (test mocks and fixtures)

All other locations must use the adapter layer for HTTP operations.

Exemption Annotation:
    If a file legitimately needs direct HTTP imports outside the allowed
    locations (rare), add a comment with the explicit exemption annotation
    on the SAME LINE as the import:

        import httpx  # omnimemory-http-exempt: <reason>

    Example:
        import httpx  # omnimemory-http-exempt: Low-level HTTP utility for adapter testing

    This annotation is intentionally specific to prevent accidental matches
    with unrelated comments containing words like "adapter".

Usage:
    python scripts/validation/validate_http_imports.py src/
    python scripts/validation/validate_http_imports.py src/omnimemory/

Added for OMN-1391: Embedding client adapters + rate limiting
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import NamedTuple

# Configure logging - default to WARNING so scripts are quiet unless debugging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class Violation(NamedTuple):
    """A validation violation."""

    file: str
    line: int
    message: str


# HTTP client import patterns to detect
HTTP_IMPORT_PATTERNS = [
    (
        # import httpx
        re.compile(r"^import\s+httpx\b"),
        "Direct 'import httpx' found - use adapter layer instead",
    ),
    (
        # from httpx import ...
        re.compile(r"^from\s+httpx\b"),
        "Direct 'from httpx' import found - use adapter layer instead",
    ),
    (
        # import requests
        re.compile(r"^import\s+requests\b"),
        "Direct 'import requests' found - use adapter layer instead",
    ),
    (
        # from requests import ...
        re.compile(r"^from\s+requests\b"),
        "Direct 'from requests' import found - use adapter layer instead",
    ),
    (
        # import urllib3
        re.compile(r"^import\s+urllib3\b"),
        "Direct 'import urllib3' found - use adapter layer instead",
    ),
    (
        # from urllib3 import ...
        re.compile(r"^from\s+urllib3\b"),
        "Direct 'from urllib3' import found - use adapter layer instead",
    ),
    (
        # import aiohttp
        re.compile(r"^import\s+aiohttp\b"),
        "Direct 'import aiohttp' found - use adapter layer instead",
    ),
    (
        # from aiohttp import ...
        re.compile(r"^from\s+aiohttp\b"),
        "Direct 'from aiohttp' import found - use adapter layer instead",
    ),
]

# Paths where HTTP imports ARE allowed (the contract boundary)
ALLOWED_PATHS = [
    # Adapter layer - the only allowed exit hatch
    "handlers/adapters/",
    # Legacy clients directory - allowed during migration
    # TODO: Remove this allowance after migration to adapters is complete
    "nodes/memory_retrieval_effect/clients/",
    # Kreuzberg effect client - HTTP transport boundary for document extraction (OMN-2733)
    "nodes/kreuzberg_parse_effect/clients/",
    # Tests can mock HTTP clients
    "tests/",
]

# Skip patterns (lines that should be ignored)
SKIP_PATTERNS = [
    # TYPE_CHECKING blocks are fine (type hints only, not runtime)
    re.compile(r"if\s+TYPE_CHECKING"),
    # Explicit HTTP boundary exemption annotation
    # Format: # omnimemory-http-exempt: <reason>
    # This is intentionally specific to prevent accidental matches with
    # unrelated comments like "# We use adapter pattern here"
    re.compile(r"#\s*omnimemory-http-exempt:", re.IGNORECASE),
]


def is_path_allowed(filepath: Path) -> bool:
    """Check if the file path is in an allowed location for HTTP imports."""
    # Use posix path for consistent segment matching across platforms
    filepath_posix = filepath.as_posix()
    # Check each allowed path as a proper path segment (not substring)
    for allowed in ALLOWED_PATHS:
        # Ensure we match complete path segments by checking for / boundaries
        if f"/{allowed}" in filepath_posix or filepath_posix.startswith(allowed):
            return True
    return False


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file for HTTP import violations."""
    # Skip files in allowed locations
    if is_path_allowed(filepath):
        return []

    try:
        content = filepath.read_text(encoding="utf-8")
    except PermissionError:
        logging.warning("Permission denied reading file: %s", filepath)
        return []
    except FileNotFoundError:
        logging.warning("File not found (possibly deleted during scan): %s", filepath)
        return []
    except OSError as e:
        logging.warning("OS error reading file %s: %s", filepath, e)
        return []
    except UnicodeDecodeError as e:
        logging.warning("Unicode decode error in file %s: %s", filepath, e)
        return []

    violations: list[Violation] = []
    in_type_checking_block = False
    indent_level = 0

    for line_num, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()

        # Track TYPE_CHECKING blocks (imports there are fine)
        if "if TYPE_CHECKING" in line:
            in_type_checking_block = True
            indent_level = len(line) - len(line.lstrip())
            continue

        # Exit TYPE_CHECKING block when indentation decreases
        if in_type_checking_block and stripped and not stripped.startswith("#"):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                in_type_checking_block = False

        # Skip if in TYPE_CHECKING block
        if in_type_checking_block:
            continue

        # Skip if line matches skip patterns
        if any(skip.search(line) for skip in SKIP_PATTERNS):
            continue

        # Check for HTTP import patterns
        for pattern, message in HTTP_IMPORT_PATTERNS:
            if pattern.search(stripped):
                violations.append(
                    Violation(
                        str(filepath),
                        line_num,
                        f"{message}. "
                        f"HTTP imports are only allowed in: {', '.join(ALLOWED_PATHS)}",
                    )
                )
                break  # Only one violation per line

    return violations


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enforce HTTP import boundary - detect direct HTTP client imports"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src/"],
        help="Files or directories to check",
    )
    args = parser.parse_args()

    files_to_check: list[Path] = []

    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == ".py":
            files_to_check.append(path)
        elif path.is_dir():
            files_to_check.extend(path.rglob("*.py"))
        else:
            logging.warning("Skipping invalid path: %s", path)

    all_violations: list[Violation] = []
    for filepath in files_to_check:
        violations = validate_file(filepath)
        all_violations.extend(violations)

    if all_violations:
        print(f"Found {len(all_violations)} HTTP import boundary violation(s):")
        print()
        for v in all_violations:
            print(f"  {v.file}:{v.line}:")
            print(f"    {v.message}")
            print()
        print("To fix: Use EmbeddingHttpClient or other adapters from")
        print("omnimemory.handlers.adapters instead of direct HTTP imports.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
