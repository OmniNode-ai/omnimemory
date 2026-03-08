#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""ONEX Kafka Import Boundary Enforcement (ARCH-002).

Ensures that direct Kafka consumer imports (AIOKafkaConsumer, KafkaConsumer,
aiokafka, kafka-python) do not appear in src/omnimemory/nodes/. This enforces
ARCH-002: "Runtime owns all Kafka plumbing".

Nodes must consume events through the abstract EventBus SPI provided by the
runtime layer, never by instantiating Kafka consumers directly.

Exemption Annotation:
    If a file legitimately needs a direct Kafka import in nodes/ (rare),
    add a comment with the explicit exemption annotation on the SAME LINE
    as the import:

        from aiokafka import AIOKafkaConsumer  # omnimemory-kafka-exempt: <reason>

    This annotation is intentionally specific to prevent accidental matches.

Usage:
    python scripts/validation/validate_kafka_imports.py src/omnimemory/nodes/
    python scripts/validation/validate_kafka_imports.py src/

Added for OMN-1750: CI Kafka import lint guard (ARCH-002 enforcement)
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


# Kafka consumer import patterns to detect.
# Each regex uses \b (word boundary) after the package name to prevent false
# positives on prefixed names like `kafka_utils` or `aiokafka_wrapper` (where
# `_` is a word character so \b does NOT match), while still catching dotted
# submodule imports like `kafka.errors` (where `.` is NOT a word character so
# \b DOES match).
KAFKA_IMPORT_PATTERNS = [
    (
        # from aiokafka import AIOKafkaConsumer (or any other import)
        re.compile(r"^from\s+aiokafka\b"),
        "Direct 'from aiokafka' import found - use EventBus SPI instead (ARCH-002)",
    ),
    (
        # import aiokafka
        re.compile(r"^import\s+aiokafka\b"),
        "Direct 'import aiokafka' found - use EventBus SPI instead (ARCH-002)",
    ),
    (
        # from kafka import KafkaConsumer (or any other import)
        re.compile(r"^from\s+kafka\b"),
        "Direct 'from kafka' import found - use EventBus SPI instead (ARCH-002)",
    ),
    (
        # import kafka
        re.compile(r"^import\s+kafka\b"),
        "Direct 'import kafka' found - use EventBus SPI instead (ARCH-002)",
    ),
    (
        # from confluent_kafka import Consumer (or any other import)
        re.compile(r"^from\s+confluent_kafka\b"),
        "Direct 'from confluent_kafka' import found - use EventBus SPI instead (ARCH-002)",
    ),
    (
        # import confluent_kafka
        re.compile(r"^import\s+confluent_kafka\b"),
        "Direct 'import confluent_kafka' found - use EventBus SPI instead (ARCH-002)",
    ),
]

# Only scan omnimemory/nodes/ directory - ARCH-002 applies to node code specifically.
# The runtime layer (which IS allowed to use Kafka directly) lives elsewhere.
# Using the fully-qualified path avoids false positives on other packages' nodes/.
ENFORCED_PATHS = [
    "omnimemory/nodes/",
]

# Pattern to detect TYPE_CHECKING guard blocks (both bare and qualified forms)
TYPE_CHECKING_GUARD = re.compile(r"^\s*(?:if|elif)\s+(?:typing\.)?TYPE_CHECKING\b")

# Skip patterns (lines that should be ignored)
SKIP_PATTERNS = [
    # Explicit Kafka boundary exemption annotation
    # Format: # omnimemory-kafka-exempt: <reason>
    re.compile(r"#\s*omnimemory-kafka-exempt:", re.IGNORECASE),
]


def is_in_enforced_path(filepath: Path) -> bool:
    """Check if the file path is within an enforced directory for Kafka lint."""
    filepath_posix = filepath.as_posix()
    for enforced in ENFORCED_PATHS:
        if f"/{enforced}" in filepath_posix or filepath_posix.startswith(enforced):
            return True
    return False


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file for Kafka import violations."""
    # Only enforce in nodes/ directory
    if not is_in_enforced_path(filepath):
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
        if TYPE_CHECKING_GUARD.search(line):
            in_type_checking_block = True
            indent_level = len(line) - len(line.lstrip())
            continue

        # Exit TYPE_CHECKING block when indentation returns to the same level.
        # This correctly handles the common `if TYPE_CHECKING: ... else: ...`
        # pattern: the `else:` branch contains runtime imports that SHOULD be
        # checked, so exiting the TYPE_CHECKING block here is the right
        # behavior. The only theoretical limitation is a contrived `else:`
        # branch that itself contains only type-checking-time imports, but
        # that pattern does not occur in practice and would be a code smell
        # regardless. This trade-off keeps the heuristic simple and correct
        # for all real-world usage.
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

        # Check for Kafka import patterns
        for pattern, message in KAFKA_IMPORT_PATTERNS:
            if pattern.search(stripped):
                violations.append(
                    Violation(
                        str(filepath),
                        line_num,
                        f"{message}. "
                        f"Nodes must use the EventBus SPI from the runtime layer.",
                    )
                )
                break  # Only one violation per line

    return violations


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Enforce ARCH-002 Kafka import boundary - detect direct Kafka "
            "consumer imports in nodes/"
        ),
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

    # Sort for deterministic output across runs
    files_to_check = sorted(set(files_to_check))

    all_violations: list[Violation] = []
    for filepath in files_to_check:
        violations = validate_file(filepath)
        all_violations.extend(violations)

    if all_violations:
        print(f"Found {len(all_violations)} Kafka import boundary violation(s):")
        print()
        for v in all_violations:
            print(f"  {v.file}:{v.line}:")
            print(f"    {v.message}")
            print()
        print("ARCH-002: Runtime owns all Kafka plumbing.")
        print("Nodes must use the EventBus SPI (event_bus.subscribe/publish)")
        print("instead of importing Kafka clients directly.")
        return 1

    print(f"No Kafka import violations found. Checked {len(files_to_check)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
