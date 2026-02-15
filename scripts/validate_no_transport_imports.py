#!/usr/bin/env python3
"""AST-based validator for transport/I/O library imports in omnimemory.

This script enforces the architectural boundary defined in ARCH-002:
Nodes never touch Kafka directly. Runtime owns all Kafka plumbing.

Unlike grep-based validators (validate_kafka_imports.py), this script uses
Python's AST module to correctly detect and allow imports inside
TYPE_CHECKING blocks, which are legal since they create no runtime
dependencies.

This is the AST-based counterpart to validate_kafka_imports.py which uses
regex patterns. Both enforce ARCH-002 but this script provides more accurate
detection by parsing the actual Python syntax tree.

Usage:
    python scripts/validate_no_transport_imports.py
    python scripts/validate_no_transport_imports.py --verbose
    python scripts/validate_no_transport_imports.py --exclude src/omnimemory/runtime

Exit codes:
    0 = no violations
    1 = violations found

Reference: omniintelligence PR #67 (transport import guard)
Added for OMN-2218: Phase 7 CI Infrastructure Alignment
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Banned transport/I/O modules that cannot be imported at runtime in omnimemory nodes
# These create runtime dependencies on external I/O libraries
# Per ARCH-002: Nodes declare intent via contracts, runtime owns all Kafka plumbing
BANNED_MODULES: frozenset[str] = frozenset(
    {
        # HTTP clients
        "aiohttp",
        "httpx",
        "requests",
        "urllib3",
        # Kafka clients
        "kafka",
        "aiokafka",
        "confluent_kafka",
        # Redis clients
        "redis",
        "aioredis",
        # Database clients
        "asyncpg",
        "psycopg2",
        "psycopg",
        "aiomysql",
        # Message queues
        "pika",
        "aio_pika",
        "kombu",
        "celery",
        # gRPC (import name is "grpc", not "grpcio" which is the PyPI package name)
        "grpc",
        # WebSocket
        "websockets",
        "wsproto",
    }
)

# Directories to skip during traversal (standard Python/build artifacts)
SKIP_DIRECTORIES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        ".env",
        "build",
        "dist",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".eggs",
    }
)

# Directory suffixes that should be skipped (e.g., "foo.egg-info" matches ".egg-info")
SKIP_DIRECTORY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".egg-info",
    }
)


@dataclass(frozen=True)
class Violation:
    """Represents a banned import violation."""

    file_path: Path
    line_number: int
    module_name: str
    import_statement: str

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_number}: Banned transport import: {self.module_name}"


@dataclass(frozen=True)
class FileProcessingError:
    """Represents an error encountered while processing a file.

    These are non-fatal warnings that indicate a file could not be fully processed,
    but should not fail the overall validation run (errors do not cause exit code 1).
    """

    file_path: Path
    error_type: str
    error_message: str

    def __str__(self) -> str:
        return f"{self.file_path}: [{self.error_type}] {self.error_message}"


class TransportImportChecker(ast.NodeVisitor):
    """AST visitor that detects banned transport imports outside TYPE_CHECKING blocks.

    This visitor tracks:
    1. Imports of TYPE_CHECKING (direct or aliased like `import typing as t`)
    2. Entry/exit from TYPE_CHECKING guarded blocks
    3. All import statements, flagging those importing banned modules at runtime
    """

    def __init__(self, source_code: str) -> None:
        self.source_lines = source_code.splitlines()
        self.violations: list[Violation] = []
        self._in_type_checking_block = False
        self._type_checking_module_aliases: set[str] = set()
        self._type_checking_constant_aliases: set[str] = set()

    def _get_source_line(self, lineno: int) -> str:
        """Get the source line for a given line number (1-indexed)."""
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def _extract_root_module(self, module_name: str) -> str:
        """Extract the root module from a potentially dotted module path."""
        return module_name.split(".")[0]

    def _is_type_checking_guard(self, node: ast.If) -> bool:
        """Detect if an If node is a TYPE_CHECKING guard.

        Handles:
        - `if TYPE_CHECKING:` (direct import)
        - `if TC:` (when `from typing import TYPE_CHECKING as TC`)
        - `if typing.TYPE_CHECKING:` (module-qualified)
        - `if t.TYPE_CHECKING:` (when `import typing as t`)
        """
        test = node.test

        if isinstance(test, ast.Name) and (
            test.id == "TYPE_CHECKING"
            or test.id in self._type_checking_constant_aliases
        ):
            return True

        if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
            if isinstance(test.value, ast.Name):
                if (
                    test.value.id == "typing"
                    or test.value.id in self._type_checking_module_aliases
                ):
                    return True
            return False

        return False

    def visit_Import(self, node: ast.Import) -> None:
        """Handle `import X` and `import X as Y` statements."""
        for alias in node.names:
            if alias.name == "typing" and alias.asname:
                self._type_checking_module_aliases.add(alias.asname)

        if not self._in_type_checking_block:
            for alias in node.names:
                root_module = self._extract_root_module(alias.name)
                if root_module in BANNED_MODULES:
                    self.violations.append(
                        Violation(
                            file_path=Path(),  # Will be set by caller
                            line_number=node.lineno,
                            module_name=root_module,
                            import_statement=self._get_source_line(node.lineno),
                        )
                    )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle `from X import Y` statements."""
        if node.level > 0:
            self.generic_visit(node)
            return

        if node.module is None:
            self.generic_visit(node)
            return

        for alias in node.names:
            if alias.name == "TYPE_CHECKING":
                if alias.asname:
                    self._type_checking_constant_aliases.add(alias.asname)

        if not self._in_type_checking_block:
            root_module = self._extract_root_module(node.module)
            if root_module in BANNED_MODULES:
                self.violations.append(
                    Violation(
                        file_path=Path(),  # Will be set by caller
                        line_number=node.lineno,
                        module_name=root_module,
                        import_statement=self._get_source_line(node.lineno),
                    )
                )

        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """Handle If statements, detecting TYPE_CHECKING guards."""
        if self._is_type_checking_guard(node):
            old_state = self._in_type_checking_block
            self._in_type_checking_block = True
            for child in node.body:
                self.visit(child)
            self._in_type_checking_block = old_state
            for child in node.orelse:
                self.visit(child)
        else:
            self.generic_visit(node)


def iter_python_files(
    root_dir: Path, excludes: set[Path], *, verbose: bool = False
) -> Iterator[Path]:
    """Iterate over all Python files in a directory, skipping excluded paths."""
    for path in root_dir.rglob("*.py"):
        if any(skip_dir in path.parts for skip_dir in SKIP_DIRECTORIES):
            continue
        if any(
            part.endswith(suffix)
            for part in path.parts
            for suffix in SKIP_DIRECTORY_SUFFIXES
        ):
            continue

        should_exclude = False
        for exclude_path in excludes:
            try:
                path.relative_to(exclude_path)
                should_exclude = True
                break
            except ValueError:
                try:
                    path_parts = path.parts
                    exclude_parts = exclude_path.parts
                    exclude_len = len(exclude_parts)
                    for i in range(len(path_parts) - exclude_len + 1):
                        if path_parts[i : i + exclude_len] == exclude_parts:
                            should_exclude = True
                            break
                    if should_exclude:
                        break
                except (TypeError, AttributeError) as e:
                    if verbose:
                        print(
                            f"  [debug] Exclusion match error for {path} "
                            f"with exclude {exclude_path}: {e}",
                            file=sys.stderr,
                        )

        if not should_exclude:
            yield path


def check_file(
    file_path: Path,
) -> tuple[list[Violation], list[FileProcessingError]]:
    """Check a single Python file for banned transport imports."""
    violations: list[Violation] = []
    errors: list[FileProcessingError] = []

    try:
        source_code = file_path.read_text(encoding="utf-8")
    except PermissionError as e:
        errors.append(
            FileProcessingError(
                file_path=file_path,
                error_type="PermissionError",
                error_message=f"Cannot read file: {e}",
            )
        )
        return violations, errors
    except UnicodeDecodeError as e:
        errors.append(
            FileProcessingError(
                file_path=file_path,
                error_type="UnicodeDecodeError",
                error_message=f"File is not valid UTF-8 (possibly binary): {e}",
            )
        )
        return violations, errors
    except OSError as e:
        errors.append(
            FileProcessingError(
                file_path=file_path,
                error_type="OSError",
                error_message=f"Could not read file: {e}",
            )
        )
        return violations, errors

    if not source_code.strip():
        return violations, errors

    try:
        tree = ast.parse(source_code, filename=str(file_path))
    except SyntaxError as e:
        errors.append(
            FileProcessingError(
                file_path=file_path,
                error_type="SyntaxError",
                error_message=f"Invalid Python syntax: {e.msg} (line {e.lineno})",
            )
        )
        return violations, errors

    checker = TransportImportChecker(source_code)
    checker.visit(tree)

    for v in checker.violations:
        violations.append(
            Violation(
                file_path=file_path,
                line_number=v.line_number,
                module_name=v.module_name,
                import_statement=v.import_statement,
            )
        )

    return violations, errors


def main() -> int:
    """Main entry point for the transport import validator CLI."""
    parser = argparse.ArgumentParser(
        description="Validate no banned transport/I/O imports in omnimemory nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Banned modules:
  HTTP: aiohttp, httpx, requests, urllib3
  Kafka: kafka, aiokafka, confluent_kafka
  Redis: redis, aioredis
  Database: asyncpg, psycopg2, psycopg, aiomysql
  MQ: pika, aio_pika, kombu, celery
  gRPC: grpc
  WebSocket: websockets, wsproto

TYPE_CHECKING guarded imports are allowed.

Per ARCH-002: Nodes never touch Kafka directly. Runtime owns all Kafka plumbing.
""",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("src/omnimemory"),
        help="Source directory to scan (default: src/omnimemory)",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        dest="excludes",
        metavar="PATH",
        help="Exclude a file or directory (can be specified multiple times)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show import statement snippets for each violation",
    )

    args = parser.parse_args()

    src_dir = args.src_dir
    if not src_dir.exists():
        print(f"Error: Source directory does not exist: {src_dir}", file=sys.stderr)
        return 1

    if not src_dir.is_dir():
        print(f"Error: Source path is not a directory: {src_dir}", file=sys.stderr)
        return 1

    excludes = set(args.excludes)
    all_violations: list[Violation] = []
    all_errors: list[FileProcessingError] = []
    file_count = 0

    print(f"Checking for transport/I/O library imports in {src_dir}...")

    for file_path in iter_python_files(src_dir, excludes, verbose=args.verbose):
        file_count += 1
        violations, errors = check_file(file_path)
        all_violations.extend(violations)
        all_errors.extend(errors)

    if all_errors:
        print("\nWarnings (file processing errors):", file=sys.stderr)
        for err in all_errors:
            print(f"  {err}", file=sys.stderr)
        print(file=sys.stderr)

    if all_violations:
        print("\nERROR: Found transport/I/O library imports in omnimemory!")
        print()
        print("Violations:")
        for v in all_violations:
            print(f"  {v}")
            if args.verbose:
                print(f"    -> {v.import_statement}")
        print()
        print("Architectural Invariant: Nodes never touch Kafka directly.")
        print("Transport and I/O libraries belong in infrastructure layers.")
        print("Per ARCH-002: Runtime owns all Kafka plumbing.")
        print()
        print("Solutions:")
        print("  1. Define a protocol for the capability you need")
        print("  2. Implement the protocol in an infrastructure package")
        print("  3. Use TYPE_CHECKING guards for type-only imports")
        print()
        print(
            f"Total: {len(all_violations)} violation(s), "
            f"{len(all_errors)} error(s) in {file_count} files scanned"
        )
        return 1

    if all_errors:
        print(
            f"No transport/I/O library imports found in omnimemory "
            f"({file_count} files scanned, {len(all_errors)} file(s) could not be processed)"
        )
    else:
        print(
            f"No transport/I/O library imports found in omnimemory "
            f"({file_count} files scanned)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
