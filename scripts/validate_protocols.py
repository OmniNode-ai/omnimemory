#!/usr/bin/env python3
"""Protocol validation CLI wrapper for omnimemory repository."""

import argparse
import sys
from pathlib import Path

# Import from omnibase_core
from omnibase_core.validation import (
    ProtocolMigrator,
    audit_protocols,
    check_against_spi,
)


def main():
    parser = argparse.ArgumentParser(
        description="Validate protocols in omnimemory repository"
    )
    parser.add_argument(
        "--mode",
        choices=["audit", "spi-check", "migration-plan"],
        default="audit",
        help="Validation mode",
    )
    parser.add_argument(
        "--spi-path", default="../omnibase_spi", help="Path to omnibase_spi"
    )

    args = parser.parse_args()

    if args.mode == "audit":
        result = audit_protocols(".")
        if result.success:
            print(
                f"✅ Repository validation passed: "
                f"{result.protocols_found} protocols found"
            )
        else:
            print("❌ Repository validation failed:")
            for violation in result.violations:
                print(f"   • {violation}")
            sys.exit(1)

    elif args.mode == "spi-check":
        if not Path(args.spi_path).exists():
            print(f"⚠️  SPI path not found: {args.spi_path}")
            print("✅ Skipping SPI duplication check")
            return

        result = check_against_spi(".", args.spi_path)
        if result.success:
            print("✅ No duplicates found with SPI")
        else:
            print("⚠️  Duplicates found with SPI:")
            for dup in result.exact_duplicates:
                print(f"   • {dup.protocols[0].name}")
            sys.exit(1)

    elif args.mode == "migration-plan":
        if not Path(args.spi_path).exists():
            print(f"⚠️  SPI path not found: {args.spi_path}")
            print("Cannot create migration plan without SPI")
            sys.exit(1)

        migrator = ProtocolMigrator(".", args.spi_path)
        plan = migrator.create_migration_plan()
        migrator.print_migration_plan(plan)

        if not plan.can_proceed():
            sys.exit(1)


if __name__ == "__main__":
    main()
