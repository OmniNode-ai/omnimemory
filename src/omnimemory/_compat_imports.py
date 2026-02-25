# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Compatibility imports for ONEX type definitions.

This module provides type aliases that work correctly with mypy by using
simple type definitions that avoid conditional type alias issues.
"""

# Always import our local types
from omnimemory.enums.enum_error_code import EnumOmniMemoryErrorCode
from omnimemory.enums.enum_severity import EnumSeverity

# Define type aliases using the local types
# This avoids conditional type alias issues with mypy
# At runtime, these work with omnibase_core types via duck typing
type ErrorCodeType = EnumOmniMemoryErrorCode | str
type SeverityType = EnumSeverity

__all__ = ["ErrorCodeType", "SeverityType", "EnumOmniMemoryErrorCode", "EnumSeverity"]
