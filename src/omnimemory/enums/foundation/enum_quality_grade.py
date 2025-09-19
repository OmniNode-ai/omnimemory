"""
Quality grade enumeration for success metrics.
"""

from enum import Enum


class EnumQualityGrade(str, Enum):
    """Quality grade enumeration for overall quality assessment."""

    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C_PLUS = "C+"
    C = "C"
    D = "D"
    F = "F"
