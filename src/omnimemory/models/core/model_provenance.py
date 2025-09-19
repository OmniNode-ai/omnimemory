"""
Provenance tracking model following ONEX standards.
"""

from .model_provenance_chain import ModelProvenanceChain

# Import and re-export models for backwards compatibility
from .model_provenance_entry import ModelProvenanceEntry

__all__ = ["ModelProvenanceEntry", "ModelProvenanceChain"]
