"""
Notes model following ONEX standards.
"""

# Import and re-export models for backwards compatibility
from .model_note import ModelNote
from .model_notes_collection import ModelNotesCollection

__all__ = ["ModelNote", "ModelNotesCollection"]
