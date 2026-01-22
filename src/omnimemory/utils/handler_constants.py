# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Constants for semantic compute handler operations.

This module contains internal constants used by the semantic compute handler
for entity extraction, complexity scoring, and text analysis. These constants
are implementation details and should not be exported from the handlers package.

Constants are organized into logical groups:
    - Entity extraction: Stopwords and confidence thresholds
    - Complexity scoring: Normalization parameters for text metrics

.. note::
    These constants are internal implementation details. They may change
    without notice between versions. Do not depend on them externally.

.. versionadded:: 0.1.0
    Extracted from handler_semantic_compute.py for maintainability (OMN-1390).
"""

from __future__ import annotations

# =============================================================================
# Entity Extraction Constants
# =============================================================================

KEY_CONCEPT_CONFIDENCE_THRESHOLD: float = 0.8
"""Confidence threshold for promoting entities to key concepts.

Higher than the general entity confidence threshold to ensure only
high-confidence entities are included as key concepts in analysis results.

Used by:
    HandlerSemanticCompute._extract_key_concepts()
"""

SENTENCE_STARTING_STOPWORDS: frozenset[str] = frozenset(
    {
        # -------------------------------------------------------------------------
        # Articles and determiners
        # -------------------------------------------------------------------------
        "The",
        "A",
        "An",
        # -------------------------------------------------------------------------
        # Demonstratives
        # -------------------------------------------------------------------------
        "This",
        "That",
        "These",
        "Those",
        # -------------------------------------------------------------------------
        # Pronouns
        # -------------------------------------------------------------------------
        "It",
        "He",
        "She",
        "We",
        "They",
        "I",
        "You",
        # -------------------------------------------------------------------------
        # Common sentence starters (conjunctive adverbs and transitions)
        # -------------------------------------------------------------------------
        "However",
        "Therefore",
        "Furthermore",
        "Moreover",
        "Nevertheless",
        "Meanwhile",
        "Additionally",
        "Consequently",
        "Subsequently",
        "Otherwise",
        "Accordingly",
        "Similarly",
        "Likewise",
        "Indeed",
        "Hence",
        "Thus",
        "Regardless",
        "Nonetheless",
        "Notwithstanding",
        # -------------------------------------------------------------------------
        # Question words (interrogatives)
        # -------------------------------------------------------------------------
        "What",
        "When",
        "Where",
        "Who",
        "Why",
        "How",
        "Which",
        # -------------------------------------------------------------------------
        # Other common sentence starters
        # -------------------------------------------------------------------------
        "There",
        "Here",
        "If",
        "As",
        "So",
        "But",
        "And",
        "Or",
        "Yet",
        "For",
        "Nor",
        "After",
        "Before",
        "Because",
        "Although",
        "While",
        "Since",
        "Until",
        "Unless",
        "Once",
        "Now",
        "Then",
        "Also",
        "First",
        "Second",
        "Third",
        "Finally",
        "Next",
        "Last",
        "Many",
        "Most",
        "Some",
        "All",
        "Any",
        "Each",
        "Every",
        "Both",
        "Few",
        "Several",
        "Such",
        "No",
        "Not",
        "Only",
        "Just",
        "Even",
        "Still",
        "Already",
        # -------------------------------------------------------------------------
        # Adverbial starters
        # -------------------------------------------------------------------------
        "Recently",
        "Currently",
        "Actually",
        "Basically",
        "Essentially",
        "Generally",
        "Usually",
        "Obviously",
        "Clearly",
        "Certainly",
        "Probably",
        "Perhaps",
        "Maybe",
        "Possibly",
        # -------------------------------------------------------------------------
        # Temporal starters (common in sentence-start context, not entity names)
        # -------------------------------------------------------------------------
        "Today",
        "Tomorrow",
        "Yesterday",
    }
)
"""Common words that start sentences but are not named entities.

Used by the heuristic entity extraction to filter out false positives.
These words frequently appear at the start of sentences due to English
grammar patterns, not because they are proper nouns or named entities.

Categories:
    - Articles and determiners (The, A, An)
    - Demonstratives (This, That, These, Those)
    - Pronouns (It, He, She, We, They, I, You)
    - Conjunctive adverbs (However, Therefore, Furthermore, etc.)
    - Question words (What, When, Where, Who, Why, How, Which)
    - Quantifiers (Many, Most, Some, All, Any, Each, Every, etc.)
    - Temporal markers (Today, Tomorrow, Yesterday)
    - Adverbial starters (Recently, Currently, Actually, etc.)

.. note::
    This list is intentionally conservative. Proper nouns that happen to
    match these words (e.g., "The Beatles") will have "The" filtered,
    but "Beatles" will still be captured as an entity.

Used by:
    HandlerSemanticCompute._extract_entities_heuristic()
"""

TOPIC_EXTRACTION_STOPWORDS: frozenset[str] = frozenset(
    {
        # -------------------------------------------------------------------------
        # Articles and determiners
        # -------------------------------------------------------------------------
        "the",
        "a",
        "an",
        # -------------------------------------------------------------------------
        # Verbs: be forms
        # -------------------------------------------------------------------------
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        # -------------------------------------------------------------------------
        # Verbs: have forms
        # -------------------------------------------------------------------------
        "have",
        "has",
        "had",
        # -------------------------------------------------------------------------
        # Verbs: do forms
        # -------------------------------------------------------------------------
        "do",
        "does",
        "did",
        # -------------------------------------------------------------------------
        # Modal verbs
        # -------------------------------------------------------------------------
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        # -------------------------------------------------------------------------
        # Prepositions
        # -------------------------------------------------------------------------
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        # -------------------------------------------------------------------------
        # Conjunctions
        # -------------------------------------------------------------------------
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        # -------------------------------------------------------------------------
        # Adverbs and modifiers
        # -------------------------------------------------------------------------
        "not",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        # -------------------------------------------------------------------------
        # Demonstratives and pronouns
        # -------------------------------------------------------------------------
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
    }
)
"""Common English stopwords for topic extraction.

Used by the heuristic topic extraction to filter out high-frequency words
that carry little semantic meaning for topic identification. These words
are filtered BEFORE frequency analysis to improve topic quality.

All words are lowercase since topic extraction normalizes text to lowercase
before processing.

Categories:
    - Articles and determiners (the, a, an)
    - Verbs: be forms (is, are, was, were, be, been, being)
    - Verbs: have forms (have, has, had)
    - Verbs: do forms (do, does, did)
    - Modal verbs (will, would, could, should, may, might, must, etc.)
    - Prepositions (to, of, in, for, on, with, at, by, from, etc.)
    - Conjunctions (and, but, or, nor, so, yet, both, either, neither)
    - Adverbs and modifiers (not, only, own, same, than, too, very, just)
    - Demonstratives and pronouns (this, that, these, those, it, its)

.. note::
    Unlike SENTENCE_STARTING_STOPWORDS which uses title case for
    sentence-start detection, this set uses lowercase since topic
    extraction normalizes all text to lowercase.

Used by:
    HandlerSemanticCompute._extract_topics_heuristic()
"""

# =============================================================================
# Complexity Score Constants
# =============================================================================
# These constants control the normalization of text complexity metrics.
# The formula is: score = (value - MIN) / RANGE, clamped to [0, 1].

COMPLEXITY_WORD_LEN_MIN: int = 2
"""Minimum word length for complexity normalization.

Words shorter than this length contribute 0 to word complexity.
Typical English words average approximately 5 characters.

Used by:
    HandlerSemanticCompute._compute_complexity_score()
"""

COMPLEXITY_WORD_LEN_RANGE: int = 8
"""Range for word length complexity normalization.

Words with length >= (MIN + RANGE) contribute 1.0 to word complexity.
With MIN=2 and RANGE=8, words of 10+ characters are fully complex.

Used by:
    HandlerSemanticCompute._compute_complexity_score()
"""

COMPLEXITY_SENTENCE_LEN_MIN: int = 5
"""Minimum sentence length (in words) for complexity normalization.

Sentences shorter than this word count contribute 0 to sentence complexity.
Typical English sentences average 15-20 words.

Used by:
    HandlerSemanticCompute._compute_complexity_score()
"""

COMPLEXITY_SENTENCE_LEN_RANGE: int = 30
"""Range for sentence length complexity normalization.

Sentences with word count >= (MIN + RANGE) contribute 1.0 to sentence complexity.
With MIN=5 and RANGE=30, sentences of 35+ words are fully complex.

Used by:
    HandlerSemanticCompute._compute_complexity_score()
"""

__all__ = [
    "KEY_CONCEPT_CONFIDENCE_THRESHOLD",
    "SENTENCE_STARTING_STOPWORDS",
    "TOPIC_EXTRACTION_STOPWORDS",
    "COMPLEXITY_WORD_LEN_MIN",
    "COMPLEXITY_WORD_LEN_RANGE",
    "COMPLEXITY_SENTENCE_LEN_MIN",
    "COMPLEXITY_SENTENCE_LEN_RANGE",
]
