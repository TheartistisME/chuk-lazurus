"""Package-neutral shared constants.

These constants are used by both CLI and introspection code.  They live at the
package root so torch-only imports can access them without importing the
`chuk_lazarus.introspection` package and triggering heavyweight runtime
dependencies.
"""

from __future__ import annotations

from enum import Enum


class LayerPhase(str, Enum):
    """Layer phase classifications for MoE analysis."""

    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


class LayerPhaseDefaults:
    """Default layer boundaries for phase classification."""

    EARLY_END: int = 8
    MIDDLE_END: int = 16


class PatternCategory(str, Enum):
    """Pattern categories for MoE trigram analysis."""

    ARITHMETIC = "arithmetic"
    CODE = "code"
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    ANALOGY = "analogy"
    HYPERNYM = "hypernym"
    COMPARISON = "comparison"
    CAUSATION = "causation"
    CONDITIONAL = "conditional"
    QUESTION = "question"
    NEGATION = "negation"
    TEMPORAL = "temporal"
    QUANTIFICATION = "quantification"
    CONTEXT_SWITCH = "context_switch"
    POSITION = "position"
    COORDINATION = "coordination"


class Domain(str, Enum):
    """Domain categories for expert analysis."""

    MATH = "math"
    CODE = "code"
    LANGUAGE = "language"
    REASONING = "reasoning"


class TokenType(str, Enum):
    """Semantic token type classifications for MoE analysis."""

    NUM = "NUM"
    OP = "OP"
    BR = "BR"
    PN = "PN"
    QUOTE = "QUOTE"
    KW = "KW"
    BOOL = "BOOL"
    TYPE = "TYPE"
    VAR = "VAR"
    SYN = "SYN"
    ANT = "ANT"
    AS = "AS"
    TO = "TO"
    CAUSE = "CAUSE"
    COND = "COND"
    THAN = "THAN"
    QW = "QW"
    ANS = "ANS"
    NEG = "NEG"
    TIME = "TIME"
    QUANT = "QUANT"
    COMP = "COMP"
    COORD = "COORD"
    NOUN = "NOUN"
    ADJ = "ADJ"
    VERB = "VERB"
    FUNC = "FUNC"
    CAP = "CAP"
    CW = "CW"
    WS = "WS"


__all__ = [
    "Domain",
    "LayerPhase",
    "LayerPhaseDefaults",
    "PatternCategory",
    "TokenType",
]
