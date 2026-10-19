"""UCL token definitions.

UCL = Unified Code Language. A C++-inspired, object-capable systems language
with AI-native low-level primitives (an ASM/C "overlay" for AI data).
This module is the single source of truth for the lexical vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class T(Enum):
    # literals
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    IDENT = auto()
    # keywords
    FN = auto()
    LET = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    TRUE = auto()
    FALSE = auto()
    CLASS = auto()
    BREAK = auto()
    CONTINUE = auto()
    MATCH = auto()
    CASE = auto()
    UNDERSCORE = auto()  # _ (wildcard pattern)
    # type keywords (incl. AI-native primitives)
    TYPE = auto()
    # punctuation / operators
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    SEMI = auto()
    COLON = auto()
    ARROW = auto()       # ->
    FAT_ARROW = auto()   # =>
    DOTDOTDOT = auto()   # ...
    ASSIGN = auto()      # =
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    PIPE = auto()        # |
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    EQ = auto()          # ==
    NE = auto()          # !=
    AND = auto()         # &&
    OR = auto()          # ||
    NOT = auto()         # !
    DOT = auto()
    EOF = auto()


KEYWORDS = {
    "fn": T.FN,
    "let": T.LET,
    "return": T.RETURN,
    "if": T.IF,
    "else": T.ELSE,
    "while": T.WHILE,
    "for": T.FOR,
    "in": T.IN,
    "true": T.TRUE,
    "false": T.FALSE,
    "class": T.CLASS,
    "break": T.BREAK,
    "continue": T.CONTINUE,
    "match": T.MATCH,
    "case": T.CASE,
}

# Built-in type names. The AI-native primitives are what make UCL "unified for
# AI data": they are first-class low-level types, not library wrappers.
TYPE_NAMES = {
    # scalar / C-like
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
    "f16", "f32", "f64", "bool", "void", "str", "ptr",
    # AI-native primitives (the UCL differentiator)
    "tensor", "weights", "tokenstream", "graph", "quant",
}


@dataclass
class Token:
    type: T
    value: object
    line: int
    col: int

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"
