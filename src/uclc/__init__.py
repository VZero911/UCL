"""uclc — the UCL (Unified Code Language) reference compiler toolchain.

Pipeline: source(.ucl) -> lexer -> parser -> AST -> IR lowering -> evaluator.
Native/bytecode backends are on the roadmap; v0 ships a tested front-end plus a
reference tree-walking evaluator.
"""
from __future__ import annotations

from .lexer import tokenize
from .parser import parse
from .ir import lower
from .vm import run, Interpreter
from .checker import check

__version__ = "0.2.0"
__all__ = ["tokenize", "parse", "lower", "run", "check", "Interpreter", "__version__"]


def compile_source(src: str):
    """Front-end: source text -> AST."""
    return parse(tokenize(src))
