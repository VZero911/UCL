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

__version__ = "0.1.0"
__all__ = ["tokenize", "parse", "lower", "run", "Interpreter", "__version__"]


def compile_source(src: str):
    """Front-end: source text -> AST."""
    return parse(tokenize(src))
