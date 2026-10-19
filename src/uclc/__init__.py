"""uclc — the UCL (Unified Code Language) reference compiler toolchain.

Pipeline: source(.ucl) -> lexer -> parser -> AST -> IR lowering -> optimizer -> VM / native codegen.
v0.11 adds IR optimizer (constant folding, propagation, strength reduction, DCE).
"""
from __future__ import annotations

from .lexer import tokenize
from .parser import parse
from .ir import lower
from .optimizer import optimize
from .vm import run, Interpreter
from .checker import check
from .codegen import compile_source_to_elf, compile_to_elf

__version__ = "0.11.0"
__all__ = [
    "tokenize", "parse", "lower", "optimize", "run", "check", "Interpreter",
    "compile_source_to_elf", "compile_to_elf", "__version__",
]


def compile_source(src: str):
    """Front-end: source text -> AST."""
    return parse(tokenize(src))
