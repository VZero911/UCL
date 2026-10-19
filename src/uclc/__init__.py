"""uclc — the UCL (Unified Code Language) reference compiler toolchain.

Pipeline: source(.ucl) -> lexer -> parser -> AST -> IR lowering -> VM / native codegen.
v0.10 adds register allocator for native codegen (linear scan, spill support).
"""
from __future__ import annotations

from .lexer import tokenize
from .parser import parse
from .ir import lower
from .vm import run, Interpreter
from .checker import check
from .codegen import compile_source_to_elf, compile_to_elf

__version__ = "0.10.0"
__all__ = [
    "tokenize", "parse", "lower", "run", "check", "Interpreter",
    "compile_source_to_elf", "compile_to_elf", "__version__",
]


def compile_source(src: str):
    """Front-end: source text -> AST."""
    return parse(tokenize(src))
