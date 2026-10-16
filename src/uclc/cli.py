"""uclc command-line driver.

Usage:
    uclc run     <file.ucl>     run the program (calls main)
    uclc tokens  <file.ucl>     dump the token stream
    uclc ast     <file.ucl>     dump the parsed AST
    uclc emit-ir <file.ucl>     lower to textual low-level IR
    uclc check   <file.ucl>     parse only; report OK or the first error
"""
from __future__ import annotations

import sys
from typing import List

from . import __version__
from .lexer import tokenize
from .parser import parse
from .ir import lower
from .vm import Interpreter


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    if argv[0] in ("-v", "--version", "version"):
        print(f"uclc {__version__}")
        return 0
    if len(argv) < 2:
        print("error: missing <file.ucl>", file=sys.stderr)
        return 2
    cmd, path = argv[0], argv[1]
    try:
        src = _read(path)
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return 2

    try:
        if cmd == "tokens":
            for t in tokenize(src):
                print(t)
            return 0
        if cmd == "check":
            parse(tokenize(src))
            print("OK")
            return 0
        if cmd == "ast":
            prog = parse(tokenize(src))
            for d in prog.decls:
                print(d)
            return 0
        if cmd == "emit-ir":
            print(lower(parse(tokenize(src))))
            return 0
        if cmd == "run":
            prog = parse(tokenize(src))
            interp = Interpreter()
            result = interp.run(prog)
            if result is not None:
                return int(result) if isinstance(result, int) else 0
            return 0
    except SyntaxError as e:
        print(f"compile error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # runtime
        print(f"runtime error: {e}", file=sys.stderr)
        return 1

    print(f"error: unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
