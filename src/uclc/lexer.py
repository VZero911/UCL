"""UCL lexer: source text -> token stream.

Hand-written scanner (no regex engine dependency) so it stays portable and
easy to lower to a native scanner later. Supports // and /* */ comments,
int/float/string literals, identifiers, keywords, type names and operators.
"""
from __future__ import annotations

from typing import List

from .tokens import KEYWORDS, TYPE_NAMES, T, Token


class LexError(SyntaxError):
    pass


_TWO_CHAR = {
    "->": T.ARROW,
    "=>": T.FAT_ARROW,
    "==": T.EQ,
    "!=": T.NE,
    "<=": T.LE,
    ">=": T.GE,
    "&&": T.AND,
    "||": T.OR,
}
_ONE_CHAR = {
    "(": T.LPAREN, ")": T.RPAREN,
    "{": T.LBRACE, "}": T.RBRACE,
    "[": T.LBRACKET, "]": T.RBRACKET,
    ",": T.COMMA, ";": T.SEMI, ":": T.COLON,
    "=": T.ASSIGN, "+": T.PLUS, "-": T.MINUS,
    "*": T.STAR, "/": T.SLASH, "%": T.PERCENT,
    "<": T.LT, ">": T.GT, "!": T.NOT, ".": T.DOT, "|": T.PIPE,
}


class Lexer:
    def __init__(self, src: str):
        self.src = src
        self.i = 0
        self.line = 1
        self.col = 1

    def _peek(self, k: int = 0) -> str:
        j = self.i + k
        return self.src[j] if j < len(self.src) else ""

    def _advance(self) -> str:
        c = self.src[self.i]
        self.i += 1
        if c == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return c

    def tokenize(self) -> List[Token]:
        toks: List[Token] = []
        while self.i < len(self.src):
            c = self._peek()
            if c in " \t\r\n":
                self._advance()
                continue
            # comments
            if c == "/" and self._peek(1) == "/":
                while self.i < len(self.src) and self._peek() != "\n":
                    self._advance()
                continue
            if c == "/" and self._peek(1) == "*":
                self._advance(); self._advance()
                while self.i < len(self.src) and not (self._peek() == "*" and self._peek(1) == "/"):
                    self._advance()
                if self.i >= len(self.src):
                    raise LexError(f"unterminated block comment at line {self.line}")
                self._advance(); self._advance()
                continue
            line, col = self.line, self.col
            if c.isdigit() or (c == "." and self._peek(1).isdigit()):
                toks.append(self._number(line, col))
                continue
            if c == "_" and not (self._peek(1).isalnum() or self._peek(1) == "_"):
                self._advance()
                toks.append(Token(T.UNDERSCORE, "_", line, col))
                continue
            if c.isalpha() or c == "_":
                toks.append(self._ident(line, col))
                continue
            if c == '"':
                toks.append(self._string(line, col))
                continue
            # three-char token: ...
            if c == "." and self._peek(1) == "." and self._peek(2) == ".":
                self._advance(); self._advance(); self._advance()
                toks.append(Token(T.DOTDOTDOT, "...", line, col))
                continue
            two = c + self._peek(1)
            if two in _TWO_CHAR:
                self._advance(); self._advance()
                toks.append(Token(_TWO_CHAR[two], two, line, col))
                continue
            if c in _ONE_CHAR:
                self._advance()
                toks.append(Token(_ONE_CHAR[c], c, line, col))
                continue
            raise LexError(f"unexpected character {c!r} at line {line}:{col}")
        toks.append(Token(T.EOF, None, self.line, self.col))
        return toks

    def _number(self, line: int, col: int) -> Token:
        start = self.i
        is_float = False
        while self._peek().isdigit():
            self._advance()
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()
            while self._peek().isdigit():
                self._advance()
        if self._peek() in ("e", "E"):
            is_float = True
            self._advance()
            if self._peek() in ("+", "-"):
                self._advance()
            if not self._peek().isdigit():
                raise LexError(f"malformed exponent at line {line}:{col}")
            while self._peek().isdigit():
                self._advance()
        text = self.src[start:self.i]
        if is_float:
            return Token(T.FLOAT, float(text), line, col)
        return Token(T.INT, int(text), line, col)

    def _ident(self, line: int, col: int) -> Token:
        start = self.i
        while self._peek().isalnum() or self._peek() == "_":
            self._advance()
        text = self.src[start:self.i]
        if text in KEYWORDS:
            return Token(KEYWORDS[text], text, line, col)
        if text in TYPE_NAMES:
            return Token(T.TYPE, text, line, col)
        return Token(T.IDENT, text, line, col)

    def _string(self, line: int, col: int) -> Token:
        self._advance()  # opening quote
        out = []
        while self.i < len(self.src) and self._peek() != '"':
            c = self._advance()
            if c == "\\":
                esc = self._advance()
                out.append({"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}.get(esc, esc))
            else:
                out.append(c)
        if self.i >= len(self.src):
            raise LexError(f"unterminated string at line {line}:{col}")
        self._advance()  # closing quote
        return Token(T.STRING, "".join(out), line, col)


def tokenize(src: str) -> List[Token]:
    return Lexer(src).tokenize()
