"""UCL parser: token stream -> AST (recursive descent, Pratt-style binary ops).

Covers the v0 language subset: top-level `fn` and `class` declarations;
let/assign/return/if/while/expr statements; full expression grammar with
C-like precedence. Generic type arguments (e.g. tensor<f32,[N,M]>) are parsed
and skipped to the base type name for now (full type system is roadmap).
"""
from __future__ import annotations

from typing import List, Optional

from . import ast
from .tokens import T, Token


class ParseError(SyntaxError):
    pass


# binary operator precedence (higher binds tighter)
_PREC = {
    T.OR: 1,
    T.AND: 2,
    T.EQ: 3, T.NE: 3,
    T.LT: 4, T.GT: 4, T.LE: 4, T.GE: 4,
    T.PLUS: 5, T.MINUS: 5,
    T.STAR: 6, T.SLASH: 6, T.PERCENT: 6,
}
_OP_TEXT = {
    T.OR: "||", T.AND: "&&", T.EQ: "==", T.NE: "!=",
    T.LT: "<", T.GT: ">", T.LE: "<=", T.GE: ">=",
    T.PLUS: "+", T.MINUS: "-", T.STAR: "*", T.SLASH: "/", T.PERCENT: "%",
}


class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.i = 0

    # -- token helpers ------------------------------------------------------
    def _peek(self, k: int = 0) -> Token:
        j = min(self.i + k, len(self.toks) - 1)
        return self.toks[j]

    def _at(self, tt: T) -> bool:
        return self._peek().type == tt

    def _advance(self) -> Token:
        t = self.toks[self.i]
        if self.i < len(self.toks) - 1:
            self.i += 1
        return t

    def _expect(self, tt: T) -> Token:
        if not self._at(tt):
            got = self._peek()
            raise ParseError(f"expected {tt.name}, got {got.type.name} ({got.value!r}) at line {got.line}:{got.col}")
        return self._advance()

    def _match(self, tt: T) -> bool:
        if self._at(tt):
            self._advance()
            return True
        return False

    # -- entry --------------------------------------------------------------
    def parse(self) -> ast.Program:
        decls: List[ast.Node] = []
        while not self._at(T.EOF):
            if self._at(T.FN):
                decls.append(self._fn())
            elif self._at(T.CLASS):
                decls.append(self._class())
            else:
                t = self._peek()
                raise ParseError(f"expected 'fn' or 'class' at top level, got {t.type.name} at line {t.line}")
        return ast.Program(decls)

    # -- type ---------------------------------------------------------------
    def _type(self) -> str:
        if self._at(T.TYPE) or self._at(T.IDENT):
            base = self._advance().value
        else:
            t = self._peek()
            raise ParseError(f"expected a type at line {t.line}:{t.col}")
        # optional generic args: skip balanced <...>
        if self._at(T.LT):
            depth = 0
            while True:
                tk = self._advance()
                if tk.type == T.LT:
                    depth += 1
                elif tk.type == T.GT:
                    depth -= 1
                    if depth == 0:
                        break
                elif tk.type == T.EOF:
                    raise ParseError("unterminated generic type argument list")
        return base

    # -- declarations -------------------------------------------------------
    def _fn(self) -> ast.Fn:
        self._expect(T.FN)
        name = self._expect(T.IDENT).value
        self._expect(T.LPAREN)
        params: List[ast.Param] = []
        if not self._at(T.RPAREN):
            params.append(self._param())
            while self._match(T.COMMA):
                params.append(self._param())
        self._expect(T.RPAREN)
        ret: Optional[str] = None
        if self._match(T.ARROW):
            ret = self._type()
        body = self._block()
        return ast.Fn(name, params, ret, body)

    def _param(self) -> ast.Param:
        # accepts `name: type` or C-style `type name`
        if self._at(T.IDENT) and self._peek(1).type == T.COLON:
            name = self._advance().value
            self._expect(T.COLON)
            ty = self._type()
            return ast.Param(name, ty)
        ty = self._type()
        name = self._expect(T.IDENT).value
        return ast.Param(name, ty)

    def _class(self) -> ast.Class:
        self._expect(T.CLASS)
        name = self._expect(T.IDENT).value
        self._expect(T.LBRACE)
        fields: List[ast.Field] = []
        methods: List[ast.Fn] = []
        while not self._at(T.RBRACE) and not self._at(T.EOF):
            if self._at(T.FN):
                methods.append(self._fn())
            else:
                # field: `type name;` or `name: type;`
                if self._at(T.IDENT) and self._peek(1).type == T.COLON:
                    fname = self._advance().value
                    self._expect(T.COLON)
                    fty = self._type()
                else:
                    fty = self._type()
                    fname = self._expect(T.IDENT).value
                self._expect(T.SEMI)
                fields.append(ast.Field(fname, fty))
        self._expect(T.RBRACE)
        return ast.Class(name, fields, methods)

    # -- statements ---------------------------------------------------------
    def _block(self) -> ast.Block:
        self._expect(T.LBRACE)
        stmts: List[ast.Node] = []
        while not self._at(T.RBRACE) and not self._at(T.EOF):
            stmts.append(self._stmt())
        self._expect(T.RBRACE)
        return ast.Block(stmts)

    def _stmt(self) -> ast.Node:
        if self._at(T.LET):
            return self._let()
        if self._at(T.RETURN):
            self._advance()
            val = None if self._at(T.SEMI) else self._expr()
            self._expect(T.SEMI)
            return ast.Return(val)
        if self._at(T.IF):
            return self._if()
        if self._at(T.WHILE):
            return self._while()
        if self._at(T.LBRACE):
            return self._block()
        # assignment or expression statement
        if self._at(T.IDENT) and self._peek(1).type == T.ASSIGN:
            name = self._advance().value
            self._expect(T.ASSIGN)
            val = self._expr()
            self._expect(T.SEMI)
            return ast.Assign(name, val)
        expr = self._expr()
        self._expect(T.SEMI)
        return ast.ExprStmt(expr)

    def _let(self) -> ast.Let:
        self._expect(T.LET)
        name = self._expect(T.IDENT).value
        ty: Optional[str] = None
        if self._match(T.COLON):
            ty = self._type()
        self._expect(T.ASSIGN)
        val = self._expr()
        self._expect(T.SEMI)
        return ast.Let(name, ty, val)

    def _if(self) -> ast.If:
        self._expect(T.IF)
        cond = self._expr()
        then = self._block()
        otherwise: Optional[ast.Block] = None
        if self._match(T.ELSE):
            otherwise = self._block() if self._at(T.LBRACE) else ast.Block([self._if()])
        return ast.If(cond, then, otherwise)

    def _while(self) -> ast.While:
        self._expect(T.WHILE)
        cond = self._expr()
        body = self._block()
        return ast.While(cond, body)

    # -- expressions (precedence climbing) ----------------------------------
    def _expr(self, min_prec: int = 0) -> ast.Node:
        left = self._unary()
        while True:
            tt = self._peek().type
            prec = _PREC.get(tt)
            if prec is None or prec < min_prec:
                break
            self._advance()
            right = self._expr(prec + 1)
            left = ast.Binary(_OP_TEXT[tt], left, right)
        return left

    def _unary(self) -> ast.Node:
        if self._at(T.MINUS) or self._at(T.NOT):
            op = self._advance().value
            return ast.Unary(op, self._unary())
        return self._postfix()

    def _postfix(self) -> ast.Node:
        # member access and method calls: obj.field / obj.method(args) / chained
        node = self._primary()
        while self._at(T.DOT):
            self._advance()
            name = self._expect(T.IDENT).value
            if self._at(T.LPAREN):
                self._advance()
                args: List[ast.Node] = []
                if not self._at(T.RPAREN):
                    args.append(self._expr())
                    while self._match(T.COMMA):
                        args.append(self._expr())
                self._expect(T.RPAREN)
                node = ast.MethodCall(node, name, args)
            else:
                node = ast.Member(node, name)
        return node

    def _primary(self) -> ast.Node:
        t = self._peek()
        if t.type == T.INT:
            self._advance(); return ast.IntLit(t.value)
        if t.type == T.FLOAT:
            self._advance(); return ast.FloatLit(t.value)
        if t.type == T.STRING:
            self._advance(); return ast.StrLit(t.value)
        if t.type == T.TRUE:
            self._advance(); return ast.BoolLit(True)
        if t.type == T.FALSE:
            self._advance(); return ast.BoolLit(False)
        if t.type == T.LPAREN:
            self._advance()
            e = self._expr()
            self._expect(T.RPAREN)
            return e
        if t.type in (T.IDENT, T.TYPE):
            self._advance()
            if self._at(T.LPAREN):
                self._advance()
                args: List[ast.Node] = []
                if not self._at(T.RPAREN):
                    args.append(self._expr())
                    while self._match(T.COMMA):
                        args.append(self._expr())
                self._expect(T.RPAREN)
                return ast.Call(t.value, args)
            return ast.Var(t.value)
        raise ParseError(f"unexpected token {t.type.name} ({t.value!r}) at line {t.line}:{t.col}")


def parse(tokens: List[Token]) -> ast.Program:
    return Parser(tokens).parse()
