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

    @staticmethod
    def _span(node: ast.Node, tok: Token) -> ast.Node:
        node.line = tok.line
        node.col = tok.col
        return node

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
        prog = ast.Program(decls)
        prog.line = 1
        prog.col = 1
        return prog

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
        tok = self._expect(T.FN)
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
        return self._span(ast.Fn(name, params, ret, body), tok)

    def _param(self) -> ast.Param:
        # accepts `name: type` or C-style `type name`
        if self._at(T.IDENT) and self._peek(1).type == T.COLON:
            tok = self._advance()
            name = tok.value
            self._expect(T.COLON)
            ty = self._type()
            return self._span(ast.Param(name, ty), tok)
        tok = self._peek()
        ty = self._type()
        name = self._expect(T.IDENT).value
        return self._span(ast.Param(name, ty), tok)

    def _class(self) -> ast.Class:
        tok = self._expect(T.CLASS)
        name = self._expect(T.IDENT).value
        self._expect(T.LBRACE)
        fields: List[ast.Field] = []
        methods: List[ast.Fn] = []
        while not self._at(T.RBRACE) and not self._at(T.EOF):
            if self._at(T.FN):
                methods.append(self._fn())
            else:
                # field: `type name;` or `name: type;`
                ftok = self._peek()
                if self._at(T.IDENT) and self._peek(1).type == T.COLON:
                    fname = self._advance().value
                    self._expect(T.COLON)
                    fty = self._type()
                else:
                    fty = self._type()
                    fname = self._expect(T.IDENT).value
                self._expect(T.SEMI)
                fields.append(self._span(ast.Field(fname, fty), ftok))
        self._expect(T.RBRACE)
        return self._span(ast.Class(name, fields, methods), tok)

    # -- statements ---------------------------------------------------------
    def _block(self) -> ast.Block:
        tok = self._expect(T.LBRACE)
        stmts: List[ast.Node] = []
        while not self._at(T.RBRACE) and not self._at(T.EOF):
            stmts.append(self._stmt())
        self._expect(T.RBRACE)
        return self._span(ast.Block(stmts), tok)

    def _stmt(self) -> ast.Node:
        if self._at(T.LET):
            return self._let()
        if self._at(T.RETURN):
            tok = self._advance()
            val = None if self._at(T.SEMI) else self._expr()
            self._expect(T.SEMI)
            return self._span(ast.Return(val), tok)
        if self._at(T.IF):
            return self._if()
        if self._at(T.WHILE):
            return self._while()
        if self._at(T.FOR):
            return self._for_in()
        if self._at(T.BREAK):
            tok = self._advance()
            self._expect(T.SEMI)
            return self._span(ast.Break(), tok)
        if self._at(T.CONTINUE):
            tok = self._advance()
            self._expect(T.SEMI)
            return self._span(ast.Continue(), tok)
        if self._at(T.MATCH):
            return self._match_stmt()
        if self._at(T.LBRACE):
            return self._block()
        # assignment or expression statement
        if self._at(T.IDENT) and self._peek(1).type == T.ASSIGN:
            tok = self._advance()
            self._expect(T.ASSIGN)
            val = self._expr()
            self._expect(T.SEMI)
            return self._span(ast.Assign(tok.value, val), tok)
        expr = self._expr()
        if self._at(T.ASSIGN) and isinstance(expr, ast.Member):
            self._advance()
            val = self._expr()
            self._expect(T.SEMI)
            return self._span(ast.MemberAssign(expr.obj, expr.name, val), expr)
        self._expect(T.SEMI)
        node = ast.ExprStmt(expr)
        node.line = expr.line
        node.col = expr.col
        return node

    def _let(self) -> ast.Let:
        tok = self._expect(T.LET)
        name = self._expect(T.IDENT).value
        ty: Optional[str] = None
        if self._match(T.COLON):
            ty = self._type()
        self._expect(T.ASSIGN)
        val = self._expr()
        self._expect(T.SEMI)
        return self._span(ast.Let(name, ty, val), tok)

    def _if(self) -> ast.If:
        tok = self._expect(T.IF)
        cond = self._expr()
        then = self._block()
        otherwise: Optional[ast.Block] = None
        if self._match(T.ELSE):
            if self._at(T.LBRACE):
                otherwise = self._block()
            else:
                inner = self._if()
                otherwise = ast.Block([inner])
                otherwise.line = inner.line
                otherwise.col = inner.col
        return self._span(ast.If(cond, then, otherwise), tok)

    def _while(self) -> ast.While:
        tok = self._expect(T.WHILE)
        cond = self._expr()
        body = self._block()
        return self._span(ast.While(cond, body), tok)

    def _for_in(self) -> ast.ForIn:
        tok = self._expect(T.FOR)
        var_name = self._expect(T.IDENT).value
        self._expect(T.IN)
        iterable = self._expr()
        body = self._block()
        return self._span(ast.ForIn(var_name, iterable, body), tok)

    # -- match/case ---------------------------------------------------------
    def _match_stmt(self) -> ast.Match:
        tok = self._expect(T.MATCH)
        expr = self._expr()
        self._expect(T.LBRACE)
        cases: List[ast.MatchCase] = []
        while not self._at(T.RBRACE) and not self._at(T.EOF):
            if self._at(T.CASE):
                cases.append(self._match_case())
            else:
                cases.append(self._match_arm_stmt())
        self._expect(T.RBRACE)
        return self._span(ast.Match(expr, cases), tok)

    def _match_case(self) -> ast.MatchCase:
        tok = self._expect(T.CASE)
        pattern = self._pattern()
        guard: Optional[ast.Node] = None
        if self._at(T.IF):
            self._advance()
            guard = self._expr()
        if self._at(T.FAT_ARROW):
            self._expect(T.FAT_ARROW)
        else:
            self._expect(T.ARROW)
        body = self._stmt()
        return self._span(ast.MatchCase(pattern, guard, body), tok)

    def _match_arm_stmt(self) -> ast.MatchCase:
        """Parse a match arm without the 'case' keyword: pattern [if guard] => stmt"""
        tok = self._peek()
        pattern = self._pattern()
        guard: Optional[ast.Node] = None
        if self._at(T.IF):
            self._advance()
            guard = self._expr()
        self._expect(T.FAT_ARROW)
        body = self._stmt()
        return self._span(ast.MatchCase(pattern, guard, body), tok)

    def _match_expr(self) -> ast.MatchExpr:
        """Parse match as expression: match expr { pattern => expr, ... }"""
        tok = self._expect(T.MATCH)
        expr = self._expr()
        self._expect(T.LBRACE)
        arms: List[ast.MatchCase] = []
        while not self._at(T.RBRACE) and not self._at(T.EOF):
            arm_tok = self._peek()
            pattern = self._pattern()
            guard: Optional[ast.Node] = None
            if self._at(T.IF):
                self._advance()
                guard = self._expr()
            self._expect(T.FAT_ARROW)
            body = self._expr()
            arms.append(self._span(ast.MatchCase(pattern, guard, body), arm_tok))
            self._match(T.COMMA)  # optional trailing comma
        self._expect(T.RBRACE)
        return self._span(ast.MatchExpr(expr, arms), tok)

    def _pattern(self) -> ast.Node:
        t = self._peek()
        if t.type == T.UNDERSCORE:
            self._advance()
            return self._span(ast.WildcardPattern(), t)
        if t.type == T.INT:
            self._advance()
            return self._span(ast.LitPattern(self._span(ast.IntLit(t.value), t)), t)
        if t.type == T.FLOAT:
            self._advance()
            return self._span(ast.LitPattern(self._span(ast.FloatLit(t.value), t)), t)
        if t.type == T.STRING:
            self._advance()
            return self._span(ast.LitPattern(self._span(ast.StrLit(t.value), t)), t)
        if t.type == T.TRUE:
            self._advance()
            return self._span(ast.LitPattern(self._span(ast.BoolLit(True), t)), t)
        if t.type == T.FALSE:
            self._advance()
            return self._span(ast.LitPattern(self._span(ast.BoolLit(False), t)), t)
        if t.type == T.MINUS and self._peek(1).type in (T.INT, T.FLOAT):
            self._advance()
            num = self._advance()
            if num.type == T.INT:
                return self._span(ast.LitPattern(self._span(ast.IntLit(-num.value), t)), t)
            return self._span(ast.LitPattern(self._span(ast.FloatLit(-num.value), t)), t)
        if t.type == T.LBRACKET:
            return self._array_pattern()
        if t.type == T.LBRACE:
            return self._map_pattern()
        if t.type == T.LPAREN:
            return self._tuple_pattern()
        if t.type == T.DOTDOTDOT:
            return self._rest_pattern()
        if t.type == T.IDENT:
            self._advance()
            return self._span(ast.VarPattern(t.value), t)
        raise ParseError(f"expected pattern, got {t.type.name} ({t.value!r}) at line {t.line}:{t.col}")

    def _array_pattern(self) -> ast.ArrayPattern:
        tok = self._expect(T.LBRACKET)
        elements: List[ast.Node] = []
        if not self._at(T.RBRACKET):
            elements.append(self._pattern())
            while self._match(T.COMMA):
                if self._at(T.RBRACKET):
                    break  # allow trailing comma
                elements.append(self._pattern())
        self._expect(T.RBRACKET)
        return self._span(ast.ArrayPattern(elements), tok)

    def _tuple_pattern(self) -> ast.TuplePattern:
        tok = self._expect(T.LPAREN)
        elements: List[ast.Node] = []
        if not self._at(T.RPAREN):
            elements.append(self._pattern())
            while self._match(T.COMMA):
                if self._at(T.RPAREN):
                    break
                elements.append(self._pattern())
        self._expect(T.RPAREN)
        return self._span(ast.TuplePattern(elements), tok)

    def _rest_pattern(self) -> ast.RestPattern:
        tok = self._expect(T.DOTDOTDOT)
        name = self._expect(T.IDENT).value
        return self._span(ast.RestPattern(name), tok)

    def _map_pattern(self) -> ast.MapPattern:
        tok = self._expect(T.LBRACE)
        keys: List[ast.Node] = []
        patterns: List[ast.Node] = []
        if not self._at(T.RBRACE):
            k = self._expect(T.STRING)
            keys.append(self._span(ast.StrLit(k.value), k))
            self._expect(T.COLON)
            patterns.append(self._pattern())
            while self._match(T.COMMA):
                k = self._expect(T.STRING)
                keys.append(self._span(ast.StrLit(k.value), k))
                self._expect(T.COLON)
                patterns.append(self._pattern())
        self._expect(T.RBRACE)
        return self._span(ast.MapPattern(keys, patterns), tok)

    # -- lambda ------------------------------------------------------------
    def _lambda(self) -> ast.Lambda:
        tok = self._expect(T.PIPE)
        params: List[str] = []
        if not self._at(T.PIPE):
            params.append(self._expect(T.IDENT).value)
            while self._match(T.COMMA):
                params.append(self._expect(T.IDENT).value)
        self._expect(T.PIPE)
        if self._match(T.ARROW):
            body = self._expr()
        else:
            body = self._block()
        return self._span(ast.Lambda(params, body), tok)

    # -- expressions (precedence climbing) ----------------------------------
    def _expr(self, min_prec: int = 0) -> ast.Node:
        left = self._unary()
        while True:
            tt = self._peek().type
            prec = _PREC.get(tt)
            if prec is None or prec < min_prec:
                break
            op_tok = self._advance()
            right = self._expr(prec + 1)
            left = self._span(ast.Binary(_OP_TEXT[tt], left, right), op_tok)
        return left

    def _unary(self) -> ast.Node:
        if self._at(T.MINUS) or self._at(T.NOT):
            tok = self._advance()
            return self._span(ast.Unary(tok.value, self._unary()), tok)
        return self._postfix()

    def _postfix(self) -> ast.Node:
        node = self._primary()
        while self._at(T.DOT) or self._at(T.LBRACKET):
            if self._at(T.DOT):
                dot_tok = self._advance()
                name = self._expect(T.IDENT).value
                if self._at(T.LPAREN):
                    self._advance()
                    args: List[ast.Node] = []
                    if not self._at(T.RPAREN):
                        args.append(self._expr())
                        while self._match(T.COMMA):
                            args.append(self._expr())
                    self._expect(T.RPAREN)
                    node = self._span(ast.MethodCall(node, name, args), dot_tok)
                else:
                    node = self._span(ast.Member(node, name), dot_tok)
            else:
                br_tok = self._advance()
                idx = self._expr()
                self._expect(T.RBRACKET)
                node = self._span(ast.Index(node, idx), br_tok)
        return node

    def _primary(self) -> ast.Node:
        t = self._peek()
        if t.type == T.INT:
            self._advance(); return self._span(ast.IntLit(t.value), t)
        if t.type == T.FLOAT:
            self._advance(); return self._span(ast.FloatLit(t.value), t)
        if t.type == T.STRING:
            self._advance(); return self._span(ast.StrLit(t.value), t)
        if t.type == T.TRUE:
            self._advance(); return self._span(ast.BoolLit(True), t)
        if t.type == T.FALSE:
            self._advance(); return self._span(ast.BoolLit(False), t)
        if t.type == T.LPAREN:
            self._advance()
            e = self._expr()
            self._expect(T.RPAREN)
            return e
        if t.type == T.LBRACKET:
            self._advance()
            elems: List[ast.Node] = []
            if not self._at(T.RBRACKET):
                elems.append(self._expr())
                while self._match(T.COMMA):
                    elems.append(self._expr())
            self._expect(T.RBRACKET)
            return self._span(ast.ArrayLit(elems), t)
        if t.type == T.MATCH:
            return self._match_expr()
        if t.type == T.PIPE:
            return self._lambda()
        if t.type == T.OR:
            self._advance()
            if self._at(T.ARROW):
                self._advance()
                body = self._expr()
            else:
                body = self._block()
            return self._span(ast.Lambda([], body), t)
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
                return self._span(ast.Call(t.value, args), t)
            return self._span(ast.Var(t.value), t)
        raise ParseError(f"unexpected token {t.type.name} ({t.value!r}) at line {t.line}:{t.col}")


def parse(tokens: List[Token]) -> ast.Program:
    return Parser(tokens).parse()
