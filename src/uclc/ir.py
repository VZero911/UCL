"""UCL low-level IR lowering.

Lowers the AST of each function into a flat, three-address linear IR — the
"ASM/C overlay" layer. This is intentionally close to a register machine so a
later backend can map it to real assembly / a bytecode VM. v0 emits textual IR
(`uclc --emit-ir`); native codegen is on the roadmap.

IR instruction forms (textual):
    tN = const <value>
    tN = load <name>
    store <name>, tS
    tN = <op> tA, tB          ; binary
    tN = neg tA / not tA      ; unary
    tN = call <fn>(tA, tB...)
    ret [tN]
    label Lk:
    jmp Lk
    jmpf tC, Lk               ; jump if tC is false
"""
from __future__ import annotations

from typing import Dict, List

from . import ast


class IRFunction:
    def __init__(self, name: str, params: List[str]):
        self.name = name
        self.params = params
        self.code: List[str] = []
        self._t = 0
        self._l = 0

    def tmp(self) -> str:
        self._t += 1
        return f"t{self._t - 1}"

    def label(self) -> str:
        self._l += 1
        return f"L{self._l - 1}"

    def emit(self, line: str) -> None:
        self.code.append(line)

    def render(self) -> str:
        head = f"func {self.name}({', '.join(self.params)}):"
        body = "\n".join(("  " + c) if not c.endswith(":") else c for c in self.code)
        return head + "\n" + body


class Lowerer:
    def __init__(self) -> None:
        self.funcs: List[IRFunction] = []

    def lower_program(self, prog: ast.Program) -> List[IRFunction]:
        for d in prog.decls:
            if isinstance(d, ast.Fn):
                self.funcs.append(self._fn(d))
            elif isinstance(d, ast.Class):
                for m in d.methods:
                    mm = self._fn(m)
                    mm.name = f"{d.name}.{mm.name}"
                    self.funcs.append(mm)
        return self.funcs

    def _fn(self, fn: ast.Fn) -> IRFunction:
        f = IRFunction(fn.name, [p.name for p in fn.params])
        self._block(f, fn.body)
        if not f.code or not f.code[-1].startswith("ret"):
            f.emit("ret")
        return f

    def _block(self, f: IRFunction, block: ast.Block) -> None:
        for s in block.stmts:
            self._stmt(f, s)

    def _stmt(self, f: IRFunction, s: ast.Node) -> None:
        if isinstance(s, ast.Let):
            t = self._expr(f, s.value)
            f.emit(f"store {s.name}, {t}")
        elif isinstance(s, ast.Assign):
            t = self._expr(f, s.value)
            f.emit(f"store {s.name}, {t}")
        elif isinstance(s, ast.Return):
            if s.value is None:
                f.emit("ret")
            else:
                t = self._expr(f, s.value)
                f.emit(f"ret {t}")
        elif isinstance(s, ast.ExprStmt):
            self._expr(f, s.expr)
        elif isinstance(s, ast.Block):
            self._block(f, s)
        elif isinstance(s, ast.If):
            c = self._expr(f, s.cond)
            l_else = f.label()
            l_end = f.label()
            f.emit(f"jmpf {c}, {l_else}")
            self._block(f, s.then)
            f.emit(f"jmp {l_end}")
            f.emit(f"{l_else}:")
            if s.otherwise is not None:
                self._block(f, s.otherwise)
            f.emit(f"{l_end}:")
        elif isinstance(s, ast.While):
            l_top = f.label()
            l_end = f.label()
            f.emit(f"{l_top}:")
            c = self._expr(f, s.cond)
            f.emit(f"jmpf {c}, {l_end}")
            self._block(f, s.body)
            f.emit(f"jmp {l_top}")
            f.emit(f"{l_end}:")
        elif isinstance(s, ast.ForIn):
            arr = self._expr(f, s.iterable)
            idx = f.tmp(); f.emit(f"{idx} = const 0")
            ln = f.tmp(); f.emit(f"{ln} = call len({arr})")
            l_top = f.label(); l_end = f.label()
            f.emit(f"{l_top}:")
            cmp = f.tmp(); f.emit(f"{cmp} = < {idx}, {ln}")
            f.emit(f"jmpf {cmp}, {l_end}")
            elem = f.tmp(); f.emit(f"{elem} = index {arr}, {idx}")
            f.emit(f"store {s.var_name}, {elem}")
            self._block(f, s.body)
            inc = f.tmp(); one = f.tmp()
            f.emit(f"{one} = const 1"); f.emit(f"{inc} = + {idx}, {one}")
            f.emit(f"store {idx}, {inc}"); f.emit(f"{idx} = load {idx}")
            f.emit(f"jmp {l_top}")
            f.emit(f"{l_end}:")
        else:  # pragma: no cover - defensive
            raise TypeError(f"cannot lower statement {type(s).__name__}")

    def _expr(self, f: IRFunction, e: ast.Node) -> str:
        if isinstance(e, ast.IntLit):
            t = f.tmp(); f.emit(f"{t} = const {e.value}"); return t
        if isinstance(e, ast.FloatLit):
            t = f.tmp(); f.emit(f"{t} = const {e.value}"); return t
        if isinstance(e, ast.StrLit):
            t = f.tmp(); f.emit(f"{t} = const {e.value!r}"); return t
        if isinstance(e, ast.BoolLit):
            t = f.tmp(); f.emit(f"{t} = const {str(e.value).lower()}"); return t
        if isinstance(e, ast.Var):
            t = f.tmp(); f.emit(f"{t} = load {e.name}"); return t
        if isinstance(e, ast.Unary):
            a = self._expr(f, e.operand)
            t = f.tmp()
            f.emit(f"{t} = {'neg' if e.op == '-' else 'not'} {a}")
            return t
        if isinstance(e, ast.Binary):
            a = self._expr(f, e.left)
            b = self._expr(f, e.right)
            t = f.tmp()
            f.emit(f"{t} = {e.op} {a}, {b}")
            return t
        if isinstance(e, ast.Call):
            args = [self._expr(f, a) for a in e.args]
            t = f.tmp()
            f.emit(f"{t} = call {e.callee}({', '.join(args)})")
            return t
        if isinstance(e, ast.Member):
            o = self._expr(f, e.obj)
            t = f.tmp()
            f.emit(f"{t} = getfield {o}, {e.name}")
            return t
        if isinstance(e, ast.ArrayLit):
            elems = [self._expr(f, el) for el in e.elements]
            t = f.tmp()
            f.emit(f"{t} = array [{', '.join(elems)}]")
            return t
        if isinstance(e, ast.Index):
            o = self._expr(f, e.obj)
            i = self._expr(f, e.index)
            t = f.tmp()
            f.emit(f"{t} = index {o}, {i}")
            return t
        if isinstance(e, ast.MethodCall):
            o = self._expr(f, e.obj)
            args = [self._expr(f, a) for a in e.args]
            t = f.tmp()
            f.emit(f"{t} = callmethod {o}.{e.name}({', '.join(args)})")
            return t
        raise TypeError(f"cannot lower expression {type(e).__name__}")  # pragma: no cover


def lower(prog: ast.Program) -> str:
    funcs = Lowerer().lower_program(prog)
    return "\n\n".join(fn.render() for fn in funcs)
