"""UCL evaluator (tree-walking interpreter for the v0 subset).

Runs a parsed Program by invoking `main()`. This is the reference semantics the
IR/native backends must match. AI-native ops (tensor/weights/...) are parsed
and lowerable but not yet evaluable here — calling an unimplemented AI builtin
raises a clear UCLRuntimeError (tracked on the roadmap).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import ast


class UCLRuntimeError(RuntimeError):
    pass


class _Return(Exception):
    def __init__(self, value: Any):
        self.value = value


class Env:
    def __init__(self, parent: Optional["Env"] = None):
        self.vars: Dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        e: Optional[Env] = self
        while e is not None:
            if name in e.vars:
                return e.vars[name]
            e = e.parent
        raise UCLRuntimeError(f"undefined variable '{name}'")

    def set_existing(self, name: str, value: Any) -> None:
        e: Optional[Env] = self
        while e is not None:
            if name in e.vars:
                e.vars[name] = value
                return
            e = e.parent
        raise UCLRuntimeError(f"assignment to undefined variable '{name}'")

    def define(self, name: str, value: Any) -> None:
        self.vars[name] = value


class Interpreter:
    def __init__(self) -> None:
        self.funcs: Dict[str, ast.Fn] = {}
        self.output: List[str] = []

    def run(self, prog: ast.Program, entry: str = "main") -> Any:
        for d in prog.decls:
            if isinstance(d, ast.Fn):
                self.funcs[d.name] = d
            elif isinstance(d, ast.Class):
                for m in d.methods:
                    self.funcs[f"{d.name}.{m.name}"] = m
        if entry not in self.funcs:
            raise UCLRuntimeError(f"no entry function '{entry}'")
        return self._call(self.funcs[entry], [])

    # -- calls --------------------------------------------------------------
    def _call(self, fn: ast.Fn, args: List[Any]) -> Any:
        if len(args) != len(fn.params):
            raise UCLRuntimeError(f"{fn.name}: expected {len(fn.params)} args, got {len(args)}")
        env = Env()
        for p, a in zip(fn.params, args):
            env.define(p.name, a)
        try:
            self._exec_block(fn.body, env)
        except _Return as r:
            return r.value
        return None

    # -- statements ---------------------------------------------------------
    def _exec_block(self, block: ast.Block, env: Env) -> None:
        inner = Env(env)
        for s in block.stmts:
            self._exec(s, inner)

    def _exec(self, s: ast.Node, env: Env) -> None:
        if isinstance(s, ast.Let):
            env.define(s.name, self._eval(s.value, env))
        elif isinstance(s, ast.Assign):
            env.set_existing(s.name, self._eval(s.value, env))
        elif isinstance(s, ast.Return):
            raise _Return(None if s.value is None else self._eval(s.value, env))
        elif isinstance(s, ast.ExprStmt):
            self._eval(s.expr, env)
        elif isinstance(s, ast.Block):
            self._exec_block(s, env)
        elif isinstance(s, ast.If):
            if self._truthy(self._eval(s.cond, env)):
                self._exec_block(s.then, env)
            elif s.otherwise is not None:
                self._exec_block(s.otherwise, env)
        elif isinstance(s, ast.While):
            while self._truthy(self._eval(s.cond, env)):
                self._exec_block(s.body, env)
        else:  # pragma: no cover
            raise UCLRuntimeError(f"cannot execute {type(s).__name__}")

    # -- expressions --------------------------------------------------------
    def _eval(self, e: ast.Node, env: Env) -> Any:
        if isinstance(e, ast.IntLit):
            return e.value
        if isinstance(e, ast.FloatLit):
            return e.value
        if isinstance(e, ast.StrLit):
            return e.value
        if isinstance(e, ast.BoolLit):
            return e.value
        if isinstance(e, ast.Var):
            return env.get(e.name)
        if isinstance(e, ast.Unary):
            v = self._eval(e.operand, env)
            return -v if e.op == "-" else (not self._truthy(v))
        if isinstance(e, ast.Binary):
            return self._binary(e, env)
        if isinstance(e, ast.Call):
            return self._call_expr(e, env)
        if isinstance(e, (ast.Member, ast.MethodCall)):
            # object model (classes/instances) is parsed + lowerable but not yet
            # evaluable by the reference interpreter — see docs/ROADMAP.md
            raise UCLRuntimeError(
                "object member/method evaluation not yet implemented in v0 (roadmap); "
                "use `uclc check` / `uclc emit-ir` for object code"
            )
        raise UCLRuntimeError(f"cannot evaluate {type(e).__name__}")  # pragma: no cover

    def _binary(self, e: ast.Binary, env: Env) -> Any:
        op = e.op
        if op == "&&":
            return self._truthy(self._eval(e.left, env)) and self._truthy(self._eval(e.right, env))
        if op == "||":
            return self._truthy(self._eval(e.left, env)) or self._truthy(self._eval(e.right, env))
        a = self._eval(e.left, env)
        b = self._eval(e.right, env)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if isinstance(a, int) and isinstance(b, int):
                if b == 0:
                    raise UCLRuntimeError("integer division by zero")
                return a // b
            return a / b
        if op == "%":
            return a % b
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        if op == "<":
            return a < b
        if op == ">":
            return a > b
        if op == "<=":
            return a <= b
        if op == ">=":
            return a >= b
        raise UCLRuntimeError(f"unknown operator {op}")  # pragma: no cover

    def _call_expr(self, e: ast.Call, env: Env) -> Any:
        if e.callee == "print":
            vals = [self._eval(a, env) for a in e.args]
            line = " ".join(self._stringify(v) for v in vals)
            self.output.append(line)
            print(line)
            return None
        if e.callee in self.funcs:
            args = [self._eval(a, env) for a in e.args]
            return self._call(self.funcs[e.callee], args)
        # AI-native builtins are declared in the spec but not yet evaluable
        if e.callee in ("tensor", "weights", "tokenstream", "graph", "dot", "matmul", "quant"):
            raise UCLRuntimeError(f"AI-native builtin '{e.callee}' not yet evaluable in v0 (roadmap)")
        raise UCLRuntimeError(f"unknown function '{e.callee}'")

    @staticmethod
    def _truthy(v: Any) -> bool:
        return bool(v)

    @staticmethod
    def _stringify(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)


def run(prog: ast.Program, entry: str = "main") -> Any:
    return Interpreter().run(prog, entry)
