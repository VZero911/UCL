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


class UCLInstance:
    def __init__(self, class_name: str, fields: Dict[str, Any]):
        self.class_name = class_name
        self.fields = dict(fields)

    def __repr__(self) -> str:
        return f"{self.class_name}{{{', '.join(f'{k}={v}' for k, v in self.fields.items())}}}"


class Interpreter:
    def __init__(self) -> None:
        self.funcs: Dict[str, ast.Fn] = {}
        self.classes: Dict[str, ast.Class] = {}
        self.output: List[str] = []

    def run(self, prog: ast.Program, entry: str = "main") -> Any:
        for d in prog.decls:
            if isinstance(d, ast.Fn):
                self.funcs[d.name] = d
            elif isinstance(d, ast.Class):
                self.classes[d.name] = d
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
        elif isinstance(s, ast.ForIn):
            iterable = self._eval(s.iterable, env)
            if not hasattr(iterable, "__iter__"):
                raise UCLRuntimeError(f"for..in target is not iterable: {type(iterable).__name__}")
            for item in iterable:
                inner = Env(env)
                inner.define(s.var_name, item)
                for stmt in s.body.stmts:
                    self._exec(stmt, inner)
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
        if isinstance(e, ast.Member):
            obj = self._eval(e.obj, env)
            if isinstance(obj, UCLInstance):
                if e.name in obj.fields:
                    return obj.fields[e.name]
                raise UCLRuntimeError(f"no field '{e.name}' on {obj.class_name}")
            raise UCLRuntimeError(f"member access on non-object: {type(obj).__name__}")
        if isinstance(e, ast.ArrayLit):
            return [self._eval(el, env) for el in e.elements]
        if isinstance(e, ast.Index):
            obj = self._eval(e.obj, env)
            idx = self._eval(e.index, env)
            if not isinstance(idx, int):
                raise UCLRuntimeError(f"array index must be integer, got {type(idx).__name__}")
            if not isinstance(obj, (list, str)):
                raise UCLRuntimeError(f"cannot index into {type(obj).__name__}")
            if idx < 0 or idx >= len(obj):
                raise UCLRuntimeError(f"index {idx} out of bounds (length {len(obj)})")
            return obj[idx]
        if isinstance(e, ast.MethodCall):
            obj = self._eval(e.obj, env)
            if isinstance(obj, UCLInstance):
                method_key = f"{obj.class_name}.{e.name}"
                if method_key not in self.funcs:
                    raise UCLRuntimeError(f"no method '{e.name}' on {obj.class_name}")
                fn = self.funcs[method_key]
                args = [self._eval(a, env) for a in e.args]
                method_env = Env()
                for field_name, field_val in obj.fields.items():
                    method_env.define(field_name, field_val)
                if len(args) != len(fn.params):
                    raise UCLRuntimeError(f"{method_key}: expected {len(fn.params)} args, got {len(args)}")
                for p, a in zip(fn.params, args):
                    method_env.define(p.name, a)
                try:
                    self._exec_block(fn.body, method_env)
                except _Return as r:
                    return r.value
                return None
            raise UCLRuntimeError(f"method call on non-object: {type(obj).__name__}")
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
        if e.callee == "len":
            if len(e.args) != 1:
                raise UCLRuntimeError("len() takes exactly one argument")
            v = self._eval(e.args[0], env)
            if not isinstance(v, (list, str)):
                raise UCLRuntimeError(f"len() requires array or str, got {type(v).__name__}")
            return len(v)
        if e.callee == "range":
            args = [self._eval(a, env) for a in e.args]
            if len(args) == 1:
                return list(range(int(args[0])))
            if len(args) == 2:
                return list(range(int(args[0]), int(args[1])))
            if len(args) == 3:
                return list(range(int(args[0]), int(args[1]), int(args[2])))
            raise UCLRuntimeError("range() takes 1-3 arguments")
        if e.callee == "push":
            if len(e.args) != 2:
                raise UCLRuntimeError("push(array, value) takes exactly 2 arguments")
            arr = self._eval(e.args[0], env)
            val = self._eval(e.args[1], env)
            if not isinstance(arr, list):
                raise UCLRuntimeError("push() first arg must be an array")
            arr.append(val)
            return None
        if e.callee in self.classes:
            cls = self.classes[e.callee]
            defaults: Dict[str, Any] = {}
            for f in cls.fields:
                if f.type_name in ("i32", "i64", "i8", "i16", "u8", "u16", "u32", "u64"):
                    defaults[f.name] = 0
                elif f.type_name in ("f32", "f64", "f16"):
                    defaults[f.name] = 0.0
                elif f.type_name == "bool":
                    defaults[f.name] = False
                elif f.type_name == "str":
                    defaults[f.name] = ""
                else:
                    defaults[f.name] = None
            return UCLInstance(e.callee, defaults)
        if e.callee in self.funcs:
            args = [self._eval(a, env) for a in e.args]
            return self._call(self.funcs[e.callee], args)
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
