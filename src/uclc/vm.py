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


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


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


class UCLLambda:
    def __init__(self, params: list, body, env: "Env"):
        self.params = params
        self.body = body
        self.env = env

    def __repr__(self) -> str:
        return f"<lambda({', '.join(self.params)})>"


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
    def _call_lambda(self, lam: UCLLambda, args: List[Any]) -> Any:
        if len(args) != len(lam.params):
            raise UCLRuntimeError(f"lambda expected {len(lam.params)} args, got {len(args)}")
        env = Env(lam.env)
        for name, val in zip(lam.params, args):
            env.define(name, val)
        if isinstance(lam.body, ast.Block):
            try:
                self._exec_block(lam.body, env)
            except _Return as r:
                return r.value
            return None
        return self._eval(lam.body, env)

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
                try:
                    self._exec_block(s.body, env)
                except _Break:
                    break
                except _Continue:
                    continue
        elif isinstance(s, ast.ForIn):
            iterable = self._eval(s.iterable, env)
            if not hasattr(iterable, "__iter__"):
                raise UCLRuntimeError(f"for..in target is not iterable: {type(iterable).__name__}")
            for item in iterable:
                inner = Env(env)
                inner.define(s.var_name, item)
                try:
                    for stmt in s.body.stmts:
                        self._exec(stmt, inner)
                except _Break:
                    break
                except _Continue:
                    continue
        elif isinstance(s, ast.Break):
            raise _Break()
        elif isinstance(s, ast.Continue):
            raise _Continue()
        elif isinstance(s, ast.Match):
            val = self._eval(s.expr, env)
            for case in s.cases:
                bindings: Dict[str, Any] = {}
                if self._match_pattern(case.pattern, val, bindings):
                    case_env = Env(env)
                    for k, v in bindings.items():
                        case_env.define(k, v)
                    if case.guard is not None:
                        if not self._truthy(self._eval(case.guard, case_env)):
                            continue
                    self._exec(case.body, case_env)
                    return
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
            if isinstance(obj, dict):
                if idx not in obj:
                    raise UCLRuntimeError(f"key {idx!r} not found in map")
                return obj[idx]
            if not isinstance(idx, int):
                raise UCLRuntimeError(f"array index must be integer, got {type(idx).__name__}")
            if not isinstance(obj, (list, str)):
                raise UCLRuntimeError(f"cannot index into {type(obj).__name__}")
            if idx < 0 or idx >= len(obj):
                raise UCLRuntimeError(f"index {idx} out of bounds (length {len(obj)})")
            return obj[idx]
        if isinstance(e, ast.MatchExpr):
            val = self._eval(e.expr, env)
            for arm in e.arms:
                bindings: Dict[str, Any] = {}
                if self._match_pattern(arm.pattern, val, bindings):
                    arm_env = Env(env)
                    for k, v in bindings.items():
                        arm_env.define(k, v)
                    if arm.guard is not None:
                        if not self._truthy(self._eval(arm.guard, arm_env)):
                            continue
                    return self._eval(arm.body, arm_env)
            return None
        if isinstance(e, ast.Lambda):
            return UCLLambda(e.params, e.body, env)
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
            if not isinstance(v, (list, str, dict)):
                raise UCLRuntimeError(f"len() requires array, str or map, got {type(v).__name__}")
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
        if e.callee == "split":
            if len(e.args) not in (1, 2):
                raise UCLRuntimeError("split(str [, sep]) takes 1-2 arguments")
            s = self._eval(e.args[0], env)
            if not isinstance(s, str):
                raise UCLRuntimeError(f"split() requires str, got {type(s).__name__}")
            sep = self._eval(e.args[1], env) if len(e.args) == 2 else " "
            return s.split(sep)
        if e.callee == "upper":
            if len(e.args) != 1:
                raise UCLRuntimeError("upper(str) takes 1 argument")
            s = self._eval(e.args[0], env)
            if not isinstance(s, str):
                raise UCLRuntimeError(f"upper() requires str, got {type(s).__name__}")
            return s.upper()
        if e.callee == "lower":
            if len(e.args) != 1:
                raise UCLRuntimeError("lower(str) takes 1 argument")
            s = self._eval(e.args[0], env)
            if not isinstance(s, str):
                raise UCLRuntimeError(f"lower() requires str, got {type(s).__name__}")
            return s.lower()
        if e.callee == "contains":
            if len(e.args) != 2:
                raise UCLRuntimeError("contains(haystack, needle) takes 2 arguments")
            h = self._eval(e.args[0], env)
            n = self._eval(e.args[1], env)
            if isinstance(h, str):
                return str(n) in h
            if isinstance(h, list):
                return n in h
            raise UCLRuntimeError(f"contains() requires str or array, got {type(h).__name__}")
        if e.callee == "substr":
            args = [self._eval(a, env) for a in e.args]
            if len(args) < 2 or len(args) > 3:
                raise UCLRuntimeError("substr(str, start [, length]) takes 2-3 arguments")
            s = args[0]
            if not isinstance(s, str):
                raise UCLRuntimeError(f"substr() requires str, got {type(s).__name__}")
            start = int(args[1])
            if len(args) == 3:
                return s[start:start + int(args[2])]
            return s[start:]
        if e.callee == "trim":
            if len(e.args) != 1:
                raise UCLRuntimeError("trim(str) takes 1 argument")
            s = self._eval(e.args[0], env)
            if not isinstance(s, str):
                raise UCLRuntimeError(f"trim() requires str, got {type(s).__name__}")
            return s.strip()
        if e.callee == "to_str":
            if len(e.args) != 1:
                raise UCLRuntimeError("to_str(value) takes 1 argument")
            return self._stringify(self._eval(e.args[0], env))
        if e.callee == "to_int":
            if len(e.args) != 1:
                raise UCLRuntimeError("to_int(value) takes 1 argument")
            v = self._eval(e.args[0], env)
            return int(v)
        if e.callee == "to_float":
            if len(e.args) != 1:
                raise UCLRuntimeError("to_float(value) takes 1 argument")
            v = self._eval(e.args[0], env)
            return float(v)
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
        if e.callee == "map":
            args = [self._eval(a, env) for a in e.args]
            if len(args) % 2 != 0:
                raise UCLRuntimeError("map() takes an even number of args: map(k1, v1, k2, v2, ...)")
            m: dict = {}
            for i in range(0, len(args), 2):
                m[args[i]] = args[i + 1]
            return m
        if e.callee == "get":
            if len(e.args) not in (2, 3):
                raise UCLRuntimeError("get(map, key [, default]) takes 2-3 arguments")
            args = [self._eval(a, env) for a in e.args]
            if not isinstance(args[0], dict):
                raise UCLRuntimeError(f"get() requires map, got {type(args[0]).__name__}")
            default = args[2] if len(args) == 3 else None
            return args[0].get(args[1], default)
        if e.callee == "set":
            if len(e.args) != 3:
                raise UCLRuntimeError("set(map, key, value) takes 3 arguments")
            m = self._eval(e.args[0], env)
            k = self._eval(e.args[1], env)
            v = self._eval(e.args[2], env)
            if not isinstance(m, dict):
                raise UCLRuntimeError(f"set() requires map, got {type(m).__name__}")
            m[k] = v
            return None
        if e.callee == "keys":
            if len(e.args) != 1:
                raise UCLRuntimeError("keys(map) takes 1 argument")
            m = self._eval(e.args[0], env)
            if not isinstance(m, dict):
                raise UCLRuntimeError(f"keys() requires map, got {type(m).__name__}")
            return list(m.keys())
        if e.callee == "values":
            if len(e.args) != 1:
                raise UCLRuntimeError("values(map) takes 1 argument")
            m = self._eval(e.args[0], env)
            if not isinstance(m, dict):
                raise UCLRuntimeError(f"values() requires map, got {type(m).__name__}")
            return list(m.values())
        if e.callee == "has_key":
            if len(e.args) != 2:
                raise UCLRuntimeError("has_key(map, key) takes 2 arguments")
            m = self._eval(e.args[0], env)
            k = self._eval(e.args[1], env)
            if not isinstance(m, dict):
                raise UCLRuntimeError(f"has_key() requires map, got {type(m).__name__}")
            return k in m
        if e.callee == "remove":
            if len(e.args) != 2:
                raise UCLRuntimeError("remove(map, key) takes 2 arguments")
            m = self._eval(e.args[0], env)
            k = self._eval(e.args[1], env)
            if not isinstance(m, dict):
                raise UCLRuntimeError(f"remove() requires map, got {type(m).__name__}")
            return m.pop(k, None)
        if e.callee == "map_arr":
            if len(e.args) != 2:
                raise UCLRuntimeError("map_arr(array, fn) takes 2 arguments")
            arr = self._eval(e.args[0], env)
            fn = self._eval(e.args[1], env)
            if not isinstance(arr, list):
                raise UCLRuntimeError(f"map_arr() first arg must be array, got {type(arr).__name__}")
            if not isinstance(fn, UCLLambda):
                raise UCLRuntimeError("map_arr() second arg must be a lambda")
            return [self._call_lambda(fn, [x]) for x in arr]
        if e.callee == "filter_arr":
            if len(e.args) != 2:
                raise UCLRuntimeError("filter_arr(array, fn) takes 2 arguments")
            arr = self._eval(e.args[0], env)
            fn = self._eval(e.args[1], env)
            if not isinstance(arr, list):
                raise UCLRuntimeError(f"filter_arr() first arg must be array, got {type(arr).__name__}")
            if not isinstance(fn, UCLLambda):
                raise UCLRuntimeError("filter_arr() second arg must be a lambda")
            return [x for x in arr if self._truthy(self._call_lambda(fn, [x]))]
        try:
            val = env.get(e.callee)
            if isinstance(val, UCLLambda):
                args = [self._eval(a, env) for a in e.args]
                return self._call_lambda(val, args)
        except UCLRuntimeError:
            pass
        if e.callee in ("tensor", "weights", "tokenstream", "graph", "dot", "matmul", "quant"):
            raise UCLRuntimeError(f"AI-native builtin '{e.callee}' not yet evaluable in v0 (roadmap)")
        raise UCLRuntimeError(f"unknown function '{e.callee}'")

    def _match_pattern(self, pat: ast.Node, val: Any, bindings: Dict[str, Any]) -> bool:
        if isinstance(pat, ast.WildcardPattern):
            return True
        if isinstance(pat, ast.LitPattern):
            lit = pat.value
            if isinstance(lit, ast.IntLit):
                return val == lit.value
            if isinstance(lit, ast.FloatLit):
                return val == lit.value
            if isinstance(lit, ast.StrLit):
                return val == lit.value
            if isinstance(lit, ast.BoolLit):
                return val == lit.value
            return False
        if isinstance(pat, ast.VarPattern):
            bindings[pat.name] = val
            return True
        if isinstance(pat, ast.TuplePattern):
            if not isinstance(val, list):
                return False
            if len(val) != len(pat.elements):
                return False
            for p, v in zip(pat.elements, val):
                if not self._match_pattern(p, v, bindings):
                    return False
            return True
        if isinstance(pat, ast.ArrayPattern):
            if not isinstance(val, list):
                return False
            # check for rest pattern (...name)
            rest_idx = None
            for i, elem in enumerate(pat.elements):
                if isinstance(elem, ast.RestPattern):
                    rest_idx = i
                    break
            if rest_idx is not None:
                fixed_count = len(pat.elements) - 1
                if len(val) < fixed_count:
                    return False
                # match elements before rest
                for i in range(rest_idx):
                    if not self._match_pattern(pat.elements[i], val[i], bindings):
                        return False
                # bind rest
                after_count = len(pat.elements) - rest_idx - 1
                rest_end = len(val) - after_count if after_count > 0 else len(val)
                bindings[pat.elements[rest_idx].name] = val[rest_idx:rest_end]
                # match elements after rest
                for j in range(after_count):
                    pat_idx = rest_idx + 1 + j
                    val_idx = len(val) - after_count + j
                    if not self._match_pattern(pat.elements[pat_idx], val[val_idx], bindings):
                        return False
                return True
            if len(val) != len(pat.elements):
                return False
            for p, v in zip(pat.elements, val):
                if not self._match_pattern(p, v, bindings):
                    return False
            return True
        if isinstance(pat, ast.MapPattern):
            if not isinstance(val, dict):
                return False
            for k_node, v_pat in zip(pat.keys, pat.patterns):
                key = k_node.value
                if key not in val:
                    return False
                if not self._match_pattern(v_pat, val[key], bindings):
                    return False
            return True
        return False

    @staticmethod
    def _truthy(v: Any) -> bool:
        return bool(v)

    @staticmethod
    def _stringify(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, UCLLambda):
            return f"<lambda({', '.join(v.params)})>"
        return str(v)


def run(prog: ast.Program, entry: str = "main") -> Any:
    return Interpreter().run(prog, entry)
