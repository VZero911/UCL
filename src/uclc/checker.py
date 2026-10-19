"""UCL static type checker (v0.2).

Walks the AST, builds scoped type environments, and emits diagnostics for type
errors. AI-native types and class types are handled gracefully; full object
model checking is deferred to v0.3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from . import ast
from .types import (
    UclType, IntType, FloatType, BoolType, StrType, VoidType,
    TensorType, FnType, ClassType,
    parse_type, promote, assignable, is_numeric,
    BOOL, VOID, STR, I64, F64, F32,
)

# AI-native builtins and their return types
_AI_BUILTINS: Dict[str, UclType] = {
    "tensor": TensorType(F32),
    "weights": TensorType(F32),
    "tokenstream": STR,
    "graph": VOID,
    "quant": IntType(8, False),
    "dot": F32,
    "matmul": TensorType(F32),
}


@dataclass
class TypeDiag:
    message: str
    line: int
    col: int
    severity: str  # "error" | "warning"

    def __str__(self) -> str:
        return f"{self.line}:{self.col}: {self.severity}: {self.message}"


class TypeEnv:
    """Scoped variable -> type mapping."""

    def __init__(self, parent: Optional[TypeEnv] = None):
        self.vars: Dict[str, UclType] = {}
        self.parent = parent

    def get(self, name: str) -> Optional[UclType]:
        if name in self.vars:
            return self.vars[name]
        return self.parent.get(name) if self.parent else None

    def define(self, name: str, ty: UclType) -> None:
        self.vars[name] = ty


class Checker:
    def __init__(self) -> None:
        self.diags: List[TypeDiag] = []
        self.funcs: Dict[str, FnType] = {}
        self.classes: Dict[str, ClassType] = {}

    # -- helpers --------------------------------------------------------------

    def _error(self, msg: str, node: ast.Node) -> None:
        self.diags.append(TypeDiag(msg, node.line, node.col, "error"))

    def _warning(self, msg: str, node: ast.Node) -> None:
        self.diags.append(TypeDiag(msg, node.line, node.col, "warning"))

    def _resolve(self, name: Optional[str]) -> Optional[UclType]:
        """Resolve a type name string to a UclType (builtins + classes)."""
        if name is None:
            return None
        t = parse_type(name)
        if t is not None:
            return t
        return self.classes.get(name)

    # -- top-level ------------------------------------------------------------

    def check_program(self, prog: ast.Program) -> List[TypeDiag]:
        # pass 1a: register all classes (so functions can reference them)
        for d in prog.decls:
            if isinstance(d, ast.Class):
                self._register_class(d)

        # pass 1b: register top-level function signatures
        for d in prog.decls:
            if isinstance(d, ast.Fn):
                self._register_fn(d)

        # pass 2: check bodies
        for d in prog.decls:
            if isinstance(d, ast.Fn):
                self._check_fn(d)
            elif isinstance(d, ast.Class):
                for m in d.methods:
                    self._check_fn(m, class_name=d.name)

        return self.diags

    def _register_class(self, cls: ast.Class) -> None:
        fields: Dict[str, UclType] = {}
        for f in cls.fields:
            ft = self._resolve(f.type_name)
            if ft is None:
                self._error(f"unknown field type '{f.type_name}'", f)
                ft = VOID
            fields[f.name] = ft

        methods: Dict[str, FnType] = {}
        for m in cls.methods:
            param_types = self._resolve_params(m)
            ret = self._resolve(m.ret_type) or VOID
            fn_t = FnType(param_types, ret)
            methods[m.name] = fn_t
            self.funcs[f"{cls.name}.{m.name}"] = fn_t

        self.classes[cls.name] = ClassType(cls.name, fields, methods)

    def _register_fn(self, fn: ast.Fn) -> None:
        param_types = self._resolve_params(fn)
        ret = self._resolve(fn.ret_type) or VOID
        if fn.ret_type and ret == VOID and fn.ret_type != "void":
            # _resolve returned None -> unknown type -> we defaulted to VOID
            self._error(f"unknown return type '{fn.ret_type}'", fn)
        self.funcs[fn.name] = FnType(param_types, ret)

    def _resolve_params(self, fn: ast.Fn) -> List[UclType]:
        out: List[UclType] = []
        for p in fn.params:
            pt = self._resolve(p.type_name)
            if pt is None:
                self._error(f"unknown param type '{p.type_name}'", p)
                pt = VOID
            out.append(pt)
        return out

    # -- function body --------------------------------------------------------

    def _check_fn(self, fn: ast.Fn, class_name: Optional[str] = None) -> None:
        env = TypeEnv()
        # inject class fields into method scope
        if class_name and class_name in self.classes:
            for fname, ftype in self.classes[class_name].fields.items():
                env.define(fname, ftype)
        for p in fn.params:
            pt = self._resolve(p.type_name)
            if pt:
                env.define(p.name, pt)
        fn_key = f"{class_name}.{fn.name}" if class_name else fn.name
        fn_type = self.funcs.get(fn_key)
        self._check_block(fn.body, env, fn_type)

    # -- statements -----------------------------------------------------------

    def _check_block(self, block: ast.Block, env: TypeEnv,
                     fn_type: Optional[FnType]) -> None:
        inner = TypeEnv(env)
        for s in block.stmts:
            self._check_stmt(s, inner, fn_type)

    def _check_stmt(self, s: ast.Node, env: TypeEnv,
                    fn_type: Optional[FnType]) -> None:
        if isinstance(s, ast.Let):
            self._check_let(s, env)
        elif isinstance(s, ast.Assign):
            self._check_assign(s, env)
        elif isinstance(s, ast.Return):
            self._check_return(s, env, fn_type)
        elif isinstance(s, ast.ExprStmt):
            self._check_expr(s.expr, env)
        elif isinstance(s, ast.Block):
            self._check_block(s, env, fn_type)
        elif isinstance(s, ast.If):
            self._check_expr(s.cond, env)
            self._check_block(s.then, env, fn_type)
            if s.otherwise:
                self._check_block(s.otherwise, env, fn_type)
        elif isinstance(s, ast.While):
            self._check_expr(s.cond, env)
            self._check_block(s.body, env, fn_type)
        elif isinstance(s, ast.ForIn):
            iter_ty = self._check_expr(s.iterable, env)
            inner = TypeEnv(env)
            from .types import ArrayType
            if isinstance(iter_ty, ArrayType):
                inner.define(s.var_name, iter_ty.elem)
            else:
                inner.define(s.var_name, VOID)
            self._check_block(s.body, inner, fn_type)

    def _check_let(self, s: ast.Let, env: TypeEnv) -> None:
        init_type = self._check_expr(s.value, env)
        if s.type_name:
            declared = self._resolve(s.type_name)
            if declared is None:
                self._error(f"unknown type '{s.type_name}'", s)
                env.define(s.name, init_type or VOID)
            elif init_type and not assignable(declared, init_type):
                self._error(f"cannot assign {init_type} to {declared}", s)
                env.define(s.name, declared)
            else:
                env.define(s.name, declared)
        else:
            # infer from initialiser
            env.define(s.name, init_type or VOID)

    def _check_assign(self, s: ast.Assign, env: TypeEnv) -> None:
        val_type = self._check_expr(s.value, env)
        var_type = env.get(s.name)
        if var_type is None:
            self._error(f"assignment to undefined variable '{s.name}'", s)
        elif val_type and not assignable(var_type, val_type):
            self._error(f"cannot assign {val_type} to {var_type}", s)

    def _check_return(self, s: ast.Return, env: TypeEnv,
                      fn_type: Optional[FnType]) -> None:
        ret_type = VOID if s.value is None else (self._check_expr(s.value, env) or VOID)
        if fn_type and not assignable(fn_type.ret, ret_type):
            self._error(
                f"return type {ret_type} doesn't match declared {fn_type.ret}", s)

    # -- expressions ----------------------------------------------------------

    def _check_expr(self, e: ast.Node, env: TypeEnv) -> Optional[UclType]:
        if isinstance(e, ast.IntLit):
            return I64
        if isinstance(e, ast.FloatLit):
            return F64
        if isinstance(e, ast.StrLit):
            return STR
        if isinstance(e, ast.BoolLit):
            return BOOL
        if isinstance(e, ast.Var):
            ty = env.get(e.name)
            if ty is None:
                self._error(f"undefined variable '{e.name}'", e)
            return ty
        if isinstance(e, ast.Unary):
            return self._check_unary(e, env)
        if isinstance(e, ast.Binary):
            return self._check_binary(e, env)
        if isinstance(e, ast.Call):
            return self._check_call(e, env)
        if isinstance(e, ast.ArrayLit):
            from .types import ArrayType
            if not e.elements:
                return ArrayType(VOID)
            elem_ty = self._check_expr(e.elements[0], env) or VOID
            for el in e.elements[1:]:
                self._check_expr(el, env)
            return ArrayType(elem_ty)
        if isinstance(e, ast.Index):
            from .types import ArrayType
            obj_ty = self._check_expr(e.obj, env)
            idx_ty = self._check_expr(e.index, env)
            if idx_ty and not isinstance(idx_ty, IntType):
                self._warning(f"array index should be integer, got {idx_ty}", e)
            if isinstance(obj_ty, ArrayType):
                return obj_ty.elem
            if obj_ty == STR:
                return STR
            return None
        if isinstance(e, ast.Member):
            return self._check_member(e, env)
        if isinstance(e, ast.MethodCall):
            return self._check_method_call(e, env)
        return None

    def _check_unary(self, e: ast.Unary, env: TypeEnv) -> Optional[UclType]:
        operand = self._check_expr(e.operand, env)
        if operand is None:
            return None
        if e.op == "-":
            if not is_numeric(operand):
                self._error(f"unary '-' requires numeric type, got {operand}", e)
            return operand
        if e.op == "!":
            return BOOL
        return operand

    def _check_binary(self, e: ast.Binary, env: TypeEnv) -> Optional[UclType]:
        left = self._check_expr(e.left, env)
        right = self._check_expr(e.right, env)
        if left is None or right is None:
            return None

        if e.op in ("==", "!=", "<", ">", "<=", ">="):
            if is_numeric(left) and is_numeric(right):
                return BOOL
            if left == right:
                return BOOL
            self._error(f"cannot compare {left} with {right}", e)
            return BOOL

        if e.op in ("&&", "||"):
            return BOOL

        # arithmetic
        if is_numeric(left) and is_numeric(right):
            result = promote(left, right)
            if result is None:
                self._error(f"incompatible types {left} and {right} for '{e.op}'", e)
            return result

        if isinstance(left, StrType) and isinstance(right, StrType) and e.op == "+":
            return STR

        self._error(f"operator '{e.op}' not supported for {left} and {right}", e)
        return left

    def _check_call(self, e: ast.Call, env: TypeEnv) -> Optional[UclType]:
        # builtin print: accepts any args, returns void
        if e.callee == "print":
            for a in e.args:
                self._check_expr(a, env)
            return VOID

        # class constructor
        if e.callee in self.classes:
            for a in e.args:
                self._check_expr(a, env)
            return self.classes[e.callee]

        # AI-native builtins
        if e.callee in _AI_BUILTINS:
            for a in e.args:
                self._check_expr(a, env)
            return _AI_BUILTINS[e.callee]

        # user-defined function
        fn_type = self.funcs.get(e.callee)
        if fn_type is None:
            self._error(f"unknown function '{e.callee}'", e)
            for a in e.args:
                self._check_expr(a, env)
            return None

        if len(e.args) != len(fn_type.params):
            self._error(
                f"'{e.callee}' expects {len(fn_type.params)} args, got {len(e.args)}", e)
        for i, a in enumerate(e.args):
            arg_type = self._check_expr(a, env)
            if i < len(fn_type.params) and arg_type:
                if not assignable(fn_type.params[i], arg_type):
                    self._error(
                        f"arg {i + 1} of '{e.callee}': expected {fn_type.params[i]}, got {arg_type}", e)
        return fn_type.ret

    def _check_member(self, e: ast.Member, env: TypeEnv) -> Optional[UclType]:
        obj_type = self._check_expr(e.obj, env)
        if isinstance(obj_type, ClassType):
            if e.name in obj_type.fields:
                return obj_type.fields[e.name]
            self._error(f"'{obj_type.name}' has no field '{e.name}'", e)
        return None

    def _check_method_call(self, e: ast.MethodCall, env: TypeEnv) -> Optional[UclType]:
        obj_type = self._check_expr(e.obj, env)
        if isinstance(obj_type, ClassType):
            if e.name in obj_type.methods:
                fn_type = obj_type.methods[e.name]
                if len(e.args) != len(fn_type.params):
                    self._error(
                        f"'{obj_type.name}.{e.name}' expects {len(fn_type.params)} args, "
                        f"got {len(e.args)}", e)
                for i, a in enumerate(e.args):
                    arg_type = self._check_expr(a, env)
                    if i < len(fn_type.params) and arg_type:
                        if not assignable(fn_type.params[i], arg_type):
                            self._error(
                                f"arg {i + 1} of '{obj_type.name}.{e.name}': "
                                f"expected {fn_type.params[i]}, got {arg_type}", e)
                return fn_type.ret
            self._error(f"'{obj_type.name}' has no method '{e.name}'", e)
        # graceful fallback for non-class types (e.g. tensor method calls)
        for a in e.args:
            self._check_expr(a, env)
        return None


def check(prog: ast.Program) -> List[TypeDiag]:
    """Run the static type checker on a parsed program. Returns diagnostics."""
    return Checker().check_program(prog)
