"""UCL type system — type representations, parsing and compatibility rules."""
from __future__ import annotations

from typing import Dict, List, Optional


class UclType:
    """Base for all UCL types."""

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self).__name__)

    def __repr__(self) -> str:
        return type(self).__name__


# -- scalar types -------------------------------------------------------------

class IntType(UclType):
    def __init__(self, bits: int, signed: bool):
        self.bits = bits
        self.signed = signed

    def __eq__(self, o: object) -> bool:
        return isinstance(o, IntType) and self.bits == o.bits and self.signed == o.signed

    def __hash__(self) -> int:
        return hash(("int", self.bits, self.signed))

    def __repr__(self) -> str:
        return f"{'i' if self.signed else 'u'}{self.bits}"


class FloatType(UclType):
    def __init__(self, bits: int):
        self.bits = bits

    def __eq__(self, o: object) -> bool:
        return isinstance(o, FloatType) and self.bits == o.bits

    def __hash__(self) -> int:
        return hash(("float", self.bits))

    def __repr__(self) -> str:
        return f"f{self.bits}"


class BoolType(UclType):
    def __repr__(self) -> str:
        return "bool"

    def __hash__(self) -> int:
        return hash("bool")


class StrType(UclType):
    def __repr__(self) -> str:
        return "str"

    def __hash__(self) -> int:
        return hash("str")


class VoidType(UclType):
    def __repr__(self) -> str:
        return "void"

    def __hash__(self) -> int:
        return hash("void")


class PtrType(UclType):
    def __repr__(self) -> str:
        return "ptr"

    def __hash__(self) -> int:
        return hash("ptr")


# -- compound types -----------------------------------------------------------

class TensorType(UclType):
    def __init__(self, dtype: UclType, shape: Optional[list] = None):
        self.dtype = dtype
        self.shape = shape or []

    def __eq__(self, o: object) -> bool:
        return isinstance(o, TensorType) and self.dtype == o.dtype and self.shape == o.shape

    def __hash__(self) -> int:
        return hash(("tensor", self.dtype, tuple(self.shape)))

    def __repr__(self) -> str:
        return f"tensor<{self.dtype}, {self.shape}>"


class ArrayType(UclType):
    def __init__(self, elem: UclType):
        self.elem = elem

    def __eq__(self, o: object) -> bool:
        return isinstance(o, ArrayType) and self.elem == o.elem

    def __hash__(self) -> int:
        return hash(("array", self.elem))

    def __repr__(self) -> str:
        return f"[{self.elem}]"


class FnType(UclType):
    def __init__(self, params: List[UclType], ret: UclType):
        self.params = params
        self.ret = ret

    def __eq__(self, o: object) -> bool:
        return isinstance(o, FnType) and self.params == o.params and self.ret == o.ret

    def __hash__(self) -> int:
        return hash(("fn", tuple(self.params), self.ret))

    def __repr__(self) -> str:
        return f"fn({', '.join(repr(p) for p in self.params)}) -> {self.ret}"


class ClassType(UclType):
    def __init__(self, name: str, fields: Optional[Dict[str, UclType]] = None,
                 methods: Optional[Dict[str, FnType]] = None):
        self.name = name
        self.fields: Dict[str, UclType] = fields or {}
        self.methods: Dict[str, FnType] = methods or {}

    def __eq__(self, o: object) -> bool:
        return isinstance(o, ClassType) and self.name == o.name

    def __hash__(self) -> int:
        return hash(("class", self.name))

    def __repr__(self) -> str:
        return self.name


# -- singleton instances ------------------------------------------------------

BOOL = BoolType()
STR = StrType()
VOID = VoidType()
PTR = PtrType()
I64 = IntType(64, True)
I32 = IntType(32, True)
F32 = FloatType(32)
F64 = FloatType(64)

# -- builtin type map ---------------------------------------------------------

_BUILTIN_TYPES: Dict[str, UclType] = {
    "i8": IntType(8, True), "i16": IntType(16, True),
    "i32": I32, "i64": I64,
    "u8": IntType(8, False), "u16": IntType(16, False),
    "u32": IntType(32, False), "u64": IntType(64, False),
    "f16": FloatType(16), "f32": F32, "f64": F64,
    "bool": BOOL, "void": VOID, "str": STR, "ptr": PTR,
    # AI-native primitives (bare name -> default parameterisation)
    "tensor": TensorType(F32), "weights": TensorType(F32),
    "tokenstream": STR, "graph": VOID, "quant": IntType(8, False),
}


def parse_type(name: str) -> Optional[UclType]:
    """Resolve a type-name string (from the AST) to a UclType, or None."""
    return _BUILTIN_TYPES.get(name)


# -- promotion / assignability ------------------------------------------------

def is_numeric(t: UclType) -> bool:
    return isinstance(t, (IntType, FloatType))


def promote(a: UclType, b: UclType) -> Optional[UclType]:
    """Common type after numeric promotion, or None if incompatible."""
    if a == b:
        return a
    if isinstance(a, IntType) and isinstance(b, IntType):
        if a.bits != b.bits:
            return a if a.bits > b.bits else b
        return a if not a.signed else b  # same width: prefer unsigned
    if isinstance(a, FloatType) and isinstance(b, FloatType):
        return a if a.bits >= b.bits else b
    if isinstance(a, IntType) and isinstance(b, FloatType):
        return b
    if isinstance(a, FloatType) and isinstance(b, IntType):
        return a
    return None


def assignable(target: UclType, source: UclType) -> bool:
    """Can *source* be implicitly assigned to a slot of type *target*?"""
    if target == source:
        return True
    # int <-> int (widening is lossless; narrowing is impl-defined in v0.2)
    if isinstance(target, IntType) and isinstance(source, IntType):
        return True
    # float <-> float
    if isinstance(target, FloatType) and isinstance(source, FloatType):
        return True
    # int -> float promotion
    if isinstance(target, FloatType) and isinstance(source, IntType):
        return True
    return False
