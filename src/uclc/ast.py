"""UCL abstract syntax tree node definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class Node:
    pass


# ---- expressions -----------------------------------------------------------
@dataclass
class IntLit(Node):
    value: int


@dataclass
class FloatLit(Node):
    value: float


@dataclass
class StrLit(Node):
    value: str


@dataclass
class BoolLit(Node):
    value: bool


@dataclass
class Var(Node):
    name: str


@dataclass
class Unary(Node):
    op: str
    operand: Node


@dataclass
class Binary(Node):
    op: str
    left: Node
    right: Node


@dataclass
class Call(Node):
    callee: str
    args: List[Node] = field(default_factory=list)


@dataclass
class Member(Node):
    obj: Node
    name: str


@dataclass
class MethodCall(Node):
    obj: Node
    name: str
    args: List[Node] = field(default_factory=list)


# ---- statements ------------------------------------------------------------
@dataclass
class Let(Node):
    name: str
    type_name: Optional[str]
    value: Node


@dataclass
class Assign(Node):
    name: str
    value: Node


@dataclass
class Return(Node):
    value: Optional[Node]


@dataclass
class ExprStmt(Node):
    expr: Node


@dataclass
class Block(Node):
    stmts: List[Node] = field(default_factory=list)


@dataclass
class If(Node):
    cond: Node
    then: Block
    otherwise: Optional[Block]


@dataclass
class While(Node):
    cond: Node
    body: Block


# ---- declarations ----------------------------------------------------------
@dataclass
class Param(Node):
    name: str
    type_name: str


@dataclass
class Fn(Node):
    name: str
    params: List[Param]
    ret_type: Optional[str]
    body: Block


@dataclass
class Field(Node):
    name: str
    type_name: str


@dataclass
class Class(Node):
    name: str
    fields: List[Field] = field(default_factory=list)
    methods: List[Fn] = field(default_factory=list)


@dataclass
class Program(Node):
    decls: List[Node] = field(default_factory=list)
