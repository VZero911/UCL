"""UCL abstract syntax tree node definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class Node:
    line: int = 0
    col: int = 0


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


@dataclass
class ForIn(Node):
    var_name: str
    iterable: Node
    body: Block


@dataclass
class Break(Node):
    pass


@dataclass
class Continue(Node):
    pass


@dataclass
class ArrayLit(Node):
    elements: List[Node] = field(default_factory=list)


@dataclass
class MapLit(Node):
    keys: List[Node] = field(default_factory=list)
    values: List[Node] = field(default_factory=list)


@dataclass
class Lambda(Node):
    params: List[str] = field(default_factory=list)
    body: Node = None


@dataclass
class Index(Node):
    obj: Node
    index: Node


# ---- pattern matching -------------------------------------------------------
@dataclass
class LitPattern(Node):
    value: Node  # IntLit, FloatLit, StrLit, BoolLit

@dataclass
class WildcardPattern(Node):
    pass

@dataclass
class VarPattern(Node):
    name: str

@dataclass
class ArrayPattern(Node):
    elements: List[Node] = field(default_factory=list)  # list of patterns

@dataclass
class TuplePattern(Node):
    elements: List[Node] = field(default_factory=list)  # list of patterns

@dataclass
class RestPattern(Node):
    name: str  # variable name to bind the rest of the array

@dataclass
class MapPattern(Node):
    keys: List[Node] = field(default_factory=list)    # StrLit keys
    patterns: List[Node] = field(default_factory=list) # value patterns

@dataclass
class MatchCase(Node):
    pattern: Node           # a pattern node
    guard: Optional[Node]   # optional guard expression (if cond)
    body: Node              # statement to execute

@dataclass
class Match(Node):
    expr: Node                                        # expression being matched
    cases: List[MatchCase] = field(default_factory=list)

@dataclass
class MatchExpr(Node):
    """Match as expression: each arm body is an expression, returns matched value."""
    expr: Node
    arms: List[MatchCase] = field(default_factory=list)


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
