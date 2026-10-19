"""UCL IR optimizer — multi-pass optimization on three-address IR.

Passes:
  1. constant_fold_propagate — fold constant expressions, propagate through temps/vars
  2. strength_reduce — replace expensive ops with cheaper equivalents
  3. dead_code_eliminate — remove assignments to unused temps

Usage:
    from uclc.optimizer import optimize
    optimized = optimize(ir_funcs)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from .ir import IRFunction

_INT_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a // b if b != 0 else None,
    "%": lambda a, b: a % b if b != 0 else None,
    "==": lambda a, b: int(a == b),
    "!=": lambda a, b: int(a != b),
    "<": lambda a, b: int(a < b),
    ">": lambda a, b: int(a > b),
    "<=": lambda a, b: int(a <= b),
    ">=": lambda a, b: int(a >= b),
}

_RE_ASSIGN = re.compile(r"^(t\d+)\s*=\s*(.+)$")
_RE_STORE = re.compile(r"^store\s+(\S+),\s*(t\d+)$")
_RE_RET = re.compile(r"^ret(?:\s+(t\d+))?$")
_RE_LABEL = re.compile(r"^(L\d+):$")
_RE_JMP = re.compile(r"^jmp\s+(L\d+)$")
_RE_JMPF = re.compile(r"^jmpf\s+(t\d+),\s*(L\d+)$")

_RE_CONST = re.compile(r"^const\s+(.+)$")
_RE_LOAD = re.compile(r"^load\s+(\S+)$")
_RE_NEG = re.compile(r"^neg\s+(t\d+)$")
_RE_NOT = re.compile(r"^not\s+(t\d+)$")
_RE_BINOP = re.compile(r"^([+\-*/%]|==|!=|<=|>=|<|>)\s+(t\d+),\s*(t\d+)$")
_RE_CALL = re.compile(r"^call\s+")
_RE_CALLMETHOD = re.compile(r"^callmethod\s+")
_RE_TEMP = re.compile(r"\bt\d+\b")


def _parse_int(s: str) -> Optional[int]:
    s = s.strip()
    if s == "true":
        return 1
    if s == "false":
        return 0
    if s.startswith("'") or s.startswith('"'):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _has_side_effects(rhs: str) -> bool:
    rhs = rhs.strip()
    return bool(_RE_CALL.match(rhs) or _RE_CALLMETHOD.match(rhs))


def _collect_used_temps(code: List[str]) -> Set[str]:
    used: Set[str] = set()
    for line in code:
        s = line.strip()

        m = _RE_STORE.match(s)
        if m:
            used.add(m.group(2))
            continue

        m = _RE_RET.match(s)
        if m and m.group(1):
            used.add(m.group(1))
            continue

        m = _RE_JMPF.match(s)
        if m:
            used.add(m.group(1))
            continue

        m = _RE_ASSIGN.match(s)
        if m:
            for t in _RE_TEMP.findall(m.group(2)):
                used.add(t)
            continue

    return used


def _has_backward_jumps(code: List[str]) -> bool:
    label_pos: Dict[str, int] = {}
    for i, line in enumerate(code):
        m = _RE_LABEL.match(line.strip())
        if m:
            label_pos[m.group(1)] = i
    for i, line in enumerate(code):
        m = _RE_JMP.match(line.strip())
        if m and m.group(1) in label_pos and label_pos[m.group(1)] <= i:
            return True
    return False


def constant_fold_propagate(fn: IRFunction) -> bool:
    """Fold constant expressions and propagate constants through temps/vars."""
    changed = False
    const_temps: Dict[str, int] = {}
    const_vars: Dict[str, int] = {}

    store_count: Dict[str, int] = {}
    for line in fn.code:
        m = _RE_STORE.match(line.strip())
        if m:
            store_count[m.group(1)] = store_count.get(m.group(1), 0) + 1

    loops = _has_backward_jumps(fn.code)
    new_code: List[str] = []

    for line in fn.code:
        s = line.strip()

        m = _RE_ASSIGN.match(s)
        if m:
            dst, rhs = m.group(1), m.group(2).strip()

            mc = _RE_CONST.match(rhs)
            if mc:
                val = _parse_int(mc.group(1))
                if val is not None:
                    const_temps[dst] = val
                new_code.append(s)
                continue

            ml = _RE_LOAD.match(rhs)
            if ml:
                var = ml.group(1)
                if not loops and store_count.get(var, 0) == 1 and var in const_vars:
                    const_temps[dst] = const_vars[var]
                    new_code.append(f"{dst} = const {const_vars[var]}")
                    changed = True
                    continue
                new_code.append(s)
                continue

            mn = _RE_NEG.match(rhs)
            if mn and mn.group(1) in const_temps:
                val = -const_temps[mn.group(1)]
                const_temps[dst] = val
                new_code.append(f"{dst} = const {val}")
                changed = True
                continue

            mnt = _RE_NOT.match(rhs)
            if mnt and mnt.group(1) in const_temps:
                val = 0 if const_temps[mnt.group(1)] else 1
                const_temps[dst] = val
                new_code.append(f"{dst} = const {val}")
                changed = True
                continue

            mb = _RE_BINOP.match(rhs)
            if mb:
                op, a, b = mb.group(1), mb.group(2), mb.group(3)
                if a in const_temps and b in const_temps and op in _INT_OPS:
                    result = _INT_OPS[op](const_temps[a], const_temps[b])
                    if result is not None:
                        const_temps[dst] = result
                        new_code.append(f"{dst} = const {result}")
                        changed = True
                        continue
                new_code.append(s)
                continue

            mcp = re.match(r"^(t\d+)$", rhs)
            if mcp and mcp.group(1) in const_temps:
                const_temps[dst] = const_temps[mcp.group(1)]

            new_code.append(s)
            continue

        ms = _RE_STORE.match(s)
        if ms and ms.group(2) in const_temps:
            const_vars[ms.group(1)] = const_temps[ms.group(2)]

        if _RE_LABEL.match(s):
            const_vars.clear()

        new_code.append(s)

    fn.code = new_code
    return changed


def strength_reduce(fn: IRFunction) -> bool:
    """Replace expensive ops with cheaper equivalents."""
    changed = False
    const_temps: Dict[str, int] = {}

    for line in fn.code:
        m = _RE_ASSIGN.match(line.strip())
        if m:
            mc = _RE_CONST.match(m.group(2).strip())
            if mc:
                val = _parse_int(mc.group(1))
                if val is not None:
                    const_temps[m.group(1)] = val

    new_code: List[str] = []
    for line in fn.code:
        s = line.strip()
        m = _RE_ASSIGN.match(s)
        if m:
            dst, rhs = m.group(1), m.group(2).strip()
            mb = _RE_BINOP.match(rhs)
            if mb:
                op, a, b = mb.group(1), mb.group(2), mb.group(3)
                a_val = const_temps.get(a)
                b_val = const_temps.get(b)

                if op == "*":
                    if b_val == 0 or a_val == 0:
                        new_code.append(f"{dst} = const 0")
                        const_temps[dst] = 0
                        changed = True
                        continue
                    if b_val == 2:
                        new_code.append(f"{dst} = + {a}, {a}")
                        changed = True
                        continue
                    if a_val == 2:
                        new_code.append(f"{dst} = + {b}, {b}")
                        changed = True
                        continue
                    if b_val == -1:
                        new_code.append(f"{dst} = neg {a}")
                        changed = True
                        continue
                    if a_val == -1:
                        new_code.append(f"{dst} = neg {b}")
                        changed = True
                        continue

        new_code.append(s)

    fn.code = new_code
    return changed


def dead_code_eliminate(fn: IRFunction) -> bool:
    """Remove assignments to temps that are never used."""
    used = _collect_used_temps(fn.code)
    changed = False
    new_code: List[str] = []

    for line in fn.code:
        s = line.strip()
        m = _RE_ASSIGN.match(s)
        if m:
            dst, rhs = m.group(1), m.group(2).strip()
            if dst not in used and not _has_side_effects(rhs):
                changed = True
                continue
        new_code.append(s)

    fn.code = new_code
    return changed


def optimize_function(fn: IRFunction, max_iterations: int = 10) -> IRFunction:
    """Apply all optimization passes until convergence."""
    for _ in range(max_iterations):
        c1 = constant_fold_propagate(fn)
        c2 = strength_reduce(fn)
        c3 = dead_code_eliminate(fn)
        if not (c1 or c2 or c3):
            break
    return fn


def optimize(ir_funcs: List[IRFunction]) -> List[IRFunction]:
    """Optimize a list of IR functions. Returns the same list, mutated."""
    for fn in ir_funcs:
        optimize_function(fn)
    return ir_funcs
