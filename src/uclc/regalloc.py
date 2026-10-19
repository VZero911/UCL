"""Linear-scan register allocator for UCL x86-64 codegen.

Computes live intervals for IR temporaries and assigns them to callee-saved
x86-64 registers (rbx, r12-r15).  Temps that don't fit spill to the stack
(existing codegen behaviour).  String temporaries are excluded from
allocation since they require paired (pointer, length) storage.

No register allocator yet -- correctness first, speed later.  This module
is the "speed later" part: opt-in via ``use_regalloc=True``.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from .ir import IRFunction

# Callee-saved registers available for allocation.
# These survive function calls, so allocated temps are safe across call sites.
ALLOCATABLE_REGS: List[str] = ["rbx", "r12", "r13", "r14", "r15"]

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class LiveInterval:
    """Half-open live interval [start, end] for one IR temporary."""
    __slots__ = ("name", "start", "end")

    def __init__(self, name: str, start: int, end: int) -> None:
        self.name = name
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"LiveInterval({self.name!r}, {self.start}, {self.end})"


class RegisterAllocation:
    """Result of register allocation for one function."""

    def __init__(self) -> None:
        self.reg_map: Dict[str, str] = {}       # temp -> register name
        self.spill_map: Dict[str, int] = {}      # temp -> placeholder (stack)
        self.used_callee_saved: List[str] = []   # ordered for prologue/epilogue

    def is_in_register(self, name: str) -> bool:
        return name in self.reg_map


# ---------------------------------------------------------------------------
# IR analysis helpers
# ---------------------------------------------------------------------------

_TEMP_PAT = re.compile(r"\bt(\d+)\b")


def _is_temp(name: str) -> bool:
    """Return True if *name* looks like a compiler temporary (``t<digits>``)."""
    return len(name) > 1 and name[0] == "t" and name[1:].isdigit()


def _parse_defs_uses(line: str) -> Tuple[Optional[str], List[str]]:
    """Return ``(defined_temp_or_None, [used_temps])`` for one IR line."""
    line = line.strip()
    if not line or line.endswith(":"):
        return None, []

    # jmp Lk
    if line.startswith("jmp "):
        return None, []

    # jmpf tC, Lk
    if line.startswith("jmpf "):
        parts = line.split()
        cond = parts[1].rstrip(",")
        return None, [cond] if _is_temp(cond) else []

    # store <name>, tS
    if line.startswith("store "):
        parts = line.split(None, 2)
        target = parts[1].rstrip(",")
        val = parts[2].strip() if len(parts) > 2 else ""
        defined = target if _is_temp(target) else None
        uses: List[str] = [val] if _is_temp(val) else []
        return defined, uses

    # ret [tN]
    if line.startswith("ret"):
        parts = line.split()
        if len(parts) > 1 and _is_temp(parts[1]):
            return None, [parts[1]]
        return None, []

    # tN = <rhs>
    m = re.match(r"^(t\d+)\s*=\s*(.*)", line)
    if m:
        dst = m.group(1)
        rhs = m.group(2).strip()
        used = [f"t{n}" for n in _TEMP_PAT.findall(rhs)]
        return dst, used

    return None, []


# ---------------------------------------------------------------------------
# String-temp and store-target identification
# ---------------------------------------------------------------------------

def identify_string_temps(ir_func: IRFunction) -> Set[str]:
    """Identify temps that may hold string values (ptr+len pairs).

    Uses iterative dataflow: string-ness propagates through stores, loads,
    concatenation, and copy assignments.
    """
    string_names: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for line in ir_func.code:
            line = line.strip()
            # const string literal
            m = re.match(r'^(t\d+)\s*=\s*const\s+(\'.*\'|".*")', line)
            if m and m.group(1) not in string_names:
                string_names.add(m.group(1))
                changed = True
                continue
            # store X, T where T is string -> X is string
            m = re.match(r"^store\s+(\S+),\s*(t\d+)", line)
            if m and m.group(2) in string_names and m.group(1) not in string_names:
                string_names.add(m.group(1))
                changed = True
                continue
            # T = load X where X is string -> T is string
            m = re.match(r"^(t\d+)\s*=\s*load\s+(\S+)", line)
            if m and m.group(2) in string_names and m.group(1) not in string_names:
                string_names.add(m.group(1))
                changed = True
                continue
            # T = + A, B where both are strings (concatenation)
            m = re.match(r"^(t\d+)\s*=\s*\+\s*(t\d+),\s*(t\d+)", line)
            if m and m.group(2) in string_names and m.group(3) in string_names:
                if m.group(1) not in string_names:
                    string_names.add(m.group(1))
                    changed = True
                    continue
            # T = A  (copy from match expression)
            m = re.match(r"^(t\d+)\s*=\s*(t\d+)\s*$", line)
            if m and m.group(2) in string_names and m.group(1) not in string_names:
                string_names.add(m.group(1))
                changed = True
                continue
    return string_names


def identify_store_targets(ir_func: IRFunction) -> Set[str]:
    """Identify temps that are targets of ``store`` instructions.

    These temps are mutated (e.g. loop variables) and are excluded from
    register allocation for simplicity.
    """
    targets: Set[str] = set()
    for line in ir_func.code:
        line = line.strip()
        if line.startswith("store "):
            parts = line.split(None, 2)
            target = parts[1].rstrip(",")
            if _is_temp(target):
                targets.add(target)
    return targets


# ---------------------------------------------------------------------------
# Loop detection (back-edge)
# ---------------------------------------------------------------------------

def _detect_loops(code: List[str]) -> List[Tuple[int, int]]:
    """Detect back-edge loops and return ``(start_line, end_line)`` pairs."""
    labels: Dict[str, int] = {}
    loops: List[Tuple[int, int]] = []
    for i, line in enumerate(code):
        line = line.strip()
        if line.endswith(":"):
            labels[line[:-1]] = i
        elif line.startswith("jmp "):
            target = line.split()[1]
            if target in labels and labels[target] <= i:
                loops.append((labels[target], i))
    return loops


# ---------------------------------------------------------------------------
# Liveness analysis
# ---------------------------------------------------------------------------

def compute_live_intervals(
    ir_func: IRFunction,
    exclude: Optional[Set[str]] = None,
) -> List[LiveInterval]:
    """Compute live intervals for each temp, excluding specified names.

    Intervals are extended to cover any loop they overlap with, ensuring
    correctness when a temp defined outside a loop is used inside it.

    Returns intervals sorted by start position.
    """
    exclude = exclude or set()
    first_def: Dict[str, int] = {}
    last_use: Dict[str, int] = {}

    for i, line in enumerate(ir_func.code):
        defined, uses = _parse_defs_uses(line)
        if defined and _is_temp(defined) and defined not in exclude:
            if defined not in first_def:
                first_def[defined] = i
            last_use[defined] = max(last_use.get(defined, i), i)
        for u in uses:
            if _is_temp(u) and u not in exclude:
                if u not in first_def:
                    first_def[u] = i  # use before def (edge case)
                last_use[u] = max(last_use.get(u, i), i)

    # Extend intervals to cover loops they overlap with
    loops = _detect_loops(ir_func.code)

    intervals: List[LiveInterval] = []
    for name in first_def:
        if name in last_use:
            start = first_def[name]
            end = last_use[name]
            # Extend for every loop the interval touches
            changed = True
            while changed:
                changed = False
                for ls, le in loops:
                    if start <= le and end >= ls:
                        ns, ne = min(start, ls), max(end, le)
                        if ns != start or ne != end:
                            start, end = ns, ne
                            changed = True
            intervals.append(LiveInterval(name, start, end))

    intervals.sort(key=lambda iv: iv.start)
    return intervals


# ---------------------------------------------------------------------------
# Linear-scan register allocation
# ---------------------------------------------------------------------------

def linear_scan(
    intervals: List[LiveInterval],
    num_regs: int = len(ALLOCATABLE_REGS),
) -> RegisterAllocation:
    """Assign registers to intervals using the linear-scan algorithm.

    Spills the interval with the farthest endpoint when registers are
    exhausted.  Returns a :class:`RegisterAllocation` with ``reg_map``
    and ``spill_map``.
    """
    result = RegisterAllocation()
    if not intervals:
        return result

    regs = list(ALLOCATABLE_REGS[:num_regs])
    free: List[str] = list(regs)
    active: List[LiveInterval] = []  # kept sorted by end

    for iv in intervals:
        # Expire intervals that ended before this one starts
        still_active: List[LiveInterval] = []
        for a in active:
            if a.end < iv.start:
                free.append(result.reg_map[a.name])
            else:
                still_active.append(a)
        active = still_active

        if free:
            reg = free.pop(0)
            result.reg_map[iv.name] = reg
            active.append(iv)
            active.sort(key=lambda a: a.end)
        else:
            # Need to spill: pick the active with the farthest endpoint
            if active and active[-1].end > iv.end:
                spill = active.pop()  # last = farthest end
                reg = result.reg_map.pop(spill.name)
                result.spill_map[spill.name] = 0
                result.reg_map[iv.name] = reg
                active.append(iv)
                active.sort(key=lambda a: a.end)
            else:
                # Current interval spills
                result.spill_map[iv.name] = 0

    result.used_callee_saved = sorted(
        set(result.reg_map.values()),
        key=lambda r: ALLOCATABLE_REGS.index(r),
    )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def allocate_registers(ir_func: IRFunction) -> RegisterAllocation:
    """Full register-allocation pipeline for one IR function.

    Excludes string temps and store-target temps, computes live intervals,
    and runs the linear-scan allocator.
    """
    exclude = identify_string_temps(ir_func) | identify_store_targets(ir_func)
    intervals = compute_live_intervals(ir_func, exclude=exclude)
    return linear_scan(intervals)
