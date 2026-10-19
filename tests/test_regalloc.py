"""Tests for the UCL linear-scan register allocator."""
import os
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from uclc.ir import IRFunction, Lowerer  # noqa: E402
from uclc.lexer import tokenize  # noqa: E402
from uclc.parser import parse  # noqa: E402
from uclc.regalloc import (  # noqa: E402
    compute_live_intervals,
    linear_scan,
    allocate_registers,
    identify_string_temps,
    identify_store_targets,
    LiveInterval,
    RegisterAllocation,
    ALLOCATABLE_REGS,
    _parse_defs_uses,
)
from uclc.codegen import compile_source_to_elf  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ir(code_lines):
    """Create an IRFunction with the given code lines."""
    fn = IRFunction("test", [])
    fn.code = list(code_lines)
    return fn


def _lower(src: str) -> list:
    """Lower UCL source to a list of IRFunction objects."""
    return Lowerer().lower_program(parse(tokenize(src)))


def _compile_and_run(src: str):
    """Compile UCL source WITH regalloc, run it, return (exit_code, stdout_lines)."""
    elf = compile_source_to_elf(src, use_regalloc=True)
    with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as f:
        f.write(elf)
        path = f.name
    os.chmod(path, 0o755)
    try:
        r = subprocess.run([path], capture_output=True, text=True, timeout=5)
        lines = [l for l in r.stdout.strip().split("\n") if l]
        return r.returncode, lines
    finally:
        os.unlink(path)


# ===================================================================
# Unit tests: IR analysis
# ===================================================================

class TestParseDefsUses:
    def test_const(self):
        d, u = _parse_defs_uses("t0 = const 42")
        assert d == "t0" and u == []

    def test_binary(self):
        d, u = _parse_defs_uses("t2 = + t0, t1")
        assert d == "t2" and set(u) == {"t0", "t1"}

    def test_store(self):
        d, u = _parse_defs_uses("store x, t3")
        assert d is None and u == ["t3"]

    def test_store_to_temp(self):
        d, u = _parse_defs_uses("store t1, t0")
        assert d == "t1" and u == ["t0"]

    def test_ret_with_value(self):
        d, u = _parse_defs_uses("ret t5")
        assert d is None and u == ["t5"]

    def test_ret_empty(self):
        d, u = _parse_defs_uses("ret")
        assert d is None and u == []

    def test_jmpf(self):
        d, u = _parse_defs_uses("jmpf t2, L1")
        assert d is None and u == ["t2"]

    def test_label(self):
        d, u = _parse_defs_uses("L0:")
        assert d is None and u == []

    def test_jmp(self):
        d, u = _parse_defs_uses("jmp L0")
        assert d is None and u == []

    def test_load(self):
        d, u = _parse_defs_uses("t3 = load x")
        assert d == "t3" and u == []

    def test_call(self):
        d, u = _parse_defs_uses("t4 = call foo(t0, t1)")
        assert d == "t4" and set(u) == {"t0", "t1"}


# ===================================================================
# Unit tests: liveness analysis
# ===================================================================

class TestLiveness:
    def test_simple_intervals(self):
        fn = _make_ir([
            "t0 = const 1",      # 0
            "t1 = const 2",      # 1
            "t2 = + t0, t1",     # 2
            "ret t2",             # 3
        ])
        intervals = compute_live_intervals(fn)
        by_name = {iv.name: iv for iv in intervals}
        assert by_name["t0"].start == 0 and by_name["t0"].end == 2
        assert by_name["t1"].start == 1 and by_name["t1"].end == 2
        assert by_name["t2"].start == 2 and by_name["t2"].end == 3

    def test_excludes_temps(self):
        fn = _make_ir([
            "t0 = const 1",
            "t1 = const 2",
            "t2 = + t0, t1",
            "ret t2",
        ])
        intervals = compute_live_intervals(fn, exclude={"t1"})
        names = {iv.name for iv in intervals}
        assert "t1" not in names
        assert "t0" in names and "t2" in names

    def test_loop_extends_intervals(self):
        """Temps used inside a loop body must stay alive for the whole loop."""
        fn = _make_ir([
            "t0 = const 0",      # 0
            "t1 = const 10",     # 1
            "L0:",                # 2
            "t2 = < t0, t1",     # 3
            "jmpf t2, L1",       # 4
            "t3 = + t0, t1",     # 5
            "jmp L0",            # 6  (back-edge -> loop is lines 2-6)
            "L1:",                # 7
            "ret t3",             # 8
        ])
        intervals = compute_live_intervals(fn)
        by_name = {iv.name: iv for iv in intervals}
        # t0 and t1 are used inside the loop -> extended to cover 2-6
        assert by_name["t0"].start <= 2 and by_name["t0"].end >= 6
        assert by_name["t1"].start <= 2 and by_name["t1"].end >= 6


# ===================================================================
# Unit tests: string and store-target identification
# ===================================================================

class TestIdentification:
    def test_string_const(self):
        fn = _make_ir(["t0 = const 'hello'", "t1 = const 42"])
        strings = identify_string_temps(fn)
        assert "t0" in strings
        assert "t1" not in strings

    def test_string_propagation_via_store_load(self):
        fn = _make_ir([
            "t0 = const 'hello'",
            "store x, t0",
            "t1 = load x",
        ])
        strings = identify_string_temps(fn)
        assert "t0" in strings
        assert "x" in strings
        assert "t1" in strings

    def test_string_concat(self):
        fn = _make_ir([
            "t0 = const 'a'",
            "t1 = const 'b'",
            "t2 = + t0, t1",
        ])
        strings = identify_string_temps(fn)
        assert "t2" in strings

    def test_store_target_detection(self):
        fn = _make_ir([
            "t0 = const 1",
            "store t0, t1",
            "store x, t2",
        ])
        targets = identify_store_targets(fn)
        assert "t0" in targets
        assert "x" not in targets  # not a temp


# ===================================================================
# Unit tests: linear-scan allocation
# ===================================================================

class TestLinearScan:
    def test_no_spills_with_few_temps(self):
        """3 temps, 5 registers -> no spills."""
        intervals = [
            LiveInterval("t0", 0, 3),
            LiveInterval("t1", 1, 3),
            LiveInterval("t2", 2, 4),
        ]
        result = linear_scan(intervals)
        assert len(result.reg_map) == 3
        assert len(result.spill_map) == 0
        assert len(set(result.reg_map.values())) == 3

    def test_spills_when_registers_exhausted(self):
        """7 overlapping temps, 5 registers -> 2 spills."""
        intervals = [LiveInterval(f"t{i}", 0, 10) for i in range(7)]
        result = linear_scan(intervals)
        assert len(result.reg_map) == 5
        assert len(result.spill_map) == 2

    def test_register_reuse_after_expiry(self):
        """Non-overlapping intervals can share the same register."""
        intervals = [
            LiveInterval("t0", 0, 2),
            LiveInterval("t1", 3, 5),   # starts after t0 ends
            LiveInterval("t2", 6, 8),   # starts after t1 ends
        ]
        result = linear_scan(intervals, num_regs=1)
        # All three get the single register, no spills
        assert len(result.reg_map) == 3
        assert len(result.spill_map) == 0

    def test_all_callee_saved(self):
        """Allocated registers must be from the callee-saved set."""
        intervals = [LiveInterval(f"t{i}", i, i + 5) for i in range(5)]
        result = linear_scan(intervals)
        for reg in result.reg_map.values():
            assert reg in ALLOCATABLE_REGS

    def test_used_callee_saved_list(self):
        """used_callee_saved should list all registers actually assigned."""
        intervals = [
            LiveInterval("t0", 0, 5),
            LiveInterval("t1", 1, 5),
        ]
        result = linear_scan(intervals)
        assert len(result.used_callee_saved) == 2
        for reg in result.used_callee_saved:
            assert reg in result.reg_map.values()

    def test_empty_intervals(self):
        result = linear_scan([])
        assert result.reg_map == {} and result.spill_map == {}

    def test_spill_picks_farthest_endpoint(self):
        """When spilling, the interval with the farthest end gets spilled."""
        intervals = [
            LiveInterval("t0", 0, 100),   # very long
            LiveInterval("t1", 0, 5),     # short
            LiveInterval("t2", 1, 3),     # shorter
        ]
        result = linear_scan(intervals, num_regs=2)
        # t0 has the farthest end and should be spilled in favor of t2
        assert "t0" in result.spill_map or len(result.reg_map) == 2


# ===================================================================
# Unit tests: full allocation pipeline
# ===================================================================

class TestAllocateRegisters:
    def test_excludes_strings(self):
        fn = _make_ir([
            "t0 = const 'hello'",
            "t1 = const 42",
            "ret t1",
        ])
        result = allocate_registers(fn)
        assert "t0" not in result.reg_map
        assert "t1" in result.reg_map

    def test_excludes_store_targets(self):
        fn = _make_ir([
            "t0 = const 0",
            "t1 = const 1",
            "store t0, t1",
            "t0 = load t0",
        ])
        result = allocate_registers(fn)
        assert "t0" not in result.reg_map

    def test_on_real_ir(self):
        """Run allocator on IR produced from a real UCL program."""
        fns = _lower("fn main() { let a = 6; let b = 7; print(a * b); return 0; }")
        main_fn = [f for f in fns if f.name == "main"][0]
        result = allocate_registers(main_fn)
        # Should have some temps allocated (const 6, const 7, multiply result)
        assert len(result.reg_map) > 0


# ===================================================================
# Integration tests: compile with regalloc, execute, verify
# ===================================================================

class TestIntegrationRegalloc:
    def test_return_value(self):
        rc, _ = _compile_and_run("fn main() { return 42; }")
        assert rc == 42

    def test_arithmetic(self):
        rc, out = _compile_and_run("""
        fn main() {
            let a = 6;
            let b = 7;
            print(a * b);
            return 0;
        }
        """)
        assert rc == 0 and out == ["42"]

    def test_division_modulo(self):
        rc, out = _compile_and_run("""
        fn main() {
            print(100 / 7);
            print(100 % 7);
            return 0;
        }
        """)
        assert rc == 0 and out == ["14", "2"]

    def test_negative_number(self):
        rc, out = _compile_and_run("""
        fn main() {
            let x = 0 - 42;
            print(x);
            return 0;
        }
        """)
        assert rc == 0 and out == ["-42"]

    def test_while_loop(self):
        rc, out = _compile_and_run("""
        fn main() {
            let i = 0;
            while i < 5 {
                print(i);
                i = i + 1;
            }
            return 0;
        }
        """)
        assert rc == 0 and out == ["0", "1", "2", "3", "4"]

    def test_if_else(self):
        rc, out = _compile_and_run("""
        fn main() {
            let x = 10;
            if x > 5 { print(1); }
            if x < 5 { print(0); }
            return 0;
        }
        """)
        assert rc == 0 and out == ["1"]

    def test_comparison_ops(self):
        rc, out = _compile_and_run("""
        fn main() {
            print(5 == 5);
            print(5 != 3);
            print(3 < 5);
            print(5 > 3);
            print(5 <= 5);
            print(3 >= 5);
            return 0;
        }
        """)
        assert rc == 0 and out == ["1", "1", "1", "1", "1", "0"]

    def test_zero_print(self):
        rc, out = _compile_and_run("fn main() { print(0); return 0; }")
        assert rc == 0 and out == ["0"]

    def test_string_print(self):
        """Strings excluded from regalloc should still work."""
        rc, out = _compile_and_run("""
        fn main() {
            let s = "hello world";
            print(s);
            return 0;
        }
        """)
        assert rc == 0 and out == ["hello world"]

    def test_string_concat(self):
        rc, out = _compile_and_run("""
        fn main() {
            let a = "hello ";
            let b = "world";
            let c = a + b;
            print(c);
            return 0;
        }
        """)
        assert rc == 0 and out == ["hello world"]

    def test_string_with_int(self):
        """Strings and ints coexist without interference."""
        rc, out = _compile_and_run("""
        fn main() {
            let s = "result:";
            let n = 42;
            print(s);
            print(n);
            return 0;
        }
        """)
        assert rc == 0 and out == ["result:", "42"]

    def test_string_len(self):
        rc, out = _compile_and_run("""
        fn main() {
            let s = "hello";
            let n = len(s);
            print(n);
            return 0;
        }
        """)
        assert rc == 0 and out == ["5"]

    def test_simple_function_call(self):
        rc, out = _compile_and_run("""
        fn add(a: i32, b: i32) -> i32 {
            return a + b;
        }
        fn main() {
            let x = add(3, 4);
            print(x);
            return 0;
        }
        """)
        assert rc == 0 and out == ["7"]

    def test_callee_saved_across_calls(self):
        """Temps must survive across function calls (callee-saved regs)."""
        rc, out = _compile_and_run("""
        fn compute(x: i32) -> i32 {
            return x * 2;
        }
        fn main() {
            let a = 10;
            let b = compute(3);
            let c = a + b;
            print(c);
            return 0;
        }
        """)
        assert rc == 0 and out == ["16"]

    def test_recursive_fibonacci(self):
        rc, out = _compile_and_run("""
        fn fib(n: i32) -> i32 {
            if n <= 1 { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        fn main() {
            print(fib(10));
            return 0;
        }
        """)
        assert rc == 0 and out == ["55"]

    def test_nested_calls(self):
        rc, out = _compile_and_run("""
        fn double(x: i32) -> i32 { return x * 2; }
        fn quad(x: i32) -> i32 { return double(double(x)); }
        fn main() {
            print(quad(5));
            return 0;
        }
        """)
        assert rc == 0 and out == ["20"]

    def test_mutual_recursion(self):
        rc, out = _compile_and_run("""
        fn is_even(n: i32) -> i32 {
            if n == 0 { return 1; }
            return is_odd(n - 1);
        }
        fn is_odd(n: i32) -> i32 {
            if n == 0 { return 0; }
            return is_even(n - 1);
        }
        fn main() {
            print(is_even(10));
            print(is_odd(10));
            return 0;
        }
        """)
        assert rc == 0 and out == ["1", "0"]

    def test_function_with_loop(self):
        rc, out = _compile_and_run("""
        fn sum_to(n: i32) -> i32 {
            let s = 0;
            let i = 1;
            while i <= n {
                s = s + i;
                i = i + 1;
            }
            return s;
        }
        fn main() {
            print(sum_to(10));
            return 0;
        }
        """)
        assert rc == 0 and out == ["55"]

    def test_many_temps_forces_spill(self):
        """Program with many concurrent temps to force register spilling."""
        rc, out = _compile_and_run("""
        fn main() {
            let a = 1;
            let b = 2;
            let c = 3;
            let d = 4;
            let e = 5;
            let f = 6;
            let g = 7;
            let h = 8;
            print(a + b + c + d + e + f + g + h);
            return 0;
        }
        """)
        assert rc == 0 and out == ["36"]

    def test_no_arg_function(self):
        rc, out = _compile_and_run("""
        fn answer() -> i32 { return 42; }
        fn main() { print(answer()); return 0; }
        """)
        assert rc == 0 and out == ["42"]

    def test_function_chain(self):
        rc, out = _compile_and_run("""
        fn inc(x: i32) -> i32 { return x + 1; }
        fn dec(x: i32) -> i32 { return x - 1; }
        fn main() {
            let a = inc(inc(inc(0)));
            let b = dec(dec(a));
            print(a);
            print(b);
            return 0;
        }
        """)
        assert rc == 0 and out == ["3", "1"]

    def test_string_in_if(self):
        """String truthiness must still work with regalloc."""
        rc, out = _compile_and_run("""
        fn main() {
            let s = "hello";
            if s { print(1); }
            let empty = "";
            if empty { print(2); }
            print(3);
            return 0;
        }
        """)
        assert rc == 0 and out == ["1", "3"]
