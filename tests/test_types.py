"""UCL v0.2 type system tests: type representations, checker, CLI integration."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uclc.lexer import tokenize  # noqa: E402
from uclc.parser import parse  # noqa: E402
from uclc.checker import check, TypeDiag  # noqa: E402
from uclc.types import (  # noqa: E402
    IntType, FloatType, BoolType, StrType, VoidType, PtrType,
    TensorType, FnType, ClassType,
    parse_type, promote, assignable, is_numeric,
    BOOL, STR, VOID, PTR, I32, I64, F32, F64,
)
from uclc import cli  # noqa: E402
import pytest  # noqa: E402


# ---- type representations ---------------------------------------------------

def test_int_type_eq():
    assert IntType(32, True) == IntType(32, True)
    assert IntType(32, True) != IntType(64, True)
    assert IntType(32, True) != IntType(32, False)


def test_int_type_repr():
    assert repr(IntType(32, True)) == "i32"
    assert repr(IntType(8, False)) == "u8"
    assert repr(IntType(64, True)) == "i64"


def test_float_type_eq():
    assert FloatType(32) == FloatType(32)
    assert FloatType(32) != FloatType(64)


def test_float_type_repr():
    assert repr(FloatType(32)) == "f32"
    assert repr(FloatType(64)) == "f64"


def test_singleton_types():
    assert BOOL == BoolType()
    assert STR == StrType()
    assert VOID == VoidType()
    assert PTR == PtrType()
    assert repr(BOOL) == "bool"
    assert repr(STR) == "str"
    assert repr(VOID) == "void"
    assert repr(PTR) == "ptr"


def test_tensor_type():
    t = TensorType(F32, [4, 4])
    assert t.dtype == F32
    assert t.shape == [4, 4]
    assert t == TensorType(F32, [4, 4])
    assert t != TensorType(F64, [4, 4])


def test_fn_type():
    ft = FnType([I64, I64], I64)
    assert ft == FnType([I64, I64], I64)
    assert ft != FnType([I64], I64)
    assert "fn(" in repr(ft)


def test_class_type():
    ct = ClassType("Foo", {"x": I32}, {"bar": FnType([], VOID)})
    assert ct == ClassType("Foo")
    assert ct != ClassType("Bar")
    assert repr(ct) == "Foo"


def test_type_hashing():
    s = {I32, I64, F32, F64, BOOL, STR, VOID, PTR}
    assert len(s) == 8


# ---- parse_type -------------------------------------------------------------

def test_parse_type_scalars():
    assert parse_type("i32") == I32
    assert parse_type("u64") == IntType(64, False)
    assert parse_type("f32") == F32
    assert parse_type("bool") == BOOL
    assert parse_type("str") == STR
    assert parse_type("void") == VOID
    assert parse_type("ptr") == PTR


def test_parse_type_ai_native():
    assert isinstance(parse_type("tensor"), TensorType)
    assert isinstance(parse_type("weights"), TensorType)
    assert parse_type("tokenstream") == STR
    assert parse_type("quant") == IntType(8, False)


def test_parse_type_unknown():
    assert parse_type("foobar") is None
    assert parse_type("MyClass") is None


# ---- promotion -------------------------------------------------------------

def test_promote_same_type():
    assert promote(I64, I64) == I64
    assert promote(F32, F32) == F32


def test_promote_int_widening():
    assert promote(I32, I64) == I64
    assert promote(I64, I32) == I64


def test_promote_int_to_float():
    assert promote(I32, F64) == F64
    assert promote(F32, I64) == F32


def test_promote_float_widening():
    assert promote(F32, F64) == F64


def test_promote_incompatible():
    assert promote(BOOL, I64) is None
    assert promote(STR, F32) is None


# ---- assignability ----------------------------------------------------------

def test_assignable_same():
    assert assignable(I64, I64)
    assert assignable(BOOL, BOOL)
    assert assignable(STR, STR)


def test_assignable_int_to_int():
    assert assignable(I32, I64)
    assert assignable(I64, I32)


def test_assignable_float_to_float():
    assert assignable(F32, F64)


def test_assignable_int_to_float():
    assert assignable(F64, I32)


def test_assignable_incompatible():
    assert not assignable(I64, STR)
    assert not assignable(BOOL, I64)
    assert not assignable(STR, I64)


def test_is_numeric():
    assert is_numeric(I64)
    assert is_numeric(F32)
    assert not is_numeric(BOOL)
    assert not is_numeric(STR)


# ---- checker: valid programs (no errors) ------------------------------------

def _check_src(src: str):
    return check(parse(tokenize(src)))


def _ex(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "examples", name)


def _check_file(name: str):
    with open(_ex(name), encoding="utf-8") as f:
        return _check_src(f.read())


def test_check_hello_no_errors():
    assert _check_file("hello.ucl") == []


def test_check_fib_no_errors():
    assert _check_file("fib.ucl") == []


def test_check_factorial_no_errors():
    assert _check_file("factorial.ucl") == []


def test_check_fizzbuzz_no_errors():
    assert _check_file("fizzbuzz.ucl") == []


def test_check_prime_no_errors():
    assert _check_file("prime.ucl") == []


def test_check_neuron_no_errors():
    assert _check_file("neuron.ucl") == []


def test_check_minimal_void_fn():
    assert _check_src("fn main() { }") == []


def test_check_let_inferred():
    assert _check_src("fn main() { let x = 5; print(x); }") == []


def test_check_let_annotated():
    assert _check_src("fn main() { let x: i64 = 5; print(x); }") == []


def test_check_bool_return():
    diags = _check_src("fn test() -> bool { return true; } fn main() { test(); }")
    assert diags == []


def test_check_string_ops():
    assert _check_src('fn main() { let s: str = "hello"; print(s); }') == []


# ---- checker: error detection -----------------------------------------------

def test_err_wrong_return_type():
    diags = _check_src('fn main() -> i32 { return "hello"; }')
    errors = [d for d in diags if d.severity == "error"]
    assert len(errors) >= 1
    assert any("return type" in d.message for d in errors)


def test_err_wrong_arg_type():
    diags = _check_src("""
        fn add(a: i64, b: i64) -> i64 { return a + b; }
        fn main() { add("x", "y"); }
    """)
    errors = [d for d in diags if d.severity == "error"]
    assert len(errors) >= 1


def test_err_arity_mismatch():
    diags = _check_src("""
        fn add(a: i64, b: i64) -> i64 { return a + b; }
        fn main() { add(1); }
    """)
    errors = [d for d in diags if d.severity == "error"]
    assert any("expects 2" in d.message for d in errors)


def test_err_arity_too_many():
    diags = _check_src("""
        fn inc(a: i64) -> i64 { return a + 1; }
        fn main() { inc(1, 2, 3); }
    """)
    errors = [d for d in diags if d.severity == "error"]
    assert any("expects 1" in d.message for d in errors)


def test_err_undefined_variable():
    diags = _check_src("fn main() { print(xyz); }")
    errors = [d for d in diags if d.severity == "error"]
    assert any("undefined" in d.message for d in errors)


def test_err_type_annotation_mismatch():
    diags = _check_src("fn main() { let x: bool = 42; }")
    errors = [d for d in diags if d.severity == "error"]
    assert any("cannot assign" in d.message for d in errors)


def test_err_assign_type_mismatch():
    diags = _check_src('fn main() { let x: str = "hi"; x = 42; }')
    errors = [d for d in diags if d.severity == "error"]
    assert any("cannot assign" in d.message for d in errors)


def test_err_unknown_function():
    diags = _check_src("fn main() { nonexistent(1); }")
    errors = [d for d in diags if d.severity == "error"]
    assert any("unknown function" in d.message for d in errors)


def test_err_unknown_type():
    diags = _check_src("fn main() { let x: FooBarBaz = 1; }")
    errors = [d for d in diags if d.severity == "error"]
    assert any("unknown type" in d.message for d in errors)


def test_err_binary_op_str_minus():
    diags = _check_src('fn main() { let x = "a" - "b"; }')
    errors = [d for d in diags if d.severity == "error"]
    assert len(errors) >= 1


def test_err_compare_str_int():
    diags = _check_src('fn main() { let x = "a" < 1; }')
    errors = [d for d in diags if d.severity == "error"]
    assert any("cannot compare" in d.message for d in errors)


def test_err_unary_minus_bool():
    diags = _check_src("fn main() { let x = -true; }")
    errors = [d for d in diags if d.severity == "error"]
    assert any("unary" in d.message for d in errors)


def test_err_assign_undefined():
    diags = _check_src("fn main() { ghost = 42; }")
    errors = [d for d in diags if d.severity == "error"]
    assert any("undefined" in d.message for d in errors)


# ---- int/float promotion in expressions ------------------------------------

def test_int_promotion_in_binary():
    # i32 + i64 should work without error
    diags = _check_src("""
        fn take32(x: i32) -> i32 { return x; }
        fn main() {
            let a: i32 = 1;
            let b: i64 = 2;
            let c = a + b;
        }
    """)
    assert diags == []


def test_int_to_float_promotion():
    diags = _check_src("""
        fn main() {
            let a: i32 = 1;
            let b: f64 = 2.0;
            let c = a + b;
        }
    """)
    assert diags == []


# ---- AST span info ---------------------------------------------------------

def test_ast_fn_has_span():
    prog = parse(tokenize("fn main() { let x = 1; }"))
    fn = prog.decls[0]
    assert fn.line == 1 and fn.col == 1


def test_ast_let_has_span():
    prog = parse(tokenize("fn main() { let x = 1; }"))
    let_stmt = prog.decls[0].body.stmts[0]
    assert let_stmt.line > 0 and let_stmt.col > 0


def test_ast_literal_has_span():
    prog = parse(tokenize("fn main() { let x = 42; }"))
    lit = prog.decls[0].body.stmts[0].value
    assert lit.line > 0


def test_diag_has_source_span():
    diags = _check_src("fn main() { print(xyz); }")
    assert len(diags) > 0
    assert diags[0].line > 0 and diags[0].col > 0


# ---- CLI typecheck command --------------------------------------------------

def test_cli_typecheck_ok(capsys):
    assert cli.main(["typecheck", _ex("hello.ucl")]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_typecheck_fib_ok(capsys):
    assert cli.main(["typecheck", _ex("fib.ucl")]) == 0


def test_cli_typecheck_neuron_ok(capsys):
    assert cli.main(["typecheck", _ex("neuron.ucl")]) == 0


def test_cli_typecheck_all_examples(capsys):
    for name in ("hello.ucl", "fib.ucl", "factorial.ucl", "fizzbuzz.ucl", "prime.ucl", "neuron.ucl"):
        assert cli.main(["typecheck", _ex(name)]) == 0


def test_cli_typecheck_error(capsys, tmp_path):
    bad = tmp_path / "bad.ucl"
    bad.write_text('fn main() -> i32 { return "oops"; }')
    assert cli.main(["typecheck", str(bad)]) == 1
    captured = capsys.readouterr()
    assert "error" in captured.err
