"""UCL compiler test suite: lexer, parser, IR lowering, evaluator, CLI."""
import io
import os
import sys
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uclc.lexer import tokenize, LexError  # noqa: E402
from uclc.tokens import T  # noqa: E402
from uclc.parser import parse, ParseError  # noqa: E402
from uclc import ast as A  # noqa: E402
from uclc.ir import lower  # noqa: E402
from uclc.vm import Interpreter, UCLRuntimeError  # noqa: E402
from uclc import cli, compile_source  # noqa: E402
import pytest  # noqa: E402


# ---- lexer -----------------------------------------------------------------
def test_lex_basic_tokens():
    toks = tokenize("fn main() -> i32 { return 0; }")
    types = [t.type for t in toks]
    assert T.FN in types and T.ARROW in types and T.TYPE in types and types[-1] == T.EOF


def test_lex_numbers():
    toks = tokenize("1 2.5 3e2 0.1")
    assert toks[0].type == T.INT and toks[0].value == 1
    assert toks[1].type == T.FLOAT and toks[1].value == 2.5
    assert toks[2].type == T.FLOAT and toks[2].value == 300.0


def test_lex_string_and_escapes():
    toks = tokenize(r'"a\nb\"c"')
    assert toks[0].type == T.STRING and toks[0].value == 'a\nb"c'


def test_lex_comments_ignored():
    toks = tokenize("fn x() {} // trailing\n/* block */ fn y() {}")
    names = [t.value for t in toks if t.type == T.IDENT]
    assert names == ["x", "y"]


def test_lex_unterminated_string():
    with pytest.raises(LexError):
        tokenize('"oops')


def test_lex_ai_native_types_are_type_tokens():
    toks = tokenize("tensor weights tokenstream graph")
    assert all(t.type == T.TYPE for t in toks[:4])


# ---- parser ----------------------------------------------------------------
def test_parse_fn_signature():
    prog = parse(tokenize("fn add(a: i64, b: i64) -> i64 { return a + b; }"))
    fn = prog.decls[0]
    assert isinstance(fn, A.Fn) and fn.name == "add"
    assert [p.name for p in fn.params] == ["a", "b"] and fn.ret_type == "i64"


def test_parse_precedence():
    prog = parse(tokenize("fn m() { let x = 1 + 2 * 3; }"))
    let = prog.decls[0].body.stmts[0]
    # top node is + with right = (2 * 3)
    assert isinstance(let.value, A.Binary) and let.value.op == "+"
    assert isinstance(let.value.right, A.Binary) and let.value.right.op == "*"


def test_parse_if_while():
    prog = parse(tokenize("fn m() { while a < 10 { if a { return 1; } else { return 2; } } }"))
    w = prog.decls[0].body.stmts[0]
    assert isinstance(w, A.While) and isinstance(w.body.stmts[0], A.If)


def test_parse_class_with_fields_and_methods():
    prog = parse(tokenize("class N { bias: f32; fn forward(x: f32) -> f32 { return x; } }"))
    c = prog.decls[0]
    assert isinstance(c, A.Class) and c.fields[0].name == "bias" and c.methods[0].name == "forward"


def test_parse_member_and_method_call():
    prog = parse(tokenize("fn m() { let y = n.forward(x); let z = n.bias; }"))
    stmts = prog.decls[0].body.stmts
    assert isinstance(stmts[0].value, A.MethodCall) and stmts[0].value.name == "forward"
    assert isinstance(stmts[1].value, A.Member) and stmts[1].value.name == "bias"


def test_parse_generic_type_args_skipped():
    prog = parse(tokenize("fn m(x: tensor<f32, [4]>) -> f32 { return 0.0; }"))
    assert prog.decls[0].params[0].type_name == "tensor"


def test_parse_error_missing_semicolon():
    with pytest.raises(ParseError):
        parse(tokenize("fn m() { let x = 1 }"))


# ---- IR --------------------------------------------------------------------
def test_ir_lowers_arithmetic():
    ir = lower(parse(tokenize("fn m() { let x = 1 + 2; }")))
    assert "const 1" in ir and "+ " in ir and "store x" in ir


def test_ir_lowers_control_flow_labels():
    ir = lower(parse(tokenize("fn m() { while 1 { } }")))
    assert "jmpf" in ir and "jmp" in ir and "L0:" in ir


def test_ir_lowers_method_call():
    ir = lower(parse(tokenize("fn m() { let y = n.forward(x); }")))
    assert "callmethod" in ir and "getfield" not in ir  # forward is a call, not a field


# ---- evaluator -------------------------------------------------------------
def _run_capture(src):
    interp = Interpreter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = interp.run(parse(tokenize(src)))
    return rc, interp.output


def test_eval_hello():
    rc, out = _run_capture('fn main() -> i32 { print("hi"); return 0; }')
    assert rc == 0 and out == ["hi"]


def test_eval_arithmetic_and_vars():
    rc, out = _run_capture('fn main() { let a = 6; let b = 7; print(a * b); }')
    assert out == ["42"]


def test_eval_recursion_fib():
    src = """
    fn fib(n: i64) -> i64 { if n < 2 { return n; } return fib(n-1) + fib(n-2); }
    fn main() -> i32 { print(fib(10)); return 0; }
    """
    rc, out = _run_capture(src)
    assert out == ["55"]


def test_eval_while_loop():
    src = "fn main() { let i = 0; let s = 0; while i < 5 { s = s + i; i = i + 1; } print(s); }"
    _, out = _run_capture(src)
    assert out == ["10"]


def test_eval_bool_and_short_circuit():
    _, out = _run_capture('fn main() { if true && (1 < 2) { print("yes"); } }')
    assert out == ["yes"]


def test_eval_int_div_by_zero():
    with pytest.raises(UCLRuntimeError):
        _run_capture("fn main() { print(1 / 0); }")


def test_eval_string_upper():
    _, out = _run_capture('fn main() { print(upper("hello")); }')
    assert out == ["HELLO"]


def test_eval_string_lower():
    _, out = _run_capture('fn main() { print(lower("WORLD")); }')
    assert out == ["world"]


def test_eval_string_split():
    _, out = _run_capture('fn main() { let parts = split("a,b,c", ","); print(len(parts)); print(parts[0]); }')
    assert out == ["3", "a"]


def test_eval_string_contains():
    _, out = _run_capture('fn main() { if contains("hello world", "world") { print("yes"); } }')
    assert out == ["yes"]


def test_eval_string_substr():
    _, out = _run_capture('fn main() { print(substr("abcdef", 2, 3)); }')
    assert out == ["cde"]


def test_eval_string_trim():
    _, out = _run_capture('fn main() { print(trim("  hi  ")); }')
    assert out == ["hi"]


def test_eval_to_str():
    _, out = _run_capture('fn main() { print(to_str(42)); }')
    assert out == ["42"]


def test_eval_to_int():
    _, out = _run_capture('fn main() { let x = to_int("7"); print(x + 3); }')
    assert out == ["10"]


def test_eval_to_float():
    _, out = _run_capture('fn main() { let x = to_float("3.14"); print(x); }')
    assert out == ["3.14"]


def test_eval_contains_array():
    _, out = _run_capture('fn main() { let arr = [1, 2, 3]; if contains(arr, 2) { print("found"); } }')
    assert out == ["found"]


def test_eval_ai_builtin_is_roadmap_error():
    with pytest.raises(UCLRuntimeError):
        _run_capture("fn main() { let t = tensor(1.0); }")


# ---- CLI -------------------------------------------------------------------
def _ex(path):
    return os.path.join(os.path.dirname(__file__), "..", "examples", path)


def test_cli_run_hello(capsys):
    assert cli.main(["run", _ex("hello.ucl")]) == 0
    assert "Hello" in capsys.readouterr().out


def test_cli_check_ok(capsys):
    assert cli.main(["check", _ex("fib.ucl")]) == 0


def test_cli_check_neuron_object_code(capsys):
    # object/AI code must at least PARSE (check) even though it can't run yet
    assert cli.main(["check", _ex("neuron.ucl")]) == 0


def test_cli_emit_ir(capsys):
    assert cli.main(["emit-ir", _ex("neuron.ucl")]) == 0
    assert "callmethod" in capsys.readouterr().out


def test_cli_version(capsys):
    assert cli.main(["--version"]) == 0
    assert "uclc" in capsys.readouterr().out


def test_compile_source_helper():
    prog = compile_source("fn main() { }")
    assert isinstance(prog, A.Program)


# ---- runnable examples (procedural subset) ---------------------------------
def _run_example(name):
    with open(_ex(name), "r", encoding="utf-8") as fh:
        return _run_capture(fh.read())


def test_example_fizzbuzz():
    rc, out = _run_example("fizzbuzz.ucl")
    assert rc == 0
    assert out == [
        "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
        "11", "Fizz", "13", "14", "FizzBuzz",
    ]


def test_example_factorial_recursive_matches_iterative():
    rc, out = _run_example("factorial.ucl")
    assert rc == 0 and len(out) == 12
    assert out[0] == "fact 1 = 1 iter 1"
    assert out[-1] == "fact 12 = 479001600 iter 479001600"
    for line in out:  # recursive and iterative results agree on every row
        rec, it = line.split(" = ")[1].split(" iter ")
        assert rec == it


def test_example_prime_below_50():
    rc, out = _run_example("prime.ucl")
    assert rc == 0
    primes = [int(line.split()[1]) for line in out[:-1]]
    assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    assert out[-1] == "count 15"


def test_cli_run_examples(capsys):
    for name in ("fizzbuzz.ucl", "factorial.ucl", "prime.ucl"):
        assert cli.main(["run", _ex(name)]) == 0
    assert "FizzBuzz" in capsys.readouterr().out
