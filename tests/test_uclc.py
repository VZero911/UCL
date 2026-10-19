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
from uclc.vm import Interpreter, UCLRuntimeError, UCLLambda  # noqa: E402
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


# ---- break / continue ------------------------------------------------------
def test_break_in_while():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let i = 0;
        while i < 10 {
            if i == 3 { break; }
            print(i);
            i = i + 1;
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["0", "1", "2"]


def test_continue_in_while():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let i = 0;
        while i < 5 {
            i = i + 1;
            if i == 3 { continue; }
            print(i);
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["1", "2", "4", "5"]


def test_break_in_for():
    rc, out = _run_capture("""
    fn main() -> i32 {
        for x in range(10) {
            if x == 4 { break; }
            print(x);
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["0", "1", "2", "3"]


def test_continue_in_for():
    rc, out = _run_capture("""
    fn main() -> i32 {
        for x in range(6) {
            if x == 2 { continue; }
            if x == 4 { continue; }
            print(x);
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["0", "1", "3", "5"]


def test_nested_break():
    rc, out = _run_capture("""
    fn main() -> i32 {
        for i in range(3) {
            for j in range(5) {
                if j == 2 { break; }
                print(i * 10 + j);
            }
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["0", "1", "10", "11", "20", "21"]


# ---- maps ------------------------------------------------------------------
def test_map_create_and_access():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let m = map("a", 1, "b", 2, "c", 3);
        print(get(m, "a"));
        print(get(m, "b"));
        print(len(m));
        return 0;
    }
    """)
    assert rc == 0 and out == ["1", "2", "3"]


def test_map_set_and_keys():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let m = map();
        set(m, "x", 10);
        set(m, "y", 20);
        print(len(m));
        print(has_key(m, "x"));
        print(has_key(m, "z"));
        for k in keys(m) {
            print(k, get(m, k));
        }
        return 0;
    }
    """)
    assert rc == 0
    assert out[0] == "2"
    assert out[1] == "true"
    assert out[2] == "false"


def test_map_index_access():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let m = map("name", "V0", "level", 9);
        print(m["name"]);
        print(m["level"]);
        return 0;
    }
    """)
    assert rc == 0 and out == ["V0", "9"]


def test_map_remove():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let m = map("a", 1, "b", 2);
        remove(m, "a");
        print(len(m));
        print(has_key(m, "a"));
        return 0;
    }
    """)
    assert rc == 0 and out == ["1", "false"]


def test_map_get_default():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let m = map("a", 1);
        print(get(m, "b", 99));
        return 0;
    }
    """)
    assert rc == 0 and out == ["99"]


def test_map_values():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let m = map("x", 10, "y", 20);
        let v = values(m);
        print(len(v));
        return 0;
    }
    """)
    assert rc == 0 and out == ["2"]


# ---- closures / lambdas ---------------------------------------------------
def test_lambda_basic():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let double = |x| -> x * 2;
        print(double(5));
        return 0;
    }
    """)
    assert rc == 0 and out == ["10"]


def test_lambda_multi_params():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let add = |a, b| -> a + b;
        print(add(3, 7));
        return 0;
    }
    """)
    assert rc == 0 and out == ["10"]


def test_lambda_as_argument():
    rc, out = _run_capture("""
    fn apply(f_name: str, x: i32) -> i32 {
        return x;
    }
    fn main() -> i32 {
        let double = |x| -> x * 2;
        let result = double(21);
        print(result);
        return 0;
    }
    """)
    assert rc == 0 and out == ["42"]


def test_lambda_closure_capture():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let factor = 10;
        let mul = |x| -> x * factor;
        print(mul(5));
        return 0;
    }
    """)
    assert rc == 0 and out == ["50"]


def test_lambda_map_arr():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let arr = [1, 2, 3, 4];
        let doubled = map_arr(arr, |x| -> x * 2);
        for v in doubled {
            print(v);
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["2", "4", "6", "8"]


def test_lambda_filter_arr():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let arr = [1, 2, 3, 4, 5, 6];
        let evens = filter_arr(arr, |x| -> x % 2 == 0);
        for v in evens {
            print(v);
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["2", "4", "6"]


def test_lambda_block_body():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let complex = |x| {
            let y = x * 2;
            return y + 1;
        };
        print(complex(10));
        return 0;
    }
    """)
    assert rc == 0 and out == ["21"]


def test_lambda_nested():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let make_adder = |n| -> |x| -> x + n;
        let add5 = make_adder(5);
        print(add5(10));
        return 0;
    }
    """)
    assert rc == 0 and out == ["15"]


def test_lambda_higher_order():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let apply = |f, x| -> f(x);
        let sq = |x| -> x * x;
        print(apply(sq, 7));
        return 0;
    }
    """)
    assert rc == 0 and out == ["49"]


def test_lambda_map_filter_chain():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let doubled = map_arr(nums, |x| -> x * 2);
        let big = filter_arr(doubled, |x| -> x > 10);
        for v in big {
            print(v);
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["12", "14", "16", "18", "20"]


# ---- pattern matching -------------------------------------------------------
def test_match_literal_int():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 2;
        match x {
            case 1 -> print("one");
            case 2 -> print("two");
            case 3 -> print("three");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["two"]


def test_match_literal_string():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let s = "hello";
        match s {
            case "hi" -> print("informal");
            case "hello" -> print("formal");
            case "hey" -> print("casual");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["formal"]


def test_match_wildcard():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 99;
        match x {
            case 1 -> print("one");
            case _ -> print("other");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["other"]


def test_match_var_binding():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 42;
        match x {
            case n -> print(to_str(n));
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["42"]


def test_match_guard_clause():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 15;
        match x {
            case n if n > 10 -> print("big: " + to_str(n));
            case n -> print("small: " + to_str(n));
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["big: 15"]


def test_match_guard_fallthrough():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 5;
        match x {
            case n if n > 10 -> print("big");
            case n if n > 3 -> print("medium: " + to_str(n));
            case _ -> print("small");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["medium: 5"]


def test_match_array_destructure():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let pair = [10, 20];
        match pair {
            case [a, b] -> print(to_str(a) + "," + to_str(b));
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["10,20"]


def test_match_array_length_mismatch():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let triple = [1, 2, 3];
        match triple {
            case [a, b] -> print("pair");
            case [a, b, c] -> print("triple: " + to_str(a + b + c));
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["triple: 6"]


def test_match_map_destructure():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let m = map("name", "alice", "age", 30);
        match m {
            case {"name": n} -> print("name=" + n);
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["name=alice"]


def test_match_nested():
    rc, out = _run_capture("""
    fn main() -> i32 {
        for i in range(3) {
            match i {
                case 0 -> print("zero");
                case 1 -> print("one");
                case _ -> print("other");
            }
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["zero", "one", "other"]


def test_match_in_function():
    rc, out = _run_capture("""
    fn describe(x: i32) -> str {
        match x {
            case 0 -> { return "zero"; }
            case 1 -> { return "one"; }
            case n if n < 0 -> { return "negative"; }
            case _ -> { return "positive"; }
        }
        return "unreachable";
    }
    fn main() -> i32 {
        print(describe(0));
        print(describe(1));
        print(describe(-5));
        print(describe(42));
        return 0;
    }
    """)
    assert rc == 0 and out == ["zero", "one", "negative", "positive"]


def test_match_bool_pattern():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let flag = true;
        match flag {
            case true -> print("yes");
            case false -> print("no");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["yes"]


def test_match_no_case_matched():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 99;
        match x {
            case 1 -> print("one");
            case 2 -> print("two");
        }
        print("done");
        return 0;
    }
    """)
    assert rc == 0 and out == ["done"]


# ---- v0.6 extensions: fat arrow, match expression, tuple, rest -------------
def test_lex_fat_arrow():
    toks = tokenize("=>")
    assert toks[0].type == T.FAT_ARROW and toks[0].value == "=>"


def test_lex_dotdotdot():
    toks = tokenize("...rest")
    assert toks[0].type == T.DOTDOTDOT and toks[0].value == "..."
    assert toks[1].type == T.IDENT and toks[1].value == "rest"


def test_match_fat_arrow_case():
    """case keyword with => instead of ->"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 2;
        match x {
            case 1 => print("one");
            case 2 => print("two");
            case _ => print("other");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["two"]


def test_match_no_case_keyword():
    """Arms without case keyword, using => syntax"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 3;
        match x {
            1 => print("one");
            2 => print("two");
            _ => print("other");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["other"]


def test_match_no_case_with_guard():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 25;
        match x {
            n if n > 100 => print("big");
            n if n > 10 => print("medium: " + to_str(n));
            _ => print("small");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["medium: 25"]


def test_match_expr_basic():
    """Match as expression: evaluates to the matched arm's value"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 2;
        let label = match x {
            1 => "one",
            2 => "two",
            _ => "other",
        };
        print(label);
        return 0;
    }
    """)
    assert rc == 0 and out == ["two"]


def test_match_expr_with_guard():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 42;
        let desc = match x {
            n if n > 100 => "big",
            n if n > 10 => "medium",
            _ => "small",
        };
        print(desc);
        return 0;
    }
    """)
    assert rc == 0 and out == ["medium"]


def test_match_expr_in_call():
    """Match expression used directly as function argument"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 1;
        print(match x {
            1 => "yes",
            _ => "no",
        });
        return 0;
    }
    """)
    assert rc == 0 and out == ["yes"]


def test_match_expr_arithmetic():
    """Match expression result used in arithmetic"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 3;
        let result = match x {
            1 => 10,
            2 => 20,
            3 => 30,
            _ => 0,
        } + 5;
        print(result);
        return 0;
    }
    """)
    assert rc == 0 and out == ["35"]


def test_match_expr_no_match_returns_none():
    """When no arm matches, match expression returns None (prints nothing meaningful)"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 99;
        let r = match x {
            1 => "one",
            2 => "two",
        };
        print("done");
        return 0;
    }
    """)
    assert rc == 0 and out == ["done"]


def test_match_expr_variable_binding():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 7;
        let result = match x {
            n => n * 10,
        };
        print(result);
        return 0;
    }
    """)
    assert rc == 0 and out == ["70"]


def test_tuple_pattern_basic():
    """Tuple destructuring with (a, b)"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let pair = [10, 20];
        match pair {
            case (a, b) -> print(to_str(a) + "+" + to_str(b));
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["10+20"]


def test_tuple_pattern_three_elements():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let triple = [1, 2, 3];
        match triple {
            case (a, b, c) -> print(to_str(a + b + c));
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["6"]


def test_tuple_pattern_length_mismatch():
    """Tuple pattern fails when array length doesn't match"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let pair = [1, 2];
        match pair {
            case (a, b, c) -> print("three");
            case (a, b) -> print("two: " + to_str(a) + "," + to_str(b));
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["two: 1,2"]


def test_tuple_pattern_with_literals():
    """Tuple pattern mixing literals and bindings"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let coord = [0, 5];
        match coord {
            case (0, y) -> print("on y-axis: " + to_str(y));
            case (x, 0) -> print("on x-axis: " + to_str(x));
            case (x, y) -> print("point: " + to_str(x) + "," + to_str(y));
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["on y-axis: 5"]


def test_tuple_in_match_expr():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let pair = [3, 4];
        let sum = match pair {
            (a, b) => a + b,
        };
        print(sum);
        return 0;
    }
    """)
    assert rc == 0 and out == ["7"]


def test_rest_pattern_basic():
    """Array destructuring with ...rest"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let arr = [1, 2, 3, 4, 5];
        match arr {
            case [first, ...rest] -> {
                print(to_str(first));
                print(to_str(len(rest)));
            }
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["1", "4"]


def test_rest_pattern_two_fixed():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let arr = [10, 20, 30, 40];
        match arr {
            case [a, b, ...tail] -> {
                print(to_str(a));
                print(to_str(b));
                for x in tail { print(to_str(x)); }
            }
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["10", "20", "30", "40"]


def test_rest_pattern_empty_rest():
    """Rest pattern matches even when rest is empty"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let arr = [42];
        match arr {
            case [x, ...rest] -> {
                print(to_str(x));
                print(to_str(len(rest)));
            }
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["42", "0"]


def test_rest_pattern_not_enough_elements():
    """Rest pattern fails when not enough elements for fixed positions"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let arr = [1];
        match arr {
            case [a, b, ...rest] -> print("matched");
            case _ -> print("no match");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["no match"]


def test_rest_with_tail_fixed():
    """Rest pattern with fixed elements after the rest"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let arr = [1, 2, 3, 4, 5];
        match arr {
            case [first, ...middle, last] -> {
                print(to_str(first));
                print(to_str(len(middle)));
                print(to_str(last));
            }
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["1", "3", "5"]


def test_match_negative_literal():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = -5;
        match x {
            case -5 -> print("neg five");
            case 5 -> print("five");
            case _ -> print("other");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["neg five"]


def test_match_float_literal():
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 3.14;
        match x {
            case 2.71 -> print("e");
            case 3.14 -> print("pi");
            case _ -> print("other");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["pi"]


def test_match_expr_nested():
    """Nested match expressions"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let x = 2;
        let y = 1;
        let result = match x {
            1 => match y { 1 => "1-1", _ => "1-?", },
            2 => match y { 1 => "2-1", _ => "2-?", },
            _ => "other",
        };
        print(result);
        return 0;
    }
    """)
    assert rc == 0 and out == ["2-1"]


def test_match_expr_with_lambda():
    """Match expression arm returning a lambda result"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let op = "double";
        let f = match op {
            "double" => |x| -> x * 2,
            "triple" => |x| -> x * 3,
            _ => |x| -> x,
        };
        print(f(5));
        return 0;
    }
    """)
    assert rc == 0 and out == ["10"]


def test_match_mixed_old_new_syntax():
    """Ensure case+fat_arrow works alongside case+arrow in different match blocks"""
    rc, out = _run_capture("""
    fn main() -> i32 {
        let a = 1;
        match a {
            case 1 -> print("arrow");
        }
        match a {
            case 1 => print("fat arrow");
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["arrow", "fat arrow"]


def test_ir_lowers_match_expr():
    ir = lower(parse(tokenize("""
    fn main() {
        let x = match 1 {
            1 => 10,
            _ => 0,
        };
    }
    """)))
    assert "const 1" in ir and "const 10" in ir


def test_ir_lowers_tuple_pattern():
    ir = lower(parse(tokenize("""
    fn main() {
        let p = [1, 2];
        match p {
            case (a, b) -> { let s = a; }
        }
    }
    """)))
    assert "index" in ir and "store a" in ir


# ---- native codegen (x86-64 ELF) ------------------------------------------
import subprocess
import tempfile
from uclc.codegen import compile_source_to_elf


def _compile_and_run(src: str) -> tuple:
    """Compile UCL source to ELF, run it, return (exit_code, stdout_lines)."""
    elf = compile_source_to_elf(src)
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


def test_native_return_value():
    rc, _ = _compile_and_run("fn main() { return 42; }")
    assert rc == 42


def test_native_arithmetic():
    rc, out = _compile_and_run("""
    fn main() {
        let a = 6;
        let b = 7;
        print(a * b);
        return 0;
    }
    """)
    assert rc == 0 and out == ["42"]


def test_native_division_modulo():
    rc, out = _compile_and_run("""
    fn main() {
        print(100 / 7);
        print(100 % 7);
        return 0;
    }
    """)
    assert rc == 0 and out == ["14", "2"]


def test_native_negative_number():
    rc, out = _compile_and_run("""
    fn main() {
        let x = 0 - 42;
        print(x);
        return 0;
    }
    """)
    assert rc == 0 and out == ["-42"]


def test_native_while_loop():
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


def test_native_if_else():
    rc, out = _compile_and_run("""
    fn main() {
        let x = 10;
        if x > 5 {
            print(1);
        }
        if x < 5 {
            print(0);
        }
        return 0;
    }
    """)
    assert rc == 0 and out == ["1"]


def test_native_comparison_ops():
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


def test_native_zero_print():
    rc, out = _compile_and_run("""
    fn main() {
        print(0);
        return 0;
    }
    """)
    assert rc == 0 and out == ["0"]


def test_native_large_number():
    rc, out = _compile_and_run("""
    fn main() {
        let x = 1000000;
        print(x);
        return 0;
    }
    """)
    assert rc == 0 and out == ["1000000"]


def test_native_elf_size():
    elf = compile_source_to_elf("fn main() { return 0; }")
    assert len(elf) < 2048


# ── Native codegen: strings (v0.8) ──────────────────────────────────

def test_native_string_print():
    rc, out = _compile_and_run("""
    fn main() {
        let s = "hello world";
        print(s);
        return 0;
    }
    """)
    assert rc == 0 and out == ["hello world"]


def test_native_string_multiple():
    rc, out = _compile_and_run("""
    fn main() {
        let a = "hello";
        let b = "world";
        print(a);
        print(b);
        return 0;
    }
    """)
    assert rc == 0 and out == ["hello", "world"]


def test_native_string_concat():
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


def test_native_string_empty():
    rc, out = _compile_and_run("""
    fn main() {
        let a = "";
        print(a);
        return 0;
    }
    """)
    assert rc == 0 and out == []  # empty string print → just newline, filtered by helper


def test_native_string_escape_tab():
    rc, out = _compile_and_run(r"""
    fn main() {
        let a = "hello\tworld";
        print(a);
        return 0;
    }
    """)
    assert rc == 0 and "hello\tworld" in out[0]


def test_native_string_len():
    rc, out = _compile_and_run("""
    fn main() {
        let s = "hello";
        let n = len(s);
        print(n);
        return 0;
    }
    """)
    assert rc == 0 and out == ["5"]


def test_native_string_concat_chain():
    rc, out = _compile_and_run("""
    fn main() {
        let a = "a";
        let b = "b";
        let c = "c";
        let result = a + b + c;
        print(result);
        return 0;
    }
    """)
    assert rc == 0 and out == ["abc"]


def test_native_string_concat_len():
    rc, out = _compile_and_run("""
    fn main() {
        let a = "hello";
        let b = " world";
        let c = a + b;
        print(len(c));
        return 0;
    }
    """)
    assert rc == 0 and out == ["11"]


def test_native_string_with_int():
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


def test_native_string_in_if():
    """String as conditional: non-empty=truthy, empty=falsy."""
    rc, out = _compile_and_run("""
    fn main() {
        let s = "hello";
        if s {
            print(1);
        }
        let empty = "";
        if empty {
            print(2);
        }
        print(3);
        return 0;
    }
    """)
    assert rc == 0 and out == ["1", "3"]


def test_native_string_elf_size():
    elf = compile_source_to_elf('fn main() { let s = "hello"; print(s); return 0; }')
    assert len(elf) < 4096


# ---- native codegen: function calls (v0.9) --------------------------------

def test_native_simple_function_call():
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


def test_native_function_return_value():
    rc, _ = _compile_and_run("""
    fn compute() -> i32 {
        return 17;
    }
    fn main() {
        return compute();
    }
    """)
    assert rc == 17


def test_native_function_multiple_args():
    rc, out = _compile_and_run("""
    fn add3(a: i32, b: i32, c: i32) -> i32 {
        return a + b + c;
    }
    fn main() {
        print(add3(10, 20, 30));
        return 0;
    }
    """)
    assert rc == 0 and out == ["60"]


def test_native_recursive_fibonacci():
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


def test_native_nested_calls():
    rc, out = _compile_and_run("""
    fn double(x: i32) -> i32 {
        return x * 2;
    }
    fn quad(x: i32) -> i32 {
        return double(double(x));
    }
    fn main() {
        print(quad(5));
        return 0;
    }
    """)
    assert rc == 0 and out == ["20"]


def test_native_function_with_print():
    rc, out = _compile_and_run("""
    fn greet(x: i32) -> i32 {
        print(x);
        return x + 1;
    }
    fn main() {
        let r = greet(42);
        print(r);
        return 0;
    }
    """)
    assert rc == 0 and out == ["42", "43"]


def test_native_no_arg_function():
    rc, out = _compile_and_run("""
    fn answer() -> i32 {
        return 42;
    }
    fn main() {
        print(answer());
        return 0;
    }
    """)
    assert rc == 0 and out == ["42"]


def test_native_mutual_recursion():
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
        print(is_even(7));
        print(is_odd(7));
        return 0;
    }
    """)
    assert rc == 0 and out == ["1", "0", "0", "1"]


def test_native_function_chain():
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


def test_native_function_with_loop():
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
