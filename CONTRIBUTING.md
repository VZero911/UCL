# Contributing to UCL

Thanks for helping build UCL. The reference toolchain is intentionally small and
stdlib-only, so changes should stay readable and dependency-free.

## Requirements

- Python 3.10+ (CI runs 3.11)
- `pytest` for the test suite (pulled in by the `test` extra)
- `make` (optional; every target is a thin wrapper over a plain command)

## Build, test, lint

```bash
make install     # pip install -e ".[test]"  (editable install + pytest)
make test        # python3 -m pytest -q
make lint        # byte-compile src/tests and parse-check every example
make examples    # run every runnable example program
make run FILE=examples/fib.ucl
```

Without `make`:

```bash
pip install -e ".[test]"
python3 -m pytest -q
PYTHONPATH=src python3 -m uclc.cli run examples/fizzbuzz.ucl
```

CI (`.github/workflows/ci.yml`) runs the test suite on every push and pull request.
A change is ready when `make test` and `make lint` both pass locally.

## The `uclc` CLI

| Command | What it does |
|---|---|
| `uclc run <file.ucl>` | Run the program (calls `main`, exit code = returned `i32`) |
| `uclc check <file.ucl>` | Parse only; prints `OK` or the first error |
| `uclc ast <file.ucl>` | Dump the parsed AST |
| `uclc emit-ir <file.ucl>` | Lower to the textual three-address IR |
| `uclc tokens <file.ucl>` | Dump the token stream |
| `uclc --version` | Print the version |

Without installing, use `PYTHONPATH=src python3 -m uclc.cli <command> <file>`.

## Project layout

```
src/uclc/
  tokens.py    token kinds and keyword table
  lexer.py     source text -> tokens
  parser.py    tokens -> AST (precedence climbing, if/while/fn/class)
  ast.py       AST node definitions
  ir.py        AST -> three-address IR (textual)
  vm.py        reference tree-walking evaluator (procedural subset)
  cli.py       the uclc command-line driver
  __init__.py  public API (tokenize, parse, lower, run, Interpreter)
tests/test_uclc.py   lexer, parser, IR, evaluator, CLI and example tests
examples/            .ucl programs (see below)
docs/                SPEC.md, GRAMMAR.ebnf, ROADMAP.md, THECORE_INTEGRATION.md
```

## Adding an example

1. Put the program in `examples/<name>.ucl` with a short header comment.
2. If it is procedural and runnable today, add a test in `tests/test_uclc.py`
   that asserts its exact output. `make examples` picks it up automatically.
3. If it uses object or AI-native features that the evaluator cannot run yet,
   note that in the header (as `neuron.ucl` does). `make lint` still parse-checks it.

## Changing the language

- Update `docs/GRAMMAR.ebnf` and `docs/SPEC.md` alongside any syntax change.
- Add a parser test and an evaluator test for new behaviour.
- Keep the IR and the evaluator in agreement: the evaluator is the reference semantics.
- Roadmap items live in `docs/ROADMAP.md`; note in the PR which phase a change belongs to.

## Pull requests

- Keep PRs focused; one feature or fix per PR.
- Run `make test && make lint` before pushing.
- Add a line to `CHANGELOG.md` under `Unreleased`.
- New runtime dependencies are not accepted for the reference toolchain.

By contributing you agree that your work is released under the MIT license (see `LICENSE`).
