# Changelog

All notable changes to UCL are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- GitHub Actions CI (`.github/workflows/ci.yml`): runs `pytest` on Python 3.11 for every push and pull request.
- `Makefile` with `install`, `test`, `run`, `lint`, `examples` and `clean` targets.
- `CONTRIBUTING.md` covering setup, the `uclc` CLI, project layout and how to add examples.
- Runnable examples: `fizzbuzz.ucl` (loops, modulo, else-if), `factorial.ucl` (recursive and iterative), `prime.ucl` (trial-division primality).
- `test` optional extra (`pip install -e ".[test]"`); no runtime dependencies added.

## [0.2.0] - 2026-10-19

### Added
- **Static type system** (`src/uclc/types.py`): `UclType` hierarchy with `IntType`, `FloatType`, `BoolType`, `StrType`, `VoidType`, `PtrType`, `TensorType`, `FnType`, `ClassType`. Type parsing via `parse_type()`, numeric promotion rules, and assignability checks.
- **Static type checker** (`src/uclc/checker.py`): walks the AST with scoped type environments. Checks let bindings, function params and return types, binary/unary ops, function call arity and arg types, assignments, class field/method access, and AI-native builtins. Collects `TypeDiag` diagnostics with source spans.
- **AST span info**: `Node` base class now carries `line` and `col`; the parser populates them from token positions on every node.
- **`uclc typecheck`** CLI command: reports diagnostics with `file:line:col: severity: message` format; exits 0 on success, 1 on errors.
- 58 new tests in `tests/test_types.py` (91 total): type representations, promotion, assignability, checker on all examples (incl. `neuron.ucl`), error detection (wrong return type, arity mismatch, undefined variables, type mismatches), span info, and CLI integration.

## [0.1.0] - 2026-10-17

Initial release: the UCL reference compiler front-end and evaluator.

### Added
- Lexer with comments, int/float/string literals, operators, and AI-native type keywords (`tensor`, `weights`, `tokenstream`, `graph`, `quant`).
- Parser for functions, classes (fields and methods), `let`, assignment, `return`, `if`/`else`, `while`, full expression precedence, member and method calls, and generic type arguments.
- Three-address IR lowering for all of the above (`uclc emit-ir`).
- Reference tree-walking evaluator for the procedural subset (integer/float arithmetic, comparisons, short-circuit `&&`/`||`, recursion, `print`).
- `uclc` CLI with `run`, `check`, `ast`, `tokens`, `emit-ir` and `--version`.
- Examples: `hello.ucl`, `fib.ucl`, `neuron.ucl` (parse and lower only).
- Language spec, EBNF grammar, roadmap and THECORE integration notes under `docs/`.
- MIT license.

### Known limitations
- No static type checking yet (see the v0.2 roadmap item).
- Object member and method evaluation is not implemented; `uclc check` and `uclc emit-ir` work on object code.
- AI-native builtins are reserved and lower to IR but cannot be evaluated yet.
- Integer literals are decimal only.
