# Changelog

All notable changes to UCL are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.11.0] - 2026-10-19

### Added
- **IR optimizer module** (`optimizer.py`): multi-pass optimization on three-address IR.
- **Constant folding**: evaluates arithmetic/comparison ops on constant operands at compile time.
- **Constant propagation**: propagates constants through single-assignment variables (straight-line code).
- **Strength reduction**: replaces x*2 with x+x, x*0 with 0, x*-1 with neg x.
- **Dead code elimination**: removes assignments to temps that are never used (preserves side effects).
- Iterative pass application until convergence (max 10 iterations).
- Wired into `codegen.py` via `use_optimizer=True` parameter.
- `optimize` exported from `uclc` package.
- 15 new tests (259 total): constant folding, propagation, DCE, strength reduction, native+optimizer end-to-end.

## [0.10.0] - 2026-10-19

### Added
- **Register allocator for native codegen**: linear scan register allocation with spill support.
- `regalloc.py` module (317 lines): `RegisterAllocation` class with interference graph and coloring.
- Spill slots on stack for register pressure beyond available physical registers.
- Integrated into `codegen.py` — optional `regalloc` parameter for `FunctionCompiler`.
- 50 new tests (244 total): register allocation unit tests, spill scenarios, complex expression chains.

## [0.9.0] - 2026-10-19

### Added
- **Function calls in native codegen**: `call funcname(args...)` compiles to x86-64 `call` instruction with System V AMD64 ABI parameter passing (rdi, rsi, rdx, rcx, r8, r9 for the first 6 args).
- **Recursive calls**: direct and mutual recursion work correctly (callee-saved rbp, proper stack frames).
- **Nested calls**: `f(g(x))` chains work — inner call's return value in rax feeds the outer call's argument.
- **Multi-function compilation**: all functions get their own labels; non-main functions emit proper `ret`.
- **No-arg functions**: zero-parameter functions compile and call correctly.
- **Functions with loops**: callees can use while loops, if/else, and all existing codegen features.
- 10 new native codegen function-call tests (194 total): simple call, return value, multiple args, fibonacci recursion, nested calls, function with print, no-arg, mutual recursion, call chain, function with loop.

## [0.8.0] - 2026-10-19

### Added
- **String support in native codegen**: string literals compile to ELF data section.
- **String print**: `print(s)` writes string data + newline via write(2) syscall.
- **String concatenation**: `s1 + s2` concatenates strings in a stack buffer.
- **String len**: `len(s)` returns the byte length of a string.
- **Escape sequences**: `\n`, `\t`, `\\` in string literals.
- **StringTable**: intern-based deduplication of string constants.
- Strings represented as (pointer, length) pairs on the stack — no null terminator, no GC.
- **String truthiness**: strings used as conditions (if/while) test length, not pointer — empty strings are falsy.
- 11 new native codegen string tests (184 total across both test suites).

## [0.7.0] - 2026-10-19

### Added
- **Native x86-64 codegen**: `uclc compile <file.ucl>` produces a standalone static ELF executable.
- **Minimal ELF builder**: generates valid ELF64 headers with a single PT_LOAD segment (~200-1.5K bytes output).
- **Stack-based register allocation**: all temporaries and locals live on the stack (correctness-first design).
- **Supported operations**: const (int), load/store, binary (`+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`), unary (`neg`, `not`), while loops, if/else, `ret`.
- **Native print**: integer-to-string conversion + write(2) syscall, handles negative numbers and zero.
- **codegen.py module**: `X86Emitter`, `FunctionCompiler`, `compile_to_elf()`, `compile_source_to_elf()`.
- 11 new native codegen tests (145 total): return values, arithmetic, division/modulo, negative numbers, while loops, if/else, all comparison operators, zero, large numbers, ELF size check.

## [0.6.0] - 2026-10-19

### Added
- **Pattern matching**: `match expr { case pattern -> stmt; }` with first-match semantics.
- **Fat arrow syntax**: `match expr { pattern => stmt; }` — arms without the `case` keyword, using `=>`. Both `case pattern ->` and `pattern =>` syntaxes supported.
- **Match as expression**: `let x = match expr { pattern => value, ... };` — match evaluates to the first matching arm's value, usable anywhere an expression is expected.
- **Literal patterns**: match on int, float, string, bool values (including negative numbers).
- **Wildcard pattern**: `_` matches anything.
- **Variable binding**: `case n -> ...` captures the matched value into a new variable.
- **Guard clauses**: `case n if n > 10 -> ...` adds a boolean condition to any case.
- **Array destructuring**: `case [x, y] -> ...` matches arrays by length and binds elements.
- **Tuple destructuring**: `case (a, b) -> ...` matches arrays with parenthesized syntax.
- **Rest pattern**: `case [first, ...rest] -> ...` binds remaining elements. Supports `[first, ...middle, last]`.
- **Map destructuring**: `case {"key": v} -> ...` matches maps by key presence and binds values.
- New tokens: `FAT_ARROW` (`=>`), `DOTDOTDOT` (`...`).
- IR lowering for Match, MatchExpr, TuplePattern, and RestPattern.
- Type checker support for Match and MatchExpr nodes.
- `pattern_match.ucl` example demonstrating all pattern matching features.
- 41 tests for pattern matching (173 total with native codegen): literal int/string/bool, wildcard, variable binding, guards, tuple destructuring, rest patterns, match expression, fat arrow syntax, nested match, IR lowering.

## [0.5.0] - 2026-10-19

### Added
- **Closures / lambdas**: first-class anonymous functions with lexical capture. Expression body `|x| -> x * 2` and block body `|x| { return x * 2; }`. Multi-param `|a, b| -> a + b`. Zero-param `|| -> 42`.
- **Nested lambdas**: closures capture their enclosing scope, enabling factory patterns like `|n| -> |x| -> x + n`.
- **Higher-order functions**: lambdas can be passed as arguments and called from other lambdas.
- **`map_arr(arr, fn)`**: transform each element of an array with a lambda.
- **`filter_arr(arr, fn)`**: keep elements where the lambda returns truthy.
- `closures.ucl` example demonstrating all closure features.
- 10 new tests (122 total): basic lambda, multi-param, closure capture, block body, nested, higher-order, map_arr, filter_arr, chain.

## [0.4.0] - 2026-10-19

### Added
- **break/continue**: full loop control flow in both `while` and `for..in` loops, including nested loops. IR lowering emits correct jump targets.
- **Hash maps**: `map(k1, v1, ...)` constructor, `get(m, k [, default])`, `set(m, k, v)`, `keys(m)`, `values(m)`, `has_key(m, k)`, `remove(m, k)`. Map indexing via `m["key"]`.
- `len()` now works on maps (returns key count).
- `break_continue.ucl` and `maps.ucl` examples.
- 11 new tests (112 total): 5 break/continue + 6 map tests.

## [0.3.0] - 2026-10-19

### Added
- **String builtins**: `upper(s)`, `lower(s)`, `trim(s)`, `split(s [, sep])`, `contains(haystack, needle)`, `substr(s, start [, len])` — all work on strings, `contains` also on arrays.
- **Type conversion builtins**: `to_str(v)`, `to_int(v)`, `to_float(v)` for runtime conversions.
- `strings.ucl` and `arrays.ucl` examples demonstrating the new builtins.
- 10 new tests for string/conversion builtins (101 total).
- GitHub Actions CI, Makefile, CONTRIBUTING.md, runnable examples (carried from unreleased).

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
