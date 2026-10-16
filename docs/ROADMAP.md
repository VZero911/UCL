# UCL Roadmap

UCL is a long-horizon project. v0.1 ships a real, tested front-end + reference
evaluator; the phases below build it toward a production AI systems language.

## v0.1 — front-end + reference semantics ✅ (now)
- Lexer, parser, three-address IR, tree-walking evaluator for the procedural subset.
- `uclc` CLI: `run`, `check`, `ast`, `tokens`, `emit-ir`.
- Examples (5 runnable) + 33-test suite.

## v0.2 — type system
- Static type checking for scalars; function signature checking.
- Tensor shape + dtype checking (`tensor<f32,[N,M]>` enforced).
- Diagnostics with source spans.

## v0.3 — object model execution
- Evaluate classes/instances (fields, methods, `self`), construction.
- Make `examples/neuron.ucl` runnable end-to-end.

## v0.4 — AI-native runtime
- Implement `tensor`, `weights`, `tokenstream`, `graph`, `dot`, `matmul`, `quant`.
- CPU arena backend; `@device` placement annotations.

## v0.5 — native/bytecode backend
- Lower the IR to a compact bytecode VM, then to native (LLVM or direct asm).
- `uclc build` producing a standalone artifact.

## v0.6 — THECORE integration
- `.ucl` as a first-class module format for THECORE's AI layer (see
  [THECORE_INTEGRATION.md](THECORE_INTEGRATION.md)).

## Known v0.1 limitations
- No type checking yet (parse + lower + run only).
- Object member/method evaluation raises a roadmap error (parses + lowers fine).
- AI builtins are reserved + lowerable but not evaluable.
- Integer literals are decimal only.
