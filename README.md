# UCL — Unified Code Language

**One low-level language, unified for AI.** UCL is a C++-inspired, object-capable
systems language whose distinguishing feature is that **AI data is a first-class
low-level type**: tensors, weight-blocks, token-streams and graphs are language
primitives, not library wrappers. Think of it as an **ASM/C-level overlay built
for AI** — close to the metal, but AI-native from the ground up.

```ucl
// hello.ucl
fn main() -> i32 {
    print("Hello, V0 — Unified Code Language");
    return 0;
}
```

## Why

Today AI systems glue together a high-level host language (Python), a tensor
library (C++/CUDA kernels), a serialization format, and a deployment runtime.
UCL collapses that stack: one language, one `.ucl` source, one compiler, with
AI data types that lower directly to a compact low-level IR.

- **AI-native primitives** — `tensor<f32,[N,M]>`, `weights`, `tokenstream`, `graph`, `quant` are built in.
- **C++-inspired objects** — `class`, fields, methods, familiar control flow.
- **Low-level by design** — compiles to a flat three-address IR (an "ASM overlay"), aimed at native/bytecode backends.
- **Small, readable toolchain** — a hand-written lexer + parser + IR + reference evaluator you can actually read.

## Install / run

No dependencies beyond Python 3.10+ for the reference toolchain.

```bash
export PYTHONPATH=src
python3 -m uclc.cli run     examples/fib.ucl     # run a program
python3 -m uclc.cli check   examples/neuron.ucl  # parse / type-check
python3 -m uclc.cli emit-ir examples/fib.ucl     # show the low-level IR
python3 -m uclc.cli tokens  examples/hello.ucl   # dump the token stream
```

More runnable programs: `examples/fizzbuzz.ucl`, `examples/factorial.ucl`, `examples/prime.ucl`.

## Status (v0.1)

| Stage | State |
|---|---|
| Lexer | ✅ complete (comments, int/float/string, operators, AI type keywords) |
| Parser | ✅ fns, classes, let/assign/return/if/while, full expr precedence, member/method calls, generic type args |
| IR lowering | ✅ three-address IR for all of the above |
| Evaluator | ✅ reference tree-walking interpreter for the procedural subset |
| Objects / AI ops | ⏳ parse + lower today; evaluation + native codegen on the roadmap |

33 tests green (`make test` or `python3 -m pytest -q`). Runnable examples: `make examples`. Contributor guide: [CONTRIBUTING.md](CONTRIBUTING.md) · Changes: [CHANGELOG.md](CHANGELOG.md). See [docs/ROADMAP.md](docs/ROADMAP.md).

## Docs
- [Language spec](docs/SPEC.md) · [Grammar (EBNF)](docs/GRAMMAR.ebnf) · [Roadmap](docs/ROADMAP.md) · [THECORE integration](docs/THECORE_INTEGRATION.md)

## License
MIT — see [LICENSE](LICENSE).
