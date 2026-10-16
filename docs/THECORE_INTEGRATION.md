# UCL × THECORE

THECORE is a secure OS whose AI layer is intended to run on UCL. This note
sketches how `.ucl` fits in. It is forward-looking (UCL is at v0.1); nothing
here is wired yet.

## The idea
THECORE's AI-layer modules — model loaders, inference kernels, data pipelines —
are written in `.ucl` and compiled by `uclc` to the low-level IR, so the OS runs
one audited toolchain for its AI compute instead of a Python+C+++CUDA stack.

## Why UCL fits THECORE
- **One language, low-level**: THECORE favors a small, auditable surface. A single
  `.ucl` → IR → native path is easier to review than a multi-runtime stack.
- **AI data is native**: tensors/weights/tokenstreams are language types, so the
  OS can reason about (and sandbox) AI memory as first-class objects.
- **Deterministic lowering**: the three-address IR (`uclc emit-ir`) gives THECORE
  a stable, inspectable intermediate form to verify or instrument.

## Integration surface (planned)
| THECORE need | UCL mechanism |
|---|---|
| AI module format | `.ucl` source + `.uclir` compiled IR |
| Loadable unit | `uclc build` artifact (roadmap v0.5) |
| Memory isolation | tensor/weights handles over a THECORE-provided arena |
| Device placement | `@device(cpu|gpu)` annotations (roadmap v0.4) |

## Milestones
1. `uclc emit-ir` stable IR format frozen (v0.2).
2. THECORE loads a `.uclir` module and runs a trivial `fn` (v0.5/0.6).
3. An AI-layer kernel (e.g. a quantized matmul) authored in `.ucl` (v0.6).
