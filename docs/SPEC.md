# UCL Language Specification (v0.1 draft)

UCL (Unified Code Language) is a statically-structured, C++-inspired systems
language with AI-native low-level types. This document describes the v0 surface
language; sections marked *(roadmap)* are designed but not yet implemented.

## 1. Lexical structure
- **Comments**: `// line` and `/* block */`.
- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`.
- **Integer literals**: `42`, decimal. *(roadmap: `0x`, `0b`, `_` separators)*
- **Float literals**: `2.5`, `3e2`, `0.1`.
- **String literals**: `"..."` with escapes `\n \t \r \" \\`.
- **Keywords**: `fn let return if else while true false class`.
- **Type keywords**: see §3.

## 2. Program structure
A program is a sequence of top-level declarations, each a **function** or a
**class**. Execution starts at `fn main()`.

```ucl
fn main() -> i32 { return 0; }
```

## 3. Types
### Scalar / C-like
`i8 i16 i32 i64 u8 u16 u32 u64 f16 f32 f64 bool void str ptr`

### AI-native primitives (the UCL differentiator)
| Type | Meaning |
|---|---|
| `tensor<DT, [dims...]>` | dense n-dimensional array, dtype + shape in the type |
| `weights` | a parameter block (a model's learnable state) |
| `tokenstream` | an ordered stream of discrete tokens |
| `graph` | a computation/dataflow graph value |
| `quant` | a quantized scalar/block (dtype carries scale/zero-point) *(roadmap)* |

Generic type arguments are parsed today; full shape/dtype checking is *(roadmap)*.

## 4. Declarations
### Functions
```ucl
fn name(param: Type, ...) -> RetType { ...stmts... }
```
Parameters accept `name: Type` or C-style `Type name`. The return arrow is
optional (defaults to `void`).

### Classes
```ucl
class Neuron {
    bias: f32;                       // field
    w: tensor<f32, [0]>;             // field
    fn forward(x: tensor<f32,[0]>) -> f32 { return dot(w, x) + bias; }  // method
}
```

## 5. Statements
`let name[: Type] = expr;` · `name = expr;` · `return [expr];` ·
`if cond { } else { }` · `while cond { }` · block `{ }` · expression statement.

## 6. Expressions & operators
Precedence (low→high): `||`, `&&`, `== !=`, `< > <= >=`, `+ -`, `* / %`, unary
`- !`, postfix `. ( )`. `&&`/`||` short-circuit. Member access `a.b` and method
calls `a.m(args)` are postfix and chainable.

## 7. Built-in functions
- `print(args...)` — write values to stdout (implemented).
- AI builtins `tensor() weights() tokenstream() graph() dot() matmul() quant()`
  are reserved and lower to IR; evaluation is *(roadmap)*.

## 8. Compilation model
`source(.ucl) → tokens → AST → three-address IR → {reference evaluator | native backend (roadmap)}`.
The IR is a flat register/vote machine form (see `uclc emit-ir`) chosen so a
later pass can map it onto real assembly or a compact bytecode VM.

## 9. Memory & execution *(roadmap)*
Value semantics for scalars; tensors/weights are reference-counted handles over
a flat arena. Ownership/borrow rules and a `@device` annotation for CPU/GPU
placement are planned.
