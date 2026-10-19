"""UCL native code generation — x86-64 Linux ELF backend.

Translates the flat three-address IR into x86-64 machine code and wraps it in
a minimal static ELF executable. Native codegen (v0.7+).

Supported IR operations:
  - const (int, bool, string literals)
  - load / store (stack-allocated locals)
  - binary: +, -, *, /, %, ==, !=, <, >, <=, >=
  - binary + for strings (concatenation via stack buffer)
  - unary: neg, not
  - ret (with optional value → exit code)
  - label / jmp / jmpf
  - call print(<tmp>) — int or string, mapped to write(1, ...) syscall
  - call len(<tmp>) — string length
  - call funcname(args...) — user-defined function calls (System V AMD64 ABI)

Calling convention: System V AMD64 ABI for syscalls and function calls. The
first 6 integer args pass in rdi, rsi, rdx, rcx, r8, r9; return value in rax.
All temporaries and locals live on the stack by default.  With
``use_regalloc=True`` a linear-scan register allocator assigns callee-saved
registers (rbx, r12-r15) to integer temps and spills the rest to the stack.

Strings are stored as (pointer, length) pairs. String literals live in a data
section appended after the code. Concatenation allocates on the stack.

Usage:
    from uclc.codegen import compile_to_elf
    elf_bytes = compile_to_elf(ir_functions)
    open("a.out", "wb").write(elf_bytes)
"""
from __future__ import annotations

import struct
import re
from typing import Dict, List, Optional

from .ir import IRFunction, Lowerer
from .regalloc import RegisterAllocation, allocate_registers
from .optimizer import optimize
from . import ast


class StringTable:
    """Collects string literals and assigns data-section offsets."""

    def __init__(self) -> None:
        self.entries: Dict[str, int] = {}
        self.data = bytearray()

    def intern(self, s: str) -> int:
        if s in self.entries:
            return self.entries[s]
        off = len(self.data)
        self.entries[s] = off
        self.data.extend(s.encode("utf-8"))
        return off

    def size(self) -> int:
        return len(self.data)


class X86Emitter:
    """Emits raw x86-64 machine code bytes."""

    def __init__(self) -> None:
        self.code = bytearray()
        self._patches: List[tuple] = []
        self._labels: Dict[str, int] = {}
        self.strings = StringTable()

    def pos(self) -> int:
        return len(self.code)

    def emit(self, *bs: int) -> None:
        self.code.extend(bs)

    def emit_bytes(self, data: bytes) -> None:
        self.code.extend(data)

    def mark_label(self, name: str) -> None:
        self._labels[name] = self.pos()

    def emit_jmp(self, label: str) -> None:
        self.emit(0xE9)
        self._patches.append((self.pos(), label, 4))
        self.emit(0, 0, 0, 0)

    def emit_jz(self, label: str) -> None:
        self.emit(0x0F, 0x84)
        self._patches.append((self.pos(), label, 4))
        self.emit(0, 0, 0, 0)

    def resolve_labels(self) -> None:
        for offset, label, size in self._patches:
            target = self._labels[label]
            rel = target - (offset + size)
            self.code[offset:offset + size] = struct.pack("<i", rel)

    # --- convenience instructions ---

    def mov_rax_imm(self, val: int) -> None:
        self.emit(0x48, 0xB8)
        self.emit_bytes(struct.pack("<q", val))

    def push_rax(self) -> None:
        self.emit(0x50)

    def pop_rax(self) -> None:
        self.emit(0x58)

    def pop_rcx(self) -> None:
        self.emit(0x59)

    def pop_rdx(self) -> None:
        self.emit(0x5A)

    def pop_rdi(self) -> None:
        self.emit(0x5F)

    def pop_rsi(self) -> None:
        self.emit(0x5E)

    def push_rbp(self) -> None:
        self.emit(0x55)

    def pop_rbp(self) -> None:
        self.emit(0x5D)

    def mov_rbp_rsp(self) -> None:
        self.emit(0x48, 0x89, 0xE5)

    def mov_rsp_rbp(self) -> None:
        self.emit(0x48, 0x89, 0xEC)

    def sub_rsp_imm8(self, n: int) -> None:
        self.emit(0x48, 0x83, 0xEC, n & 0xFF)

    def sub_rsp_imm32(self, n: int) -> None:
        self.emit(0x48, 0x81, 0xEC)
        self.emit_bytes(struct.pack("<i", n))

    def mov_rax_rbp_off(self, off: int) -> None:
        if -128 <= off < 128:
            self.emit(0x48, 0x8B, 0x45, off & 0xFF)
        else:
            self.emit(0x48, 0x8B, 0x85)
            self.emit_bytes(struct.pack("<i", off))

    def mov_rbp_off_rax(self, off: int) -> None:
        if -128 <= off < 128:
            self.emit(0x48, 0x89, 0x45, off & 0xFF)
        else:
            self.emit(0x48, 0x89, 0x85)
            self.emit_bytes(struct.pack("<i", off))

    def add_rax_rcx(self) -> None:
        self.emit(0x48, 0x01, 0xC8)

    def sub_rax_rcx(self) -> None:
        self.emit(0x48, 0x29, 0xC8)

    def imul_rax_rcx(self) -> None:
        self.emit(0x48, 0x0F, 0xAF, 0xC1)

    def cqo(self) -> None:
        self.emit(0x48, 0x99)

    def idiv_rcx(self) -> None:
        self.emit(0x48, 0xF7, 0xF9)

    def neg_rax(self) -> None:
        self.emit(0x48, 0xF7, 0xD8)

    def not_logical_rax(self) -> None:
        self.emit(0x48, 0x85, 0xC0)
        self.emit(0x0F, 0x94, 0xC0)
        self.emit(0x48, 0x0F, 0xB6, 0xC0)

    def cmp_rax_rcx(self) -> None:
        self.emit(0x48, 0x39, 0xC8)

    def test_rax_rax(self) -> None:
        self.emit(0x48, 0x85, 0xC0)

    def setcc(self, cc: int) -> None:
        self.emit(0x0F, cc, 0xC0)
        self.emit(0x48, 0x0F, 0xB6, 0xC0)

    def syscall(self) -> None:
        self.emit(0x0F, 0x05)

    def ret(self) -> None:
        self.emit(0xC3)

    def emit_call(self, label: str) -> None:
        """Emit a relative call to a labeled function."""
        self.emit(0xE8)
        self._patches.append((self.pos(), label, 4))
        self.emit(0, 0, 0, 0)

    def push_rdi(self) -> None:
        self.emit(0x57)

    def push_rsi(self) -> None:
        self.emit(0x56)

    def push_rdx(self) -> None:
        self.emit(0x52)

    def push_rcx(self) -> None:
        self.emit(0x51)

    def push_r8(self) -> None:
        self.emit(0x41, 0x50)

    def push_r9(self) -> None:
        self.emit(0x41, 0x51)

    def mov_rdi_rax(self) -> None:
        self.emit(0x48, 0x89, 0xC7)

    def mov_rsi_rax(self) -> None:
        self.emit(0x48, 0x89, 0xC6)

    def mov_rdx_rax(self) -> None:
        self.emit(0x48, 0x89, 0xC2)

    def mov_rcx_rax(self) -> None:
        self.emit(0x48, 0x89, 0xC1)

    def mov_r8_rax(self) -> None:
        self.emit(0x49, 0x89, 0xC0)

    def mov_r9_rax(self) -> None:
        self.emit(0x49, 0x89, 0xC1)

    def mov_rax_rdi(self) -> None:
        self.emit(0x48, 0x89, 0xF8)

    def mov_rax_rsi(self) -> None:
        self.emit(0x48, 0x89, 0xF0)

    def mov_rax_rdx(self) -> None:
        self.emit(0x48, 0x89, 0xD0)

    def mov_rax_rcx(self) -> None:
        self.emit(0x48, 0x89, 0xC8)

    def mov_rax_r8(self) -> None:
        self.emit(0x4C, 0x89, 0xC0)

    def mov_rax_r9(self) -> None:
        self.emit(0x4C, 0x89, 0xC8)


# System V AMD64 ABI: first 6 integer args
ABI_ARG_REGS = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]

SETCC_MAP = {
    "==": 0x94,  # sete
    "!=": 0x95,  # setne
    "<":  0x9C,  # setl
    ">":  0x9F,  # setg
    "<=": 0x9E,  # setle
    ">=": 0x9D,  # setge
}

# x86-64 encodings for callee-saved register <-> rax moves
_MOV_RAX_FROM: Dict[str, bytes] = {
    "rbx": bytes([0x48, 0x89, 0xD8]),  # mov rax, rbx
    "r12": bytes([0x4C, 0x89, 0xE0]),  # mov rax, r12
    "r13": bytes([0x4C, 0x89, 0xE8]),  # mov rax, r13
    "r14": bytes([0x4C, 0x89, 0xF0]),  # mov rax, r14
    "r15": bytes([0x4C, 0x89, 0xF8]),  # mov rax, r15
}
_MOV_TO_FROM_RAX: Dict[str, bytes] = {
    "rbx": bytes([0x48, 0x89, 0xC3]),  # mov rbx, rax
    "r12": bytes([0x49, 0x89, 0xC4]),  # mov r12, rax
    "r13": bytes([0x49, 0x89, 0xC5]),  # mov r13, rax
    "r14": bytes([0x49, 0x89, 0xC6]),  # mov r14, rax
    "r15": bytes([0x49, 0x89, 0xC7]),  # mov r15, rax
}
# (rex, modrm_disp8, modrm_disp32) for mov reg,[rbp+off] / mov [rbp+off],reg
_REG_RBP_MODRM: Dict[str, tuple] = {
    "rbx": (0x48, 0x5D, 0x9D),
    "r12": (0x4C, 0x65, 0xA5),
    "r13": (0x4C, 0x6D, 0xAD),
    "r14": (0x4C, 0x75, 0xB5),
    "r15": (0x4C, 0x7D, 0xBD),
}

IR_LINE = re.compile(
    r"^(?:"
    r"(?P<dst>t\d+)\s*=\s*(?P<rhs>.*)"
    r"|store\s+(?P<sname>\S+),\s*(?P<sval>t\d+)"
    r"|ret(?:\s+(?P<retval>t\d+))?"
    r"|(?P<label>L\d+):"
    r"|jmp\s+(?P<jmptgt>L\d+)"
    r"|jmpf\s+(?P<jcond>t\d+),\s*(?P<jftgt>L\d+)"
    r")$"
)


class FunctionCompiler:
    """Compiles one IRFunction to x86-64."""

    # Maps register name to (mov_rax_from_reg) method name on the emitter
    _LOAD_FROM_REG = {
        "rdi": "mov_rax_rdi", "rsi": "mov_rax_rsi", "rdx": "mov_rax_rdx",
        "rcx": "mov_rax_rcx", "r8": "mov_rax_r8", "r9": "mov_rax_r9",
    }
    _STORE_TO_REG = {
        "rdi": "mov_rdi_rax", "rsi": "mov_rsi_rax", "rdx": "mov_rdx_rax",
        "rcx": "mov_rcx_rax", "r8": "mov_r8_rax", "r9": "mov_r9_rax",
    }

    def __init__(self, fn: IRFunction, asm: X86Emitter, is_entry: bool = False,
                 data_base_addr: int = 0, func_names: Optional[set] = None,
                 regalloc: Optional[RegisterAllocation] = None) -> None:
        self.fn = fn
        self.asm = asm
        self.is_entry = is_entry
        self.data_base_addr = data_base_addr
        self.func_names = func_names or set()
        self.regalloc = regalloc
        self.slots: Dict[str, int] = {}
        self.slot_is_str: Dict[str, bool] = {}
        self._next_slot = -8
        self._cs_slots: Dict[str, int] = {}  # callee-saved reg -> stack offset

    def _alloc(self, name: str, is_str: bool = False) -> int:
        if name not in self.slots:
            self.slots[name] = self._next_slot
            self._next_slot -= 16  # always 16 bytes to accommodate strings
        return self.slots[name]

    def _slot(self, name: str) -> int:
        return self._alloc(name)

    # -- register-allocation helpers ------------------------------------------

    def _is_reg_allocated(self, name: str) -> bool:
        return self.regalloc is not None and name in self.regalloc.reg_map

    def _load_to_rax(self, name: str) -> None:
        """Load value into rax -- from register if allocated, else from stack."""
        if self.regalloc and name in self.regalloc.reg_map:
            self.asm.emit_bytes(_MOV_RAX_FROM[self.regalloc.reg_map[name]])
        else:
            self.asm.mov_rax_rbp_off(self._slot(name))

    def _store_from_rax(self, name: str) -> None:
        """Store rax -- to register if allocated, else to stack."""
        if self.regalloc and name in self.regalloc.reg_map:
            self.asm.emit_bytes(_MOV_TO_FROM_RAX[self.regalloc.reg_map[name]])
        else:
            self.asm.mov_rbp_off_rax(self._slot(name))

    def _save_callee_regs(self) -> None:
        """Save callee-saved registers to their dedicated stack slots."""
        if not self.regalloc:
            return
        for reg in self.regalloc.used_callee_saved:
            off = self._cs_slots[reg]
            rex, modrm8, modrm32 = _REG_RBP_MODRM[reg]
            if -128 <= off < 128:
                self.asm.emit(rex, 0x89, modrm8, off & 0xFF)
            else:
                self.asm.emit(rex, 0x89, modrm32)
                self.asm.emit_bytes(struct.pack("<i", off))

    def _restore_callee_regs(self) -> None:
        """Restore callee-saved registers from their dedicated stack slots."""
        if not self.regalloc:
            return
        for reg in self.regalloc.used_callee_saved:
            off = self._cs_slots[reg]
            rex, modrm8, modrm32 = _REG_RBP_MODRM[reg]
            if -128 <= off < 128:
                self.asm.emit(rex, 0x8B, modrm8, off & 0xFF)
            else:
                self.asm.emit(rex, 0x8B, modrm32)
                self.asm.emit_bytes(struct.pack("<i", off))

    def compile(self) -> None:
        for p in self.fn.params:
            self._alloc(p)

        for line in self.fn.code:
            self._alloc_line(line.strip())

        # Reserve stack slots for callee-saved register spills
        if self.regalloc:
            for reg in self.regalloc.used_callee_saved:
                cs_name = f"__cs_{reg}"
                self._alloc(cs_name)
                self._cs_slots[reg] = self._slot(cs_name)

        frame_size = (-self._next_slot + 8)
        frame_size = (frame_size + 15) & ~15

        self.asm.push_rbp()
        self.asm.mov_rbp_rsp()
        if frame_size <= 127:
            self.asm.sub_rsp_imm8(frame_size)
        else:
            self.asm.sub_rsp_imm32(frame_size)

        # Save callee-saved registers to stack
        self._save_callee_regs()

        # Load parameters from ABI registers into stack slots
        for i, p in enumerate(self.fn.params):
            if i < len(ABI_ARG_REGS):
                reg = ABI_ARG_REGS[i]
                getattr(self.asm, self._LOAD_FROM_REG[reg])()
                self.asm.mov_rbp_off_rax(self._slot(p))

        for line in self.fn.code:
            self._compile_line(line.strip())

    def _alloc_line(self, line: str) -> None:
        m = IR_LINE.match(line)
        if not m:
            return
        if m.group("dst"):
            self._alloc(m.group("dst"))
        if m.group("sname"):
            self._alloc(m.group("sname"))

    def _compile_line(self, line: str) -> None:
        m = IR_LINE.match(line)
        if not m:
            return

        if m.group("label") is not None:
            self.asm.mark_label(f"{self.fn.name}_{m.group('label')}")
            return

        if m.group("jmptgt") is not None:
            self.asm.emit_jmp(f"{self.fn.name}_{m.group('jmptgt')}")
            return

        if m.group("jcond") is not None:
            cond = m.group("jcond")
            if self.slot_is_str.get(cond):
                # For strings, truthiness = non-zero length
                self.asm.mov_rax_rbp_off(self._slot(cond) - 8)
            else:
                self._load_to_rax(cond)
            self.asm.test_rax_rax()
            self.asm.emit_jz(f"{self.fn.name}_{m.group('jftgt')}")
            return

        if m.group("sname") is not None:
            val = m.group("sval")
            sname = m.group("sname")
            self._load_to_rax(val)
            self.asm.mov_rbp_off_rax(self._slot(sname))
            # copy string length too (only when source is stack-based)
            if not self._is_reg_allocated(val):
                self.asm.mov_rax_rbp_off(self._slot(val) - 8)
                self.asm.mov_rbp_off_rax(self._slot(sname) - 8)
            if self.slot_is_str.get(val):
                self.slot_is_str[sname] = True
            return

        retval = m.group("retval")
        if "ret" in line and m.group("dst") is None and m.group("sname") is None:
            if retval:
                self._load_to_rax(retval)
            else:
                self.asm.mov_rax_imm(0)
            self._restore_callee_regs()
            self.asm.mov_rsp_rbp()
            self.asm.pop_rbp()
            if not self.is_entry:
                self.asm.ret()
            return

        dst = m.group("dst")
        rhs = m.group("rhs").strip()

        if rhs.startswith("const "):
            val_str = rhs[6:].strip()
            if val_str in ("true", "false"):
                val = 1 if val_str == "true" else 0
                self.asm.mov_rax_imm(val)
                self._store_from_rax(dst)
            elif val_str.startswith("'") or val_str.startswith('"'):
                s = val_str[1:-1]
                s = s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
                str_off = self.asm.strings.intern(s)
                str_addr = self.data_base_addr + str_off
                self.asm.mov_rax_imm(str_addr)
                self.asm.mov_rbp_off_rax(self._slot(dst))
                self.asm.mov_rax_imm(len(s.encode("utf-8")))
                self.asm.mov_rbp_off_rax(self._slot(dst) - 8)
                self.slot_is_str[dst] = True
            else:
                try:
                    val = int(val_str)
                except ValueError:
                    try:
                        val = int(float(val_str))
                    except ValueError:
                        val = 0
                self.asm.mov_rax_imm(val)
                self._store_from_rax(dst)
            return

        if rhs.startswith("load "):
            name = rhs[5:].strip()
            self.asm.mov_rax_rbp_off(self._slot(name))
            self._store_from_rax(dst)
            # copy string length too (only for stack-based destinations)
            if not self._is_reg_allocated(dst):
                self.asm.mov_rax_rbp_off(self._slot(name) - 8)
                self.asm.mov_rbp_off_rax(self._slot(dst) - 8)
            if self.slot_is_str.get(name):
                self.slot_is_str[dst] = True
            return

        if rhs.startswith("neg "):
            src = rhs[4:].strip()
            self._load_to_rax(src)
            self.asm.neg_rax()
            self._store_from_rax(dst)
            return

        if rhs.startswith("not "):
            src = rhs[4:].strip()
            self._load_to_rax(src)
            self.asm.not_logical_rax()
            self._store_from_rax(dst)
            return

        if rhs.startswith("call print("):
            arg_str = rhs[11:-1].strip()
            # Spill register-allocated arg to stack for print routines
            if self._is_reg_allocated(arg_str):
                self._load_to_rax(arg_str)
                self.asm.mov_rbp_off_rax(self._slot(arg_str))
            off = self._slot(arg_str)
            if self.slot_is_str.get(arg_str):
                self._emit_print_str(off)
            else:
                self._emit_print_int(off)
            self._store_from_rax(dst)
            return

        if rhs.startswith("call len("):
            arg_str = rhs[9:-1].strip()
            if self.slot_is_str.get(arg_str):
                self.asm.mov_rax_rbp_off(self._slot(arg_str) - 8)
            else:
                self.asm.mov_rax_imm(0)
            self._store_from_rax(dst)
            return

        if rhs.startswith("call "):
            self._emit_user_call(dst, rhs)
            return

        for op in ("==", "!=", "<=", ">=", "<", ">", "+", "-", "*", "/", "%"):
            parts = None
            if rhs.startswith(f"{op} "):
                rest = rhs[len(op)+1:]
                parts = rest.split(", ", 1)
            if not parts or len(parts) != 2:
                continue
            if parts and len(parts) == 2:
                a, b = parts[0].strip(), parts[1].strip()

                # String concatenation
                if op == "+" and self.slot_is_str.get(a) and self.slot_is_str.get(b):
                    self._emit_str_concat(dst, a, b)
                    return

                self._load_to_rax(a)
                self.asm.push_rax()
                self._load_to_rax(b)
                self.asm.emit(0x48, 0x89, 0xC1)  # mov rcx, rax
                self.asm.pop_rax()

                if op == "+":
                    self.asm.add_rax_rcx()
                elif op == "-":
                    self.asm.sub_rax_rcx()
                elif op == "*":
                    self.asm.imul_rax_rcx()
                elif op == "/":
                    self.asm.cqo()
                    self.asm.idiv_rcx()
                elif op == "%":
                    self.asm.cqo()
                    self.asm.idiv_rcx()
                    self.asm.emit(0x48, 0x89, 0xD0)  # mov rax, rdx
                elif op in SETCC_MAP:
                    self.asm.cmp_rax_rcx()
                    self.asm.setcc(SETCC_MAP[op])

                self._store_from_rax(dst)
                return

    def _emit_user_call(self, dst: str, rhs: str) -> None:
        """Compile a call to a user-defined function via System V AMD64 ABI."""
        # Parse: call funcname(t0, t1, ...)
        rest = rhs[5:]  # strip "call "
        paren = rest.index("(")
        func_name = rest[:paren].strip()
        args_str = rest[paren+1:-1].strip()
        args = [a.strip() for a in args_str.split(",")] if args_str else []

        if func_name not in self.func_names:
            # Unknown function — emit 0 as fallback (builtins handled above)
            self.asm.mov_rax_imm(0)
            self._store_from_rax(dst)
            return

        # Load args into ABI registers (first 6 in registers, rest on stack)
        # We need to be careful: loading arg N into its register might clobber
        # a register that holds arg M. So load all values to stack first, then
        # pop into registers.
        stack_args = []
        for i, arg in enumerate(args):
            if i < len(ABI_ARG_REGS):
                # Push value onto stack temporarily
                self._load_to_rax(arg)
                self.asm.push_rax()
            else:
                stack_args.append(arg)

        # Push extra args (>6) in reverse order for stack-based passing
        for arg in reversed(stack_args):
            self._load_to_rax(arg)
            self.asm.push_rax()

        # Pop register args in reverse order into ABI registers
        reg_count = min(len(args), len(ABI_ARG_REGS))
        for i in range(reg_count - 1, -1, -1):
            self.asm.pop_rax()
            getattr(self.asm, self._STORE_TO_REG[ABI_ARG_REGS[i]])()

        # Ensure 16-byte stack alignment before call
        # RSP must be 16-byte aligned at the call instruction.
        # push rbp + sub rsp makes it aligned; extra stack args may misalign.
        if len(stack_args) % 2 != 0:
            self.asm.sub_rsp_imm8(8)

        self.asm.emit_call(func_name)

        # Clean up stack args
        cleanup = len(stack_args) * 8
        if len(stack_args) % 2 != 0:
            cleanup += 8
        if cleanup:
            if cleanup <= 127:
                self.asm.emit(0x48, 0x83, 0xC4, cleanup & 0xFF)  # add rsp, N
            else:
                self.asm.emit(0x48, 0x81, 0xC4)
                self.asm.emit_bytes(struct.pack("<i", cleanup))

        # Return value is in rax — store to destination
        self._store_from_rax(dst)

    def _emit_str_concat(self, dst: str, a: str, b: str) -> None:
        """Concatenate two strings into a stack buffer."""
        asm = self.asm
        # r8 = a.ptr, r9 = a.len, r10 = b.ptr, r11 = b.len
        asm.mov_rax_rbp_off(self._slot(a))
        asm.emit(0x49, 0x89, 0xC0)  # mov r8, rax
        asm.mov_rax_rbp_off(self._slot(a) - 8)
        asm.emit(0x49, 0x89, 0xC1)  # mov r9, rax
        asm.mov_rax_rbp_off(self._slot(b))
        asm.emit(0x49, 0x89, 0xC2)  # mov r10, rax
        asm.mov_rax_rbp_off(self._slot(b) - 8)
        asm.emit(0x49, 0x89, 0xC3)  # mov r11, rax

        # total_len = r9 + r11, allocate on stack (aligned to 16)
        asm.emit(0x4C, 0x89, 0xC8)  # mov rax, r9
        asm.emit(0x4C, 0x01, 0xD8)  # add rax, r11
        asm.emit(0x48, 0x89, 0xC1)  # mov rcx, rax (total_len in rcx)
        # round up to 16
        asm.emit(0x48, 0x83, 0xC0, 0x0F)  # add rax, 15
        asm.emit(0x48, 0x83, 0xE0, 0xF0)  # and rax, ~15
        asm.emit(0x48, 0x29, 0xC4)  # sub rsp, rax (allocate)
        asm.emit(0x48, 0x89, 0xE6)  # mov rsi, rsp (buf ptr in rsi)

        # save total_len and buf ptr
        asm.emit(0x50)  # push rax (aligned size, for later restore)
        asm.emit(0x51)  # push rcx (total_len)
        asm.emit(0x56)  # push rsi (buf ptr)

        # copy first string: rep movsb (rdi=dst, rsi=src, rcx=count)
        asm.emit(0x48, 0x89, 0xF7)  # mov rdi, rsi (dst = buf)
        asm.emit(0x4C, 0x89, 0xC6)  # mov rsi, r8  (src = a.ptr)
        asm.emit(0x4C, 0x89, 0xC9)  # mov rcx, r9  (count = a.len)
        asm.emit(0xF3, 0xA4)        # rep movsb

        # copy second string (rdi already points past first string)
        asm.emit(0x4C, 0x89, 0xD6)  # mov rsi, r10 (src = b.ptr)
        asm.emit(0x4C, 0x89, 0xD9)  # mov rcx, r11 (count = b.len)
        asm.emit(0xF3, 0xA4)        # rep movsb

        # restore buf ptr and total_len
        asm.emit(0x5E)  # pop rsi (buf ptr)
        asm.emit(0x59)  # pop rcx (total_len)
        asm.emit(0x58)  # pop rax (aligned size, don't restore stack yet)

        # store result: dst.ptr = buf, dst.len = total_len
        asm.emit(0x48, 0x89, 0xF0)  # mov rax, rsi
        asm.mov_rbp_off_rax(self._slot(dst))
        asm.emit(0x48, 0x89, 0xC8)  # mov rax, rcx
        asm.mov_rbp_off_rax(self._slot(dst) - 8)
        self.slot_is_str[dst] = True

    def _emit_print_str(self, val_off: int) -> None:
        """Print a string to stdout: write(1, ptr, len) + newline."""
        a = self.asm
        # load ptr into rsi, len into rdx
        a.mov_rax_rbp_off(val_off)
        a.emit(0x48, 0x89, 0xC6)                            # mov rsi, rax (ptr)
        a.mov_rax_rbp_off(val_off - 8)
        a.emit(0x48, 0x89, 0xC2)                            # mov rdx, rax (len)
        # write(1, ptr, len)
        a.emit(0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00)  # mov rax, 1 (sys_write)
        a.emit(0x48, 0xC7, 0xC7, 0x01, 0x00, 0x00, 0x00)  # mov rdi, 1 (stdout)
        a.syscall()
        # write newline via stack buffer
        a.sub_rsp_imm8(16)
        a.emit(0xC6, 0x04, 0x24, 0x0A)                     # mov byte [rsp], '\n'
        a.emit(0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00)  # mov rax, 1
        a.emit(0x48, 0xC7, 0xC7, 0x01, 0x00, 0x00, 0x00)  # mov rdi, 1
        a.emit(0x48, 0x89, 0xE6)                            # mov rsi, rsp
        a.emit(0x48, 0xC7, 0xC2, 0x01, 0x00, 0x00, 0x00)  # mov rdx, 1
        a.syscall()
        a.emit(0x48, 0x83, 0xC4, 0x10)                     # add rsp, 16
        a.mov_rax_imm(0)

    def _emit_print_int(self, val_off: int) -> None:
        """Print an integer to stdout using write(2) syscall.

        Uses a 24-byte buffer on the stack. Converts right-to-left, then
        writes the whole thing with one syscall + a newline.
        """
        a = self.asm
        a.mov_rax_rbp_off(val_off)

        # r8 = abs(value), r9 = 1 if negative else 0
        a.emit(0x49, 0x89, 0xC0)                       # mov r8, rax
        a.emit(0x4D, 0x31, 0xC9)                       # xor r9, r9
        a.emit(0x4D, 0x85, 0xC0)                       # test r8, r8
        a.emit(0x79, 0x06)                              # jns +6
        a.emit(0x49, 0xF7, 0xD8)                       # neg r8
        a.emit(0x49, 0xFF, 0xC1)                       # inc r9

        # allocate 24 bytes for digit buffer, r10 = end pointer
        a.sub_rsp_imm8(24)
        a.emit(0x4C, 0x8D, 0x54, 0x24, 23)            # lea r10, [rsp+23]

        # store newline at the end
        a.emit(0x41, 0xC6, 0x02, 0x0A)                # mov byte [r10], '\n'

        # r11 = digit count (starts at 1 for newline)
        a.emit(0x49, 0xC7, 0xC3, 0x01, 0x00, 0x00, 0x00)  # mov r11, 1

        # mov rcx, 10 (divisor)
        a.emit(0x48, 0xC7, 0xC1, 0x0A, 0x00, 0x00, 0x00)  # mov rcx, 10

        # digit loop
        loop_top = a.pos()
        a.emit(0x4C, 0x89, 0xC0)                       # mov rax, r8
        a.emit(0x48, 0x31, 0xD2)                       # xor rdx, rdx
        a.emit(0x48, 0xF7, 0xF1)                       # div rcx  (rax=quotient, rdx=remainder)
        a.emit(0x49, 0x89, 0xC0)                       # mov r8, rax (quotient back)
        a.emit(0x48, 0x83, 0xC2, 0x30)                # add rdx, '0'
        a.emit(0x49, 0xFF, 0xCA)                       # dec r10
        a.emit(0x41, 0x88, 0x12)                       # mov [r10], dl
        a.emit(0x49, 0xFF, 0xC3)                       # inc r11
        a.emit(0x4D, 0x85, 0xC0)                       # test r8, r8
        rel = loop_top - (a.pos() + 2)
        a.emit(0x75, rel & 0xFF)                        # jnz loop_top

        # if negative, prepend '-'
        a.emit(0x4D, 0x85, 0xC9)                       # test r9, r9
        a.emit(0x74, 0x08)                              # jz +8
        a.emit(0x49, 0xFF, 0xCA)                       # dec r10
        a.emit(0x41, 0xC6, 0x02, 0x2D)                # mov byte [r10], '-'
        a.emit(0x49, 0xFF, 0xC3)                       # inc r11

        # write(1, r10, r11)
        a.emit(0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00)  # mov rax, 1 (sys_write)
        a.emit(0x48, 0xC7, 0xC7, 0x01, 0x00, 0x00, 0x00)  # mov rdi, 1 (stdout)
        a.emit(0x4C, 0x89, 0xD6)                       # mov rsi, r10 (buf)
        a.emit(0x4C, 0x89, 0xDA)                       # mov rdx, r11 (len)
        a.syscall()

        # restore stack
        a.emit(0x48, 0x83, 0xC4, 24)                   # add rsp, 24
        a.mov_rax_imm(0)


def _build_elf(code: bytes, entry_offset: int, data: bytes = b"") -> bytes:
    """Wrap machine code in a minimal static ELF64 executable.

    String data is appended after the code in the same PT_LOAD segment.
    The data_base_addr returned tells the codegen where strings will live.
    """
    LOAD_ADDR = 0x400000
    ELF_HDR_SIZE = 64
    PHDR_SIZE = 56

    code_offset = ELF_HDR_SIZE + PHDR_SIZE
    total_payload = len(code) + len(data)
    file_size = code_offset + total_payload
    entry_addr = LOAD_ADDR + code_offset + entry_offset

    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2    # 64-bit
    ehdr[5] = 1    # little-endian
    ehdr[6] = 1    # ELF version
    ehdr[7] = 0    # System V ABI
    struct.pack_into("<H", ehdr, 16, 2)      # e_type: ET_EXEC
    struct.pack_into("<H", ehdr, 18, 0x3E)   # e_machine: x86-64
    struct.pack_into("<I", ehdr, 20, 1)      # e_version
    struct.pack_into("<Q", ehdr, 24, entry_addr)  # e_entry
    struct.pack_into("<Q", ehdr, 32, ELF_HDR_SIZE)  # e_phoff
    struct.pack_into("<Q", ehdr, 40, 0)      # e_shoff
    struct.pack_into("<I", ehdr, 48, 0)      # e_flags
    struct.pack_into("<H", ehdr, 52, ELF_HDR_SIZE)  # e_ehsize
    struct.pack_into("<H", ehdr, 54, PHDR_SIZE)  # e_phentsize
    struct.pack_into("<H", ehdr, 56, 1)      # e_phnum
    struct.pack_into("<H", ehdr, 58, 0)      # e_shentsize
    struct.pack_into("<H", ehdr, 60, 0)      # e_shnum
    struct.pack_into("<H", ehdr, 62, 0)      # e_shstrndx

    phdr = bytearray(56)
    struct.pack_into("<I", phdr, 0, 1)       # p_type: PT_LOAD
    struct.pack_into("<I", phdr, 4, 5)       # p_flags: PF_R | PF_X
    struct.pack_into("<Q", phdr, 8, 0)       # p_offset
    struct.pack_into("<Q", phdr, 16, LOAD_ADDR)  # p_vaddr
    struct.pack_into("<Q", phdr, 24, LOAD_ADDR)  # p_paddr
    struct.pack_into("<Q", phdr, 32, file_size)  # p_filesz
    struct.pack_into("<Q", phdr, 40, file_size)  # p_memsz
    struct.pack_into("<Q", phdr, 48, 0x1000)  # p_align

    return bytes(ehdr) + bytes(phdr) + code + data


def compile_to_elf(ir_funcs: List[IRFunction], *, use_regalloc: bool = False,
                   use_optimizer: bool = False) -> bytes:
    """Compile IR functions to a standalone ELF executable.

    Two-pass for string literals: first pass with placeholder address, second
    pass patches string addresses once code size is known.
    """
    LOAD_ADDR = 0x400000
    ELF_HDR_SIZE = 64
    PHDR_SIZE = 56
    HEADERS_SIZE = ELF_HDR_SIZE + PHDR_SIZE
    # Placeholder — will be patched
    PLACEHOLDER_DATA_ADDR = 0xDEAD0000

    if use_optimizer:
        optimize(ir_funcs)

    asm = X86Emitter()
    compilers: List[FunctionCompiler] = []

    main_fn = None
    others = []
    func_names = set()
    for fn in ir_funcs:
        func_names.add(fn.name)
        if fn.name == "main":
            main_fn = fn
        else:
            others.append(fn)

    for fn in others:
        asm.mark_label(fn.name)
        ra = allocate_registers(fn) if use_regalloc else None
        fc = FunctionCompiler(fn, asm, data_base_addr=PLACEHOLDER_DATA_ADDR,
                              func_names=func_names, regalloc=ra)
        fc.compile()
        compilers.append(fc)

    entry_offset = asm.pos()

    if main_fn:
        ra = allocate_registers(main_fn) if use_regalloc else None
        fc = FunctionCompiler(main_fn, asm, is_entry=True,
                              data_base_addr=PLACEHOLDER_DATA_ADDR,
                              func_names=func_names, regalloc=ra)
        fc.compile()
        compilers.append(fc)
    else:
        asm.mov_rax_imm(0)

    # exit syscall with return value in rax
    asm.emit(0x48, 0x89, 0xC7)  # mov rdi, rax
    asm.emit(0x48, 0xC7, 0xC0, 0x3C, 0x00, 0x00, 0x00)  # mov rax, 60 (sys_exit)
    asm.syscall()

    asm.resolve_labels()

    str_data = bytes(asm.strings.data) if asm.strings.size() > 0 else b""

    if str_data:
        # Compute real data base address: after headers + code in the segment
        real_data_addr = LOAD_ADDR + HEADERS_SIZE + len(asm.code)
        # Patch all placeholder addresses in the code
        placeholder_bytes = struct.pack("<q", PLACEHOLDER_DATA_ADDR)
        for s, off in asm.strings.entries.items():
            old_addr = PLACEHOLDER_DATA_ADDR + off
            new_addr = real_data_addr + off
            old_bytes = struct.pack("<q", old_addr)
            new_bytes = struct.pack("<q", new_addr)
            # Replace all occurrences in machine code
            pos = 0
            while True:
                idx = asm.code.find(old_bytes, pos)
                if idx == -1:
                    break
                asm.code[idx:idx+8] = new_bytes
                pos = idx + 8

    return _build_elf(bytes(asm.code), entry_offset, str_data)


def compile_source_to_elf(source: str, *, use_regalloc: bool = False,
                          use_optimizer: bool = False) -> bytes:
    """Full pipeline: UCL source -> ELF binary."""
    from .lexer import tokenize
    from .parser import parse

    prog = parse(tokenize(source))
    ir_funcs = Lowerer().lower_program(prog)
    return compile_to_elf(ir_funcs, use_regalloc=use_regalloc,
                          use_optimizer=use_optimizer)
