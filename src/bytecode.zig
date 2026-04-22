//! Zua bytecode — register-based, Lua/Luau-style.
//!
//! Each instruction is (op, a, b, c) where `a` is conventionally the
//! destination register and `b`/`c` are source registers or constant-pool
//! indices depending on the op. Padding this out to fixed-width operands
//! keeps the dispatch loop simple at the cost of a little size — we'll
//! revisit once we have measurements.
//!
//! VM-1 subset: only primitive arithmetic / comparison / unary / loads /
//! return. No branches, no calls, no heap access. Those come with later
//! VM phases.

const std = @import("std");

pub const Op = enum(u8) {
    // ---- loads ----
    /// A <- nil
    load_nil,
    /// A <- true
    load_true,
    /// A <- false
    load_false,
    /// A <- int_consts[B]
    load_int_const,
    /// A <- float_consts[B]
    load_float_const,
    /// A <- B (register-to-register copy)
    move,

    // ---- arithmetic ----
    /// A <- B op C
    add,
    sub,
    mul,
    div,
    mod,

    // ---- unary ----
    /// A <- -B
    neg,
    /// A <- not B (truthiness-based)
    not_op,

    // ---- comparison (result is a boolean Value in A) ----
    eq,
    neq,
    lt,
    gt,
    le,
    ge,

    // ---- control ----
    /// return the Value in register A; halts the VM
    ret,
};

pub const Inst = struct {
    op: Op,
    a: u16,
    b: u16,
    c: u16,
};

pub const Program = struct {
    instructions: []const Inst,
    int_consts: []const i64,
    float_consts: []const f64,
    /// Upper bound on register indices used by `instructions`. The VM
    /// allocates a register file sized to hold this many values.
    register_count: u16,

    /// Owns nothing when allocated via an arena; caller is responsible for
    /// arena lifetime. For heap-allocation-based builds later we can add a
    /// deinit, but arena ownership is the only use today.
    pub fn dump(self: *const Program, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        for (self.instructions, 0..) |inst, i| {
            try writer.print("{d:4}  {s} a={d} b={d} c={d}\n", .{ i, @tagName(inst.op), inst.a, inst.b, inst.c });
        }
    }
};
