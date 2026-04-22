//! AST → bytecode compiler (VM-1 subset).
//!
//! For VM-1 we only compile expressions built from literals, unary
//! operators, and binary operators. Short-circuit logical `and`/`or`
//! need branch instructions that don't exist yet, so we reject them here
//! and pick them up when control-flow ops land.
//!
//! Register allocation is a plain bump allocator. The invariant is that
//! `next_reg` points one past the highest currently-live register, so
//! callers of `compileExpr` can read "the result went into `dst`, and
//! everything above `dst` is free to reuse". After a compound expression,
//! we reset `next_reg = dst + 1` to release the temporaries the children
//! used. Simple and enough for the non-mutating, non-flow-aware slice of
//! the language we're compiling today.

const std = @import("std");
const ast = @import("ast.zig");
const bytecode = @import("bytecode.zig");

const Expr = ast.Expr;
const Op = bytecode.Op;
const Inst = bytecode.Inst;
const Program = bytecode.Program;

pub const CompileError = error{
    /// The AST node is valid in the surface grammar but the compiler
    /// doesn't yet know how to lower it. VM-1 returns this for almost
    /// everything except primitive arithmetic; later phases will whittle
    /// this down.
    UnsupportedExpr,
    InvalidNumericLiteral,
    /// Expression nests too deep to fit into the 16-bit register space.
    /// 65,536 registers is a lot; hitting this almost certainly means
    /// recursion without proper release.
    TooManyRegisters,
} || std.mem.Allocator.Error;

pub const Compiler = struct {
    arena: std.mem.Allocator,
    insts: std.ArrayList(Inst),
    int_consts: std.ArrayList(i64),
    float_consts: std.ArrayList(f64),
    next_reg: u16,
    max_reg: u16,

    pub fn init(arena: std.mem.Allocator) Compiler {
        return .{
            .arena = arena,
            .insts = .empty,
            .int_consts = .empty,
            .float_consts = .empty,
            .next_reg = 0,
            .max_reg = 0,
        };
    }

    /// Compile a single expression and terminate with a RET. Suitable for
    /// the eval-an-expression driver in tests and the upcoming REPL.
    pub fn compileExpressionAsProgram(self: *Compiler, expr: *const Expr) CompileError!Program {
        const result_reg = try self.compileExpr(expr);
        try self.emit(.{ .op = .ret, .a = result_reg, .b = 0, .c = 0 });
        return .{
            .instructions = self.insts.items,
            .int_consts = self.int_consts.items,
            .float_consts = self.float_consts.items,
            .register_count = self.max_reg,
        };
    }

    fn compileExpr(self: *Compiler, expr: *const Expr) CompileError!u16 {
        const dst = try self.reserve();
        switch (expr.data) {
            .int_literal => |s| {
                const val = try parseIntLexeme(s);
                const idx = try self.addIntConst(val);
                try self.emit(.{ .op = .load_int_const, .a = dst, .b = idx, .c = 0 });
            },
            .float_literal => |s| {
                const val = try parseFloatLexeme(s);
                const idx = try self.addFloatConst(val);
                try self.emit(.{ .op = .load_float_const, .a = dst, .b = idx, .c = 0 });
            },
            .bool_literal => |b| {
                try self.emit(.{
                    .op = if (b) .load_true else .load_false,
                    .a = dst,
                    .b = 0,
                    .c = 0,
                });
            },
            .nil_literal => {
                try self.emit(.{ .op = .load_nil, .a = dst, .b = 0, .c = 0 });
            },
            .unary => |u| {
                const src = try self.compileExpr(u.operand);
                const op: Op = switch (u.op) {
                    .neg => .neg,
                    .not => .not_op,
                };
                try self.emit(.{ .op = op, .a = dst, .b = src, .c = 0 });
                self.next_reg = dst + 1;
            },
            .binary => |b| {
                const lhs = try self.compileExpr(b.lhs);
                // Release lhs's temporaries before compiling rhs, keeping
                // lhs itself live. This is the whole reason for the
                // manual bump / reset dance: without it, `(1+2)*(3+4)`
                // uses a linear-in-expression-size register file.
                self.next_reg = lhs + 1;
                const rhs = try self.compileExpr(b.rhs);
                const op: Op = switch (b.op) {
                    .add => .add,
                    .sub => .sub,
                    .mul => .mul,
                    .div => .div,
                    .mod => .mod,
                    .eq => .eq,
                    .neq => .neq,
                    .lt => .lt,
                    .gt => .gt,
                    .le => .le,
                    .ge => .ge,
                    // Short-circuit ops need branch instructions that
                    // don't exist in the VM-1 opcode table. Returning
                    // here is better than silently lowering to eager
                    // evaluation, which would change semantics.
                    .logical_and, .logical_or => return CompileError.UnsupportedExpr,
                };
                try self.emit(.{ .op = op, .a = dst, .b = lhs, .c = rhs });
                self.next_reg = dst + 1;
            },
            else => return CompileError.UnsupportedExpr,
        }
        return dst;
    }

    fn reserve(self: *Compiler) CompileError!u16 {
        if (self.next_reg == std.math.maxInt(u16)) return CompileError.TooManyRegisters;
        const r = self.next_reg;
        self.next_reg += 1;
        if (self.next_reg > self.max_reg) self.max_reg = self.next_reg;
        return r;
    }

    fn emit(self: *Compiler, inst: Inst) !void {
        try self.insts.append(self.arena, inst);
    }

    /// Add an int to the constant pool, deduplicating against previous
    /// entries. Linear search is fine for the small programs we produce
    /// today; switching to a hashmap is easy later.
    fn addIntConst(self: *Compiler, val: i64) !u16 {
        for (self.int_consts.items, 0..) |c, i| {
            if (c == val) return @intCast(i);
        }
        try self.int_consts.append(self.arena, val);
        return @intCast(self.int_consts.items.len - 1);
    }

    /// Same as `addIntConst`, but deduplicates via bitcast equality so
    /// that NaN (where `a == a` is false) still dedups correctly.
    fn addFloatConst(self: *Compiler, val: f64) !u16 {
        const bits: u64 = @bitCast(val);
        for (self.float_consts.items, 0..) |c, i| {
            if (@as(u64, @bitCast(c)) == bits) return @intCast(i);
        }
        try self.float_consts.append(self.arena, val);
        return @intCast(self.float_consts.items.len - 1);
    }
};

/// Decode an integer literal lexeme. Strips visual-separator underscores
/// before delegating to `std.fmt.parseInt`, which handles `0x`/`0b`/`0o`
/// prefixes when given base 0.
fn parseIntLexeme(raw: []const u8) CompileError!i64 {
    var buf: [64]u8 = undefined;
    var len: usize = 0;
    for (raw) |c| {
        if (c == '_') continue;
        if (len >= buf.len) return CompileError.InvalidNumericLiteral;
        buf[len] = c;
        len += 1;
    }
    return std.fmt.parseInt(i64, buf[0..len], 0) catch CompileError.InvalidNumericLiteral;
}

fn parseFloatLexeme(raw: []const u8) CompileError!f64 {
    var buf: [64]u8 = undefined;
    var len: usize = 0;
    for (raw) |c| {
        if (c == '_') continue;
        if (len >= buf.len) return CompileError.InvalidNumericLiteral;
        buf[len] = c;
        len += 1;
    }
    return std.fmt.parseFloat(f64, buf[0..len]) catch CompileError.InvalidNumericLiteral;
}
