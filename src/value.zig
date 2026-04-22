//! Runtime values.
//!
//! This is the union the bytecode VM operates on. For VM-1 it only covers
//! the primitives — `nil`, booleans, 64-bit ints, 64-bit floats. Heap types
//! (strings, arrays, maps, closures, records, channels, goroutine handles)
//! land in later VM phases with GC plumbing.
//!
//! Arithmetic helpers are kept out of `Value`'s own namespace so they can
//! return errors — the methods on the union itself stay pure and infallible.
//! Numeric mixing rules are Lua-ish: `int op int → int`, `float op float →
//! float`, and mixing promotes the int side to float. Anything else
//! (`bool + int`, `nil * nil`, etc.) is `TypeMismatch`, which the VM turns
//! into a runtime error later.

const std = @import("std");

pub const Value = union(enum) {
    nil,
    boolean: bool,
    int: i64,
    float: f64,

    /// Zua truthiness: only `nil` and `false` are falsy. Numeric zero is
    /// still true. Matches the decision recorded in the language spec.
    pub fn isTruthy(v: Value) bool {
        return switch (v) {
            .nil => false,
            .boolean => |b| b,
            .int, .float => true,
        };
    }
};

pub const RuntimeError = error{
    TypeMismatch,
    DivisionByZero,
};

// ====================== arithmetic ======================

pub fn add(a: Value, b: Value) RuntimeError!Value {
    return switch (a) {
        .int => |ai| switch (b) {
            .int => |bi| .{ .int = ai + bi },
            .float => |bf| .{ .float = asFloat(ai) + bf },
            else => RuntimeError.TypeMismatch,
        },
        .float => |af| switch (b) {
            .int => |bi| .{ .float = af + asFloat(bi) },
            .float => |bf| .{ .float = af + bf },
            else => RuntimeError.TypeMismatch,
        },
        else => RuntimeError.TypeMismatch,
    };
}

pub fn sub(a: Value, b: Value) RuntimeError!Value {
    return switch (a) {
        .int => |ai| switch (b) {
            .int => |bi| .{ .int = ai - bi },
            .float => |bf| .{ .float = asFloat(ai) - bf },
            else => RuntimeError.TypeMismatch,
        },
        .float => |af| switch (b) {
            .int => |bi| .{ .float = af - asFloat(bi) },
            .float => |bf| .{ .float = af - bf },
            else => RuntimeError.TypeMismatch,
        },
        else => RuntimeError.TypeMismatch,
    };
}

pub fn mul(a: Value, b: Value) RuntimeError!Value {
    return switch (a) {
        .int => |ai| switch (b) {
            .int => |bi| .{ .int = ai * bi },
            .float => |bf| .{ .float = asFloat(ai) * bf },
            else => RuntimeError.TypeMismatch,
        },
        .float => |af| switch (b) {
            .int => |bi| .{ .float = af * asFloat(bi) },
            .float => |bf| .{ .float = af * bf },
            else => RuntimeError.TypeMismatch,
        },
        else => RuntimeError.TypeMismatch,
    };
}

pub fn div(a: Value, b: Value) RuntimeError!Value {
    return switch (a) {
        .int => |ai| switch (b) {
            .int => |bi| blk: {
                if (bi == 0) return RuntimeError.DivisionByZero;
                // @divTrunc matches how most languages spell integer
                // division when they don't otherwise commit: toward zero.
                break :blk .{ .int = @divTrunc(ai, bi) };
            },
            .float => |bf| blk: {
                if (bf == 0.0) return RuntimeError.DivisionByZero;
                break :blk .{ .float = asFloat(ai) / bf };
            },
            else => RuntimeError.TypeMismatch,
        },
        .float => |af| switch (b) {
            .int => |bi| blk: {
                if (bi == 0) return RuntimeError.DivisionByZero;
                break :blk .{ .float = af / asFloat(bi) };
            },
            .float => |bf| blk: {
                if (bf == 0.0) return RuntimeError.DivisionByZero;
                break :blk .{ .float = af / bf };
            },
            else => RuntimeError.TypeMismatch,
        },
        else => RuntimeError.TypeMismatch,
    };
}

pub fn mod(a: Value, b: Value) RuntimeError!Value {
    return switch (a) {
        .int => |ai| switch (b) {
            .int => |bi| blk: {
                if (bi == 0) return RuntimeError.DivisionByZero;
                break :blk .{ .int = @mod(ai, bi) };
            },
            .float => |bf| blk: {
                if (bf == 0.0) return RuntimeError.DivisionByZero;
                break :blk .{ .float = @mod(asFloat(ai), bf) };
            },
            else => RuntimeError.TypeMismatch,
        },
        .float => |af| switch (b) {
            .int => |bi| blk: {
                if (bi == 0) return RuntimeError.DivisionByZero;
                break :blk .{ .float = @mod(af, asFloat(bi)) };
            },
            .float => |bf| blk: {
                if (bf == 0.0) return RuntimeError.DivisionByZero;
                break :blk .{ .float = @mod(af, bf) };
            },
            else => RuntimeError.TypeMismatch,
        },
        else => RuntimeError.TypeMismatch,
    };
}

// ====================== unary ======================

pub fn neg(v: Value) RuntimeError!Value {
    return switch (v) {
        .int => |i| .{ .int = -i },
        .float => |f| .{ .float = -f },
        else => RuntimeError.TypeMismatch,
    };
}

/// Logical `not`. Works on any value via truthiness; never errors.
pub fn notOp(v: Value) Value {
    return .{ .boolean = !v.isTruthy() };
}

// ====================== comparison ======================

/// Structural equality for primitives. `int` and `float` compare
/// numerically with each other (`5 == 5.0` is true); differing non-numeric
/// tags are never equal. This can't return a RuntimeError — every pair of
/// values has a defined equality answer.
pub fn eq(a: Value, b: Value) bool {
    return switch (a) {
        .nil => b == .nil,
        .boolean => |ab| switch (b) {
            .boolean => |bb| ab == bb,
            else => false,
        },
        .int => |ai| switch (b) {
            .int => |bi| ai == bi,
            .float => |bf| asFloat(ai) == bf,
            else => false,
        },
        .float => |af| switch (b) {
            .int => |bi| af == asFloat(bi),
            .float => |bf| af == bf,
            else => false,
        },
    };
}

pub fn neq(a: Value, b: Value) bool {
    return !eq(a, b);
}

/// Ordering only makes sense for numerics. Mixed numeric types promote
/// the int side to float, matching arithmetic. Anything non-numeric is
/// `TypeMismatch`.
pub fn lt(a: Value, b: Value) RuntimeError!bool {
    return switch (a) {
        .int => |ai| switch (b) {
            .int => |bi| ai < bi,
            .float => |bf| asFloat(ai) < bf,
            else => RuntimeError.TypeMismatch,
        },
        .float => |af| switch (b) {
            .int => |bi| af < asFloat(bi),
            .float => |bf| af < bf,
            else => RuntimeError.TypeMismatch,
        },
        else => RuntimeError.TypeMismatch,
    };
}

pub fn gt(a: Value, b: Value) RuntimeError!bool {
    return lt(b, a);
}

pub fn le(a: Value, b: Value) RuntimeError!bool {
    return !(try lt(b, a));
}

pub fn ge(a: Value, b: Value) RuntimeError!bool {
    return !(try lt(a, b));
}

// ====================== helpers ======================

fn asFloat(i: i64) f64 {
    return @floatFromInt(i);
}

// ====================== tests ======================

const testing = std.testing;

test "truthiness: only nil and false are falsy" {
    try testing.expect(!(Value{ .nil = {} }).isTruthy());
    try testing.expect(!(Value{ .boolean = false }).isTruthy());
    try testing.expect((Value{ .boolean = true }).isTruthy());
    try testing.expect((Value{ .int = 0 }).isTruthy());
    try testing.expect((Value{ .int = 1 }).isTruthy());
    try testing.expect((Value{ .float = 0.0 }).isTruthy());
}

test "int arithmetic stays int" {
    try testing.expectEqual(Value{ .int = 5 }, try add(.{ .int = 2 }, .{ .int = 3 }));
    try testing.expectEqual(Value{ .int = -1 }, try sub(.{ .int = 2 }, .{ .int = 3 }));
    try testing.expectEqual(Value{ .int = 6 }, try mul(.{ .int = 2 }, .{ .int = 3 }));
    try testing.expectEqual(Value{ .int = 7 }, try div(.{ .int = 15 }, .{ .int = 2 }));
    try testing.expectEqual(Value{ .int = 1 }, try mod(.{ .int = 15 }, .{ .int = 2 }));
}

test "mixed int/float promotes to float" {
    try testing.expectEqual(Value{ .float = 5.5 }, try add(.{ .int = 2 }, .{ .float = 3.5 }));
    try testing.expectEqual(Value{ .float = 5.5 }, try add(.{ .float = 3.5 }, .{ .int = 2 }));
}

test "division by zero" {
    try testing.expectError(RuntimeError.DivisionByZero, div(.{ .int = 1 }, .{ .int = 0 }));
    try testing.expectError(RuntimeError.DivisionByZero, div(.{ .float = 1.0 }, .{ .float = 0.0 }));
    try testing.expectError(RuntimeError.DivisionByZero, mod(.{ .int = 1 }, .{ .int = 0 }));
}

test "type mismatch on non-numeric arithmetic" {
    try testing.expectError(RuntimeError.TypeMismatch, add(.nil, .{ .int = 1 }));
    try testing.expectError(RuntimeError.TypeMismatch, add(.{ .boolean = true }, .{ .int = 1 }));
}

test "equality across numeric types" {
    try testing.expect(eq(.{ .int = 5 }, .{ .float = 5.0 }));
    try testing.expect(!eq(.{ .int = 5 }, .{ .boolean = true }));
    try testing.expect(eq(.nil, .nil));
    try testing.expect(!eq(.{ .int = 0 }, .{ .boolean = false }));
}

test "ordering on mixed numeric types" {
    try testing.expect(try lt(.{ .int = 3 }, .{ .float = 3.5 }));
    try testing.expect(try ge(.{ .float = 3.5 }, .{ .int = 3 }));
    try testing.expectError(RuntimeError.TypeMismatch, lt(.{ .boolean = true }, .{ .int = 0 }));
}

test "neg and not" {
    try testing.expectEqual(Value{ .int = -5 }, try neg(.{ .int = 5 }));
    try testing.expectEqual(Value{ .float = -3.14 }, try neg(.{ .float = 3.14 }));
    try testing.expectEqual(Value{ .boolean = false }, notOp(.{ .boolean = true }));
    try testing.expectEqual(Value{ .boolean = true }, notOp(.nil));
    try testing.expectEqual(Value{ .boolean = false }, notOp(.{ .int = 0 }));
}
