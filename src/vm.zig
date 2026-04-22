//! Bytecode interpreter.
//!
//! The register file is a fixed-size array today; once closures / calls
//! enter the picture we'll swap to a dynamic stack of frames and allocate
//! per-frame. For VM-1, "one register file for one program" is enough and
//! lets the dispatch loop stay tight.

const std = @import("std");
const bytecode = @import("bytecode.zig");
const value = @import("value.zig");

const Inst = bytecode.Inst;
const Program = bytecode.Program;
const Op = bytecode.Op;
const Value = value.Value;

pub const ExecError = error{
    /// The program ran past its last instruction without returning.
    /// Valid programs end with a RET, so hitting this almost always
    /// signals a compiler bug rather than a user bug.
    MissingReturn,
    /// Register index in an instruction points past the allocated
    /// register file. Another "should never happen" — would mean the
    /// compiler under-reported `register_count`.
    RegisterOutOfRange,
} || value.RuntimeError;

pub const max_registers = 256;

pub const VM = struct {
    registers: [max_registers]Value,

    pub fn init() VM {
        return .{ .registers = [_]Value{.nil} ** max_registers };
    }

    pub fn run(self: *VM, program: *const Program) ExecError!Value {
        if (program.register_count > max_registers) {
            return ExecError.RegisterOutOfRange;
        }

        var pc: usize = 0;
        while (pc < program.instructions.len) : (pc += 1) {
            const inst = program.instructions[pc];
            switch (inst.op) {
                .load_nil => self.registers[inst.a] = .nil,
                .load_true => self.registers[inst.a] = .{ .boolean = true },
                .load_false => self.registers[inst.a] = .{ .boolean = false },
                .load_int_const => {
                    self.registers[inst.a] = .{ .int = program.int_consts[inst.b] };
                },
                .load_float_const => {
                    self.registers[inst.a] = .{ .float = program.float_consts[inst.b] };
                },
                .move => self.registers[inst.a] = self.registers[inst.b],

                .add => self.registers[inst.a] = try value.add(self.registers[inst.b], self.registers[inst.c]),
                .sub => self.registers[inst.a] = try value.sub(self.registers[inst.b], self.registers[inst.c]),
                .mul => self.registers[inst.a] = try value.mul(self.registers[inst.b], self.registers[inst.c]),
                .div => self.registers[inst.a] = try value.div(self.registers[inst.b], self.registers[inst.c]),
                .mod => self.registers[inst.a] = try value.mod(self.registers[inst.b], self.registers[inst.c]),

                .neg => self.registers[inst.a] = try value.neg(self.registers[inst.b]),
                .not_op => self.registers[inst.a] = value.notOp(self.registers[inst.b]),

                .eq => self.registers[inst.a] = .{ .boolean = value.eq(self.registers[inst.b], self.registers[inst.c]) },
                .neq => self.registers[inst.a] = .{ .boolean = value.neq(self.registers[inst.b], self.registers[inst.c]) },
                .lt => self.registers[inst.a] = .{ .boolean = try value.lt(self.registers[inst.b], self.registers[inst.c]) },
                .gt => self.registers[inst.a] = .{ .boolean = try value.gt(self.registers[inst.b], self.registers[inst.c]) },
                .le => self.registers[inst.a] = .{ .boolean = try value.le(self.registers[inst.b], self.registers[inst.c]) },
                .ge => self.registers[inst.a] = .{ .boolean = try value.ge(self.registers[inst.b], self.registers[inst.c]) },

                .ret => return self.registers[inst.a],
            }
        }
        return ExecError.MissingReturn;
    }
};

// ============================== tests ==============================
//
// These go all the way from source text through lexer, parser, compiler,
// and VM to a runtime Value. They're the canary for any pipeline-level
// breakage: a test in here failing means some stage changed a contract
// without the others keeping up.

const parser_mod = @import("parser.zig");
const compiler_mod = @import("compiler.zig");
const ast_mod = @import("ast.zig");

const testing = std.testing;

/// Parse an expression, compile it, and run the resulting program. Returns
/// whatever the program RETurned. Sets up and tears down its own arena.
fn eval(source: [:0]const u8) !Value {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var p = try parser_mod.Parser.init(source, arena);
    const expr = try p.parseExpr();
    if (p.diagnostics.items.len != 0) return error.ParseDiagnostic;

    var c = compiler_mod.Compiler.init(arena);
    const program = try c.compileExpressionAsProgram(expr);

    var machine = VM.init();
    return machine.run(&program);
}

fn expectInt(source: [:0]const u8, expected: i64) !void {
    const got = try eval(source);
    try testing.expectEqual(Value{ .int = expected }, got);
}

fn expectFloat(source: [:0]const u8, expected: f64) !void {
    const got = try eval(source);
    try testing.expectEqual(Value{ .float = expected }, got);
}

fn expectBool(source: [:0]const u8, expected: bool) !void {
    const got = try eval(source);
    try testing.expectEqual(Value{ .boolean = expected }, got);
}

// ----- literal loads round-trip -----

test "int literal" {
    try expectInt("42", 42);
    try expectInt("0xFF", 255);
    try expectInt("0b1010", 10);
    try expectInt("0o777", 511);
    try expectInt("1_000_000", 1_000_000);
}

test "float literal" {
    try expectFloat("3.14", 3.14);
    try expectFloat("1e10", 1e10);
    try expectFloat("1.5e-3", 1.5e-3);
}

test "bool and nil literals" {
    try expectBool("true", true);
    try expectBool("false", false);
    try testing.expectEqual(Value{ .nil = {} }, try eval("nil"));
}

// ----- arithmetic -----

test "int arithmetic preserves type" {
    try expectInt("1 + 2", 3);
    try expectInt("10 - 3", 7);
    try expectInt("4 * 5", 20);
    try expectInt("15 / 2", 7); // @divTrunc
    try expectInt("15 % 4", 3);
}

test "float arithmetic" {
    try expectFloat("1.5 + 2.5", 4.0);
    try expectFloat("10.0 / 4.0", 2.5);
}

test "mixed int/float promotes" {
    try expectFloat("1 + 2.5", 3.5);
    try expectFloat("10.0 / 4", 2.5);
}

test "precedence survives the lower pipeline" {
    // The parser's Pratt precedence table gets tested at its own layer,
    // but this verifies the bytecode compiler respects the tree shape
    // end to end.
    try expectInt("1 + 2 * 3", 7);
    try expectInt("(1 + 2) * 3", 9);
    try expectInt("2 * 3 + 4 * 5", 26);
    try expectInt("100 - 10 - 5", 85); // left-associative
}

// ----- unary -----

test "unary minus" {
    try expectInt("-42", -42);
    try expectInt("-(1 + 2)", -3);
    try expectInt("--5", 5); // two separate unary minus operators
    try expectFloat("-3.14", -3.14);
}

test "logical not uses truthiness" {
    try expectBool("not true", false);
    try expectBool("not false", true);
    try expectBool("not nil", true);
    try expectBool("not 0", false); // zero is truthy in Zua
    try expectBool("not not true", true);
}

// ----- comparisons produce booleans -----

test "equality comparisons" {
    try expectBool("1 == 1", true);
    try expectBool("1 == 2", false);
    try expectBool("1 != 2", true);
    try expectBool("true == true", true);
    try expectBool("true == false", false);
    try expectBool("nil == nil", true);
    // Numeric cross-type equality
    try expectBool("5 == 5.0", true);
    try expectBool("5 != 5.0", false);
    // Non-numeric mismatches are not-equal, not errors
    try expectBool("nil == false", false);
    try expectBool("true == 1", false);
}

test "ordering comparisons" {
    try expectBool("1 < 2", true);
    try expectBool("2 < 1", false);
    try expectBool("1 <= 1", true);
    try expectBool("2 > 1", true);
    try expectBool("2 >= 2", true);
    try expectBool("1.5 < 2", true);
    try expectBool("2 <= 2.0", true);
}

// ----- runtime errors bubble out of the VM -----

test "division by zero is a runtime error" {
    try testing.expectError(error.DivisionByZero, eval("1 / 0"));
    try testing.expectError(error.DivisionByZero, eval("1 % 0"));
}

test "arithmetic on non-numerics is a TypeMismatch" {
    try testing.expectError(error.TypeMismatch, eval("true + 1"));
    try testing.expectError(error.TypeMismatch, eval("nil - nil"));
}

test "ordering comparisons on non-numerics error" {
    try testing.expectError(error.TypeMismatch, eval("true < 1"));
    try testing.expectError(error.TypeMismatch, eval("nil < nil"));
}

// ----- register reuse sanity check -----

test "deeply nested arithmetic does not leak registers" {
    // Each sub-expression reserves one register, the binary op's temps
    // are released after. For a balanced tree of depth n, the peak
    // usage should be roughly n + 1, not 2^n. This test doesn't directly
    // assert register count but exercises enough depth that a leak would
    // explode the register file.
    try expectInt(
        "((1 + 2) * (3 + 4)) - ((5 * 6) - (7 + 8))",
        (((1 + 2) * (3 + 4)) - ((5 * 6) - (7 + 8))),
    );
}

// ----- unsupported expressions reject at compile time, not runtime -----

test "logical and/or are unsupported in VM-1" {
    // These parse fine but need branch ops to lower correctly — branch
    // ops arrive with VM-2.
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var p = try parser_mod.Parser.init("true and false", arena);
    const expr = try p.parseExpr();
    var c = compiler_mod.Compiler.init(arena);
    try testing.expectError(compiler_mod.CompileError.UnsupportedExpr, c.compileExpressionAsProgram(expr));
}

