//! Zua AST — the shape every later compiler stage consumes.
//!
//! Nodes are arena-allocated tagged unions. Every node carries a `Span`
//! (byte offsets into the original source) so diagnostics can point at the
//! right place, and every leaf keeps the raw source lexeme rather than a
//! decoded value. Decoding escape sequences, numeric values, and the like
//! is left to the type checker / compiler — that way the parser is the
//! single source of truth for "did this text parse" and doesn't also carry
//! "did this number fit in an i64".
//!
//! The split between `Expr`, `TypeExpr`, and `Stmt` is intentional: they
//! live in different syntactic positions with different precedence rules,
//! and keeping them as separate types makes parser errors like "expected a
//! type here" turn into type errors rather than tag mismatches.
//!
//! Recovery: the parser emits `.invalid` variants when it can't build a
//! well-formed node but wants to keep parsing. Downstream stages should
//! treat `.invalid` as "errors already reported for this subtree, don't
//! pile on" rather than re-reporting.

const std = @import("std");

/// Byte range into the source buffer. u32 is enough for ~4 GB source files,
/// which is comfortably larger than any single Zua program will ever be.
pub const Span = struct {
    start: u32,
    end: u32,

    pub fn merge(a: Span, b: Span) Span {
        return .{ .start = @min(a.start, b.start), .end = @max(a.end, b.end) };
    }
};

pub const UnaryOp = enum { neg, not };

pub const BinaryOp = enum {
    add,
    sub,
    mul,
    div,
    mod,
    eq,
    neq,
    lt,
    gt,
    le,
    ge,
    logical_and,
    logical_or,

    pub fn symbol(op: BinaryOp) []const u8 {
        return switch (op) {
            .add => "+",
            .sub => "-",
            .mul => "*",
            .div => "/",
            .mod => "%",
            .eq => "==",
            .neq => "!=",
            .lt => "<",
            .gt => ">",
            .le => "<=",
            .ge => ">=",
            .logical_and => "and",
            .logical_or => "or",
        };
    }
};

pub const AssignOp = enum {
    assign,
    add_assign,
    sub_assign,
    mul_assign,
    div_assign,
    mod_assign,

    pub fn symbol(op: AssignOp) []const u8 {
        return switch (op) {
            .assign => "=",
            .add_assign => "+=",
            .sub_assign => "-=",
            .mul_assign => "*=",
            .div_assign => "/=",
            .mod_assign => "%=",
        };
    }
};

// ====================== expressions ======================

pub const Expr = struct {
    span: Span,
    data: Data,

    pub const Data = union(enum) {
        // Literals carry raw lexeme slices (including quotes for strings and
        // base prefixes for numbers). Decoding happens in a later pass.
        int_literal: []const u8,
        float_literal: []const u8,
        string_literal: []const u8,
        /// A string with one or more `${expr}` substitutions. Parts alternate
        /// between raw text runs (may be empty) and embedded expressions.
        string_interp: []StringPart,
        bool_literal: bool,
        nil_literal,

        // References
        ident: []const u8,
        self_ref,

        // Compound
        unary: Unary,
        binary: Binary,
        call: Call,
        method_call: MethodCall,
        field: Field,
        index: Index,
        range: Range,
        try_expr: *Expr,
        catch_expr: Catch,

        // Collection literals
        array_literal: []Expr,
        map_literal: []MapEntry,

        // Control flow as expressions (Rust-style)
        block: *Block,
        if_expr: If,
        match_expr: Match,

        // Record construction: `Point { x: 1, y: 2 }`.
        record_literal: RecordLit,

        /// Anonymous function expression: `fn(x: int) -> int { x * 2 }`.
        /// Named functions are a Phase C top-level declaration — at
        /// expression position, `fn` only ever introduces a closure.
        closure: Closure,

        /// `go <call>` — launch the inner call as a goroutine. Stored as
        /// `*Expr` rather than a richer struct because the parser has
        /// already verified the pointee's shape is a call or method_call;
        /// there is no extra metadata to carry.
        go_launch: *Expr,

        /// Parser synthesised this node to represent a subtree that failed
        /// to parse. Diagnostics for the failure have already been emitted.
        invalid,
    };

    pub const Unary = struct { op: UnaryOp, operand: *Expr };
    pub const Binary = struct { op: BinaryOp, lhs: *Expr, rhs: *Expr };
    pub const Call = struct { callee: *Expr, args: []Expr };
    pub const MethodCall = struct { receiver: *Expr, name: []const u8, args: []Expr };
    pub const Field = struct { receiver: *Expr, name: []const u8 };
    pub const Index = struct { receiver: *Expr, index: *Expr };
    pub const Range = struct { start: *Expr, end: *Expr, inclusive: bool };
    pub const MapEntry = struct { key: *Expr, value: *Expr };

    /// `try_expr catch [|e|] handler`. `err_binding == null` means the
    /// user wrote `a catch b` (no binding — the error value is discarded).
    /// The handler is itself an expression (often a block) that runs when
    /// `inner` evaluates to an error.
    pub const Catch = struct {
        inner: *Expr,
        err_binding: ?[]const u8,
        err_binding_span: ?Span,
        handler: *Expr,
    };

    pub const If = struct {
        cond: *Expr,
        then_block: *Block,
        else_branch: ?ElseBranch,
    };

    /// An `else` tail is either a final block or another `if` (for
    /// `else if` chains). Representing it as a nested `*Expr` that must be
    /// an `.if_expr` lets the chain stay a simple linked list in the AST
    /// rather than a specialised array of else-ifs.
    pub const ElseBranch = union(enum) {
        block: *Block,
        if_expr: *Expr,
    };

    pub const Match = struct {
        scrutinee: *Expr,
        arms: []MatchArm,
        /// `partial match ...` sets this to false. The parser doesn't check
        /// whether the arms actually cover every case — that's the type
        /// checker's job — but it records the user's intent.
        exhaustive: bool,
    };

    /// Construction of a named record type: `Point { x: 1, y: 2 }`.
    /// Generic record types (`Pair<int, string> { ... }`) are deferred —
    /// v1 accepts only bare type names on the constructor.
    pub const RecordLit = struct {
        name: []const u8,
        name_span: Span,
        fields: []FieldInit,
    };

    /// One `name: value` (or bare-ident shorthand) inside a record literal.
    /// When `value` is null, the field is shorthand: the current scope's
    /// binding with the same name as the field supplies the value — i.e.
    /// `Point { x, y }` is sugar for `Point { x: x, y: y }`.
    pub const FieldInit = struct {
        name: []const u8,
        name_span: Span,
        value: ?*Expr,
    };

    pub const Closure = struct {
        params: []Param,
        /// Optional `-> T` return type. When absent the function is
        /// declared to return `nil` — a common case for side-effectful
        /// closures like goroutine bodies.
        return_type: ?*TypeExpr,
        body: *Block,
    };

    /// A closure / function parameter. Type annotations are optional: an
    /// unannotated parameter takes `any`, matching the gradual-typing
    /// rule for unannotated bindings elsewhere.
    pub const Param = struct {
        name: []const u8,
        name_span: Span,
        type_ann: ?*TypeExpr,
    };
};

pub const MatchArm = struct {
    pattern: Pattern,
    body: *Expr,
};

/// Patterns only appear in match arm heads for now. Array / or- /
/// range- / guard-patterns are deferred: these forms cover the types we
/// have today (records, enums with record-style variants, primitives).
pub const Pattern = struct {
    span: Span,
    data: Data,

    pub const Data = union(enum) {
        /// Literal patterns match a specific value; equality is the
        /// semantic contract (decoding still happens in a later pass).
        int_pat: []const u8,
        float_pat: []const u8,
        string_pat: []const u8,
        bool_pat: bool,
        nil_pat,

        /// `_` — matches anything, binds nothing.
        wildcard,

        /// A bare identifier — binds the scrutinee to this name. Acts as
        /// a catch-all. Note that whether a bare identifier refers to an
        /// enum variant or creates a binding is a semantic question;
        /// parsing always produces `.bind`, and the type checker decides.
        bind: Binding,

        /// `Name { field: binding, shorthand }` — matches a record (or
        /// record-style enum variant) with the listed fields. Requires
        /// explicit `Name { }` braces even for zero-field variants so
        /// that we can reliably distinguish from `.bind`.
        constructor: Constructor,

        invalid,
    };

    pub const Constructor = struct {
        name: []const u8,
        name_span: Span,
        fields: []PatternField,
    };

    pub const PatternField = struct {
        field_name: []const u8,
        field_name_span: Span,
        /// Binding name for the matched field. When null, it's shorthand:
        /// `{ radius }` means the field `radius` is bound to a local
        /// `radius`, equivalent to `{ radius: radius }`.
        binding: ?Binding,
    };
};

/// A braced `{ stmt; stmt; trailing_expr? }`. Appears both as a standalone
/// expression and as the body of `if` / `while` / `for`. When `trailing`
/// is non-null, the block evaluates to that expression; otherwise it
/// evaluates to `nil`.
pub const Block = struct {
    span: Span,
    stmts: []Stmt,
    trailing: ?*Expr,
};

pub const StringPart = union(enum) {
    /// Raw literal text between `${...}` substitutions. May be the empty
    /// string when two substitutions touch (e.g., `"${a}${b}"`).
    text: []const u8,
    /// An embedded expression — the source's `${...}` contents.
    expr: *Expr,
};

// ====================== types ======================

pub const TypeExpr = struct {
    span: Span,
    data: Data,

    pub const Data = union(enum) {
        /// A named type with zero or more generic arguments:
        /// `int`, `Point`, `Array<int>`, `Map<string, Array<int>>`.
        named: Named,
        /// `T?` — an optional. Parsed as a postfix suffix and stored as a
        /// dedicated node so the type checker doesn't have to re-discover
        /// it inside `named` each time.
        optional: *TypeExpr,
        /// `!T` — the error-as-value wrapper.
        error_type: *TypeExpr,
        /// `fn(T1, T2) -> R`.
        function: Function,
        /// `self`, valid only inside `impl` method signatures. The parser
        /// accepts it anywhere and leaves the legality check to the type
        /// stage — that way we can write tests for parse shape in isolation.
        self_type,
        invalid,
    };

    pub const Named = struct { name: []const u8, type_args: []TypeExpr };
    pub const Function = struct { params: []TypeExpr, ret: *TypeExpr };
};

// ====================== statements ======================

pub const Stmt = struct {
    span: Span,
    data: Data,

    pub const Data = union(enum) {
        /// A bare expression at statement position (e.g. a function call
        /// whose return value we discard).
        expr: *Expr,
        /// `var x = e` / `const x = e`, optionally with `: T` annotation.
        var_decl: VarDecl,
        /// `target op= value` where `op=` covers `=`, `+=`, and friends.
        assign: Assign,
        /// `return` optionally followed by an expression.
        return_stmt: ?*Expr,
        break_stmt,
        continue_stmt,
        /// `while cond { ... }`. Kept as a statement (not an expression)
        /// because its value would be unit/nil — no reason to spend a tag
        /// slot on that.
        while_stmt: While,
        /// `for binding in iterable { ... }`. Statement for the same
        /// reason as `while`.
        for_stmt: For,
        invalid,
    };

    pub const VarDecl = struct {
        is_const: bool,
        name: []const u8,
        name_span: Span,
        type_ann: ?*TypeExpr,
        value: *Expr,
    };

    pub const Assign = struct {
        op: AssignOp,
        target: *Expr,
        value: *Expr,
    };

    pub const While = struct {
        cond: *Expr,
        body: *Block,
    };

    pub const For = struct {
        pattern: ForPattern,
        iterable: *Expr,
        body: *Block,
    };
};

/// The binding(s) introduced by a `for` loop:
///   `for x in arr`         -> .single("x")
///   `for k, v in map`      -> .key_value(.{ "k", "v" })
pub const ForPattern = union(enum) {
    single: Binding,
    key_value: KeyValue,

    pub const KeyValue = struct { key: Binding, value: Binding };
};

pub const Binding = struct {
    name: []const u8,
    span: Span,
};

// ====================== top-level declarations ======================

/// A fully parsed `.zua` file — the sequence of top-level declarations.
/// The parser stops only at EOF; errors during any single declaration are
/// collected as diagnostics and an `invalid` Decl is emitted so the file
/// shape stays intact.
pub const File = struct {
    span: Span,
    decls: []Decl,
};

pub const Decl = struct {
    span: Span,
    /// `true` iff the declaration was prefixed with `export`. Whether that
    /// has any visible effect (e.g. in the module system) is a later-stage
    /// concern; the parser only records the user's intent.
    exported: bool,
    data: Data,

    pub const Data = union(enum) {
        fn_decl: FnDecl,
        record_decl: RecordDecl,
        enum_decl: EnumDecl,
        type_alias: TypeAlias,
        const_decl: ConstDecl,
        invalid,
    };
};

pub const GenericParam = struct {
    name: []const u8,
    span: Span,
    // Future work: bounds / trait constraints would live here.
};

pub const FnDecl = struct {
    name: []const u8,
    name_span: Span,
    generic_params: []GenericParam,
    params: []Expr.Param,
    return_type: ?*TypeExpr,
    body: *Block,
};

pub const RecordDecl = struct {
    name: []const u8,
    name_span: Span,
    generic_params: []GenericParam,
    fields: []FieldDecl,
};

pub const EnumDecl = struct {
    name: []const u8,
    name_span: Span,
    generic_params: []GenericParam,
    variants: []Variant,
};

/// One variant of an `enum`. `Name` alone (empty `fields`) is a zero-
/// payload variant; `Name { a: T }` carries record-style fields. No other
/// shapes are supported in v1 (no tuple-style variants).
pub const Variant = struct {
    name: []const u8,
    name_span: Span,
    fields: []FieldDecl,
};

pub const FieldDecl = struct {
    name: []const u8,
    name_span: Span,
    /// Records and variants require every field to carry a type
    /// annotation — unlike function params and `var` bindings, there's
    /// nothing to infer it from.
    type_ann: *TypeExpr,
};

pub const TypeAlias = struct {
    name: []const u8,
    name_span: Span,
    generic_params: []GenericParam,
    target: *TypeExpr,
};

pub const ConstDecl = struct {
    name: []const u8,
    name_span: Span,
    type_ann: ?*TypeExpr,
    value: *Expr,
};

// ====================== pretty-printer ======================
//
// S-expression dump for use in tests. Not a pretty-printer for humans
// (indentation would make tests too whitespace-sensitive); just a compact,
// unambiguous shape we can compare strings against.

pub fn formatExpr(expr: *const Expr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    switch (expr.data) {
        .int_literal => |v| try writer.print("(int {s})", .{v}),
        .float_literal => |v| try writer.print("(float {s})", .{v}),
        .string_literal => |v| try writer.print("(str {s})", .{v}),
        .string_interp => |parts| {
            try writer.print("(interp", .{});
            for (parts) |part| switch (part) {
                .text => |t| try writer.print(" (text \"{s}\")", .{t}),
                .expr => |e| {
                    try writer.print(" ", .{});
                    try formatExpr(e, writer);
                },
            };
            try writer.print(")", .{});
        },
        .bool_literal => |b| try writer.print("{s}", .{if (b) "true" else "false"}),
        .nil_literal => try writer.print("nil", .{}),
        .ident => |n| try writer.print("(id {s})", .{n}),
        .self_ref => try writer.print("self", .{}),
        .unary => |u| {
            const sym = switch (u.op) {
                .neg => "-",
                .not => "not",
            };
            try writer.print("({s} ", .{sym});
            try formatExpr(u.operand, writer);
            try writer.print(")", .{});
        },
        .binary => |b| {
            try writer.print("({s} ", .{b.op.symbol()});
            try formatExpr(b.lhs, writer);
            try writer.print(" ", .{});
            try formatExpr(b.rhs, writer);
            try writer.print(")", .{});
        },
        .call => |c| {
            try writer.print("(call ", .{});
            try formatExpr(c.callee, writer);
            for (c.args) |*a| {
                try writer.print(" ", .{});
                try formatExpr(a, writer);
            }
            try writer.print(")", .{});
        },
        .method_call => |m| {
            try writer.print("(mcall ", .{});
            try formatExpr(m.receiver, writer);
            try writer.print(" {s}", .{m.name});
            for (m.args) |*a| {
                try writer.print(" ", .{});
                try formatExpr(a, writer);
            }
            try writer.print(")", .{});
        },
        .field => |f| {
            try writer.print("(. ", .{});
            try formatExpr(f.receiver, writer);
            try writer.print(" {s})", .{f.name});
        },
        .index => |i| {
            try writer.print("([] ", .{});
            try formatExpr(i.receiver, writer);
            try writer.print(" ", .{});
            try formatExpr(i.index, writer);
            try writer.print(")", .{});
        },
        .range => |r| {
            const op = if (r.inclusive) "..=" else "..";
            try writer.print("({s} ", .{op});
            try formatExpr(r.start, writer);
            try writer.print(" ", .{});
            try formatExpr(r.end, writer);
            try writer.print(")", .{});
        },
        .try_expr => |inner| {
            try writer.print("(try ", .{});
            try formatExpr(inner, writer);
            try writer.print(")", .{});
        },
        .array_literal => |elems| {
            try writer.print("(array", .{});
            for (elems) |*e| {
                try writer.print(" ", .{});
                try formatExpr(e, writer);
            }
            try writer.print(")", .{});
        },
        .map_literal => |entries| {
            try writer.print("(map", .{});
            for (entries) |e| {
                try writer.print(" (", .{});
                try formatExpr(e.key, writer);
                try writer.print(" ", .{});
                try formatExpr(e.value, writer);
                try writer.print(")", .{});
            }
            try writer.print(")", .{});
        },
        .catch_expr => |c| {
            try writer.print("(catch ", .{});
            try formatExpr(c.inner, writer);
            if (c.err_binding) |name| {
                try writer.print(" |{s}| ", .{name});
            } else {
                try writer.print(" ", .{});
            }
            try formatExpr(c.handler, writer);
            try writer.print(")", .{});
        },
        .block => |b| try formatBlock(b, writer),
        .if_expr => |i| try formatIf(&i, writer),
        .match_expr => |m| try formatMatch(&m, writer),
        .record_literal => |r| try formatRecordLit(&r, writer),
        .closure => |c| try formatClosure(&c, writer),
        .go_launch => |inner| {
            try writer.print("(go ", .{});
            try formatExpr(inner, writer);
            try writer.print(")", .{});
        },
        .invalid => try writer.print("<invalid>", .{}),
    }
}

fn formatClosure(c: *const Expr.Closure, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("(fn", .{});
    for (c.params) |p| {
        try writer.print(" (", .{});
        try writer.print("{s}", .{p.name});
        if (p.type_ann) |ta| {
            try writer.print(": ", .{});
            try formatType(ta, writer);
        }
        try writer.print(")", .{});
    }
    if (c.return_type) |rt| {
        try writer.print(" -> ", .{});
        try formatType(rt, writer);
    }
    try writer.print(" ", .{});
    try formatBlock(c.body, writer);
    try writer.print(")", .{});
}

fn formatMatch(m: *const Expr.Match, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const head = if (m.exhaustive) "match" else "partial-match";
    try writer.print("({s} ", .{head});
    try formatExpr(m.scrutinee, writer);
    for (m.arms) |*arm| {
        try writer.print(" (arm ", .{});
        try formatPattern(&arm.pattern, writer);
        try writer.print(" ", .{});
        try formatExpr(arm.body, writer);
        try writer.print(")", .{});
    }
    try writer.print(")", .{});
}

fn formatRecordLit(r: *const Expr.RecordLit, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("(rec {s}", .{r.name});
    for (r.fields) |f| {
        if (f.value) |v| {
            try writer.print(" (f {s} ", .{f.name});
            try formatExpr(v, writer);
            try writer.print(")", .{});
        } else {
            // Shorthand — no value, field name implies the binding.
            try writer.print(" (f {s})", .{f.name});
        }
    }
    try writer.print(")", .{});
}

pub fn formatPattern(p: *const Pattern, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    switch (p.data) {
        .int_pat => |v| try writer.print("(intpat {s})", .{v}),
        .float_pat => |v| try writer.print("(floatpat {s})", .{v}),
        .string_pat => |v| try writer.print("(strpat {s})", .{v}),
        .bool_pat => |b| try writer.print("{s}pat", .{if (b) "true" else "false"}),
        .nil_pat => try writer.print("nilpat", .{}),
        .wildcard => try writer.print("_", .{}),
        .bind => |b| try writer.print("(bind {s})", .{b.name}),
        .constructor => |c| {
            try writer.print("(ctor {s}", .{c.name});
            for (c.fields) |f| {
                if (f.binding) |b| {
                    // Explicit rename: field `foo` bound to local `bar`.
                    try writer.print(" ({s} {s})", .{ f.field_name, b.name });
                } else {
                    // Shorthand: field name is the binding name.
                    try writer.print(" ({s})", .{f.field_name});
                }
            }
            try writer.print(")", .{});
        },
        .invalid => try writer.print("<invalid-pat>", .{}),
    }
}

pub fn formatPatternAlloc(p: *const Pattern, allocator: std.mem.Allocator) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();
    try formatPattern(p, &aw.writer);
    return try aw.toOwnedSlice();
}

// ----- declaration pretty-printers -----

pub fn formatDecl(d: *const Decl, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    if (d.exported) {
        try writer.print("(export ", .{});
        try formatDeclBody(&d.data, writer);
        try writer.print(")", .{});
    } else {
        try formatDeclBody(&d.data, writer);
    }
}

fn formatDeclBody(data: *const Decl.Data, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    switch (data.*) {
        .fn_decl => |f| try formatFnDecl(&f, writer),
        .record_decl => |r| try formatRecordDecl(&r, writer),
        .enum_decl => |e| try formatEnumDecl(&e, writer),
        .type_alias => |t| try formatTypeAlias(&t, writer),
        .const_decl => |c| try formatConstDecl(&c, writer),
        .invalid => try writer.print("<invalid-decl>", .{}),
    }
}

fn formatGenerics(gs: []const GenericParam, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    if (gs.len == 0) return;
    try writer.print(" (generics", .{});
    for (gs) |g| try writer.print(" {s}", .{g.name});
    try writer.print(")", .{});
}

fn formatFnDecl(f: *const FnDecl, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("(fn {s}", .{f.name});
    try formatGenerics(f.generic_params, writer);
    for (f.params) |p| {
        try writer.print(" (", .{});
        try writer.print("{s}", .{p.name});
        if (p.type_ann) |ta| {
            try writer.print(": ", .{});
            try formatType(ta, writer);
        }
        try writer.print(")", .{});
    }
    if (f.return_type) |rt| {
        try writer.print(" -> ", .{});
        try formatType(rt, writer);
    }
    try writer.print(" ", .{});
    try formatBlock(f.body, writer);
    try writer.print(")", .{});
}

fn formatRecordDecl(r: *const RecordDecl, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("(record {s}", .{r.name});
    try formatGenerics(r.generic_params, writer);
    for (r.fields) |fd| {
        try writer.print(" (field {s} ", .{fd.name});
        try formatType(fd.type_ann, writer);
        try writer.print(")", .{});
    }
    try writer.print(")", .{});
}

fn formatEnumDecl(e: *const EnumDecl, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("(enum {s}", .{e.name});
    try formatGenerics(e.generic_params, writer);
    for (e.variants) |v| {
        try writer.print(" (variant {s}", .{v.name});
        for (v.fields) |fd| {
            try writer.print(" (field {s} ", .{fd.name});
            try formatType(fd.type_ann, writer);
            try writer.print(")", .{});
        }
        try writer.print(")", .{});
    }
    try writer.print(")", .{});
}

fn formatTypeAlias(t: *const TypeAlias, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("(typedef {s}", .{t.name});
    try formatGenerics(t.generic_params, writer);
    try writer.print(" ", .{});
    try formatType(t.target, writer);
    try writer.print(")", .{});
}

fn formatConstDecl(c: *const ConstDecl, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("(const {s}", .{c.name});
    if (c.type_ann) |ta| {
        try writer.print(" : ", .{});
        try formatType(ta, writer);
    }
    try writer.print(" ", .{});
    try formatExpr(c.value, writer);
    try writer.print(")", .{});
}

pub fn formatFile(f: *const File, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("(file", .{});
    for (f.decls) |*d| {
        try writer.print(" ", .{});
        try formatDecl(d, writer);
    }
    try writer.print(")", .{});
}

pub fn formatDeclAlloc(d: *const Decl, allocator: std.mem.Allocator) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();
    try formatDecl(d, &aw.writer);
    return try aw.toOwnedSlice();
}

pub fn formatFileAlloc(f: *const File, allocator: std.mem.Allocator) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();
    try formatFile(f, &aw.writer);
    return try aw.toOwnedSlice();
}

pub fn formatBlock(b: *const Block, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("(block", .{});
    for (b.stmts) |*s| {
        try writer.print(" ", .{});
        try formatStmt(s, writer);
    }
    if (b.trailing) |t| {
        try writer.print(" => ", .{});
        try formatExpr(t, writer);
    }
    try writer.print(")", .{});
}

fn formatIf(i: *const Expr.If, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("(if ", .{});
    try formatExpr(i.cond, writer);
    try writer.print(" ", .{});
    try formatBlock(i.then_block, writer);
    if (i.else_branch) |eb| switch (eb) {
        .block => |b| {
            try writer.print(" else ", .{});
            try formatBlock(b, writer);
        },
        .if_expr => |e| {
            try writer.print(" else ", .{});
            try formatExpr(e, writer);
        },
    };
    try writer.print(")", .{});
}

pub fn formatType(ty: *const TypeExpr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    switch (ty.data) {
        .named => |n| {
            if (n.type_args.len == 0) {
                try writer.print("{s}", .{n.name});
            } else {
                try writer.print("{s}<", .{n.name});
                for (n.type_args, 0..) |*ta, i| {
                    if (i != 0) try writer.print(", ", .{});
                    try formatType(ta, writer);
                }
                try writer.print(">", .{});
            }
        },
        .optional => |inner| {
            try formatType(inner, writer);
            try writer.print("?", .{});
        },
        .error_type => |inner| {
            try writer.print("!", .{});
            try formatType(inner, writer);
        },
        .function => |f| {
            try writer.print("fn(", .{});
            for (f.params, 0..) |*p, i| {
                if (i != 0) try writer.print(", ", .{});
                try formatType(p, writer);
            }
            try writer.print(") -> ", .{});
            try formatType(f.ret, writer);
        },
        .self_type => try writer.print("self", .{}),
        .invalid => try writer.print("<invalid>", .{}),
    }
}

pub fn formatStmt(stmt: *const Stmt, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    switch (stmt.data) {
        .expr => |e| {
            try writer.print("(expr ", .{});
            try formatExpr(e, writer);
            try writer.print(")", .{});
        },
        .var_decl => |v| {
            try writer.print("({s} {s}", .{ if (v.is_const) "const" else "var", v.name });
            if (v.type_ann) |ta| {
                try writer.print(" : ", .{});
                try formatType(ta, writer);
            }
            try writer.print(" ", .{});
            try formatExpr(v.value, writer);
            try writer.print(")", .{});
        },
        .assign => |a| {
            try writer.print("({s} ", .{a.op.symbol()});
            try formatExpr(a.target, writer);
            try writer.print(" ", .{});
            try formatExpr(a.value, writer);
            try writer.print(")", .{});
        },
        .return_stmt => |v| {
            if (v) |e| {
                try writer.print("(return ", .{});
                try formatExpr(e, writer);
                try writer.print(")", .{});
            } else {
                try writer.print("(return)", .{});
            }
        },
        .break_stmt => try writer.print("(break)", .{}),
        .continue_stmt => try writer.print("(continue)", .{}),
        .while_stmt => |w| {
            try writer.print("(while ", .{});
            try formatExpr(w.cond, writer);
            try writer.print(" ", .{});
            try formatBlock(w.body, writer);
            try writer.print(")", .{});
        },
        .for_stmt => |f| {
            try writer.print("(for ", .{});
            switch (f.pattern) {
                .single => |b| try writer.print("{s}", .{b.name}),
                .key_value => |kv| try writer.print("(kv {s} {s})", .{ kv.key.name, kv.value.name }),
            }
            try writer.print(" ", .{});
            try formatExpr(f.iterable, writer);
            try writer.print(" ", .{});
            try formatBlock(f.body, writer);
            try writer.print(")", .{});
        },
        .invalid => try writer.print("<invalid>", .{}),
    }
}

/// Convenience: format to an owned string. Caller frees with `allocator.free`.
/// Uses `Allocating.init` + `errdefer deinit` so that an error mid-format
/// (e.g. OOM) doesn't leak the intermediate buffer.
pub fn formatExprAlloc(expr: *const Expr, allocator: std.mem.Allocator) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();
    try formatExpr(expr, &aw.writer);
    return try aw.toOwnedSlice();
}

pub fn formatStmtAlloc(stmt: *const Stmt, allocator: std.mem.Allocator) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();
    try formatStmt(stmt, &aw.writer);
    return try aw.toOwnedSlice();
}

pub fn formatTypeAlloc(ty: *const TypeExpr, allocator: std.mem.Allocator) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();
    try formatType(ty, &aw.writer);
    return try aw.toOwnedSlice();
}

// ====================== tests ======================

test "span merge" {
    const a: Span = .{ .start = 3, .end = 10 };
    const b: Span = .{ .start = 0, .end = 5 };
    const m = Span.merge(a, b);
    try std.testing.expectEqual(@as(u32, 0), m.start);
    try std.testing.expectEqual(@as(u32, 10), m.end);
}

test "format simple int literal" {
    const alloc = std.testing.allocator;
    var e: Expr = .{ .span = .{ .start = 0, .end = 2 }, .data = .{ .int_literal = "42" } };
    const s = try formatExprAlloc(&e, alloc);
    defer alloc.free(s);
    try std.testing.expectEqualStrings("(int 42)", s);
}

test "format nested binary" {
    const alloc = std.testing.allocator;
    var one: Expr = .{ .span = .{ .start = 0, .end = 1 }, .data = .{ .int_literal = "1" } };
    var two: Expr = .{ .span = .{ .start = 0, .end = 1 }, .data = .{ .int_literal = "2" } };
    var three: Expr = .{ .span = .{ .start = 0, .end = 1 }, .data = .{ .int_literal = "3" } };
    var inner: Expr = .{
        .span = .{ .start = 0, .end = 3 },
        .data = .{ .binary = .{ .op = .mul, .lhs = &two, .rhs = &three } },
    };
    var outer: Expr = .{
        .span = .{ .start = 0, .end = 5 },
        .data = .{ .binary = .{ .op = .add, .lhs = &one, .rhs = &inner } },
    };
    const s = try formatExprAlloc(&outer, alloc);
    defer alloc.free(s);
    try std.testing.expectEqualStrings("(+ (int 1) (* (int 2) (int 3)))", s);
}
