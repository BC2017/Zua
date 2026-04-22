//! Zua parser — turns tokens into an AST.
//!
//! Handwritten recursive descent for statements and types, Pratt parsing
//! for expressions (cleanest way to hand-roll a full precedence table).
//! AST nodes are arena-allocated: the caller passes in an `Allocator`
//! (typically an `ArenaAllocator`) that outlives the returned AST, and
//! every child pointer and slice lives inside that arena.
//!
//! Error handling is diagnostic-collecting, not fail-fast: when something
//! fails to parse at statement scope, the parser emits a `Diagnostic`,
//! synthesises an `.invalid` node, skips to the next `;` or `}`, and
//! resumes. That way a single program with five bugs gives you five
//! diagnostics, not one.
//!
//! Scope for this file (Phase A of the parser work):
//!   - Every expression form except record literals, closures, block/if/
//!     match expressions, and `go` launches — those land in phase B/C.
//!   - Type expressions: named + generics, optional (`T?`), error (`!T`),
//!     function types (`fn(...) -> T`), `self`.
//!   - Simple statements: `var`/`const` declarations with optional type
//!     annotation, assignment (`=` + compound forms), bare expression,
//!     `return`, `break`, `continue`. Blocks: a `{...}` sequence of those.

const std = @import("std");
const lexer = @import("lexer.zig");
const ast = @import("ast.zig");

const Token = lexer.Token;
const Tokenizer = lexer.Tokenizer;
const Expr = ast.Expr;
const TypeExpr = ast.TypeExpr;
const Stmt = ast.Stmt;
const Span = ast.Span;

pub const Diagnostic = struct {
    span: Span,
    message: []const u8,
};

pub const Error = error{InvalidSyntax} || std.mem.Allocator.Error;

pub const Parser = struct {
    /// All AST allocations and diagnostic message buffers come from here.
    /// The caller frees this with a single `deinit` on their arena.
    arena: std.mem.Allocator,
    source: [:0]const u8,
    tokens: []const Token,
    index: usize,
    diagnostics: std.ArrayList(Diagnostic),

    pub fn init(source: [:0]const u8, arena: std.mem.Allocator) !Parser {
        var list: std.ArrayList(Token) = .empty;
        var tk = Tokenizer.init(source);
        while (true) {
            const t = tk.next();
            try list.append(arena, t);
            if (t.kind == .eof) break;
        }
        return .{
            .arena = arena,
            .source = source,
            .tokens = list.items,
            .index = 0,
            .diagnostics = .empty,
        };
    }

    // ======================= token helpers =======================

    fn peek(self: *const Parser) Token {
        return self.tokens[self.index];
    }

    fn peekKind(self: *const Parser) Token.Kind {
        return self.tokens[self.index].kind;
    }

    fn peekAhead(self: *const Parser, offset: usize) Token.Kind {
        const i = self.index + offset;
        if (i >= self.tokens.len) return .eof;
        return self.tokens[i].kind;
    }

    fn advance(self: *Parser) Token {
        const t = self.tokens[self.index];
        if (t.kind != .eof) self.index += 1;
        return t;
    }

    fn match(self: *Parser, kind: Token.Kind) bool {
        if (self.peekKind() == kind) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    fn expect(self: *Parser, kind: Token.Kind, msg: []const u8) !Token {
        const t = self.peek();
        if (t.kind == kind) {
            _ = self.advance();
            return t;
        }
        try self.diag(t, msg);
        return error.InvalidSyntax;
    }

    fn skipSemis(self: *Parser) void {
        while (self.peekKind() == .semi) _ = self.advance();
    }

    // ======================= diagnostics =======================

    fn diag(self: *Parser, tok: Token, message: []const u8) !void {
        const owned = try self.arena.dupe(u8, message);
        try self.diagnostics.append(self.arena, .{
            .span = spanOf(tok),
            .message = owned,
        });
    }

    fn diagf(self: *Parser, tok: Token, comptime fmt: []const u8, args: anytype) !void {
        const msg = try std.fmt.allocPrint(self.arena, fmt, args);
        try self.diagnostics.append(self.arena, .{
            .span = spanOf(tok),
            .message = msg,
        });
    }

    // ======================= node construction =======================

    fn makeExpr(self: *Parser, data: Expr.Data, span: Span) !*Expr {
        const e = try self.arena.create(Expr);
        e.* = .{ .span = span, .data = data };
        return e;
    }

    fn makeType(self: *Parser, data: TypeExpr.Data, span: Span) !*TypeExpr {
        const t = try self.arena.create(TypeExpr);
        t.* = .{ .span = span, .data = data };
        return t;
    }

    fn invalidExpr(self: *Parser, span: Span) !*Expr {
        return self.makeExpr(.invalid, span);
    }

    // ======================= expressions (Pratt) =======================

    const Prec = enum(u8) {
        none = 0,
        or_ = 1,
        and_ = 2,
        equality = 3,
        comparison = 4,
        range = 5,
        term = 6,
        factor = 7,
        unary = 8,
        postfix = 9,
    };

    /// Precedence of the given token *as an infix operator*. Tokens that
    /// cannot appear in infix position (or that we haven't assigned a
    /// precedence for) return `.none`, terminating the Pratt loop.
    fn infixPrec(kind: Token.Kind) Prec {
        return switch (kind) {
            .kw_or => .or_,
            .kw_and => .and_,
            .eq, .neq => .equality,
            .lt, .gt, .le, .ge => .comparison,
            .dotdot, .dotdot_eq => .range,
            .plus, .minus => .term,
            .star, .slash, .percent => .factor,
            .dot, .lbracket, .lparen => .postfix,
            else => .none,
        };
    }

    pub fn parseExpr(self: *Parser) Error!*Expr {
        return self.parseExprPrec(.or_);
    }

    fn parseExprPrec(self: *Parser, min_prec: Prec) Error!*Expr {
        var left = try self.parsePrefix();
        while (true) {
            const tok = self.peek();
            const prec = infixPrec(tok.kind);
            if (@intFromEnum(prec) < @intFromEnum(min_prec)) break;
            left = try self.parseInfix(left, tok, prec);
        }
        return left;
    }

    fn parsePrefix(self: *Parser) Error!*Expr {
        const tok = self.peek();
        switch (tok.kind) {
            .int_literal => {
                _ = self.advance();
                return self.makeExpr(.{ .int_literal = tok.lexeme(self.source) }, spanOf(tok));
            },
            .float_literal => {
                _ = self.advance();
                return self.makeExpr(.{ .float_literal = tok.lexeme(self.source) }, spanOf(tok));
            },
            .string_literal => {
                _ = self.advance();
                return self.makeExpr(.{ .string_literal = tok.lexeme(self.source) }, spanOf(tok));
            },
            .string_start => return self.parseStringInterp(),
            .kw_true => {
                _ = self.advance();
                return self.makeExpr(.{ .bool_literal = true }, spanOf(tok));
            },
            .kw_false => {
                _ = self.advance();
                return self.makeExpr(.{ .bool_literal = false }, spanOf(tok));
            },
            .kw_nil => {
                _ = self.advance();
                return self.makeExpr(.nil_literal, spanOf(tok));
            },
            .kw_self => {
                _ = self.advance();
                return self.makeExpr(.self_ref, spanOf(tok));
            },
            .ident => {
                _ = self.advance();
                return self.makeExpr(.{ .ident = tok.lexeme(self.source) }, spanOf(tok));
            },
            .lparen => return self.parseParenExpr(),
            .lbracket => return self.parseArrayLit(),
            .lbrace => return self.parseMapLit(),
            .minus => return self.parseUnary(.neg),
            .kw_not => return self.parseUnary(.not),
            .kw_try => return self.parseTry(),
            else => {
                try self.diagf(tok, "expected expression, got {s}", .{@tagName(tok.kind)});
                return error.InvalidSyntax;
            },
        }
    }

    fn parseInfix(self: *Parser, left: *Expr, op_tok: Token, prec: Prec) Error!*Expr {
        switch (op_tok.kind) {
            .plus, .minus, .star, .slash, .percent,
            .eq, .neq, .lt, .gt, .le, .ge,
            .kw_and, .kw_or,
            => {
                _ = self.advance();
                // Left-associative: right operand takes precedence strictly
                // higher than `prec`. For right-associative ops we'd use
                // `prec` itself, but Zua has none of those at the moment.
                const rhs = try self.parseExprPrec(@enumFromInt(@intFromEnum(prec) + 1));
                const op = binOpFromKind(op_tok.kind);
                return self.makeExpr(
                    .{ .binary = .{ .op = op, .lhs = left, .rhs = rhs } },
                    Span.merge(left.span, rhs.span),
                );
            },
            .dotdot, .dotdot_eq => {
                _ = self.advance();
                const rhs = try self.parseExprPrec(@enumFromInt(@intFromEnum(prec) + 1));
                const inclusive = op_tok.kind == .dotdot_eq;
                return self.makeExpr(
                    .{ .range = .{ .start = left, .end = rhs, .inclusive = inclusive } },
                    Span.merge(left.span, rhs.span),
                );
            },
            .dot => return self.parseDotOrMethod(left),
            .lbracket => return self.parseIndex(left),
            .lparen => return self.parseCall(left),
            else => unreachable,
        }
    }

    fn parseUnary(self: *Parser, op: ast.UnaryOp) Error!*Expr {
        const op_tok = self.advance();
        const operand = try self.parseExprPrec(.unary);
        return self.makeExpr(
            .{ .unary = .{ .op = op, .operand = operand } },
            Span.merge(spanOf(op_tok), operand.span),
        );
    }

    fn parseTry(self: *Parser) Error!*Expr {
        const try_tok = self.advance();
        const inner = try self.parseExprPrec(.unary);
        return self.makeExpr(
            .{ .try_expr = inner },
            Span.merge(spanOf(try_tok), inner.span),
        );
    }

    fn parseParenExpr(self: *Parser) Error!*Expr {
        _ = self.advance(); // consume (
        const inner = try self.parseExpr();
        _ = try self.expect(.rparen, "expected ')' to close grouped expression");
        // Parens are a parsing construct only — they affect precedence but
        // carry no AST-level meaning, so we return the inner expression
        // with its own span.
        return inner;
    }

    fn parseArrayLit(self: *Parser) Error!*Expr {
        const start_tok = self.advance(); // consume [
        var elems: std.ArrayList(Expr) = .empty;
        if (self.peekKind() != .rbracket) {
            while (true) {
                const e = try self.parseExpr();
                try elems.append(self.arena, e.*);
                if (self.peekKind() != .comma) break;
                _ = self.advance();
                if (self.peekKind() == .rbracket) break; // allow trailing comma
            }
        }
        const end_tok = try self.expect(.rbracket, "expected ']' to close array literal");
        return self.makeExpr(
            .{ .array_literal = elems.items },
            Span.merge(spanOf(start_tok), spanOf(end_tok)),
        );
    }

    fn parseMapLit(self: *Parser) Error!*Expr {
        const start_tok = self.advance(); // consume {
        var entries: std.ArrayList(Expr.MapEntry) = .empty;
        if (self.peekKind() != .rbrace) {
            while (true) {
                const key = try self.parseExpr();
                _ = try self.expect(.colon, "expected ':' between map key and value");
                const val = try self.parseExpr();
                try entries.append(self.arena, .{ .key = key, .value = val });
                if (self.peekKind() != .comma) break;
                _ = self.advance();
                if (self.peekKind() == .rbrace) break; // trailing comma
            }
        }
        const end_tok = try self.expect(.rbrace, "expected '}' to close map literal");
        return self.makeExpr(
            .{ .map_literal = entries.items },
            Span.merge(spanOf(start_tok), spanOf(end_tok)),
        );
    }

    fn parseCall(self: *Parser, callee: *Expr) Error!*Expr {
        _ = self.advance(); // consume (
        var args: std.ArrayList(Expr) = .empty;
        if (self.peekKind() != .rparen) {
            while (true) {
                const a = try self.parseExpr();
                try args.append(self.arena, a.*);
                if (self.peekKind() != .comma) break;
                _ = self.advance();
                if (self.peekKind() == .rparen) break;
            }
        }
        const end_tok = try self.expect(.rparen, "expected ')' to close call");
        return self.makeExpr(
            .{ .call = .{ .callee = callee, .args = args.items } },
            Span.merge(callee.span, spanOf(end_tok)),
        );
    }

    fn parseIndex(self: *Parser, receiver: *Expr) Error!*Expr {
        _ = self.advance(); // consume [
        const idx = try self.parseExpr();
        const end_tok = try self.expect(.rbracket, "expected ']' to close index");
        return self.makeExpr(
            .{ .index = .{ .receiver = receiver, .index = idx } },
            Span.merge(receiver.span, spanOf(end_tok)),
        );
    }

    /// Handles both `a.b` (field access) and `a.b(args)` (method call).
    /// We look one token past the identifier to decide: if a `(` follows
    /// the `.name`, we collapse the `.` + call into a dedicated
    /// `method_call` node so later stages don't have to re-discover the
    /// method-call pattern inside a `call(field(...), ...)`.
    fn parseDotOrMethod(self: *Parser, receiver: *Expr) Error!*Expr {
        _ = self.advance(); // consume .
        const name_tok = self.peek();
        if (name_tok.kind != .ident) {
            try self.diagf(name_tok, "expected field name after '.', got {s}", .{@tagName(name_tok.kind)});
            return error.InvalidSyntax;
        }
        _ = self.advance();
        const name = name_tok.lexeme(self.source);

        if (self.peekKind() == .lparen) {
            _ = self.advance(); // consume (
            var args: std.ArrayList(Expr) = .empty;
            if (self.peekKind() != .rparen) {
                while (true) {
                    const a = try self.parseExpr();
                    try args.append(self.arena, a.*);
                    if (self.peekKind() != .comma) break;
                    _ = self.advance();
                    if (self.peekKind() == .rparen) break;
                }
            }
            const end_tok = try self.expect(.rparen, "expected ')' to close method call");
            return self.makeExpr(
                .{ .method_call = .{ .receiver = receiver, .name = name, .args = args.items } },
                Span.merge(receiver.span, spanOf(end_tok)),
            );
        }

        return self.makeExpr(
            .{ .field = .{ .receiver = receiver, .name = name } },
            Span.merge(receiver.span, spanOf(name_tok)),
        );
    }

    fn parseStringInterp(self: *Parser) Error!*Expr {
        const start_tok = self.advance(); // consume string_start
        const start_lex = start_tok.lexeme(self.source);
        // Lexeme is `"hello ${` — strip 1 leading (`"`) and 2 trailing (`${`).
        const first_text = start_lex[1 .. start_lex.len - 2];

        var parts: std.ArrayList(ast.StringPart) = .empty;
        try parts.append(self.arena, .{ .text = first_text });

        var span = spanOf(start_tok);

        while (true) {
            const expr = try self.parseExpr();
            try parts.append(self.arena, .{ .expr = expr });

            const next = self.peek();
            switch (next.kind) {
                .string_part => {
                    _ = self.advance();
                    const lex = next.lexeme(self.source);
                    // Lexeme is `}...${` — strip 1 and 2.
                    try parts.append(self.arena, .{ .text = lex[1 .. lex.len - 2] });
                },
                .string_end => {
                    _ = self.advance();
                    const lex = next.lexeme(self.source);
                    // Lexeme is `}..."` — strip 1 and 1.
                    try parts.append(self.arena, .{ .text = lex[1 .. lex.len - 1] });
                    span = Span.merge(span, spanOf(next));
                    break;
                },
                else => {
                    try self.diagf(next, "expected '}}' to close string interpolation, got {s}", .{@tagName(next.kind)});
                    return error.InvalidSyntax;
                },
            }
        }

        return self.makeExpr(.{ .string_interp = parts.items }, span);
    }

    // ======================= types =======================

    /// Parse a type expression. Starts at any primary type form and then
    /// applies postfix suffixes:
    ///   - `T?`       -> optional
    /// The `!T` prefix form is a prefix, not a postfix, so it's handled
    /// at the top level.
    pub fn parseType(self: *Parser) Error!*TypeExpr {
        const tok = self.peek();
        var ty: *TypeExpr = switch (tok.kind) {
            .bang => blk: {
                _ = self.advance();
                const inner = try self.parseType();
                break :blk try self.makeType(
                    .{ .error_type = inner },
                    Span.merge(spanOf(tok), inner.span),
                );
            },
            .kw_self => blk: {
                _ = self.advance();
                break :blk try self.makeType(.self_type, spanOf(tok));
            },
            .kw_fn => try self.parseFunctionType(),
            .ident => try self.parseNamedType(),
            else => {
                try self.diagf(tok, "expected type, got {s}", .{@tagName(tok.kind)});
                return error.InvalidSyntax;
            },
        };

        // Postfix `?` for optional — may chain (`T??` is weird but legal).
        while (self.peekKind() == .question) {
            const q = self.advance();
            ty = try self.makeType(
                .{ .optional = ty },
                Span.merge(ty.span, spanOf(q)),
            );
        }
        return ty;
    }

    fn parseNamedType(self: *Parser) Error!*TypeExpr {
        const name_tok = self.advance();
        const name = name_tok.lexeme(self.source);
        var span = spanOf(name_tok);
        var args: std.ArrayList(TypeExpr) = .empty;
        if (self.peekKind() == .lt) {
            _ = self.advance();
            if (self.peekKind() != .gt) {
                while (true) {
                    const arg = try self.parseType();
                    try args.append(self.arena, arg.*);
                    if (self.peekKind() != .comma) break;
                    _ = self.advance();
                }
            }
            const end_tok = try self.expect(.gt, "expected '>' to close generic arguments");
            span = Span.merge(span, spanOf(end_tok));
        }
        return self.makeType(
            .{ .named = .{ .name = name, .type_args = args.items } },
            span,
        );
    }

    fn parseFunctionType(self: *Parser) Error!*TypeExpr {
        const fn_tok = self.advance(); // consume `fn`
        _ = try self.expect(.lparen, "expected '(' after 'fn' in function type");
        var params: std.ArrayList(TypeExpr) = .empty;
        if (self.peekKind() != .rparen) {
            while (true) {
                const p = try self.parseType();
                try params.append(self.arena, p.*);
                if (self.peekKind() != .comma) break;
                _ = self.advance();
            }
        }
        _ = try self.expect(.rparen, "expected ')' to close function parameter types");
        _ = try self.expect(.arrow, "expected '->' before function return type");
        const ret = try self.parseType();
        return self.makeType(
            .{ .function = .{ .params = params.items, .ret = ret } },
            Span.merge(spanOf(fn_tok), ret.span),
        );
    }

    // ======================= statements =======================

    /// Parse one statement. On error, emits a diagnostic, rewinds the
    /// token stream to the next synchronization point, and returns an
    /// `invalid` statement so the surrounding parser can continue.
    pub fn parseStmt(self: *Parser) !Stmt {
        const start_tok = self.peek();
        const result = self.parseStmtInner() catch |err| switch (err) {
            error.InvalidSyntax => {
                self.syncToStmtBoundary();
                const span = Span.merge(
                    spanOf(start_tok),
                    spanOf(self.peek()),
                );
                return Stmt{ .span = span, .data = .invalid };
            },
            else => return err,
        };
        return result;
    }

    fn parseStmtInner(self: *Parser) Error!Stmt {
        const tok = self.peek();
        switch (tok.kind) {
            .kw_const => return self.parseVarDecl(true),
            .kw_var => return self.parseVarDecl(false),
            .kw_return => return self.parseReturn(),
            .kw_break => {
                _ = self.advance();
                return .{ .span = spanOf(tok), .data = .break_stmt };
            },
            .kw_continue => {
                _ = self.advance();
                return .{ .span = spanOf(tok), .data = .continue_stmt };
            },
            else => return self.parseExprOrAssignStmt(),
        }
    }

    fn parseVarDecl(self: *Parser, is_const: bool) Error!Stmt {
        const kw_tok = self.advance();
        const name_tok = try self.expect(.ident, "expected identifier after 'var' or 'const'");
        const name = name_tok.lexeme(self.source);

        var type_ann: ?*TypeExpr = null;
        if (self.peekKind() == .colon) {
            _ = self.advance();
            type_ann = try self.parseType();
        }

        _ = try self.expect(.assign, "expected '=' in variable declaration");
        const value = try self.parseExpr();

        return .{
            .span = Span.merge(spanOf(kw_tok), value.span),
            .data = .{ .var_decl = .{
                .is_const = is_const,
                .name = name,
                .name_span = spanOf(name_tok),
                .type_ann = type_ann,
                .value = value,
            } },
        };
    }

    fn parseReturn(self: *Parser) Error!Stmt {
        const kw_tok = self.advance();
        // `return` with nothing after it is "return nil" (a plain return).
        // "nothing after" means: a statement-terminating token, EOF, or `}`.
        const next = self.peekKind();
        if (next == .semi or next == .eof or next == .rbrace) {
            return .{ .span = spanOf(kw_tok), .data = .{ .return_stmt = null } };
        }
        const value = try self.parseExpr();
        return .{
            .span = Span.merge(spanOf(kw_tok), value.span),
            .data = .{ .return_stmt = value },
        };
    }

    /// An expression statement or an assignment. We parse the LHS as an
    /// expression first, then peek for `=` / `+=` / etc. If we see one,
    /// promote to an assignment and parse the RHS; otherwise treat the
    /// whole thing as a bare expression statement.
    fn parseExprOrAssignStmt(self: *Parser) Error!Stmt {
        const expr = try self.parseExpr();
        const op_kind = self.peekKind();
        const op: ?ast.AssignOp = switch (op_kind) {
            .assign => .assign,
            .plus_eq => .add_assign,
            .minus_eq => .sub_assign,
            .star_eq => .mul_assign,
            .slash_eq => .div_assign,
            .percent_eq => .mod_assign,
            else => null,
        };
        if (op) |asgn_op| {
            _ = self.advance();
            const value = try self.parseExpr();
            return .{
                .span = Span.merge(expr.span, value.span),
                .data = .{ .assign = .{ .op = asgn_op, .target = expr, .value = value } },
            };
        }
        return .{
            .span = expr.span,
            .data = .{ .expr = expr },
        };
    }

    /// Parse a `{ stmt; stmt; ... }` block. Returns the sequence of
    /// statements; the caller attaches a span / chooses how to wrap it.
    pub fn parseBlock(self: *Parser) ![]Stmt {
        _ = try self.expect(.lbrace, "expected '{' to open block");
        var stmts: std.ArrayList(Stmt) = .empty;
        self.skipSemis();
        while (self.peekKind() != .rbrace and self.peekKind() != .eof) {
            const s = try self.parseStmt();
            try stmts.append(self.arena, s);
            // Require a separator between statements: either an explicit
            // `;`, an implicit `;` (which the lexer inserts for ASI), or
            // the end of the block (`}`). Anything else means two
            // statements ran together on one line.
            switch (self.peekKind()) {
                .semi => self.skipSemis(),
                .rbrace, .eof => {},
                else => {
                    const bad = self.peek();
                    try self.diagf(bad, "expected ';' or newline between statements, got {s}", .{@tagName(bad.kind)});
                    self.syncToStmtBoundary();
                },
            }
        }
        _ = try self.expect(.rbrace, "expected '}' to close block");
        return stmts.items;
    }

    /// Recovery: skip forward until we stop *at* a token that plausibly
    /// marks the end of a statement. Leaving the `;` / `}` / EOF in place
    /// lets the caller's normal separator handling take over — otherwise
    /// consuming the boundary here would trip `parseBlock`'s "expected `;`
    /// between statements" check on the next iteration and cascade into
    /// losing the next well-formed statement. We don't descend into brace-
    /// groups either; if an error happens on line 3, eating bytes until
    /// line 50's `}` would bury the real problem under spurious closers.
    fn syncToStmtBoundary(self: *Parser) void {
        while (true) {
            switch (self.peekKind()) {
                .semi, .rbrace, .eof => return,
                else => _ = self.advance(),
            }
        }
    }
};

// ======================= free helpers =======================

fn spanOf(tok: Token) Span {
    return .{ .start = @intCast(tok.loc.start), .end = @intCast(tok.loc.end) };
}

fn binOpFromKind(kind: Token.Kind) ast.BinaryOp {
    return switch (kind) {
        .plus => .add,
        .minus => .sub,
        .star => .mul,
        .slash => .div,
        .percent => .mod,
        .eq => .eq,
        .neq => .neq,
        .lt => .lt,
        .gt => .gt,
        .le => .le,
        .ge => .ge,
        .kw_and => .logical_and,
        .kw_or => .logical_or,
        else => unreachable,
    };
}

// ======================= tests =======================

const testing = std.testing;

/// Parse an expression from `source` and compare its s-expression dump to
/// `expected`. Uses an arena so the test doesn't have to free individual
/// AST nodes — we throw the whole thing away at the end.
fn expectExpr(source: [:0]const u8, expected: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var parser = try Parser.init(source, arena);
    const e = try parser.parseExpr();
    // No trailing tokens allowed — tests should specify a complete expression.
    try testing.expectEqual(@as(usize, 0), parser.diagnostics.items.len);
    try testing.expectEqual(Token.Kind.eof, parser.peekKind());
    const rendered = try ast.formatExprAlloc(e, testing.allocator);
    defer testing.allocator.free(rendered);
    try testing.expectEqualStrings(expected, rendered);
}

fn expectType(source: [:0]const u8, expected: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var parser = try Parser.init(source, arena);
    const ty = try parser.parseType();
    try testing.expectEqual(@as(usize, 0), parser.diagnostics.items.len);
    try testing.expectEqual(Token.Kind.eof, parser.peekKind());
    const rendered = try ast.formatTypeAlloc(ty, testing.allocator);
    defer testing.allocator.free(rendered);
    try testing.expectEqualStrings(expected, rendered);
}

fn expectStmt(source: [:0]const u8, expected: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var parser = try Parser.init(source, arena);
    const s = try parser.parseStmt();
    try testing.expectEqual(@as(usize, 0), parser.diagnostics.items.len);
    // Tolerate a trailing implicit semi (Zua programs often end with a newline).
    parser.skipSemis();
    try testing.expectEqual(Token.Kind.eof, parser.peekKind());
    const rendered = try ast.formatStmtAlloc(&s, testing.allocator);
    defer testing.allocator.free(rendered);
    try testing.expectEqualStrings(expected, rendered);
}

fn expectParseError(source: [:0]const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var parser = try Parser.init(source, arena);
    _ = parser.parseExpr() catch |err| switch (err) {
        error.InvalidSyntax => {
            try testing.expect(parser.diagnostics.items.len > 0);
            return;
        },
        else => return err,
    };
    // If we got here, parseExpr succeeded when we expected a failure.
    try testing.expect(false);
}

// ----- primary expressions -----

test "int literal" {
    try expectExpr("42", "(int 42)");
    try expectExpr("0xFF", "(int 0xFF)");
    try expectExpr("1_000_000", "(int 1_000_000)");
}

test "float literal" {
    try expectExpr("3.14", "(float 3.14)");
    try expectExpr("1e10", "(float 1e10)");
}

test "string literal" {
    try expectExpr("\"hello\"", "(str \"hello\")");
}

test "bool and nil" {
    try expectExpr("true", "true");
    try expectExpr("false", "false");
    try expectExpr("nil", "nil");
}

test "identifier and self" {
    try expectExpr("foo", "(id foo)");
    try expectExpr("self", "self");
}

test "parenthesised expression has no AST node" {
    try expectExpr("(1 + 2)", "(+ (int 1) (int 2))");
    try expectExpr("((42))", "(int 42)");
}

// ----- unary / try -----

test "unary minus" {
    try expectExpr("-5", "(- (int 5))");
    try expectExpr("-(-5)", "(- (- (int 5)))");
}

test "logical not" {
    try expectExpr("not true", "(not true)");
    try expectExpr("not not x", "(not (not (id x)))");
}

test "try wraps the next expression" {
    try expectExpr("try foo()", "(try (call (id foo)))");
    try expectExpr("try x", "(try (id x))");
}

// ----- binary precedence -----

test "multiplication binds tighter than addition" {
    try expectExpr("1 + 2 * 3", "(+ (int 1) (* (int 2) (int 3)))");
    try expectExpr("1 * 2 + 3", "(+ (* (int 1) (int 2)) (int 3))");
}

test "all four levels of arithmetic" {
    // or < and < eq < cmp < range < term < factor
    try expectExpr("1 + 2 == 3 * 4", "(== (+ (int 1) (int 2)) (* (int 3) (int 4)))");
}

test "left associativity of same-precedence operators" {
    try expectExpr("1 + 2 + 3", "(+ (+ (int 1) (int 2)) (int 3))");
    try expectExpr("10 - 3 - 2", "(- (- (int 10) (int 3)) (int 2))");
    try expectExpr("a == b == c", "(== (== (id a) (id b)) (id c))");
}

test "and is tighter than or" {
    try expectExpr("a or b and c", "(or (id a) (and (id b) (id c)))");
    try expectExpr("a and b or c", "(or (and (id a) (id b)) (id c))");
}

test "comparison is tighter than logical" {
    try expectExpr("x < 1 and y > 2", "(and (< (id x) (int 1)) (> (id y) (int 2)))");
}

test "unary binds tighter than binary" {
    try expectExpr("-x + y", "(+ (- (id x)) (id y))");
    try expectExpr("not a or b", "(or (not (id a)) (id b))");
}

// ----- ranges -----

test "range half-open and inclusive" {
    try expectExpr("0..10", "(.. (int 0) (int 10))");
    try expectExpr("0..=10", "(..= (int 0) (int 10))");
}

test "range is tighter than comparison but looser than addition" {
    // `a+1..b+2` should parse as `(a+1)..(b+2)`, not `a+(1..b)+2`.
    try expectExpr("a + 1 .. b + 2", "(.. (+ (id a) (int 1)) (+ (id b) (int 2)))");
}

// ----- postfix: call, index, field, method -----

test "simple call" {
    try expectExpr("f()", "(call (id f))");
    try expectExpr("f(1, 2, 3)", "(call (id f) (int 1) (int 2) (int 3))");
}

test "call with trailing comma" {
    try expectExpr("f(1, 2,)", "(call (id f) (int 1) (int 2))");
}

test "field access" {
    try expectExpr("a.b", "(. (id a) b)");
    try expectExpr("a.b.c", "(. (. (id a) b) c)");
}

test "method call collapses .name + () into one node" {
    try expectExpr("a.b()", "(mcall (id a) b)");
    try expectExpr("a.b(1, 2)", "(mcall (id a) b (int 1) (int 2))");
}

test "field access vs method call discriminated by paren" {
    try expectExpr("a.b.c()", "(mcall (. (id a) b) c)");
    try expectExpr("a.b().c", "(. (mcall (id a) b) c)");
}

test "index" {
    try expectExpr("arr[0]", "([] (id arr) (int 0))");
    try expectExpr("m[\"key\"]", "([] (id m) (str \"key\"))");
    try expectExpr("a[b][c]", "([] ([] (id a) (id b)) (id c))");
}

test "mixed postfix chains" {
    try expectExpr("xs[0].foo()", "(mcall ([] (id xs) (int 0)) foo)");
    try expectExpr("f(a)(b)", "(call (call (id f) (id a)) (id b))");
}

// ----- collection literals -----

test "empty and one-element arrays" {
    try expectExpr("[]", "(array)");
    try expectExpr("[1]", "(array (int 1))");
}

test "array literal with trailing comma" {
    try expectExpr("[1, 2, 3,]", "(array (int 1) (int 2) (int 3))");
}

test "empty and populated map literals" {
    try expectExpr("{}", "(map)");
    try expectExpr("{\"a\": 1}", "(map ((str \"a\") (int 1)))");
    try expectExpr(
        "{\"a\": 1, \"b\": 2}",
        "(map ((str \"a\") (int 1)) ((str \"b\") (int 2)))",
    );
}

// ----- string interpolation -----

test "simple string interpolation" {
    try expectExpr("\"hi ${name}!\"", "(interp (text \"hi \") (id name) (text \"!\"))");
}

test "multiple interpolations" {
    try expectExpr(
        "\"${a}-${b}\"",
        "(interp (text \"\") (id a) (text \"-\") (id b) (text \"\"))",
    );
}

test "interpolation with complex expression" {
    try expectExpr(
        "\"= ${x + 1} =\"",
        "(interp (text \"= \") (+ (id x) (int 1)) (text \" =\"))",
    );
}

// ----- types -----

test "primitive named type" {
    try expectType("int", "int");
    try expectType("string", "string");
}

test "generic type" {
    try expectType("Array<int>", "Array<int>");
    try expectType("Map<string, int>", "Map<string, int>");
    try expectType("Map<string, Array<int>>", "Map<string, Array<int>>");
}

test "optional type" {
    try expectType("int?", "int?");
    try expectType("Array<int>?", "Array<int>?");
}

test "error type" {
    try expectType("!int", "!int");
    try expectType("!Array<int>", "!Array<int>");
}

test "function type" {
    try expectType("fn() -> int", "fn() -> int");
    try expectType("fn(int, int) -> int", "fn(int, int) -> int");
    try expectType("fn(int) -> !string", "fn(int) -> !string");
}

test "self type" {
    try expectType("self", "self");
}

// ----- statements -----

test "var and const declarations" {
    try expectStmt("var x = 5", "(var x (int 5))");
    try expectStmt("const y = 10", "(const y (int 10))");
}

test "typed declaration" {
    try expectStmt("var x: int = 5", "(var x : int (int 5))");
    try expectStmt(
        "const xs: Array<int> = [1, 2, 3]",
        "(const xs : Array<int> (array (int 1) (int 2) (int 3)))",
    );
}

test "plain assignment" {
    try expectStmt("x = 5", "(= (id x) (int 5))");
    try expectStmt("arr[0] = 1", "(= ([] (id arr) (int 0)) (int 1))");
    try expectStmt("p.x = 1", "(= (. (id p) x) (int 1))");
}

test "compound assignment" {
    try expectStmt("x += 1", "(+= (id x) (int 1))");
    try expectStmt("x -= 1", "(-= (id x) (int 1))");
    try expectStmt("x *= 2", "(*= (id x) (int 2))");
    try expectStmt("x /= 2", "(/= (id x) (int 2))");
    try expectStmt("x %= 2", "(%= (id x) (int 2))");
}

test "return with and without value" {
    try expectStmt("return", "(return)");
    try expectStmt("return 42", "(return (int 42))");
    try expectStmt("return x + 1", "(return (+ (id x) (int 1)))");
}

test "break and continue" {
    try expectStmt("break", "(break)");
    try expectStmt("continue", "(continue)");
}

test "bare expression statement" {
    try expectStmt("foo()", "(expr (call (id foo)))");
    try expectStmt("x.push(1)", "(expr (mcall (id x) push (int 1)))");
}

// ----- block parsing -----

test "block of statements produces expected sequence" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const src: [:0]const u8 =
        \\{
        \\  var x = 1
        \\  var y = 2
        \\  x = x + y
        \\  return x
        \\}
    ;
    var parser = try Parser.init(src, arena);
    const stmts = try parser.parseBlock();
    try testing.expectEqual(@as(usize, 4), stmts.len);
    try testing.expectEqual(@as(usize, 0), parser.diagnostics.items.len);

    const expected = [_][]const u8{
        "(var x (int 1))",
        "(var y (int 2))",
        "(= (id x) (+ (id x) (id y)))",
        "(return (id x))",
    };
    for (stmts, 0..) |*s, i| {
        const rendered = try ast.formatStmtAlloc(s, testing.allocator);
        defer testing.allocator.free(rendered);
        try testing.expectEqualStrings(expected[i], rendered);
    }
}

test "empty block is legal" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var parser = try Parser.init("{}", arena);
    const stmts = try parser.parseBlock();
    try testing.expectEqual(@as(usize, 0), stmts.len);
    try testing.expectEqual(@as(usize, 0), parser.diagnostics.items.len);
}

// ----- error cases -----

test "missing closing paren is a syntax error" {
    try expectParseError("(1 + 2");
}

test "trailing operator with no RHS is a syntax error" {
    try expectParseError("1 +");
}

test "unexpected starting token reports an error" {
    try expectParseError(";");
}

test "block recovers past a broken statement" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Middle statement is malformed. Outer statements should still parse,
    // with exactly one diagnostic emitted.
    const src: [:0]const u8 =
        \\{
        \\  var x = 1
        \\  var = broken
        \\  var z = 3
        \\}
    ;
    var parser = try Parser.init(src, arena);
    const stmts = try parser.parseBlock();
    try testing.expectEqual(@as(usize, 3), stmts.len);
    try testing.expect(parser.diagnostics.items.len >= 1);

    const first = try ast.formatStmtAlloc(&stmts[0], testing.allocator);
    defer testing.allocator.free(first);
    try testing.expectEqualStrings("(var x (int 1))", first);

    try testing.expectEqual(@as(Stmt.Data, .invalid), @as(Stmt.Data, stmts[1].data));

    const third = try ast.formatStmtAlloc(&stmts[2], testing.allocator);
    defer testing.allocator.free(third);
    try testing.expectEqualStrings("(var z (int 3))", third);
}

// ----- integration-ish parse of a representative program -----

test "representative block exercising many features" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const src: [:0]const u8 =
        \\{
        \\  const xs: Array<int> = [1, 2, 3]
        \\  var sum = 0
        \\  sum += xs[0]
        \\  sum += xs[1] * 2
        \\  const label = "total: ${sum}"
        \\  return label
        \\}
    ;
    var parser = try Parser.init(src, arena);
    const stmts = try parser.parseBlock();
    // This test is a smoke test — the structural tests above check the shape
    // of each individual feature. Here we just verify zero diagnostics and
    // the expected number of statements.
    try testing.expectEqual(@as(usize, 0), parser.diagnostics.items.len);
    try testing.expectEqual(@as(usize, 6), stmts.len);
}
