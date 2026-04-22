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
const Binding = ast.Binding;

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
    /// When true, the "ident followed by `{`" shape is NOT parsed as a
    /// record literal — the `{` is left for a caller that expects a block
    /// body. Set when parsing the condition of `if`/`while`/`for`, the
    /// scrutinee of `match`, and the iterable of `for`, so
    /// `if point { ... }` is always "if, cond=point, then-block=`{...}`"
    /// rather than "if, cond=`point {...}`, then-block=???". Parens reset
    /// the flag to `false` (save+restore) so that users who *do* want a
    /// record literal in a condition can write `if (Point { x: 1 }) { ... }`.
    /// This matches Rust's `no_struct_literal` lexer/parser context.
    no_record_literal: bool,

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
            .no_record_literal = false,
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
    ///
    /// `catch` sits at the same level as `or`, which matches Zig's
    /// convention. That makes `a catch b or c` parse as `(a catch b) or c`
    /// (left-to-right, same-precedence).
    fn infixPrec(kind: Token.Kind) Prec {
        return switch (kind) {
            .kw_or, .kw_catch => .or_,
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
        // Record literal postfix: `Name { field: val, ... }`. Fires only
        // on bare-ident primaries, and only when we're not in the no-
        // record-literal context (inside an if/while/for condition or a
        // match scrutinee, where `{` belongs to the body).
        if (!self.no_record_literal and
            left.data == .ident and
            self.peekKind() == .lbrace)
        {
            left = try self.parseRecordLiteral(left);
        }
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
            .lbrace => return self.parseBraceGroup(),
            .minus => return self.parseUnary(.neg),
            .kw_not => return self.parseUnary(.not),
            .kw_try => return self.parseTry(),
            .kw_if => return self.parseIf(),
            .kw_match => return self.parseMatch(true),
            .kw_partial => return self.parsePartialMatch(),
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
            .kw_catch => return self.parseCatch(left),
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
        // A parenthesised subexpression is a fresh context — in particular,
        // the no-record-literal restriction from an enclosing condition
        // doesn't apply here. This is Rust's escape hatch for writing
        // `if (Point { x: 1 }) { ... }` in condition position: the outer
        // context wants to save `{` for the then-block, but once we're
        // inside parens the ambiguity doesn't matter.
        const prev = self.no_record_literal;
        self.no_record_literal = false;
        defer self.no_record_literal = prev;

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

    /// `{...}` is ambiguous — it could be a block, a map literal, or (once
    /// we add them) a record literal. We disambiguate by behaviour, not by
    /// syntactic markers, which keeps the user-facing syntax free of sigils
    /// at the cost of a small speculative parse:
    ///   - Empty `{}` is always a block (value = nil). Empty maps can be
    ///     written as an explicit typed default or constructor later.
    ///   - A statement-starting keyword (`var`/`const`/`return`/`break`/
    ///     `continue`/`while`/`for`) commits to block mode immediately.
    ///   - Otherwise parse one expression, then look at the next token:
    ///     `:` means we were looking at a map key and go into map mode;
    ///     anything else means we were looking at the first statement (or
    ///     the trailing expression) of a block and go into block mode.
    fn parseBraceGroup(self: *Parser) Error!*Expr {
        const start_tok = self.advance(); // consume {

        if (self.peekKind() == .rbrace) {
            const end_tok = self.advance();
            return self.makeBlockExpr(&.{}, null, Span.merge(spanOf(start_tok), spanOf(end_tok)));
        }

        switch (self.peekKind()) {
            .kw_var, .kw_const, .kw_return, .kw_break, .kw_continue,
            .kw_while, .kw_for => return self.finishBlockExpr(start_tok, null),
            else => {},
        }

        const first = try self.parseExpr();

        if (self.peekKind() == .colon) {
            // Map literal: `first` is the first key.
            _ = self.advance();
            const first_val = try self.parseExpr();
            var entries: std.ArrayList(Expr.MapEntry) = .empty;
            try entries.append(self.arena, .{ .key = first, .value = first_val });
            if (self.peekKind() == .comma) {
                _ = self.advance();
                while (self.peekKind() != .rbrace) {
                    const key = try self.parseExpr();
                    _ = try self.expect(.colon, "expected ':' between map key and value");
                    const val = try self.parseExpr();
                    try entries.append(self.arena, .{ .key = key, .value = val });
                    if (self.peekKind() != .comma) break;
                    _ = self.advance();
                }
            }
            const end_tok = try self.expect(.rbrace, "expected '}' to close map literal");
            return self.makeExpr(
                .{ .map_literal = entries.items },
                Span.merge(spanOf(start_tok), spanOf(end_tok)),
            );
        }

        return self.finishBlockExpr(start_tok, first);
    }

    /// Complete parsing a block expression. `first_expr`, if non-null, is
    /// an already-parsed leading expression from the speculative disambig-
    /// uation in `parseBraceGroup` — it may be the first statement, the
    /// trailing expression, or the LHS of an assignment statement, and
    /// `finishBlock` figures out which based on what follows.
    fn finishBlockExpr(self: *Parser, start_tok: Token, first_expr: ?*Expr) Error!*Expr {
        const block = try self.finishBlock(start_tok, first_expr);
        return self.makeExpr(.{ .block = block }, block.span);
    }

    /// Parse a block that we already know is a block (no map ambiguity) —
    /// e.g. the body of an `if`/`while`/`for`. The caller hasn't consumed
    /// the `{`.
    fn parseBlockDirect(self: *Parser) Error!*ast.Block {
        const start_tok = try self.expect(.lbrace, "expected '{' to open block");
        return self.finishBlock(start_tok, null);
    }

    /// Shared block-building logic. Collects statements until `}` and
    /// ergonomically promotes a trailing expression-statement into the
    /// block's `trailing` slot.
    ///
    /// The promotion exists because with Go-style ASI, the user can't see
    /// whether a final newline inserted a `;` or not. In Rust, the explicit
    /// presence or absence of `;` after the last expression chooses between
    /// "discard" and "return". We've already given up that signal, so we
    /// commit to "the last expression is the block's value". `{ foo }` and
    /// `{ foo; }` therefore mean the same thing; users who explicitly want
    /// to discard should write `nil` or something similar at the tail.
    fn finishBlock(self: *Parser, start_tok: Token, first_expr: ?*Expr) Error!*ast.Block {
        var stmts: std.ArrayList(Stmt) = .empty;

        if (first_expr) |fe| {
            const s = try self.completeExprOrAssignStmt(fe);
            try stmts.append(self.arena, s);
            try self.consumeStmtSeparator();
        }

        while (self.peekKind() != .rbrace and self.peekKind() != .eof) {
            const s = try self.parseStmt();
            try stmts.append(self.arena, s);
            try self.consumeStmtSeparator();
        }

        const end_tok = try self.expect(.rbrace, "expected '}' to close block");

        var trailing: ?*Expr = null;
        if (stmts.items.len > 0 and stmts.items[stmts.items.len - 1].data == .expr) {
            trailing = stmts.items[stmts.items.len - 1].data.expr;
            stmts.shrinkRetainingCapacity(stmts.items.len - 1);
        }

        const block = try self.arena.create(ast.Block);
        block.* = .{
            .span = Span.merge(spanOf(start_tok), spanOf(end_tok)),
            .stmts = stmts.items,
            .trailing = trailing,
        };
        return block;
    }

    fn makeBlockExpr(self: *Parser, stmts: []Stmt, trailing: ?*Expr, span: Span) !*Expr {
        const block = try self.arena.create(ast.Block);
        block.* = .{ .span = span, .stmts = stmts, .trailing = trailing };
        return self.makeExpr(.{ .block = block }, span);
    }

    fn parseIf(self: *Parser) Error!*Expr {
        const if_tok = self.advance();
        const cond = try self.parseCondExpr();
        const then_block = try self.parseBlockDirect();

        var else_branch: ?Expr.ElseBranch = null;
        var end_span = then_block.span;
        if (self.peekKind() == .kw_else) {
            _ = self.advance();
            if (self.peekKind() == .kw_if) {
                // `else if` chains as a nested `.if_expr` rather than an
                // array of else-ifs. Keeps the AST a simple linked list
                // and means any consumer (type check, codegen) walks
                // chains without a special case.
                const nested = try self.parseIf();
                else_branch = .{ .if_expr = nested };
                end_span = nested.span;
            } else {
                const eb = try self.parseBlockDirect();
                else_branch = .{ .block = eb };
                end_span = eb.span;
            }
        }

        return self.makeExpr(
            .{ .if_expr = .{
                .cond = cond,
                .then_block = then_block,
                .else_branch = else_branch,
            } },
            Span.merge(spanOf(if_tok), end_span),
        );
    }

    /// Parse an expression in a position where a following `{` belongs
    /// to a body, not to the expression itself (condition of if/while/for,
    /// iterable of for, scrutinee of match). Record literals are disabled
    /// for the duration; parens are a valid escape hatch.
    fn parseCondExpr(self: *Parser) Error!*Expr {
        const prev = self.no_record_literal;
        self.no_record_literal = true;
        defer self.no_record_literal = prev;
        return self.parseExpr();
    }

    fn parseRecordLiteral(self: *Parser, name_expr: *Expr) Error!*Expr {
        // `name_expr` is guaranteed to be an `.ident` — the caller checked.
        const name = name_expr.data.ident;
        const name_span = name_expr.span;

        _ = self.advance(); // consume `{`

        var fields: std.ArrayList(Expr.FieldInit) = .empty;
        if (self.peekKind() != .rbrace) {
            while (true) {
                const field_tok = try self.expect(.ident, "expected field name in record literal");
                const field_name = field_tok.lexeme(self.source);
                var value: ?*Expr = null;
                if (self.peekKind() == .colon) {
                    _ = self.advance();
                    value = try self.parseExpr();
                }
                // Shorthand (no `:`): the field's value is `ident(field_name)`
                // at lookup time. We leave `value == null` in the AST so
                // the type checker can decide how to resolve it.
                try fields.append(self.arena, .{
                    .name = field_name,
                    .name_span = spanOf(field_tok),
                    .value = value,
                });
                if (self.peekKind() != .comma) break;
                _ = self.advance();
                if (self.peekKind() == .rbrace) break; // trailing comma
            }
        }
        const end_tok = try self.expect(.rbrace, "expected '}' to close record literal");
        return self.makeExpr(
            .{ .record_literal = .{
                .name = name,
                .name_span = name_span,
                .fields = fields.items,
            } },
            Span.merge(name_span, spanOf(end_tok)),
        );
    }

    /// `match scrutinee { Pattern => body, ... }`. The scrutinee is parsed
    /// in no-record-literal mode so the subsequent `{` consistently opens
    /// the arm list.
    fn parseMatch(self: *Parser, exhaustive: bool) Error!*Expr {
        const match_tok = self.advance(); // consume `match`
        const scrutinee = try self.parseCondExpr();
        _ = try self.expect(.lbrace, "expected '{' to open match arms");

        var arms: std.ArrayList(ast.MatchArm) = .empty;
        // Allow an implicit semi right after `{` (ASI from `... match x {\n`).
        self.skipSemis();
        while (self.peekKind() != .rbrace and self.peekKind() != .eof) {
            const arm = self.parseMatchArm() catch |err| switch (err) {
                error.InvalidSyntax => {
                    self.syncToMatchArmBoundary();
                    continue;
                },
                else => return err,
            };
            try arms.append(self.arena, arm);
            // Arm separator: comma, implicit semi from ASI after the body,
            // or the end of the match. Explicit semis are also tolerated.
            switch (self.peekKind()) {
                .comma => _ = self.advance(),
                .semi => {},
                .rbrace, .eof => {},
                else => {
                    const bad = self.peek();
                    try self.diagf(bad, "expected ',' or '}}' between match arms, got {s}", .{@tagName(bad.kind)});
                    self.syncToMatchArmBoundary();
                },
            }
            self.skipSemis();
        }
        const end_tok = try self.expect(.rbrace, "expected '}' to close match");

        return self.makeExpr(
            .{ .match_expr = .{
                .scrutinee = scrutinee,
                .arms = arms.items,
                .exhaustive = exhaustive,
            } },
            Span.merge(spanOf(match_tok), spanOf(end_tok)),
        );
    }

    fn parsePartialMatch(self: *Parser) Error!*Expr {
        _ = self.advance(); // consume `partial`
        if (self.peekKind() != .kw_match) {
            try self.diagf(self.peek(), "expected 'match' after 'partial', got {s}", .{@tagName(self.peekKind())});
            return error.InvalidSyntax;
        }
        return self.parseMatch(false);
    }

    fn parseMatchArm(self: *Parser) Error!ast.MatchArm {
        const pat = try self.parsePattern();
        _ = try self.expect(.fat_arrow, "expected '=>' after match pattern");
        const body = try self.parseExpr();
        return .{ .pattern = pat, .body = body };
    }

    /// Recovery within match: advance past the current arm without
    /// descending into its body — stop at the next `,` or the closing
    /// `}` of the match expression itself.
    fn syncToMatchArmBoundary(self: *Parser) void {
        while (true) {
            switch (self.peekKind()) {
                .comma, .rbrace, .eof => return,
                else => _ = self.advance(),
            }
        }
    }

    fn parsePattern(self: *Parser) Error!ast.Pattern {
        const tok = self.peek();
        switch (tok.kind) {
            .int_literal => {
                _ = self.advance();
                return .{ .span = spanOf(tok), .data = .{ .int_pat = tok.lexeme(self.source) } };
            },
            .float_literal => {
                _ = self.advance();
                return .{ .span = spanOf(tok), .data = .{ .float_pat = tok.lexeme(self.source) } };
            },
            .string_literal => {
                _ = self.advance();
                return .{ .span = spanOf(tok), .data = .{ .string_pat = tok.lexeme(self.source) } };
            },
            .kw_true => {
                _ = self.advance();
                return .{ .span = spanOf(tok), .data = .{ .bool_pat = true } };
            },
            .kw_false => {
                _ = self.advance();
                return .{ .span = spanOf(tok), .data = .{ .bool_pat = false } };
            },
            .kw_nil => {
                _ = self.advance();
                return .{ .span = spanOf(tok), .data = .nil_pat };
            },
            .ident => {
                _ = self.advance();
                const name = tok.lexeme(self.source);
                if (std.mem.eql(u8, name, "_")) {
                    return .{ .span = spanOf(tok), .data = .wildcard };
                }
                if (self.peekKind() == .lbrace) {
                    return self.parseConstructorPattern(tok, name);
                }
                return .{
                    .span = spanOf(tok),
                    .data = .{ .bind = .{ .name = name, .span = spanOf(tok) } },
                };
            },
            else => {
                try self.diagf(tok, "expected pattern, got {s}", .{@tagName(tok.kind)});
                return error.InvalidSyntax;
            },
        }
    }

    fn parseConstructorPattern(self: *Parser, name_tok: Token, name: []const u8) Error!ast.Pattern {
        _ = self.advance(); // consume `{`
        var fields: std.ArrayList(ast.Pattern.PatternField) = .empty;
        if (self.peekKind() != .rbrace) {
            while (true) {
                const field_tok = try self.expect(.ident, "expected field name in constructor pattern");
                const field_name = field_tok.lexeme(self.source);
                var binding: ?Binding = null;
                if (self.peekKind() == .colon) {
                    _ = self.advance();
                    const bind_tok = try self.expect(.ident, "expected binding name after ':' in constructor pattern");
                    binding = .{
                        .name = bind_tok.lexeme(self.source),
                        .span = spanOf(bind_tok),
                    };
                }
                try fields.append(self.arena, .{
                    .field_name = field_name,
                    .field_name_span = spanOf(field_tok),
                    .binding = binding,
                });
                if (self.peekKind() != .comma) break;
                _ = self.advance();
                if (self.peekKind() == .rbrace) break; // trailing comma
            }
        }
        const end_tok = try self.expect(.rbrace, "expected '}' to close constructor pattern");
        return .{
            .span = Span.merge(spanOf(name_tok), spanOf(end_tok)),
            .data = .{ .constructor = .{
                .name = name,
                .name_span = spanOf(name_tok),
                .fields = fields.items,
            } },
        };
    }

    fn parseCatch(self: *Parser, left: *Expr) Error!*Expr {
        _ = self.advance(); // consume `catch`

        var binding: ?[]const u8 = null;
        var binding_span: ?Span = null;
        if (self.peekKind() == .pipe) {
            _ = self.advance();
            const name_tok = try self.expect(.ident, "expected identifier after '|' in catch binding");
            binding = name_tok.lexeme(self.source);
            binding_span = spanOf(name_tok);
            _ = try self.expect(.pipe, "expected '|' to close catch binding");
        }

        // Parse the handler at strictly-greater precedence to keep catch
        // left-associative at .or_ level. `a catch b catch c` parses as
        // `(a catch b) catch c`, matching Zig's convention.
        const handler = try self.parseExprPrec(@enumFromInt(@intFromEnum(Prec.or_) + 1));

        return self.makeExpr(
            .{ .catch_expr = .{
                .inner = left,
                .err_binding = binding,
                .err_binding_span = binding_span,
                .handler = handler,
            } },
            Span.merge(left.span, handler.span),
        );
    }

    fn parseWhile(self: *Parser) Error!Stmt {
        const while_tok = self.advance();
        const cond = try self.parseCondExpr();
        const body = try self.parseBlockDirect();
        return .{
            .span = Span.merge(spanOf(while_tok), body.span),
            .data = .{ .while_stmt = .{ .cond = cond, .body = body } },
        };
    }

    fn parseFor(self: *Parser) Error!Stmt {
        const for_tok = self.advance();
        const first_name = try self.expect(.ident, "expected binding name after 'for'");

        var pattern: ast.ForPattern = undefined;
        if (self.peekKind() == .comma) {
            _ = self.advance();
            const second_name = try self.expect(.ident, "expected second binding name after ',' (for key, value loops)");
            pattern = .{ .key_value = .{
                .key = .{ .name = first_name.lexeme(self.source), .span = spanOf(first_name) },
                .value = .{ .name = second_name.lexeme(self.source), .span = spanOf(second_name) },
            } };
        } else {
            pattern = .{ .single = .{
                .name = first_name.lexeme(self.source),
                .span = spanOf(first_name),
            } };
        }

        _ = try self.expect(.kw_in, "expected 'in' after for-loop binding");
        const iterable = try self.parseCondExpr();
        const body = try self.parseBlockDirect();

        return .{
            .span = Span.merge(spanOf(for_tok), body.span),
            .data = .{ .for_stmt = .{
                .pattern = pattern,
                .iterable = iterable,
                .body = body,
            } },
        };
    }

    /// After an already-parsed expression, check for an assignment operator
    /// and either complete an assignment statement or wrap the expression
    /// as an expression statement. Mirrors the logic inside
    /// `parseExprOrAssignStmt` but for a caller that has the LHS in hand.
    fn completeExprOrAssignStmt(self: *Parser, lhs: *Expr) Error!Stmt {
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
                .span = Span.merge(lhs.span, value.span),
                .data = .{ .assign = .{ .op = asgn_op, .target = lhs, .value = value } },
            };
        }
        return .{ .span = lhs.span, .data = .{ .expr = lhs } };
    }

    /// Between two statements in a block we need a `;` (possibly an
    /// implicit one from ASI). Missing separators are reported at the
    /// offending token so the user knows where the statements ran together.
    fn consumeStmtSeparator(self: *Parser) !void {
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
            .kw_while => return self.parseWhile(),
            .kw_for => return self.parseFor(),
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
    // "Parse error" means the parser either bubbled `error.InvalidSyntax`
    // OR recovered locally (inside a match arm, block, etc.) and recorded
    // at least one diagnostic. Both count — what we care about is that
    // the malformed source was flagged rather than silently accepted.
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
    try testing.expect(parser.diagnostics.items.len > 0);
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

test "map literals need at least one key:value to disambiguate from block" {
    // Bare `{}` is now an empty block — see `parseBraceGroup`. Maps are
    // distinguished from blocks by the first expression being followed
    // by a `:`. An "empty map" would look like every other empty block
    // at the syntax level, so we resolve the tie toward block; users
    // who need an empty map can use a typed constructor later.
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

// ----- block expressions -----

test "empty braces are a block, not a map" {
    try expectExpr("{}", "(block)");
}

test "block with only a trailing expression" {
    try expectExpr("{ 42 }", "(block => (int 42))");
    try expectExpr("{ x + 1 }", "(block => (+ (id x) (int 1)))");
}

test "block with statements and no trailing" {
    try expectExpr(
        "{ var x = 1; var y = 2 }",
        // `var y = 2` is the last statement; not an expression, so no
        // promotion to trailing.
        "(block (var x (int 1)) (var y (int 2)))",
    );
}

test "last expression-statement is promoted to trailing" {
    try expectExpr(
        "{ var x = 1; x }",
        "(block (var x (int 1)) => (id x))",
    );
}

test "block promotion works across ASI newlines" {
    // With Go-style ASI, the newline before `}` inserts an implicit
    // semicolon after `x`, but promotion lifts the final
    // expression-statement back out to be the trailing expression. This
    // is the ergonomic outcome — users don't have to see the hidden `;`.
    try expectExpr(
        "{\n  var y = 2\n  y\n}",
        "(block (var y (int 2)) => (id y))",
    );
}

test "block as rvalue in var declaration" {
    try expectStmt(
        "var x = { 1 + 2 }",
        "(var x (block => (+ (int 1) (int 2))))",
    );
}

// ----- if expressions -----

test "if without else" {
    try expectExpr(
        "if x { 1 }",
        "(if (id x) (block => (int 1)))",
    );
}

test "if with else" {
    try expectExpr(
        "if x { 1 } else { 2 }",
        "(if (id x) (block => (int 1)) else (block => (int 2)))",
    );
}

test "if else-if chain flattens into nested if_expr" {
    try expectExpr(
        "if a { 1 } else if b { 2 } else { 3 }",
        "(if (id a) (block => (int 1)) else (if (id b) (block => (int 2)) else (block => (int 3))))",
    );
}

test "if as rvalue" {
    try expectStmt(
        "var y = if x > 0 { x } else { -x }",
        "(var y (if (> (id x) (int 0)) (block => (id x)) else (block => (- (id x)))))",
    );
}

test "complex condition in if" {
    try expectExpr(
        "if a and b or c { 1 }",
        "(if (or (and (id a) (id b)) (id c)) (block => (int 1)))",
    );
}

// ----- catch -----

test "catch without binding is a default-value expression" {
    try expectExpr(
        "foo() catch 0",
        "(catch (call (id foo)) (int 0))",
    );
}

test "catch with |e| binding" {
    try expectExpr(
        "foo() catch |e| e",
        "(catch (call (id foo)) |e| (id e))",
    );
}

test "catch handler can be a block" {
    try expectExpr(
        "risky() catch |err| { err }",
        "(catch (call (id risky)) |err| (block => (id err)))",
    );
}

test "chained catch is left-associative" {
    try expectExpr(
        "a() catch b() catch 0",
        "(catch (catch (call (id a)) (call (id b))) (int 0))",
    );
}

test "catch has lower precedence than arithmetic" {
    // `foo() catch 0 + 1` should parse as `(foo() catch (0 + 1))` because
    // `+` binds tighter than `catch`.
    try expectExpr(
        "foo() catch 0 + 1",
        "(catch (call (id foo)) (+ (int 0) (int 1)))",
    );
}

// ----- while -----

test "simple while" {
    try expectStmt(
        "while x > 0 { x -= 1 }",
        "(while (> (id x) (int 0)) (block (-= (id x) (int 1))))",
    );
}

test "while with empty body" {
    try expectStmt("while cond {}", "(while (id cond) (block))");
}

// ----- for -----

test "for over range" {
    // Trailing promotion lifts the final expression-statement in every
    // block — including for-loop bodies. The tail value is semantically
    // ignored by a for body, but the parser's shape is uniform.
    try expectStmt(
        "for i in 0..10 { foo(i) }",
        "(for i (.. (int 0) (int 10)) (block => (call (id foo) (id i))))",
    );
}

test "for over array" {
    // Compound assignment is not an expression-statement, so it is NOT
    // promoted — the body stays a plain statement list with no trailing.
    try expectStmt(
        "for x in xs { total += x }",
        "(for x (id xs) (block (+= (id total) (id x))))",
    );
}

test "for over map with key, value" {
    try expectStmt(
        "for k, v in m { foo(k, v) }",
        "(for (kv k v) (id m) (block => (call (id foo) (id k) (id v))))",
    );
}

test "nested for loops" {
    try expectStmt(
        "for i in 0..n { for j in 0..n { use(i, j) } }",
        "(for i (.. (int 0) (id n)) (block (for j (.. (int 0) (id n)) (block => (call (id use) (id i) (id j))))))",
    );
}

// ----- interaction tests -----

test "if inside a block, with trailing promotion" {
    // The block's last statement is an if-expression with no else; it
    // becomes the trailing. Note that if-without-else has value nil if
    // the condition is false; the parser doesn't care about that, the
    // type checker will.
    try expectExpr(
        "{ var x = 1; if x > 0 { x } else { 0 } }",
        "(block (var x (int 1)) => (if (> (id x) (int 0)) (block => (id x)) else (block => (int 0))))",
    );
}

test "for body with control flow" {
    try expectStmt(
        "for i in 0..n { if i == 5 { break } else { continue } }",
        "(for i (.. (int 0) (id n)) (block => (if (== (id i) (int 5)) (block (break)) else (block (continue)))))",
    );
}

// ----- record literals -----

fn expectPattern(source: [:0]const u8, expected: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var parser = try Parser.init(source, arena);
    const p = try parser.parsePattern();
    try testing.expectEqual(@as(usize, 0), parser.diagnostics.items.len);
    try testing.expectEqual(Token.Kind.eof, parser.peekKind());
    const rendered = try ast.formatPatternAlloc(&p, testing.allocator);
    defer testing.allocator.free(rendered);
    try testing.expectEqualStrings(expected, rendered);
}

test "empty record literal" {
    try expectExpr("Point {}", "(rec Point)");
}

test "record literal with one field" {
    try expectExpr("Point { x: 1 }", "(rec Point (f x (int 1)))");
}

test "record literal with multiple fields" {
    try expectExpr(
        "Point { x: 1, y: 2 }",
        "(rec Point (f x (int 1)) (f y (int 2)))",
    );
}

test "record literal with shorthand field" {
    // `Point { x }` is sugar for `Point { x: x }` — the local variable `x`
    // supplies the value. AST keeps shorthand distinct (value == null) so
    // the type checker can resolve scope explicitly.
    try expectExpr("Point { x }", "(rec Point (f x))");
    try expectExpr("Point { x, y }", "(rec Point (f x) (f y))");
}

test "record literal mixing explicit and shorthand fields" {
    try expectExpr(
        "Point { x: 0, y }",
        "(rec Point (f x (int 0)) (f y))",
    );
}

test "record literal with trailing comma" {
    try expectExpr(
        "Point { x: 1, y: 2, }",
        "(rec Point (f x (int 1)) (f y (int 2)))",
    );
}

test "nested record literal" {
    try expectExpr(
        "Rect { origin: Point { x: 0, y: 0 }, size: 10 }",
        "(rec Rect (f origin (rec Point (f x (int 0)) (f y (int 0)))) (f size (int 10)))",
    );
}

test "record literal as rvalue" {
    try expectStmt(
        "var p = Point { x: 1, y: 2 }",
        "(var p (rec Point (f x (int 1)) (f y (int 2))))",
    );
}

// ----- record-literal suppression in condition position -----

test "if condition is a bare ident, not a record literal" {
    // Without suppression, this would try to parse `shape { ... }` as a
    // record literal and then hit EOF looking for a then-block.
    try expectExpr(
        "if shape { 1 }",
        "(if (id shape) (block => (int 1)))",
    );
}

test "while condition is a bare ident" {
    try expectStmt(
        "while running { step() }",
        "(while (id running) (block => (call (id step))))",
    );
}

test "for iterable is a bare ident" {
    try expectStmt(
        "for x in items { foo(x) }",
        "(for x (id items) (block => (call (id foo) (id x))))",
    );
}

test "parens in condition re-enable record literals" {
    // The explicit parens are Rust's escape hatch: inside `(...)` the
    // no-record-literal restriction is lifted, so `Point { x: 1, y: 2 }`
    // parses as a record literal and the outer `{` is the if body.
    try expectExpr(
        "if (Point { x: 1, y: 2 }) { 1 }",
        "(if (rec Point (f x (int 1)) (f y (int 2))) (block => (int 1)))",
    );
}

// ----- patterns -----

test "literal patterns" {
    try expectPattern("42", "(intpat 42)");
    try expectPattern("3.14", "(floatpat 3.14)");
    try expectPattern("\"hi\"", "(strpat \"hi\")");
    try expectPattern("true", "truepat");
    try expectPattern("false", "falsepat");
    try expectPattern("nil", "nilpat");
}

test "binding and wildcard patterns" {
    try expectPattern("x", "(bind x)");
    try expectPattern("_", "_");
}

test "constructor pattern with shorthand fields" {
    try expectPattern("Circle { radius }", "(ctor Circle (radius))");
    try expectPattern("Rect { width, height }", "(ctor Rect (width) (height))");
}

test "constructor pattern with renamed bindings" {
    try expectPattern(
        "Circle { radius: r }",
        "(ctor Circle (radius r))",
    );
    try expectPattern(
        "Rect { width: w, height: h }",
        "(ctor Rect (width w) (height h))",
    );
}

test "constructor pattern with mixed shorthand and rename" {
    try expectPattern(
        "Rect { width, height: h }",
        "(ctor Rect (width) (height h))",
    );
}

test "empty-fields constructor pattern" {
    // Explicit `{}` after the name is required for zero-field variants,
    // so they can be distinguished from bare-ident bindings at parse time.
    try expectPattern("NoArgs {}", "(ctor NoArgs)");
}

// ----- match -----

test "match with a single arm" {
    try expectExpr(
        "match x { 1 => \"one\" }",
        "(match (id x) (arm (intpat 1) (str \"one\")))",
    );
}

test "match with multiple arms, trailing comma" {
    try expectExpr(
        "match x { 1 => \"one\", 2 => \"two\", _ => \"other\", }",
        "(match (id x) (arm (intpat 1) (str \"one\")) (arm (intpat 2) (str \"two\")) (arm _ (str \"other\")))",
    );
}

test "match arms across newlines (ASI does not confuse separator handling)" {
    try expectExpr(
        "match x {\n  1 => \"one\",\n  2 => \"two\"\n}",
        "(match (id x) (arm (intpat 1) (str \"one\")) (arm (intpat 2) (str \"two\")))",
    );
}

test "match with constructor patterns" {
    try expectExpr(
        "match shape { Circle { radius } => radius, Square { side: s } => s }",
        "(match (id shape) (arm (ctor Circle (radius)) (id radius)) (arm (ctor Square (side s)) (id s)))",
    );
}

test "match with block-bodied arms" {
    try expectExpr(
        "match x { 1 => { foo(); 42 }, _ => 0 }",
        "(match (id x) (arm (intpat 1) (block (expr (call (id foo))) => (int 42))) (arm _ (int 0)))",
    );
}

test "partial match sets exhaustive to false" {
    try expectExpr(
        "partial match x { 1 => \"one\" }",
        "(partial-match (id x) (arm (intpat 1) (str \"one\")))",
    );
}

test "match as rvalue" {
    try expectStmt(
        "var label = match n { 0 => \"zero\", _ => \"nonzero\" }",
        "(var label (match (id n) (arm (intpat 0) (str \"zero\")) (arm _ (str \"nonzero\"))))",
    );
}

test "match scrutinee of bare ident does not swallow arm braces" {
    // `match shape { ... }` — scrutinee is a bare `shape`, not
    // `shape { ... }` as a record literal.
    try expectExpr(
        "match shape { _ => 1 }",
        "(match (id shape) (arm _ (int 1)))",
    );
}

test "match with record literal in arm body (no-record flag is scoped to scrutinee)" {
    // The no-record-literal restriction applies only to the scrutinee;
    // arm bodies are normal expressions where record literals work.
    try expectExpr(
        "match x { _ => Point { x: 1, y: 2 } }",
        "(match (id x) (arm _ (rec Point (f x (int 1)) (f y (int 2)))))",
    );
}

// ----- error cases specific to match/patterns -----

test "match arm missing `=>` is a parse error" {
    try expectParseError("match x { 1 \"one\" }");
}

test "constructor pattern missing `:` binding is just shorthand" {
    // Not an error — `Circle { radius }` is valid shorthand.
    try expectPattern("Circle { radius }", "(ctor Circle (radius))");
}

test "`partial` without `match` is a parse error" {
    try expectParseError("partial x");
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
