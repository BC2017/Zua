//! Zua lexer.
//!
//! Converts a UTF-8 source buffer into a stream of `Token`s. Tokens only store
//! byte offsets into the source — the source must outlive the tokens. Escape
//! sequences and numeric values are *not* decoded here; later stages take the
//! raw lexeme slice and interpret it. This keeps the lexer allocation-free
//! and its tests purely structural.
//!
//! ## Token grammar (informal)
//!
//! Keywords: and, as, break, catch, const, continue, else, enum, export,
//!   false, fn, for, from, go, if, impl, import, in, interface, match,
//!   nil, not, or, partial, record, return, self, test, true, try, type,
//!   while
//!
//! Literals:
//!   int    := decimal | '0x' hex | '0b' binary | '0o' octal
//!     where each digit run may contain '_' as a visual separator
//!   float  := digits '.' digits ( [eE] [+-]? digits )?
//!           | digits [eE] [+-]? digits
//!   string := '"' ( char | escape | interp )* '"'
//!     escape := '\' ( n | t | r | 0 | \\ | " | $ | x HH | u { H+ } )
//!     interp := '${' <expression tokens> '}'
//!   bool   := true | false
//!   nil    := nil
//!
//! Punctuation / operators:
//!   ( ) { } [ ] , ; : . .. ..= -> => | ? !
//!   = + - * / % += -= *= /= %=
//!   == != < > <= >=
//!
//! Logical operators (`and`, `or`, `not`) are keywords rather than symbols,
//! which frees `!` for error-type prefixes (`!int`) and keeps the reading
//! voice closer to Zua's Lua heritage while the rest of the surface stays
//! C-family.
//!
//! Comments:
//!   // line comment, to end of line
//!   /* block comment, nested */
//!
//! ## Implicit semicolons (Go-style ASI)
//!
//! After skipping whitespace and comments, if we are about to cross a newline
//! and the previously emitted token was "statement-ending" (an ident, a
//! literal, `return`/`break`/`continue`, a closing `)`/`]`/`}`, or `?`), the
//! tokenizer inserts a synthetic `semi` token. Other newlines are discarded
//! as whitespace. This matches Go's rule and avoids the ambiguity traps of
//! JavaScript's ASI.
//!
//! ## String interpolation
//!
//! An interpolated string breaks into a sequence:
//!
//!   "hello ${name}!"
//!     -> string_start("hello ${")  ident(name)  string_end("}!")
//!
//!   "a${f()}b${g()}c"
//!     -> string_start("a${")  ident(f) lparen rparen
//!        string_part("}b${")  ident(g) lparen rparen
//!        string_end("}c")
//!
//! Nested interpolation is supported via an internal context stack (up to 16
//! levels — deeper nesting is almost certainly a mistake rather than a real
//! need). A `{` inside a substitution increments the current context's brace
//! depth; a `}` decrements it, and only closes the substitution when the
//! depth has returned to zero.
//!
//! ## Errors
//!
//! The tokenizer is resilient: on a malformed token it emits a token of kind
//! `invalid` with a span covering the offending bytes and resumes scanning.
//! This lets a driver report many lexical errors in one pass instead of
//! bailing out at the first problem.

const std = @import("std");
const testing = std.testing;

pub const Token = struct {
    kind: Kind,
    loc: Loc,

    pub const Loc = struct {
        start: usize,
        end: usize,
    };

    pub const Kind = enum {
        // Literals
        int_literal,
        float_literal,
        string_literal,
        string_start,
        string_part,
        string_end,
        ident,

        // Keywords
        kw_and,
        kw_as,
        kw_break,
        kw_catch,
        kw_const,
        kw_continue,
        kw_else,
        kw_enum,
        kw_export,
        kw_false,
        kw_fn,
        kw_for,
        kw_from,
        kw_go,
        kw_if,
        kw_impl,
        kw_import,
        kw_in,
        kw_interface,
        kw_match,
        kw_nil,
        kw_not,
        kw_or,
        kw_partial,
        kw_record,
        kw_return,
        kw_self,
        kw_test,
        kw_true,
        kw_try,
        kw_type,
        kw_while,

        // Punctuation
        lparen,
        rparen,
        lbrace,
        rbrace,
        lbracket,
        rbracket,
        comma,
        semi,
        colon,
        dot,
        dotdot,
        dotdot_eq,
        arrow,
        fat_arrow,
        pipe,
        question,
        bang,

        // Operators
        assign,
        plus,
        minus,
        star,
        slash,
        percent,
        plus_eq,
        minus_eq,
        star_eq,
        slash_eq,
        percent_eq,
        eq,
        neq,
        lt,
        gt,
        le,
        ge,

        // End / error
        eof,
        invalid,

        /// True for tokens that should cause a following newline to be
        /// treated as an implicit statement terminator. Kept next to the
        /// enum so any new "value-producing" or "block-closing" kind is
        /// considered here at the time it's added, not in a far-away
        /// switch.
        pub fn endsStatement(k: Kind) bool {
            return switch (k) {
                .int_literal,
                .float_literal,
                .string_literal,
                .string_end,
                .ident,
                .kw_true,
                .kw_false,
                .kw_nil,
                .kw_self,
                .kw_return,
                .kw_break,
                .kw_continue,
                .rparen,
                .rbrace,
                .rbracket,
                .question,
                => true,
                else => false,
            };
        }
    };

    /// Raw source bytes this token spans, including any surrounding
    /// delimiters (quotes, `${`, etc. for string pieces).
    pub fn lexeme(self: Token, source: []const u8) []const u8 {
        return source[self.loc.start..self.loc.end];
    }
};

/// Resolve a bare word to its keyword kind, or `.ident` if it is not reserved.
fn keywordKind(word: []const u8) Token.Kind {
    const entries = .{
        .{ "and", Token.Kind.kw_and },
        .{ "as", Token.Kind.kw_as },
        .{ "break", Token.Kind.kw_break },
        .{ "catch", Token.Kind.kw_catch },
        .{ "const", Token.Kind.kw_const },
        .{ "continue", Token.Kind.kw_continue },
        .{ "else", Token.Kind.kw_else },
        .{ "enum", Token.Kind.kw_enum },
        .{ "export", Token.Kind.kw_export },
        .{ "false", Token.Kind.kw_false },
        .{ "fn", Token.Kind.kw_fn },
        .{ "for", Token.Kind.kw_for },
        .{ "from", Token.Kind.kw_from },
        .{ "go", Token.Kind.kw_go },
        .{ "if", Token.Kind.kw_if },
        .{ "impl", Token.Kind.kw_impl },
        .{ "import", Token.Kind.kw_import },
        .{ "in", Token.Kind.kw_in },
        .{ "interface", Token.Kind.kw_interface },
        .{ "match", Token.Kind.kw_match },
        .{ "nil", Token.Kind.kw_nil },
        .{ "not", Token.Kind.kw_not },
        .{ "or", Token.Kind.kw_or },
        .{ "partial", Token.Kind.kw_partial },
        .{ "record", Token.Kind.kw_record },
        .{ "return", Token.Kind.kw_return },
        .{ "self", Token.Kind.kw_self },
        .{ "test", Token.Kind.kw_test },
        .{ "true", Token.Kind.kw_true },
        .{ "try", Token.Kind.kw_try },
        .{ "type", Token.Kind.kw_type },
        .{ "while", Token.Kind.kw_while },
    };
    inline for (entries) |entry| {
        if (std.mem.eql(u8, word, entry[0])) return entry[1];
    }
    return .ident;
}

pub const Tokenizer = struct {
    source: [:0]const u8,
    index: usize,
    /// Kind of the most recently emitted *real* token (not whitespace/comments
    /// and not synthetic semis). Needed to decide whether a crossed newline
    /// triggers an implicit semicolon.
    last_kind: Token.Kind,
    /// Whether the next `next()` call should emit a synthetic `semi` because
    /// we consumed trivia that contained a newline at a statement boundary.
    /// Kept as a separate flag, not emitted eagerly, so that a run of blank
    /// lines produces exactly one implicit semi rather than one per newline.
    pending_semi: bool,
    /// Buffered invalid-token location, used when trivia-scanning detects a
    /// problem (e.g. an unterminated `/* ... */` block comment) that we want
    /// to surface but cannot report in the middle of `skipTrivia` since that
    /// helper does not produce tokens itself.
    pending_invalid: ?Token.Loc,
    /// Context stack for string interpolation. Non-empty means we are
    /// currently lexing expression tokens inside a `${...}` — so `{` and `}`
    /// need to be tracked, and a `}` at depth zero resumes string-scanning.
    interp_stack: [max_interp_depth]InterpCtx,
    interp_len: u8,

    const max_interp_depth = 16;

    const InterpCtx = struct {
        /// Unbalanced `{` opens inside the current substitution. Starts at 0;
        /// the `}` that brings this back to -1 is the one that closes the
        /// substitution.
        brace_depth: i32,
    };

    pub fn init(source: [:0]const u8) Tokenizer {
        return .{
            .source = source,
            .index = 0,
            .last_kind = .eof,
            .pending_semi = false,
            .pending_invalid = null,
            .interp_stack = undefined,
            .interp_len = 0,
        };
    }

    /// Main entry point: produce the next token. Never returns an error —
    /// lexical problems are surfaced as `invalid` tokens so a driver can
    /// accumulate diagnostics.
    pub fn next(self: *Tokenizer) Token {
        // If we previously buffered a semi for ASI, emit it now before any
        // further trivia-skipping, so it sits between the line's last token
        // and the next line's first token.
        if (self.pending_semi) {
            self.pending_semi = false;
            const t = Token{ .kind = .semi, .loc = .{ .start = self.index, .end = self.index } };
            self.last_kind = .semi;
            return t;
        }

        const saw_newline = self.skipTrivia();
        // Trivia-scanning may have detected something bad (unterminated block
        // comment). Surface it before looking at the next real character,
        // since the "next character" may be an artificial EOF.
        if (self.pending_invalid) |loc| {
            self.pending_invalid = null;
            self.last_kind = .invalid;
            return Token{ .kind = .invalid, .loc = loc };
        }
        if (saw_newline and self.last_kind.endsStatement()) {
            const t = Token{ .kind = .semi, .loc = .{ .start = self.index, .end = self.index } };
            self.last_kind = .semi;
            return t;
        }

        const start = self.index;
        if (self.atEnd()) {
            return self.produce(.eof, start, start);
        }

        const c = self.source[self.index];

        // Inside an interpolation substitution: a `}` at depth 0 reopens
        // string-literal scanning rather than acting as a brace token.
        if (c == '}' and self.interp_len > 0 and
            self.interp_stack[self.interp_len - 1].brace_depth == 0)
        {
            self.index += 1; // consume the `}` itself; it's part of the next string segment's lexeme
            return self.scanStringSegment(start, .resumed);
        }

        switch (c) {
            'a'...'z', 'A'...'Z', '_' => return self.scanIdentOrKeyword(start),
            '0'...'9' => return self.scanNumber(start),
            '"' => {
                self.index += 1; // consume opening "
                return self.scanStringSegment(start, .opening);
            },
            '(' => return self.emitSingle(start, .lparen),
            ')' => return self.emitSingle(start, .rparen),
            '{' => {
                if (self.interp_len > 0) self.interp_stack[self.interp_len - 1].brace_depth += 1;
                return self.emitSingle(start, .lbrace);
            },
            '}' => {
                if (self.interp_len > 0) self.interp_stack[self.interp_len - 1].brace_depth -= 1;
                return self.emitSingle(start, .rbrace);
            },
            '[' => return self.emitSingle(start, .lbracket),
            ']' => return self.emitSingle(start, .rbracket),
            ',' => return self.emitSingle(start, .comma),
            ';' => return self.emitSingle(start, .semi),
            ':' => return self.emitSingle(start, .colon),
            '?' => return self.emitSingle(start, .question),
            '|' => return self.emitSingle(start, .pipe),
            '.' => return self.scanDot(start),
            '=' => return self.scanEq(start),
            '!' => return self.scanBang(start),
            '<' => return self.scanAngle(start, .lt, .le),
            '>' => return self.scanAngle(start, .gt, .ge),
            '+' => return self.scanOpMaybeEq(start, .plus, .plus_eq),
            '-' => return self.scanMinus(start),
            '*' => return self.scanOpMaybeEq(start, .star, .star_eq),
            '/' => return self.scanOpMaybeEq(start, .slash, .slash_eq),
            '%' => return self.scanOpMaybeEq(start, .percent, .percent_eq),
            else => {
                self.index += 1;
                return self.produce(.invalid, start, self.index);
            },
        }
    }

    // ----- trivia -----

    /// Skip whitespace and comments. Returns true if any newline was crossed.
    fn skipTrivia(self: *Tokenizer) bool {
        var saw_newline = false;
        while (!self.atEnd()) {
            const c = self.source[self.index];
            switch (c) {
                ' ', '\t', '\r' => self.index += 1,
                '\n' => {
                    saw_newline = true;
                    self.index += 1;
                },
                '/' => {
                    const next_c = self.peek(1);
                    if (next_c == '/') {
                        // Line comment: consume up to but not including the \n
                        self.index += 2;
                        while (!self.atEnd() and self.source[self.index] != '\n') {
                            self.index += 1;
                        }
                    } else if (next_c == '*') {
                        // Nested block comment. Track depth so inner /* ... */
                        // don't prematurely close the outer comment.
                        const comment_start = self.index;
                        self.index += 2;
                        var depth: u32 = 1;
                        while (!self.atEnd() and depth > 0) {
                            const bc = self.source[self.index];
                            if (bc == '/' and self.peek(1) == '*') {
                                self.index += 2;
                                depth += 1;
                            } else if (bc == '*' and self.peek(1) == '/') {
                                self.index += 2;
                                depth -= 1;
                            } else {
                                if (bc == '\n') saw_newline = true;
                                self.index += 1;
                            }
                        }
                        if (depth > 0) {
                            // Ran out of source before the comment closed.
                            // Buffer an invalid covering the whole comment
                            // so the driver gets one clear diagnostic; the
                            // tokenizer itself is now at EOF.
                            self.pending_invalid = .{ .start = comment_start, .end = self.source.len };
                            return saw_newline;
                        }
                    } else {
                        return saw_newline;
                    }
                },
                else => return saw_newline,
            }
        }
        return saw_newline;
    }

    // ----- single-byte / simple multi-byte tokens -----

    fn emitSingle(self: *Tokenizer, start: usize, kind: Token.Kind) Token {
        self.index += 1;
        return self.produce(kind, start, self.index);
    }

    fn scanDot(self: *Tokenizer, start: usize) Token {
        self.index += 1;
        if (!self.atEnd() and self.source[self.index] == '.') {
            self.index += 1;
            if (!self.atEnd() and self.source[self.index] == '=') {
                self.index += 1;
                return self.produce(.dotdot_eq, start, self.index);
            }
            return self.produce(.dotdot, start, self.index);
        }
        return self.produce(.dot, start, self.index);
    }

    fn scanEq(self: *Tokenizer, start: usize) Token {
        self.index += 1;
        if (!self.atEnd() and self.source[self.index] == '=') {
            self.index += 1;
            return self.produce(.eq, start, self.index);
        }
        if (!self.atEnd() and self.source[self.index] == '>') {
            self.index += 1;
            return self.produce(.fat_arrow, start, self.index);
        }
        return self.produce(.assign, start, self.index);
    }

    fn scanBang(self: *Tokenizer, start: usize) Token {
        self.index += 1;
        if (!self.atEnd() and self.source[self.index] == '=') {
            self.index += 1;
            return self.produce(.neq, start, self.index);
        }
        return self.produce(.bang, start, self.index);
    }

    fn scanAngle(self: *Tokenizer, start: usize, base: Token.Kind, with_eq: Token.Kind) Token {
        self.index += 1;
        if (!self.atEnd() and self.source[self.index] == '=') {
            self.index += 1;
            return self.produce(with_eq, start, self.index);
        }
        return self.produce(base, start, self.index);
    }

    fn scanOpMaybeEq(self: *Tokenizer, start: usize, base: Token.Kind, with_eq: Token.Kind) Token {
        self.index += 1;
        if (!self.atEnd() and self.source[self.index] == '=') {
            self.index += 1;
            return self.produce(with_eq, start, self.index);
        }
        return self.produce(base, start, self.index);
    }

    fn scanMinus(self: *Tokenizer, start: usize) Token {
        self.index += 1;
        if (!self.atEnd()) {
            const c = self.source[self.index];
            if (c == '=') {
                self.index += 1;
                return self.produce(.minus_eq, start, self.index);
            }
            if (c == '>') {
                self.index += 1;
                return self.produce(.arrow, start, self.index);
            }
        }
        return self.produce(.minus, start, self.index);
    }

    // ----- identifiers & keywords -----

    fn scanIdentOrKeyword(self: *Tokenizer, start: usize) Token {
        self.index += 1; // first char already validated by caller
        while (!self.atEnd()) {
            const c = self.source[self.index];
            switch (c) {
                'a'...'z', 'A'...'Z', '0'...'9', '_' => self.index += 1,
                else => break,
            }
        }
        const word = self.source[start..self.index];
        const kind = keywordKind(word);
        return self.produce(kind, start, self.index);
    }

    // ----- numbers -----

    fn scanNumber(self: *Tokenizer, start: usize) Token {
        // Detect base prefixes (0x / 0b / 0o). We still allow a leading 0
        // followed by more decimal digits (e.g. `007`), treated as decimal —
        // C-style octal-by-leading-zero is a classic footgun that we refuse.
        if (self.source[self.index] == '0' and !self.atEnd()) {
            const p = self.peek(1);
            switch (p) {
                'x', 'X' => {
                    self.index += 2;
                    return self.scanDigits(start, isHexDigit, .int_literal);
                },
                'b', 'B' => {
                    self.index += 2;
                    return self.scanDigits(start, isBinDigit, .int_literal);
                },
                'o', 'O' => {
                    self.index += 2;
                    return self.scanDigits(start, isOctDigit, .int_literal);
                },
                else => {},
            }
        }

        // Decimal integer part.
        self.consumeDigits(isDecDigit);

        var is_float = false;

        // Fraction: only if followed by a digit. This leaves `1.foo` to be
        // tokenized as `1 . foo` (method call on an int) and `1..5` to be
        // tokenized as `1 .. 5` rather than `1. .5`.
        if (!self.atEnd() and self.source[self.index] == '.' and isDecDigit(self.peek(1))) {
            is_float = true;
            self.index += 1; // consume '.'
            self.consumeDigits(isDecDigit);
        }

        // Exponent: e/E [+/-]? digits+
        if (!self.atEnd()) {
            const c = self.source[self.index];
            if (c == 'e' or c == 'E') {
                const after = self.peek(1);
                const after2 = self.peek(2);
                const has_sign = after == '+' or after == '-';
                const exp_digit = if (has_sign) after2 else after;
                if (isDecDigit(exp_digit)) {
                    is_float = true;
                    self.index += 1;
                    if (has_sign) self.index += 1;
                    self.consumeDigits(isDecDigit);
                }
            }
        }

        return self.produce(if (is_float) .float_literal else .int_literal, start, self.index);
    }

    fn scanDigits(self: *Tokenizer, start: usize, comptime is_digit: fn (u8) bool, kind: Token.Kind) Token {
        // For prefixed numbers (hex/bin/oct). We require at least one digit.
        const digits_start = self.index;
        self.consumeDigits(is_digit);
        if (self.index == digits_start) {
            // e.g. "0x" with nothing after — invalid
            return self.produce(.invalid, start, self.index);
        }
        return self.produce(kind, start, self.index);
    }

    fn consumeDigits(self: *Tokenizer, comptime is_digit: fn (u8) bool) void {
        while (!self.atEnd()) {
            const c = self.source[self.index];
            if (is_digit(c) or c == '_') {
                self.index += 1;
            } else break;
        }
    }

    fn isDecDigit(c: u8) bool {
        return c >= '0' and c <= '9';
    }
    fn isHexDigit(c: u8) bool {
        return (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
    }
    fn isBinDigit(c: u8) bool {
        return c == '0' or c == '1';
    }
    fn isOctDigit(c: u8) bool {
        return c >= '0' and c <= '7';
    }

    // ----- strings -----

    const StringMode = enum { opening, resumed };

    /// Scan a string segment. On entry, `self.index` points to the first
    /// *content* byte (just after the opening `"` for `.opening`, or just
    /// after the closing `}` of the previous substitution for `.resumed`).
    /// `start` is the byte offset of the token (i.e. the `"` or `}` itself).
    fn scanStringSegment(self: *Tokenizer, start: usize, mode: StringMode) Token {
        while (!self.atEnd()) {
            const c = self.source[self.index];
            switch (c) {
                '"' => {
                    self.index += 1;
                    const kind: Token.Kind = switch (mode) {
                        .opening => .string_literal,
                        .resumed => .string_end,
                    };
                    if (mode == .resumed) self.popInterp();
                    return self.produce(kind, start, self.index);
                },
                '\\' => {
                    // Consume the escape; its validity is left to a later
                    // pass (we only need to avoid mistaking \" or \$ for a
                    // string terminator or interpolation opener).
                    self.index += 1;
                    if (!self.atEnd()) self.index += 1;
                },
                '$' => {
                    if (self.peek(1) == '{') {
                        self.index += 2; // consume '${'
                        const kind: Token.Kind = switch (mode) {
                            .opening => .string_start,
                            .resumed => .string_part,
                        };
                        if (mode == .opening) {
                            if (!self.pushInterp()) {
                                return self.produce(.invalid, start, self.index);
                            }
                        }
                        // mode == .resumed already has a context on the stack
                        return self.produce(kind, start, self.index);
                    }
                    self.index += 1;
                },
                '\n' => {
                    // Bare newlines inside strings are rejected for v1 —
                    // require \n explicitly. Emit invalid and resync.
                    self.index += 1;
                    return self.produce(.invalid, start, self.index);
                },
                else => self.index += 1,
            }
        }
        // Ran out of source before closing delimiter.
        return self.produce(.invalid, start, self.index);
    }

    fn pushInterp(self: *Tokenizer) bool {
        if (self.interp_len >= max_interp_depth) return false;
        self.interp_stack[self.interp_len] = .{ .brace_depth = 0 };
        self.interp_len += 1;
        return true;
    }

    fn popInterp(self: *Tokenizer) void {
        if (self.interp_len > 0) self.interp_len -= 1;
    }

    // ----- plumbing -----

    fn atEnd(self: *const Tokenizer) bool {
        return self.index >= self.source.len;
    }

    fn peek(self: *const Tokenizer, offset: usize) u8 {
        const i = self.index + offset;
        if (i >= self.source.len) return 0;
        return self.source[i];
    }

    fn produce(self: *Tokenizer, kind: Token.Kind, start: usize, end: usize) Token {
        self.last_kind = kind;
        return .{ .kind = kind, .loc = .{ .start = start, .end = end } };
    }
};

// ===== tests =====

/// Collect every non-EOF token from a source. Helper used throughout the
/// test suite so assertions stay focused on token content rather than on
/// driving the tokenizer.
fn collect(source: [:0]const u8, out: []Token.Kind) usize {
    var tk = Tokenizer.init(source);
    var n: usize = 0;
    while (true) {
        const t = tk.next();
        if (t.kind == .eof) break;
        if (n >= out.len) return n;
        out[n] = t.kind;
        n += 1;
    }
    return n;
}

fn expectKinds(source: [:0]const u8, expected: []const Token.Kind) !void {
    var buf: [128]Token.Kind = undefined;
    const n = collect(source, buf[0..]);
    try testing.expectEqualSlices(Token.Kind, expected, buf[0..n]);
}

test "empty source produces only eof" {
    var tk = Tokenizer.init("");
    try testing.expectEqual(Token.Kind.eof, tk.next().kind);
}

test "whitespace-only source produces only eof" {
    var tk = Tokenizer.init("   \t\t   ");
    try testing.expectEqual(Token.Kind.eof, tk.next().kind);
}

test "all single-char punctuation" {
    try expectKinds("(){}[],;:?|", &.{
        .lparen, .rparen, .lbrace, .rbrace, .lbracket, .rbracket,
        .comma,  .semi,   .colon,  .question, .pipe,
    });
}

test "all compound operators" {
    try expectKinds("== != <= >= += -= *= /= %= -> => .. ..=", &.{
        .eq, .neq, .le, .ge, .plus_eq, .minus_eq, .star_eq, .slash_eq,
        .percent_eq, .arrow, .fat_arrow, .dotdot, .dotdot_eq,
    });
}

test "arithmetic and comparison singles" {
    try expectKinds("+ - * / % = < > ! ? .", &.{
        .plus, .minus, .star, .slash, .percent, .assign,
        .lt,   .gt,    .bang, .question, .dot,
    });
}

test "every keyword tokenises as its kind, not ident" {
    const cases = .{
        .{ "and", Token.Kind.kw_and },           .{ "as", Token.Kind.kw_as },
        .{ "break", Token.Kind.kw_break },       .{ "catch", Token.Kind.kw_catch },
        .{ "const", Token.Kind.kw_const },       .{ "continue", Token.Kind.kw_continue },
        .{ "else", Token.Kind.kw_else },         .{ "enum", Token.Kind.kw_enum },
        .{ "export", Token.Kind.kw_export },     .{ "false", Token.Kind.kw_false },
        .{ "fn", Token.Kind.kw_fn },             .{ "for", Token.Kind.kw_for },
        .{ "from", Token.Kind.kw_from },         .{ "go", Token.Kind.kw_go },
        .{ "if", Token.Kind.kw_if },             .{ "impl", Token.Kind.kw_impl },
        .{ "import", Token.Kind.kw_import },     .{ "in", Token.Kind.kw_in },
        .{ "interface", Token.Kind.kw_interface }, .{ "match", Token.Kind.kw_match },
        .{ "nil", Token.Kind.kw_nil },           .{ "not", Token.Kind.kw_not },
        .{ "or", Token.Kind.kw_or },             .{ "partial", Token.Kind.kw_partial },
        .{ "record", Token.Kind.kw_record },     .{ "return", Token.Kind.kw_return },
        .{ "self", Token.Kind.kw_self },         .{ "test", Token.Kind.kw_test },
        .{ "true", Token.Kind.kw_true },         .{ "try", Token.Kind.kw_try },
        .{ "type", Token.Kind.kw_type },         .{ "while", Token.Kind.kw_while },
    };
    inline for (cases) |c| {
        var tk = Tokenizer.init(c[0]);
        const t = tk.next();
        try testing.expectEqual(c[1], t.kind);
        try testing.expectEqual(Token.Kind.eof, tk.next().kind);
    }
}

test "identifiers that prefix a keyword are identifiers" {
    // `fnord` starts with `fn`, but the full word is not a keyword.
    try expectKinds("fnord constant returnable _private x1 _", &.{
        .ident, .ident, .ident, .ident, .ident, .ident,
    });
}

test "ident lexeme spans are accurate" {
    const src = "alpha  beta";
    var tk = Tokenizer.init(src);
    const a = tk.next();
    try testing.expectEqualStrings("alpha", a.lexeme(src));
    const b = tk.next();
    try testing.expectEqualStrings("beta", b.lexeme(src));
}

test "integer literals: decimal, hex, binary, octal, underscores" {
    try expectKinds("0 42 1_000_000 0xFF 0xDEAD_BEEF 0b1010_1010 0o755", &.{
        .int_literal, .int_literal, .int_literal,
        .int_literal, .int_literal, .int_literal, .int_literal,
    });
}

test "float literals: fractional and exponential forms" {
    try expectKinds("3.14 0.0 1_000.5 1e10 1.5e-3 2E+4", &.{
        .float_literal, .float_literal, .float_literal,
        .float_literal, .float_literal, .float_literal,
    });
}

test "int followed by .. is int then range (not malformed float)" {
    try expectKinds("1..5", &.{ .int_literal, .dotdot, .int_literal });
    try expectKinds("1..=5", &.{ .int_literal, .dotdot_eq, .int_literal });
}

test "int followed by .ident is method-call shape" {
    try expectKinds("42.method", &.{ .int_literal, .dot, .ident });
}

test "bad hex literal with no digits is invalid" {
    try expectKinds("0x ", &.{.invalid});
}

test "bare e without exponent digits is int then ident" {
    // `1e` is not a float; it's int(1) followed by ident(e).
    try expectKinds("1e", &.{ .int_literal, .ident });
    try expectKinds("1e+", &.{ .int_literal, .ident, .plus });
}

test "empty string literal" {
    try expectKinds("\"\"", &.{.string_literal});
}

test "string containing only interpolation" {
    try expectKinds("\"${x}\"", &.{ .string_start, .ident, .string_end });
}

test "simple string literal" {
    try expectKinds("\"hello\"", &.{.string_literal});
}

test "string with escape sequences" {
    try expectKinds("\"a\\nb\\\"c\\\\d\\xffe\\u{1F600}\"", &.{.string_literal});
}

test "string with simple interpolation" {
    try expectKinds("\"hi ${name}!\"", &.{ .string_start, .ident, .string_end });
}

test "string with multiple interpolations" {
    try expectKinds("\"a${f()}b${g()}c\"", &.{
        .string_start, .ident, .lparen, .rparen,
        .string_part,  .ident, .lparen, .rparen,
        .string_end,
    });
}

test "nested interpolation" {
    try expectKinds("\"x${\"y${z}w\"}v\"", &.{
        .string_start, // "x${
        .string_start, // "y${
        .ident, // z
        .string_end, // }w"
        .string_end, // }v"
    });
}

test "braces inside interpolation do not close the substitution" {
    // `${ { k: v }.k }` — the inner record-literal braces are balanced.
    try expectKinds("\"a${{\"k\": 1}}b\"", &.{
        .string_start, .lbrace, .string_literal, .colon, .int_literal, .rbrace, .string_end,
    });
}

test "escaped dollar does not open interpolation" {
    try expectKinds("\"cost: \\$5\"", &.{.string_literal});
}

test "unterminated string produces invalid" {
    try expectKinds("\"hello", &.{.invalid});
}

test "bare newline in string produces invalid" {
    try expectKinds("\"hel\nlo\"", &.{ .invalid, .ident, .invalid });
}

test "line comment is ignored" {
    try expectKinds("a // this is ignored\nb", &.{ .ident, .semi, .ident });
}

test "block comment is ignored" {
    try expectKinds("a /* ignored */ b", &.{ .ident, .ident });
}

test "nested block comments balance correctly" {
    try expectKinds("a /* outer /* inner */ still outer */ b", &.{ .ident, .ident });
}

test "unterminated block comment becomes an invalid token" {
    try expectKinds("a /* never closed", &.{ .ident, .invalid });
}

test "unterminated nested block comment is still invalid" {
    try expectKinds("a /* /* only inner closed */ ", &.{ .ident, .invalid });
}

test "line comment at end of file needs no trailing newline" {
    try expectKinds("a // trailing", &.{.ident});
}

test "ASI: newline after ident inserts implicit semi" {
    try expectKinds("a\nb", &.{ .ident, .semi, .ident });
}

test "ASI: multiple blank lines still insert only one semi" {
    try expectKinds("a\n\n\nb", &.{ .ident, .semi, .ident });
}

test "ASI: newline after binary op does NOT insert semi (continuation)" {
    try expectKinds("a +\nb", &.{ .ident, .plus, .ident });
}

test "ASI: newline after comma or open paren does NOT insert semi" {
    try expectKinds("f(\n  a,\n  b\n)", &.{
        .ident, .lparen, .ident, .comma, .ident, .rparen,
    });
}

test "ASI: newline after closing brace DOES insert semi" {
    try expectKinds("{}\n{}", &.{ .lbrace, .rbrace, .semi, .lbrace, .rbrace });
}

test "ASI: newline after return inserts semi" {
    try expectKinds("return\nfoo", &.{ .kw_return, .semi, .ident });
}

test "explicit semicolon works when written out" {
    try expectKinds("a; b; c", &.{ .ident, .semi, .ident, .semi, .ident });
}

test "trailing newline at EOF produces a final semi when applicable" {
    try expectKinds("a\n", &.{ .ident, .semi });
}

test "trailing newline at EOF produces no semi after operator" {
    try expectKinds("a +\n", &.{ .ident, .plus });
}

test "span positions are preserved across newlines" {
    const src = "foo\n  bar";
    var tk = Tokenizer.init(src);
    const foo = tk.next();
    try testing.expectEqualStrings("foo", foo.lexeme(src));
    const semi = tk.next();
    try testing.expectEqual(Token.Kind.semi, semi.kind);
    const bar = tk.next();
    try testing.expectEqualStrings("bar", bar.lexeme(src));
    // `bar` starts at byte 6: "foo\n  bar"
    //                         0123 456
    try testing.expectEqual(@as(usize, 6), bar.loc.start);
    try testing.expectEqual(@as(usize, 9), bar.loc.end);
}

test "invalid char is reported and scanning continues" {
    try expectKinds("a # b", &.{ .ident, .invalid, .ident });
}

test "representative real program" {
    // A small program exercising many features together: types, generics,
    // strings with interpolation, operators, control flow, goroutines.
    const src =
        \\fn main() {
        \\  const xs: Array<int> = [1, 2, 3]
        \\  for i in 0..xs.len {
        \\    io.print("item ${xs[i]}")
        \\  }
        \\  go ch.send(42)
        \\}
    ;
    var tk = Tokenizer.init(src);
    // We just verify it reaches EOF without any invalid tokens — structural
    // correctness of this particular sequence is exercised by the
    // feature-specific tests above.
    while (true) {
        const t = tk.next();
        try testing.expect(t.kind != .invalid);
        if (t.kind == .eof) break;
    }
}
