//! Root module for the Zua language implementation.
//!
//! Everything that the CLI (and eventually any embedder of Zua as a library)
//! uses is re-exported from this file. Each compiler/runtime stage lives in
//! its own source file and is surfaced here so that `zig build test` can
//! reach every test block in the project through the module graph.

pub const lexer = @import("lexer.zig");

// Surface every submodule's `test` blocks to `zig build test`. A bare
// `pub const lexer = @import(...)` is *not* enough — Zig only compiles test
// blocks out of files whose declarations are semantically referenced, and a
// top-level re-export doesn't force analysis of the imported file's test
// bodies. A test block that references the submodule does. Without this,
// `zig build test` silently reports "All 0 tests passed".
//
// Every new `.zig` file under `src/` that carries its own `test` blocks
// needs a `_ =` line here, or those tests are invisible to CI.
test {
    _ = lexer;
}
