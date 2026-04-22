//! Root module for the Zua language implementation.
//!
//! Everything that the CLI (and eventually any embedder of Zua as a library)
//! uses is re-exported from this file. Each compiler/runtime stage lives in
//! its own source file and is surfaced here so that `zig build test` can
//! reach every test block in the project through the module graph.
