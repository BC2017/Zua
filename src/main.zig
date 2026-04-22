//! Zua CLI entry point.
//!
//! For now this is a stub — there is no runnable Zua source yet, because the
//! lexer, parser, type checker, compiler, and VM are all still to be built.
//! As each stage comes online this CLI will grow (`zua run file.zua`,
//! `zua test`, `zua check`, etc.).

const std = @import("std");
const zua = @import("Zua");

pub fn main(init: std.process.Init) !void {
    _ = init;
    std.debug.print("zua: interpreter not yet implemented\n", .{});
}
