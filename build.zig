const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 1. ebm_layer_compiler executable
    const exe = b.addExecutable(.{
        .name = "ebm_layer_compiler",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ebm_layer_compiler.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    // 2. Unit Tests
    const test_compiler = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ebm_layer_compiler.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_test_compiler = b.addRunArtifact(test_compiler);

    const mod_walk = b.createModule(.{
        .root_source_file = b.path("src/ebm_quant_walk.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_walk = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_ebm_quant_walk.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ebm_quant_walk", .module = mod_walk },
            },
        }),
    });
    const run_test_walk = b.addRunArtifact(test_walk);

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_test_compiler.step);
    test_step.dependOn(&run_test_walk.step);
}
