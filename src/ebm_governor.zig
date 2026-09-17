//! Multi-signal Product-of-Experts write governor (Rectangle 4).
//!
//! Log-space PoE identity:
//!   -log P(candidate) = E_joint = E_NPU + E_CPU + E_GPU
//! Happy path (b2b gap < 0.0100): E_GPU = ∅ (skip), not 0.
//!   Zero would mean "GPU agrees perfectly" and would corrupt the joint.
//! GPU wakes only when b2b gap >= 0.0100 (same pin as b2b_pack local silence).
//!
//! Synchronous hard veto, before any pwrite to the cell bank:
//!   E_joint >= 500  -> HardVeto (opcode 1004)
//!   hop depth > 4   -> RefusalMaxHopExceeded (Invariant A-11)
//!
//! Invariants: A-1 17408, A-2 64, A-8 1004, A-11 hop<=4.

const std = @import("std");

pub const CELL_BYTES: usize = 17408;
pub const HEADER_BYTES: usize = 64;
pub const MAX_HOP_DEPTH: usize = 4;
pub const PROV_HOP_SHIFT: u6 = 12;
pub const HARD_VETO_ENERGY: u32 = 500;
pub const GPU_WAKE_GAP: f32 = 0.0100;
pub const OPCODE_VETO: u64 = 1004; // lexicon.LogicOp.negative_affinity_veto
pub const OPCODE_ASSERT: u64 = 1001; // lexicon.LogicOp.assert_relation

pub const GovernorError = error{
    HardVeto,
    RefusalMaxHopExceeded,
    ShortWrite,
};

/// Three named silicon terms. `gpu == null` is skip (∅), not energy 0.
pub const ExpertEnergy = struct {
    npu: u32 = 0,
    cpu: u32 = 0,
    gpu: ?u32 = null,

    pub fn gpuIsSkip(self: ExpertEnergy) bool {
        return self.gpu == null;
    }
};

pub fn hopFromFlags(flags: u64) usize {
    return @intCast((flags >> PROV_HOP_SHIFT) & 0xF);
}

pub const CellBuf = [CELL_BYTES]u8;

/// Compose the tripartite term. GPU is omitted unless the b2b gap wakes it.
pub fn compose(npu: u32, cpu: u32, b2b_gap: f32, gpu_energy: u32) ExpertEnergy {
    if (b2b_gap < GPU_WAKE_GAP) {
        return .{ .npu = npu, .cpu = cpu, .gpu = null };
    }
    return .{ .npu = npu, .cpu = cpu, .gpu = gpu_energy };
}

/// Joint log-space sum. Null GPU contributes nothing (skip, not +0-as-agreement).
pub fn jointEnergy(terms: ExpertEnergy) u32 {
    var j: u32 = terms.npu;
    j +|= terms.cpu;
    if (terms.gpu) |g| j +|= g;
    return j;
}

/// Register-level gate. No allocation. No copy.
pub fn evaluate(terms: ExpertEnergy, hop: usize) GovernorError!u32 {
    if (hop > MAX_HOP_DEPTH) return error.RefusalMaxHopExceeded;
    const j = jointEnergy(terms);
    if (j >= HARD_VETO_ENERGY) return error.HardVeto;
    return j;
}

fn stampVeto(cell: *align(64) CellBuf) void {
    std.mem.writeInt(u64, cell[0..8], OPCODE_VETO, .little);
}

pub const PoeGovernor = struct {
    terms: ExpertEnergy = .{},
    commits: u64 = 0,
    vetoes: u64 = 0,
    hop_kills: u64 = 0,
    gpu_skips: u64 = 0,
    gpu_wakes: u64 = 0,
    bytes_written: u64 = 0,

    pub fn observe(self: *PoeGovernor, terms: ExpertEnergy) void {
        self.terms = terms;
        if (terms.gpuIsSkip()) {
            self.gpu_skips += 1;
        } else {
            self.gpu_wakes += 1;
        }
    }

    /// Evaluate then writePositional 17,408 B. Veto returns without a write (zero-copy).
    pub fn tryCommit(
        self: *PoeGovernor,
        cell: *align(64) CellBuf,
        terms: ExpertEnergy,
        hop: usize,
        io: std.Io,
        file: std.Io.File,
        slot: u64,
    ) GovernorError!u32 {
        self.observe(terms);
        const j = evaluate(terms, hop) catch |err| {
            stampVeto(cell);
            switch (err) {
                error.RefusalMaxHopExceeded => self.hop_kills += 1,
                else => {},
            }
            self.vetoes += 1;
            return err;
        };
        const off: u64 = slot * CELL_BYTES;
        file.writePositionalAll(io, cell, off) catch return error.ShortWrite;
        self.commits += 1;
        self.bytes_written += CELL_BYTES;
        return j;
    }
};

comptime {
    std.debug.assert(CELL_BYTES == 17408);
    std.debug.assert(HEADER_BYTES == 64);
    std.debug.assert(MAX_HOP_DEPTH == 4);
    std.debug.assert(OPCODE_VETO == 1004);
    std.debug.assert(HARD_VETO_ENERGY == 500);
}

test "happy-path GPU is skip not zero: joint ignores a huge idle GPU term" {
    const idle = compose(40, 10, 0.0, 1_000_000);
    try std.testing.expect(idle.gpuIsSkip());
    try std.testing.expectEqual(@as(u32, 50), jointEnergy(idle));
    try std.testing.expectEqual(@as(u32, 50), try evaluate(idle, 0));
}

test "b2b gap at 0.0100 wakes GPU and the extra term can trip HardVeto" {
    const silent = compose(200, 200, 0.0099, 200);
    try std.testing.expect(silent.gpuIsSkip());
    try std.testing.expectEqual(@as(u32, 400), try evaluate(silent, 1));

    const awake = compose(200, 200, 0.0100, 200);
    try std.testing.expect(!awake.gpuIsSkip());
    try std.testing.expectEqual(@as(?u32, 200), awake.gpu);
    try std.testing.expectError(error.HardVeto, evaluate(awake, 1));
}

test "E_joint >= 500 hard-vetoes even with GPU skipped" {
    const terms = compose(250, 250, 0.0, 0);
    try std.testing.expect(terms.gpuIsSkip());
    try std.testing.expectError(error.HardVeto, evaluate(terms, 0));
    try std.testing.expectEqual(@as(u32, 499), try evaluate(compose(249, 250, 0.0, 9_999), 4));
}

test "hop 5 refuses before energy is consulted (A-11)" {
    const cheap = compose(0, 0, 0.0, 0);
    try std.testing.expectEqual(@as(u32, 0), try evaluate(cheap, 4));
    try std.testing.expectError(error.RefusalMaxHopExceeded, evaluate(cheap, 5));
}

test "tryCommit writes 17408 B on pass and writes 0 B on veto" {
    const io = std.testing.io;
    const path = "/tmp/tot_hybrid_ebm_governor_unit.cells";
    std.Io.Dir.cwd().deleteFile(io, path) catch {};
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .read = true, .truncate = true });
    defer {
        file.close(io);
        std.Io.Dir.cwd().deleteFile(io, path) catch {};
    }

    var cell: CellBuf align(64) = undefined;
    @memset(&cell, 0);
    std.mem.writeInt(u64, cell[0..8], OPCODE_ASSERT, .little);

    var gov = PoeGovernor{};
    const pass_terms = compose(40, 10, 0.0, 1_000_000);
    const j = try gov.tryCommit(&cell, pass_terms, 1, io, file, 0);
    try std.testing.expectEqual(@as(u32, 50), j);
    try std.testing.expectEqual(@as(u64, 1), gov.commits);
    try std.testing.expectEqual(@as(u64, CELL_BYTES), gov.bytes_written);
    try std.testing.expectEqual(@as(u64, 1), gov.gpu_skips);
    try std.testing.expectEqual(@as(u64, 0), gov.gpu_wakes);

    const st_pass = try file.stat(io);
    try std.testing.expectEqual(@as(u64, CELL_BYTES), st_pass.size);

    const veto_terms = compose(300, 250, 0.0, 0);
    try std.testing.expectError(error.HardVeto, gov.tryCommit(&cell, veto_terms, 1, io, file, 1));
    try std.testing.expectEqual(OPCODE_VETO, std.mem.readInt(u64, cell[0..8], .little));
    try std.testing.expectEqual(@as(u64, 1), gov.vetoes);
    try std.testing.expectEqual(@as(u64, CELL_BYTES), gov.bytes_written);

    const st_veto = try file.stat(io);
    try std.testing.expectEqual(@as(u64, CELL_BYTES), st_veto.size);

    try std.testing.expectError(error.RefusalMaxHopExceeded, gov.tryCommit(&cell, pass_terms, 5, io, file, 1));
    try std.testing.expectEqual(@as(u64, 1), gov.hop_kills);
    const st_hop = try file.stat(io);
    try std.testing.expectEqual(@as(u64, CELL_BYTES), st_hop.size);
}
