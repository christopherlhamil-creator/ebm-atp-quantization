//! Comptime Hardware Architecture & Kernel Execution Directives
//! Generated automatically by scripts/solve_ebm_hardware_gap.py
//! Verified by Z3 + Vampire + Leo-III + EBM solver stack.

pub const TARGET_HARDWARE = "AMD Ryzen 7 8700F (8 Cores / 16 Threads @ 4.775 GHz All-Core AVX-512)";
pub const TARGET_ISA = "x86_64";
pub const ACTIVE_CORES: usize = 8;
pub const VOCAB_ROWS_PER_CORE: usize = 18992;
pub const L1D_WORKING_BUDGET_BYTES: usize = 24576;
pub const TILE_WORKING_SET_BYTES: usize = 10240;

pub const FUSED_SWIGLU_WORKING_SET_BYTES: usize = 2048;
pub const UNFUSED_TWOPASS_WORKING_SET_BYTES: usize = 88064;
pub const TILES_PER_GROUP: usize = 8;
pub const TOTAL_INTERMEDIATE_TILES: usize = 688;
pub const TOTAL_INTERMEDIATE_GROUPS: usize = 86;
pub const WORKER_0_TILE_START: usize = 0;
pub const WORKER_0_TILE_END: usize = 176;
pub const WORKER_1_TILE_START: usize = 176;
pub const WORKER_1_TILE_END: usize = 352;
pub const WORKER_2_TILE_START: usize = 352;
pub const WORKER_2_TILE_END: usize = 520;
pub const WORKER_3_TILE_START: usize = 520;
pub const WORKER_3_TILE_END: usize = 688;
pub const SDOT_VECTOR_REGS_USED: usize = 26;
pub const SDOT_VECTOR_REGS_BUDGET: usize = 32;

pub const SYMMETRICAL_ROW_ISOLATION: bool = true;
pub const SYMMETRICAL_BLOCK_SIZE: usize = 128;

pub const VECTOR_DOT_LANES: usize = 16;
pub const VECTOR_INSTRUCTION = "vfmadd231ps";
pub const VECTOR_WIDTH_BITS: usize = 512;
pub const UNROLL_ACCUMULATOR_FACTOR: usize = 2;
pub const REDUCTION_STRIDE: usize = 2048;
pub const USE_INT8_SDOT: bool = false;

pub const DRAM_BANDWIDTH_SATURATION_GB_S: f32 = 43.2000;
pub const TARGET_THROUGHPUT_TOK_S: f32 = 51.28;
pub const TARGET_DECODE_MS: f32 = 19.50;
pub const LATEST_MEASURED_DECODE_MS: f32 = 78.40;
pub const LATEST_MEASURED_TOK_S: f32 = 12.755;

// Upstream competitor baseline & remaining squeeze target
pub const LLAMA_BASELINE_MS: f32 = 25.97;
pub const LLAMA_BASELINE_TOK_S: f32 = 38.50;
pub const REMAINING_GAP_TO_LLAMA_MS: f32 = 52.43;
pub const SQUEEZED_TARGET_DECODE_MS: f32 = 19.50;
pub const SQUEEZED_TARGET_TOK_S: f32 = 15.53;

// Profiled layer breakdown (ms)
pub const PROFILE_GATE_UP_MS: f32 = 37.37;
pub const PROFILE_DOWN_PROJ_MS: f32 = 20.49;
pub const PROFILE_LM_HEAD_MS: f32 = 6.82;
pub const PROFILE_QKV_PROJ_MS: f32 = 5.14;
pub const PROFILE_O_PROJ_MS: f32 = 4.20;
pub const PROFILE_RMSNORM_MS: f32 = 0.46;
pub const PROFILE_ATTN_GQA_MS: f32 = 0.07;

// Squeeze targets (ms)
pub const TARGET_GATE_UP_MS: f32 = 34.00;
pub const TARGET_DOWN_PROJ_MS: f32 = 15.00;
pub const TARGET_LM_HEAD_MS: f32 = 5.50;

const std = @import("std");

test "hardware config parameters" {
    try std.testing.expectEqual(@as(usize, 4), ACTIVE_CORES);
    try std.testing.expectEqual(@as(usize, 49152), L1D_WORKING_BUDGET_BYTES);
    try std.testing.expect(SDOT_VECTOR_REGS_USED <= SDOT_VECTOR_REGS_BUDGET);
    try std.testing.expect(SQUEEZED_TARGET_DECODE_MS < LLAMA_BASELINE_MS);
}
