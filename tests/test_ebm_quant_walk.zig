const std = @import("std");
const ebq = @import("ebm_quant_walk");

test "create path refuses safetensors" {
    const args = [_][]const u8{ "mill", "/tmp/model.safetensors" };
    try std.testing.expect(ebq.createPathRefusesSafetensors(&args));
    const ok_args = [_][]const u8{ "mill", "8", "64" };
    try std.testing.expect(!ebq.createPathRefusesSafetensors(&ok_args));
}

test "pack even-low odd-high recovers nibbles" {
    const q = [_]u8{ 0, 15, 8, 7, 1, 14 };
    var coded: [3]u8 = undefined;
    var back: [6]u8 = undefined;
    ebq.packEvenLowOddHigh(&q, &coded);
    try std.testing.expectEqual(@as(u8, 0xF0), coded[0]);
    try std.testing.expectEqual(@as(u8, 0x78), coded[1]);
    try std.testing.expectEqual(@as(u8, 0xE1), coded[2]);
    ebq.unpackEvenLowOddHigh(&coded, &back);
    try std.testing.expectEqualSlices(u8, &q, &back);
}

test "W2 affine dequant is (q-8)*scale+bias" {
    const aff = ebq.Affine{ .scale = 2.0, .bias = 0.5 };
    try std.testing.expectEqual(@as(f64, -1.5), ebq.dequant(7, aff));
    try std.testing.expectEqual(@as(f64, 2.5), ebq.dequant(9, aff));
}

test "layer EBM drops GEMV residual vs round-nearest" {
    const aff = ebq.Affine{ .scale = 1.0, .bias = 0.0 };
    var q = [_]u8{ 9, 9 };
    const x = [_]f64{ 1.0, 1.0 };
    const y_true = [_]f64{1.2};
    var y_hat: [1]f64 = undefined;
    ebq.gemvDequant(&q, &x, 1, 2, aff, &y_hat);
    const e_round = ebq.residualEnergy(&y_true, &y_hat);
    var residual = [_]f64{y_true[0] - y_hat[0]};
    const n_accept = ebq.descendLayer(&residual, &x, &q, 1, 2, aff);
    ebq.gemvDequant(&q, &x, 1, 2, aff, &y_hat);
    const e_ebm = ebq.residualEnergy(&y_true, &y_hat);
    try std.testing.expect(n_accept >= 1);
    try std.testing.expect(e_ebm < e_round);
    try std.testing.expect((q[0] == 8 and q[1] == 9) or (q[0] == 9 and q[1] == 8));
}

test "f64 to f32 to 4-bit walk energy does not rise" {
    const rows: usize = 8;
    const cols: usize = 16;
    const n = rows * cols;
    var w: [128]f64 = undefined;
    var x: [16]f64 = undefined;
    ebq.fillLattice(&w, &x, rows, cols, 0);
    var w_proj: [128]f64 = undefined;
    var w_f32: [128]f32 = undefined;
    var w_re: [128]f64 = undefined;
    var y_true: [8]f64 = undefined;
    var y_hat: [8]f64 = undefined;
    var residual: [8]f64 = undefined;
    var q: [128]u8 = undefined;
    var q_check: [128]u8 = undefined;
    var coded: [64]u8 = undefined;
    const got = ebq.walkLayer(
        &w,
        &x,
        rows,
        cols,
        1.0,
        &w_proj,
        &w_f32,
        &w_re,
        &y_true,
        &y_hat,
        &residual,
        &q,
        &q_check,
        &coded,
    );
    try std.testing.expect(got.packed_ok);
    try std.testing.expectEqual(n, got.n_weights);
    try std.testing.expectEqual(@as(usize, 64), got.pack_bytes);
    try std.testing.expect(got.energy_ebm <= got.energy_round);
    try std.testing.expect(got.energy_f32 >= 0);
    try std.testing.expect(got.energy_round >= 0);
}

test "f64 collapse to f32 is not always identity" {
    const src = [_]f64{1.0 / 3.0};
    var dst: [1]f32 = undefined;
    var back: [1]f64 = undefined;
    ebq.collapseF64ToF32(&src, &dst);
    ebq.reprojectF32ToF64(&dst, &back);
    try std.testing.expect(back[0] != src[0]);
}
