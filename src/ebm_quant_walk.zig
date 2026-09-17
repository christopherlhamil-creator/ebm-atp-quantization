//! Create-path down-walk: f64 → f32 collapse → W2 4-bit, EBM per layer.
//!
//! Energy is layer GEMV residual, not per-weight L2:
//!   E(q) = || Y_f64 - GEMV(X, dequant(q)) ||^2
//! Affine is mill_orch W2: (q-8)*scale+bias, q in 0..15, pack even-low odd-high.
//! error_rate is a caller f64; identity 1.0. Do not invent the coefficient.
//! Portable @Vector. Not CELL_BYTES. Not safetensors ingest.

const std = @import("std");

pub const EBQ_LANES: usize = 8;
pub const EbqF64 = @Vector(EBQ_LANES, f64);
pub const EbqF32 = @Vector(EBQ_LANES, f32);

pub const Affine = struct {
    scale: f64,
    bias: f64,
};

pub const LayerWalk = struct {
    energy_f32: f64,
    energy_round: f64,
    energy_ebm: f64,
    n_accept: usize,
    n_weights: usize,
    pack_bytes: usize,
    packed_ok: bool,
};

pub fn createPathRefusesSafetensors(args: []const []const u8) bool {
    for (args) |arg| {
        if (std.mem.endsWith(u8, arg, ".safetensors")) return true;
    }
    return false;
}

pub fn affineFromRange(min_v: f64, max_v: f64) Affine {
    const scale_v: f64 = if (max_v > min_v) (max_v - min_v) / 15.0 else 1.0;
    return .{ .scale = scale_v, .bias = min_v + 8.0 * scale_v };
}

pub fn dequant(q: u8, aff: Affine) f64 {
    return (@as(f64, @floatFromInt(q)) - 8.0) * aff.scale + aff.bias;
}

pub fn quantizeRoundNearest(w: f64, aff: Affine) u8 {
    const raw = (w - aff.bias) / aff.scale + 8.0;
    const qi: i32 = @intFromFloat(@round(raw));
    if (qi < 0) return 0;
    if (qi > 15) return 15;
    return @intCast(qi);
}

fn loadF64(src: []const f64, off: usize) EbqF64 {
    const arr: [EBQ_LANES]f64 = src[off..][0..EBQ_LANES].*;
    return arr;
}

fn loadF32(src: []const f32, off: usize) EbqF32 {
    const arr: [EBQ_LANES]f32 = src[off..][0..EBQ_LANES].*;
    return arr;
}

fn storeF64(dst: []f64, off: usize, v: EbqF64) void {
    dst[off..][0..EBQ_LANES].* = @as([EBQ_LANES]f64, v);
}

fn storeF32(dst: []f32, off: usize, v: EbqF32) void {
    dst[off..][0..EBQ_LANES].* = @as([EBQ_LANES]f32, v);
}

pub fn collapseF64ToF32(src: []const f64, dst: []f32) void {
    std.debug.assert(dst.len >= src.len);
    var i: usize = 0;
    while (i + EBQ_LANES <= src.len) : (i += EBQ_LANES) {
        const wide: EbqF64 = loadF64(src, i);
        const narrowed: EbqF32 = @floatCast(wide);
        storeF32(dst, i, narrowed);
    }
    while (i < src.len) : (i += 1) {
        dst[i] = @floatCast(src[i]);
    }
}

pub fn reprojectF32ToF64(src: []const f32, dst: []f64) void {
    std.debug.assert(dst.len >= src.len);
    var i: usize = 0;
    while (i + EBQ_LANES <= src.len) : (i += EBQ_LANES) {
        const narrow: EbqF32 = loadF32(src, i);
        const widened: EbqF64 = @floatCast(narrow);
        storeF64(dst, i, widened);
    }
    while (i < src.len) : (i += 1) {
        dst[i] = @floatCast(src[i]);
    }
}

pub fn packBytesNeeded(n_weights: usize) usize {
    return (n_weights + 1) / 2;
}

pub fn packEvenLowOddHigh(q: []const u8, coded: []u8) void {
    std.debug.assert(coded.len >= packBytesNeeded(q.len));
    var i: usize = 0;
    var out_i: usize = 0;
    while (i + 1 < q.len) : ({
        i += 2;
        out_i += 1;
    }) {
        coded[out_i] = (q[i] & 0x0F) | ((q[i + 1] & 0x0F) << 4);
    }
    if (i < q.len) {
        coded[out_i] = q[i] & 0x0F;
    }
}

pub fn unpackEvenLowOddHigh(coded: []const u8, q: []u8) void {
    std.debug.assert(coded.len >= packBytesNeeded(q.len));
    var i: usize = 0;
    var out_i: usize = 0;
    while (i + 1 < q.len) : ({
        i += 2;
        out_i += 1;
    }) {
        q[i] = coded[out_i] & 0x0F;
        q[i + 1] = coded[out_i] >> 4;
    }
    if (i < q.len) {
        q[i] = coded[out_i] & 0x0F;
    }
}

pub fn gemvF64(w: []const f64, x: []const f64, rows: usize, cols: usize, y: []f64) void {
    std.debug.assert(w.len >= rows * cols);
    std.debug.assert(x.len >= cols);
    std.debug.assert(y.len >= rows);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row_w = w[r * cols ..][0..cols];
        var acc: EbqF64 = @splat(0);
        var col_i: usize = 0;
        while (col_i + EBQ_LANES <= cols) : (col_i += EBQ_LANES) {
            const vw: EbqF64 = loadF64(row_w, col_i);
            const vx: EbqF64 = loadF64(x, col_i);
            acc += vw * vx;
        }
        var sum_v: f64 = @reduce(.Add, acc);
        while (col_i < cols) : (col_i += 1) {
            sum_v += row_w[col_i] * x[col_i];
        }
        y[r] = sum_v;
    }
}

pub fn gemvDequant(q: []const u8, x: []const f64, rows: usize, cols: usize, aff: Affine, y: []f64) void {
    std.debug.assert(q.len >= rows * cols);
    std.debug.assert(x.len >= cols);
    std.debug.assert(y.len >= rows);
    const v_scale: EbqF64 = @splat(aff.scale);
    const v_bias: EbqF64 = @splat(aff.bias);
    const v_off: EbqF64 = @splat(8.0);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row_q = q[r * cols ..][0..cols];
        var acc: EbqF64 = @splat(0);
        var col_i: usize = 0;
        while (col_i + EBQ_LANES <= cols) : (col_i += EBQ_LANES) {
            const raw: EbqF64 = .{
                @floatFromInt(row_q[col_i + 0]),
                @floatFromInt(row_q[col_i + 1]),
                @floatFromInt(row_q[col_i + 2]),
                @floatFromInt(row_q[col_i + 3]),
                @floatFromInt(row_q[col_i + 4]),
                @floatFromInt(row_q[col_i + 5]),
                @floatFromInt(row_q[col_i + 6]),
                @floatFromInt(row_q[col_i + 7]),
            };
            const vw = (raw - v_off) * v_scale + v_bias;
            const vx: EbqF64 = loadF64(x, col_i);
            acc += vw * vx;
        }
        var sum_v: f64 = @reduce(.Add, acc);
        while (col_i < cols) : (col_i += 1) {
            sum_v += dequant(row_q[col_i], aff) * x[col_i];
        }
        y[r] = sum_v;
    }
}

pub fn residualEnergy(y_true: []const f64, y_hat: []const f64) f64 {
    std.debug.assert(y_hat.len >= y_true.len);
    var energy_acc: f64 = 0;
    var i: usize = 0;
    while (i + EBQ_LANES <= y_true.len) : (i += EBQ_LANES) {
        const vt: EbqF64 = loadF64(y_true, i);
        const vh: EbqF64 = loadF64(y_hat, i);
        const delta_v = vt - vh;
        const sq = delta_v * delta_v;
        energy_acc += @reduce(.Add, sq);
    }
    while (i < y_true.len) : (i += 1) {
        const delta_s = y_true[i] - y_hat[i];
        energy_acc += delta_s * delta_s;
    }
    return energy_acc;
}

/// Gauss-Seidel ±1 on each nibble. Rank-1 residual update. Strict energy drop.
pub fn descendLayer(residual: []f64, x: []const f64, q: []u8, rows: usize, cols: usize, aff: Affine) usize {
    std.debug.assert(residual.len >= rows);
    std.debug.assert(x.len >= cols);
    std.debug.assert(q.len >= rows * cols);
    var n_accept: usize = 0;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var col_i: usize = 0;
        while (col_i < cols) : (col_i += 1) {
            const idx = r * cols + col_i;
            const q0: u8 = q[idx];
            const w0 = dequant(q0, aff);
            const xi = x[col_i];
            const ri = residual[r];
            var best_q = q0;
            var best_e = ri * ri;
            var best_ri = ri;
            var dq: i32 = -1;
            while (dq <= 1) : (dq += 2) {
                const qn_i = @as(i32, @intCast(q0)) + dq;
                if (qn_i < 0 or qn_i > 15) continue;
                const qn: u8 = @intCast(qn_i);
                const dw = dequant(qn, aff) - w0;
                const ri1 = ri - dw * xi;
                const e1 = ri1 * ri1;
                if (e1 < best_e) {
                    best_e = e1;
                    best_q = qn;
                    best_ri = ri1;
                }
            }
            if (best_q != q0) {
                q[idx] = best_q;
                residual[r] = best_ri;
                n_accept += 1;
            }
        }
    }
    return n_accept;
}

pub fn walkMinMax(w: []const f64) struct { min_v: f64, max_v: f64 } {
    var min_v: f64 = w[0];
    var max_v: f64 = w[0];
    for (w[1..]) |val| {
        min_v = @min(min_v, val);
        max_v = @max(max_v, val);
    }
    return .{ .min_v = min_v, .max_v = max_v };
}

pub fn walkLayer(
    w_f64: []const f64,
    x: []const f64,
    rows: usize,
    cols: usize,
    error_rate: f64,
    w_proj: []f64,
    w_f32: []f32,
    w_re: []f64,
    y_true: []f64,
    y_hat: []f64,
    residual: []f64,
    q: []u8,
    q_check: []u8,
    coded: []u8,
) LayerWalk {
    const n = rows * cols;
    std.debug.assert(w_f64.len >= n);
    std.debug.assert(w_proj.len >= n);
    std.debug.assert(w_f32.len >= n);
    std.debug.assert(w_re.len >= n);
    std.debug.assert(q.len >= n);
    std.debug.assert(q_check.len >= n);
    const nbytes = packBytesNeeded(n);
    std.debug.assert(coded.len >= nbytes);

    var i: usize = 0;
    while (i + EBQ_LANES <= n) : (i += EBQ_LANES) {
        const v: EbqF64 = loadF64(w_f64, i);
        storeF64(w_proj, i, v * @as(EbqF64, @splat(error_rate)));
    }
    while (i < n) : (i += 1) {
        w_proj[i] = w_f64[i] * error_rate;
    }

    collapseF64ToF32(w_proj[0..n], w_f32[0..n]);
    reprojectF32ToF64(w_f32[0..n], w_re[0..n]);

    gemvF64(w_proj[0..n], x, rows, cols, y_true);
    gemvF64(w_re[0..n], x, rows, cols, y_hat);
    const energy_f32 = residualEnergy(y_true[0..rows], y_hat[0..rows]);

    const mm = walkMinMax(w_re[0..n]);
    const aff = affineFromRange(mm.min_v, mm.max_v);
    var k: usize = 0;
    while (k < n) : (k += 1) {
        q[k] = quantizeRoundNearest(w_re[k], aff);
    }

    gemvDequant(q[0..n], x, rows, cols, aff, y_hat);
    const energy_round = residualEnergy(y_true[0..rows], y_hat[0..rows]);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        residual[r] = y_true[r] - y_hat[r];
    }

    const n_accept = descendLayer(residual[0..rows], x, q[0..n], rows, cols, aff);
    gemvDequant(q[0..n], x, rows, cols, aff, y_hat);
    const energy_ebm = residualEnergy(y_true[0..rows], y_hat[0..rows]);

    packEvenLowOddHigh(q[0..n], coded[0..nbytes]);
    unpackEvenLowOddHigh(coded[0..nbytes], q_check[0..n]);
    const packed_ok = std.mem.eql(u8, q[0..n], q_check[0..n]);

    return .{
        .energy_f32 = energy_f32,
        .energy_round = energy_round,
        .energy_ebm = energy_ebm,
        .n_accept = n_accept,
        .n_weights = n,
        .pack_bytes = nbytes,
        .packed_ok = packed_ok,
    };
}

pub fn fillLattice(w: []f64, x: []f64, rows: usize, cols: usize, layer: usize) void {
    const loff: f64 = @as(f64, @floatFromInt(layer)) * 0.07;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var col_i: usize = 0;
        while (col_i < cols) : (col_i += 1) {
            const t: i32 = @intCast(r * 17 + col_i * 3 + layer * 11);
            w[r * cols + col_i] = @as(f64, @floatFromInt(t)) * 0.013 - 0.41 + loff;
        }
    }
    var col_i: usize = 0;
    while (col_i < cols) : (col_i += 1) {
        const t: i32 = @intCast(col_i + 1);
        x[col_i] = @as(f64, @floatFromInt(t)) * 0.031 - 0.22;
    }
}
