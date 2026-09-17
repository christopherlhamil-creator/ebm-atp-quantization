const std = @import("std");

pub fn optimizeQwenRowChunkAvx512(
    target_y: @Vector(8, f64),
    x_activations: @Vector(8, f64),
    initial_i4: @Vector(8, i32),
    scale: f64,
) @Vector(8, i32) {
    var best_weights = initial_i4;
    var min_energy = evaluateLayerEnergy(target_y, x_activations, best_weights, scale);

    inline for (0..8) |lane| {
        var test_weights = best_weights;
        if (test_weights[lane] < 7) {
            test_weights[lane] += 1;
            const energy_up = evaluateLayerEnergy(target_y, x_activations, test_weights, scale);
            if (energy_up < min_energy) {
                min_energy = energy_up;
                best_weights = test_weights;
            } else {
                test_weights[lane] -= 1;
                if (test_weights[lane] > -8) {
                    test_weights[lane] -= 1;
                    const energy_down = evaluateLayerEnergy(target_y, x_activations, test_weights, scale);
                    if (energy_down < min_energy) {
                        min_energy = energy_down;
                        best_weights = test_weights;
                    }
                }
            }
        } else if (test_weights[lane] > -8) {
            test_weights[lane] -= 1;
            const energy_down = evaluateLayerEnergy(target_y, x_activations, test_weights, scale);
            if (energy_down < min_energy) {
                min_energy = energy_down;
                best_weights = test_weights;
            }
        }
    }
    return best_weights;
}

pub inline fn evaluateLayerEnergy(
    y_true: @Vector(8, f64),
    x_act: @Vector(8, f64),
    w_cand: @Vector(8, i32),
    scale: f64,
) f64 {
    const w_f64: @Vector(8, f64) = @floatFromInt(w_cand);
    const vec_scale: @Vector(8, f64) = @splat(scale);
    const w_dequant = w_f64 * vec_scale;
    const y_approx = x_act * w_dequant;
    const delta = y_true - y_approx;
    const squared_loss = delta * delta;
    return @reduce(.Add, squared_loss);
}

pub fn calculateBlockScale(vals: []const f64) f64 {
    var max_abs: f64 = 0.0;
    for (vals) |v| {
        const abs_v = @abs(v);
        if (abs_v > max_abs) max_abs = abs_v;
    }
    if (max_abs == 0.0) return 1.0;
    return max_abs / 7.0;
}

pub fn naiveQuantize(vals: []const f64, scale: f64) @Vector(8, i32) {
    var out: [8]i32 = undefined;
    const inv_scale = 1.0 / scale;
    for (vals, 0..) |v, i| {
        const scaled = @round(v * inv_scale);
        const int_val: i32 = @intFromFloat(scaled);
        out[i] = std.math.clamp(int_val, -8, 7);
    }
    return out;
}

fn readFile(allocator: std.mem.Allocator, path: []const u8, buf: []u8) !void {
    const path_z = try allocator.allocSentinel(u8, path.len, 0);
    defer allocator.free(path_z);
    @memcpy(path_z, path);
    const fd = try std.posix.openat(std.posix.AT.FDCWD, path_z, .{ .ACCMODE = .RDONLY }, 0);
    defer _ = std.os.linux.close(fd);

    var offset: usize = 0;
    while (offset < buf.len) {
        const n = std.os.linux.read(fd, buf.ptr + offset, buf.len - offset);
        if (n <= 0) break;
        offset += @intCast(n);
    }
    if (offset != buf.len) {
        return error.UnexpectedEndOfFile;
    }
}

fn writeFile(allocator: std.mem.Allocator, path: []const u8, data: []const u8) !void {
    const path_z = try allocator.allocSentinel(u8, path.len, 0);
    defer allocator.free(path_z);
    @memcpy(path_z, path);
    const fd = try std.posix.openat(
        std.posix.AT.FDCWD,
        path_z,
        .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true },
        0o644,
    );
    defer _ = std.os.linux.close(fd);

    var offset: usize = 0;
    while (offset < data.len) {
        const n = std.os.linux.write(fd, data.ptr + offset, data.len - offset);
        if (n <= 0) break;
        offset += @intCast(n);
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);

    if (args.len < 13) {
        std.debug.print("Usage: ebm_layer_compiler --weights-bin <path> --x-act-bin <path> --y-target-bin <path> --rows <R> --cols <C> --out-slice <path>\n", .{});
        std.process.exit(1);
    }

    var weights_path: []const u8 = "";
    var x_path: []const u8 = "";
    var y_path: []const u8 = "";
    var out_path: []const u8 = "";
    var rows: usize = 0;
    var cols: usize = 0;

    var idx: usize = 1;
    while (idx + 1 < args.len) : (idx += 2) {
        const key = args[idx];
        const val = args[idx + 1];
        if (std.mem.eql(u8, key, "--weights-bin")) weights_path = val
        else if (std.mem.eql(u8, key, "--x-act-bin")) x_path = val
        else if (std.mem.eql(u8, key, "--y-target-bin")) y_path = val
        else if (std.mem.eql(u8, key, "--out-slice")) out_path = val
        else if (std.mem.eql(u8, key, "--rows")) rows = try std.fmt.parseInt(usize, val, 10)
        else if (std.mem.eql(u8, key, "--cols")) cols = try std.fmt.parseInt(usize, val, 10);
    }

    if (rows == 0 or cols == 0 or weights_path.len == 0 or out_path.len == 0) {
        std.debug.print("Invalid arguments provided\n", .{});
        std.process.exit(1);
    }

    const total_elements = rows * cols;
    const weights = try allocator.alloc(f64, total_elements);
    defer allocator.free(weights);
    try readFile(allocator, weights_path, std.mem.sliceAsBytes(weights));

    const x_act = try allocator.alloc(f64, total_elements);
    defer allocator.free(x_act);
    try readFile(allocator, x_path, std.mem.sliceAsBytes(x_act));

    const y_targets = try allocator.alloc(f64, total_elements);
    defer allocator.free(y_targets);
    try readFile(allocator, y_path, std.mem.sliceAsBytes(y_targets));

    const packed_bytes_count = total_elements / 2;
    // Align to 16,384 sector law
    const sector_aligned_size = ((packed_bytes_count + 16383) / 16384) * 16384;
    const packed_output = try allocator.alloc(u8, sector_aligned_size);
    defer allocator.free(packed_output);
    @memset(packed_output, 0);

    var stream_idx: usize = 0;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var c: usize = 0;
        while (c < cols) : (c += 8) {
            const offset = (r * cols) + c;
            const target_y_chunk: @Vector(8, f64) = y_targets[offset..][0..8].*;
            const x_act_chunk: @Vector(8, f64) = x_act[offset..][0..8].*;

            const row_slice = weights[offset .. offset + 8];
            const scale = calculateBlockScale(row_slice);
            const initial_i4 = naiveQuantize(row_slice, scale);

            const optimized_lanes = optimizeQwenRowChunkAvx512(target_y_chunk, x_act_chunk, initial_i4, scale);

            var packed_word: u32 = 0;
            inline for (0..8) |i| {
                const bit_cast_u32 = @as(u32, @bitCast(optimized_lanes[i]));
                packed_word |= ((bit_cast_u32 & 0x0F) << (@as(u5, @intCast(i)) * 4));
            }

            @memcpy(packed_output[stream_idx .. stream_idx + 4], std.mem.asBytes(&packed_word));
            stream_idx += 4;
        }
    }

    try writeFile(allocator, out_path, packed_output);

    std.debug.print("Compiled layer: {d} bytes written (sector aligned {d} KiB)\n", .{ packed_output.len, packed_output.len / 1024 });
}

test "EBM 8-lane AVX-512 coordinate sweep decreases or preserves energy" {
    const target_y: @Vector(8, f64) = .{ 1.2, -0.4, 0.8, -1.5, 0.3, 0.9, -0.7, 0.2 };
    const x_act: @Vector(8, f64) = .{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const initial_i4: @Vector(8, i32) = .{ 4, -1, 3, -5, 1, 3, -2, 1 };
    const scale: f64 = 0.3;

    const initial_energy = evaluateLayerEnergy(target_y, x_act, initial_i4, scale);
    const optimized = optimizeQwenRowChunkAvx512(target_y, x_act, initial_i4, scale);
    const optimized_energy = evaluateLayerEnergy(target_y, x_act, optimized, scale);

    try std.testing.expect(optimized_energy <= initial_energy);
}

test "32-bit word nibble packing round-trip" {
    const lanes: @Vector(8, i32) = .{ 1, -2, 3, -4, 5, -6, 7, -8 };
    var packed_word: u32 = 0;
    inline for (0..8) |i| {
        const bit_cast_u32 = @as(u32, @bitCast(lanes[i]));
        packed_word |= ((bit_cast_u32 & 0x0F) << (@as(u5, @intCast(i)) * 4));
    }

    // Unpack and verify
    inline for (0..8) |i| {
        const raw_nibble = (packed_word >> (@as(u5, @intCast(i)) * 4)) & 0x0F;
        // Sign extend 4-bit to 32-bit signed
        const signed_val: i32 = if (raw_nibble >= 8) @as(i32, @intCast(raw_nibble)) - 16 else @as(i32, @intCast(raw_nibble));
        try std.testing.expectEqual(lanes[i], signed_val);
    }
}
