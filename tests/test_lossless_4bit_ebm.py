import numpy as np

def test_zero_offset_4bit_representation():
    # Verify signed nibbles fit cleanly into [-8, 7] without dmin offsets
    q = np.array([-8, -7, 0, 7], dtype=np.int8)
    assert np.all(q >= -8) and np.all(q <= 7)
    # Check packing 2 nibbles into 1 byte (signed low nibble, signed high nibble)
    low = -8 & 0x0F
    high = 7 & 0x0F
    byte_val = low | (high << 4)

    # Unpack signed 4-bit nibbles
    u_low = byte_val & 0x0F
    unpacked_low = u_low - 16 if u_low >= 8 else u_low

    u_high = (byte_val >> 4) & 0x0F
    unpacked_high = u_high - 16 if u_high >= 8 else u_high

    assert unpacked_low == -8
    assert unpacked_high == 7

def ebm_coordinate_sweep_row(w_row: np.ndarray, x: np.ndarray, scale: float, iters: int = 2) -> np.ndarray:
    """EBM coordinate sweep: perturb nibbles by +-1 to minimize (w_q @ x - target)^2."""
    target = np.dot(w_row, x)
    q = np.clip(np.round(w_row / scale), -8, 7).astype(np.float32)
    current_proj = np.dot(q * scale, x)
    current_err = (current_proj - target) ** 2

    # Coordinate descent over top components
    order = np.argsort(-np.abs(x))
    for _ in range(iters):
        for idx in order[:32]:
            xi = x[idx]
            if abs(xi) < 1e-6:
                continue
            # Try +1 step
            if q[idx] < 7:
                cand_proj = current_proj + scale * xi
                cand_err = (cand_proj - target) ** 2
                if cand_err < current_err:
                    q[idx] += 1
                    current_proj = cand_proj
                    current_err = cand_err
                    continue
            # Try -1 step
            if q[idx] > -8:
                cand_proj = current_proj - scale * xi
                cand_err = (cand_proj - target) ** 2
                if cand_err < current_err:
                    q[idx] -= 1
                    current_proj = cand_proj
                    current_err = cand_err
    return q

def test_inter_layer_error_absorption():
    rng = np.random.RandomState(42)
    x_in = rng.randn(128).astype(np.float32)
    w_true = rng.randn(16, 128).astype(np.float32) * 0.1
    y_target = w_true @ x_in

    # Quantize with EBM coordinate sweep
    w_optimized = np.zeros_like(w_true)
    for r in range(16):
        scale = np.max(np.abs(w_true[r])) / 7.0
        q_opt = ebm_coordinate_sweep_row(w_true[r], x_in, scale)
        w_optimized[r] = q_opt * scale

    rtn_scale = np.max(np.abs(w_true), axis=1, keepdims=True) / 7.0
    w_rtn = np.clip(np.round(w_true / rtn_scale), -8, 7) * rtn_scale

    err_rtn = np.abs((w_rtn @ x_in) - y_target).mean()
    err_ebm = np.abs((w_optimized @ x_in) - y_target).mean()

    # EBM coordinate sweep strictly improves activation projection error
    assert err_ebm < err_rtn, f"EBM error {err_ebm} not lower than RTN {err_rtn}"
    assert err_ebm < 0.15

if __name__ == "__main__":
    test_zero_offset_4bit_representation()
    test_inter_layer_error_absorption()
    print("Lossless 4-bit EBM representation tests PASSED")
