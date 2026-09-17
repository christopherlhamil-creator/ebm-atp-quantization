#!/usr/bin/env python3
"""Tests for Zero-Copy Sector Splice."""
import os
import subprocess
import sys

def test_splice_chpe_slices():
    os.makedirs("db/layers", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Create two 16 KiB dummy slices
    for l in (0, 1):
        with open(f"db/layers/layer_{l}.chpe", "wb") as f:
            f.write(b"\xAA" * 16384)

    cmd = [sys.executable, "scripts/splice_ebm_chpe.py", "--layer-count", "2", "--out-file", "models/test_model.chpe"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Splice failed with stderr: {result.stderr}"

    assert os.path.exists("models/test_model.chpe")
    # 4096 byte header + 2 * 16384 bytes
    expected_size = 4096 + (2 * 16384)
    actual_size = os.path.getsize("models/test_model.chpe")
    assert actual_size == expected_size, f"Size mismatch: got {actual_size}, expected {expected_size}"

if __name__ == "__main__":
    test_splice_chpe_slices()
    print("test_splice_chpe_slices: PASS")
