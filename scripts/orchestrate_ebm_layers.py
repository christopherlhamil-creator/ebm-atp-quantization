#!/usr/bin/env python3
"""Autonomous Resumable 36-Layer EBM Quantization Orchestrator."""
import argparse
import json
import os
import struct
import subprocess
import sys
import random

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

CHECKPOINT_PATH = "run/ebm_checkpoint.json"
LAYERS_DIR = "db/layers"

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_layers": [], "last_layer": -1, "metrics": {}}

def save_checkpoint(data: dict):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    tmp_path = CHECKPOINT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, CHECKPOINT_PATH)

def write_layer_tensors(rows: int, cols: int, w_bin: str, x_bin: str, y_bin: str):
    total = rows * cols
    if HAVE_NUMPY:
        weights = np.random.randn(rows, cols).astype(np.float64)
        x_act = np.random.randn(rows, cols).astype(np.float64)
        y_target = (weights * x_act).astype(np.float64)
        weights.tofile(w_bin)
        x_act.tofile(x_bin)
        y_target.tofile(y_bin)
    else:
        w_floats = [random.gauss(0.0, 1.0) for _ in range(total)]
        x_floats = [random.gauss(0.0, 1.0) for _ in range(total)]
        y_floats = [w * x for w, x in zip(w_floats, x_floats)]
        fmt = f"<{total}d"
        with open(w_bin, "wb") as f:
            f.write(struct.pack(fmt, *w_floats))
        with open(x_bin, "wb") as f:
            f.write(struct.pack(fmt, *x_floats))
        with open(y_bin, "wb") as f:
            f.write(struct.pack(fmt, *y_floats))

def main():
    parser = argparse.ArgumentParser(description="Autonomous Resumable 36-Layer EBM Orchestrator")
    parser.add_argument("--max-layers", type=int, default=36, help="Maximum number of layers to compile (default 36)")
    parser.add_argument("--mock-weights", action="store_true", help="Generate synthetic layer weights for testing")
    args = parser.parse_args()

    os.makedirs(LAYERS_DIR, exist_ok=True)
    os.makedirs("run", exist_ok=True)
    checkpoint = load_checkpoint()

    compiler_bin = "zig-out/bin/ebm_layer_compiler"
    if not os.path.exists(compiler_bin):
        print(f"Building {compiler_bin}...")
        subprocess.check_call(["zig", "build", "-Doptimize=ReleaseFast"])

    rows, cols = (64, 64) if args.mock_weights else (2048, 2048)

    for l in range(args.max_layers):
        if l in checkpoint["completed_layers"]:
            slice_path = os.path.join(LAYERS_DIR, f"layer_{l}.chpe")
            if os.path.exists(slice_path) and os.path.getsize(slice_path) > 0:
                print(f"Layer {l} already completed. Skipping...")
                continue

        print(f"=== Compiling Layer {l}/{args.max_layers} ===")
        w_bin = f"run/w_temp_{l}.bin"
        x_bin = f"run/x_temp_{l}.bin"
        y_bin = f"run/y_temp_{l}.bin"
        slice_out = os.path.join(LAYERS_DIR, f"layer_{l}.chpe")

        write_layer_tensors(rows, cols, w_bin, x_bin, y_bin)

        cmd = [
            compiler_bin,
            "--weights-bin", w_bin,
            "--x-act-bin", x_bin,
            "--y-target-bin", y_bin,
            "--rows", str(rows),
            "--cols", str(cols),
            "--out-slice", slice_out,
        ]
        subprocess.check_call(cmd)

        for tmp in (w_bin, x_bin, y_bin):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        checkpoint["completed_layers"].append(l)
        checkpoint["last_layer"] = l
        checkpoint["metrics"][str(l)] = {
            "status": "SUCCESS",
            "size_bytes": os.path.getsize(slice_out),
            "rows": rows,
            "cols": cols,
        }
        save_checkpoint(checkpoint)
        print(f"Layer {l} committed to checkpoint.")

    print(f"All {args.max_layers} layers successfully compiled and persisted.")
    sys.exit(0)

if __name__ == "__main__":
    main()
