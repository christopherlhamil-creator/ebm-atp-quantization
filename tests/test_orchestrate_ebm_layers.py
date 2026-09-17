#!/usr/bin/env python3
"""Tests for Resumable 36-Layer EBM Orchestrator."""
import json
import os
import subprocess
import sys

def test_layer_orchestrator_two_layer_mock():
    if os.path.exists("run/ebm_checkpoint.json"):
        os.remove("run/ebm_checkpoint.json")
    for l in (0, 1):
        p = f"db/layers/layer_{l}.chpe"
        if os.path.exists(p):
            os.remove(p)

    cmd = [sys.executable, "scripts/orchestrate_ebm_layers.py", "--max-layers", "2", "--mock-weights"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Orchestrator failed with stderr: {result.stderr}"

    assert os.path.exists("db/layers/layer_0.chpe"), "layer_0.chpe missing"
    assert os.path.exists("db/layers/layer_1.chpe"), "layer_1.chpe missing"
    assert os.path.exists("run/ebm_checkpoint.json"), "checkpoint missing"

    with open("run/ebm_checkpoint.json") as f:
        checkpoint = json.load(f)
    assert 0 in checkpoint["completed_layers"], "Layer 0 not marked complete"
    assert 1 in checkpoint["completed_layers"], "Layer 1 not marked complete"

if __name__ == "__main__":
    test_layer_orchestrator_two_layer_mock()
    print("test_layer_orchestrator_two_layer_mock: PASS")
