# tests/test_solve_ebm_hardware_gap.py
import json
import subprocess
import sys
from pathlib import Path


def test_hardware_solver_tau_t2a():
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "solve_ebm_hardware_gap.py"),
        "--arch", "neoverse-n1",
        "--cores", "4",
        "--bandwidth-gbs", "41.845",
        "--target-latency-ms", "50.0",
        "--out", "run/hardware_solver_solution.json",
    ]
    subprocess.check_call(cmd, cwd=str(repo_root))

    out_file = repo_root / "run" / "hardware_solver_solution.json"
    assert out_file.exists(), f"Missing {out_file}"
    with open(out_file) as f:
        sol = json.load(f)

    assert sol["z3_status"] == "SATISFIABLE"
    assert sol["vampire_theorems_proven"] == 15
    assert sol["leo3_theorems_proven"] == 14
    assert sol["ebm_energy"] == 0.0
    assert sol["max_archive_bytes"] <= 2_092_250_000
    assert sol["target_archive_bytes"] <= sol["max_archive_bytes"]
    assert sol["theoretical_dram_floor_ms"] <= 50.00
    assert sol["embed_precision"] == "w4"
    assert sol["attention_precision"]["kv"] == "bf16"
    assert sol["attention_precision"]["q"] == "bf16"
    assert 1 in sol["mlp_precision"]["w8_layers"]
    assert sol["seq2_verified_argmax"] == 2
    assert sol["seq2_verified_margin"] >= 0.40
    assert sol["cos_sim"] >= 0.9999


def test_hardware_solver_ebm_tolerance_contrast():
    repo_root = Path(__file__).resolve().parent.parent
    # 1. Test Loose Tolerance (0.02) -> selects Q4 Attention Q
    loose_out = repo_root / "run" / "hardware_solver_loose.json"
    cmd_loose = [
        sys.executable,
        str(repo_root / "scripts" / "solve_ebm_hardware_gap.py"),
        "--arch", "neoverse-n1",
        "--cores", "4",
        "--bandwidth-gbs", "41.845",
        "--target-latency-ms", "50.0",
        "--ebm-tolerance", "0.02",
        "--out", str(loose_out),
    ]
    subprocess.check_call(cmd_loose, cwd=str(repo_root))
    with open(loose_out) as f:
        sol_loose = json.load(f)
    assert sol_loose["attention_precision"]["q"] == "w4g128"
    assert sol_loose["embed_precision"] == "w8"

    # 2. Test Strict Tolerance (0.0001) -> Christopher's BF16 standard locks Attention Q to BF16
    strict_out = repo_root / "run" / "hardware_solver_strict.json"
    cmd_strict = [
        sys.executable,
        str(repo_root / "scripts" / "solve_ebm_hardware_gap.py"),
        "--arch", "neoverse-n1",
        "--cores", "4",
        "--bandwidth-gbs", "41.845",
        "--target-latency-ms", "50.0",
        "--ebm-tolerance", "0.0001",
        "--out", str(strict_out),
    ]
    subprocess.check_call(cmd_strict, cwd=str(repo_root))
    with open(strict_out) as f:
        sol_strict = json.load(f)
    assert sol_strict["attention_precision"]["q"] == "bf16"
    assert sol_strict["embed_precision"] == "w4"
    assert sol_strict["seq2_verified_argmax"] == 2
    assert sol_strict["seq2_verified_margin"] >= 0.40


def test_hardware_solver_unachievable_constraint():
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "solve_ebm_hardware_gap.py"),
        "--arch", "neoverse-n1",
        "--cores", "4",
        "--bandwidth-gbs", "41.845",
        "--target-latency-ms", "10.0",
        "--out", "run/hardware_solver_unachievable.json",
    ]
    res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert res.returncode != 0, f"Expected non-zero exit code, got {res.returncode}. Output: {res.stdout}"
    assert "Z3 SMT constraint violation" in res.stderr or "EBM energy" in res.stderr


def test_hardware_solver_neoverse_n1_bf16():
    repo_root = Path(__file__).resolve().parent.parent
    out_file = repo_root / "run" / "hardware_solver_bf16_solution.json"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "solve_ebm_hardware_gap.py"),
        "--profile", "neoverse_n1_bf16",
        "--out", str(out_file),
    ]
    subprocess.check_call(cmd, cwd=str(repo_root))

    assert out_file.exists(), f"Missing {out_file}"
    with open(out_file) as f:
        sol = json.load(f)

    assert sol["z3_status"] == "SATISFIABLE"
    assert sol["vampire_theorems_proven"] == 15
    assert sol["leo3_theorems_proven"] == 14
    assert sol["ebm_energy"] == 0.0
    assert sol["model"] == "Qwen2.5-3B-Instruct.bf16.raw.chpe"
    assert sol["current_decode_ms"] == 210.02
    assert sol["llama_baseline_ms"] == 165.60
    assert 44.0 <= sol["remaining_gap_ms"] <= 45.0
    assert sol["squeezed_target_ms"] <= 155.00
    assert sol["squeezed_target_tok_s"] >= 6.45
    assert sol["theoretical_dram_floor_ms"] <= 148.00
    assert sol["embed_precision"] == "bf16"
    assert sol["attention_precision"]["q"] == "bf16"
    assert sol["attention_precision"]["kv"] == "bf16"
    assert sol["attention_precision"]["o"] == "bf16"
    assert sol["mlp_precision"]["default"] == "bf16"
    assert sol["sdot_vector_regs"] == 9
    assert sol["sdot_vector_regs"] <= 32
    assert sol["working_set_bytes"] <= 49152
    assert sol["cos_sim"] >= 0.999999
    assert sol["seq2_verified_argmax"] == 2
    assert sol["token0_verified_argmax"] == 50994


if __name__ == "__main__":
    test_hardware_solver_tau_t2a()
    print("test_hardware_solver_tau_t2a: PASS")
    test_hardware_solver_ebm_tolerance_contrast()
    print("test_hardware_solver_ebm_tolerance_contrast: PASS")
    test_hardware_solver_unachievable_constraint()
    print("test_hardware_solver_unachievable_constraint: PASS")
    test_hardware_solver_neoverse_n1_bf16()
    print("test_hardware_solver_neoverse_n1_bf16: PASS")

