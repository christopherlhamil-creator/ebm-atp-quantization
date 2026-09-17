# 🏛️ EBM × ATP Automated 36-Layer Weight Compilation Architecture

**Date:** 2026-09-14  
**Author:** Christopher Hamil & Antigravity  
**Status:** Approved Design Spec  
**Deliverable Path:** `docs/superpowers/specs/2026-09-14-ebm-atp-layer-quantization-design.md`  

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Failure Mode
The Qwen2.5-3B-Instruct `.chpe` post-training quantization benchmark revealed a severe 36-layer error cascade:
- **Baseline Failure**: Final token **Cosine Similarity collapsed to 0.2667** (target $> 0.95$), and **RMS Error rose to 4.333** against the BF16 Oracle baseline.
- **Layer-wise Divergence**: Layer 0 starts with strong alignment (**0.9398** Cosine Similarity), but compounding quantization noise and variance attenuation across the 36 deep residual blocks progressively collapses the final token distribution.
- **Root Cause**: Raw affine round-to-nearest (RTN) quantization minimizes local Euclidean weight distance independently, squashing MLP matrix variance ($\text{Var}(X \hat{W}) < \text{Var}(X W)$). Across 36 sequential layers, this variance dampening acts as an unintended shrinkage prior, destroying the signal-to-noise ratio.

### 1.2 Target System
This design specifies an end-to-end automated compilation system that:
1. Validates packing invertibility, tile concurrency, and transformer scale congruence through a **Pre-Flight Formal Solver Gate** (Z3, Vampire, Leo-III).
2. Optimizes 4-bit weights row-by-row in CPU vector registers using an **AVX-512 Energy-Based Model (EBM) Compiler** that prioritizes activation variance preservation over raw Euclidean distance.
3. Drives an **Autonomous 36-Layer Sequential Down-Walk** with crash-proof layer-slice checkpointing, forcing downstream weights to actively absorb and invert upstream quantization noise.
4. Provides **Zero-Copy Sector Assembly** into the final 1.800 GB `.chpe` file under 1 second.
5. Operates with **Cross-Host Portability**, running functional tests on Pop (AVX2 fallback) and full-velocity burn on Brandys (native Zen 4 AVX-512).

---

## 2. End-to-End System Architecture

```
                                  [ FORMAL VERIFICATION ]
                                  scripts/atp_simulation_gate.py
                                  ├─ Z3: Bitvector packing & 16 KiB sector divisor
                                  ├─ Vampire: 16-thread exclusive tile ownership
                                  └─ Leo-III: Higher-order scale congruence
                                            │
                                            ▼ (Exit 0; cite_key logged to scars.sqlite)
                                  [ AUTOMATION RUNNER ]
                                  scripts/orchestrate_ebm_layers.py
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
       For Layer l = 0                                           For Layer l = 1..35
       X[0] = Clean prompt tokens                                X[l] = Forward pass through
       Y*[0] = Oracle Layer 0                                           already-quantized slices
                                                                        db/layers/layer_0..l-1.chpe
                                                                 Y*[l] = Oracle Layer l
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                              [ ZIG AVX-512 COMPILER ]
                              zig-out/bin/ebm_layer_compiler
                              ├─ Slices 8 weights: @Vector(8, f64)
                              ├─ In-register vfmadd231pd FMA projection
                              ├─ Outlier loss: delta^2 penalizes variance loss
                              ├─ 8-lane coordinate sweep: tests w_i ± 1
                              └─ Packs 8 nibbles into u32 word via bitshifts
                                            │
                                            ▼
                              [ DISK PERSISTENCE & CHECKPOINT ]
                              ├─ Writes: db/layers/layer_{l}.chpe (16 KiB sector aligned)
                              └─ Commits: run/ebm_checkpoint.json
                                            │
                                            ▼ (After Layer 35 completes)
                              [ ZERO-COPY SECTOR SPLICE ]
                              scripts/splice_ebm_chpe.py
                              ├─ Concatenates layer_0..35.chpe via sendfile() (< 1s)
                              └─ Output: models/qwen2.5-3b-ebm.chpe (1.800 GB)
                                            │
                                            ▼
                              [ MODEL QA VERIFICATION ]
                              mill_qwen3b_fwd.py
                              ├─ Argmax parity check at Token 0
                              ├─ Cosine Similarity > 0.95
                              └─ RMS Error < 1.0
```

---

## 3. Detailed Component Specifications

### 3.1 Component 1: Pre-Flight ATP Simulation Gate (`scripts/atp_simulation_gate.py`)
In accordance with Invariant A-5 and disk governance (`BLUEPRINT-20260912-FORMAL-SOLVER-ASSIGNMENT-Z3-VAMPIRE-LEO.md`), formal provers run strictly out-of-process. The inner optimization loop never halts on a solver.

1. **Z3 (SMT-LIB2 Solver)**:
   - Evaluates bitvector logic verifying that:
     $$\forall w \in [-8, 7]^8, \quad \text{unpack}(\text{pack}(w)) = w$$
     where $\text{pack}(w) = \bigvee_{i=0}^7 (w_i \ \& \ 0\text{x0F}) \ll (i \times 4)$.
   - Verifies integer geometry divisibility: $(R \times C) / 2 \pmod{16384} \equiv 0$ across all 7 projection matrices per layer (q, k, v, o, gate, up, down).
   - Assertion requirement: Z3 returns `unsat` on negation within a 30-second timeout.
2. **Vampire (First-Order ATP — TPTP FOF)**:
   - Evaluates concurrency axioms ensuring that multi-threaded SIMD workers operate with disjoint tile ownership without mutual exclusion locks on tile memory.
   - Assertion requirement: Vampire returns `Theorem` / `Refutation` within 30 seconds.
3. **Leo-III (Higher-Order ATP — TPTP THF)**:
   - Evaluates higher-order modal formulas asserting that downstream error absorption in residual connections preserves total activation norm bounds across deep layers.
   - Assertion requirement: Leo-III returns `Theorem` within 30 seconds.
4. **Failure & Success Behaviors**:
   - If any solver fails, the script outputs `inventory/EVAL-20260914-SOLVER-FAIL.md` detailing the countermodel and halts with exit code 1.
   - On success, the script records the 16-byte `cite_key` (`SHA-256(formula_bytes)[0:16]`) into `scars.sqlite` and touches `run/atp_gate.passed`.

---

### 3.2 Component 2: Register-Bound Zig EBM Compiler (`src/ebm_layer_compiler.zig`)
The weight compiler is written in Zig (0.13+ / 0.17 toolchain) and compiled with `-Doptimize=ReleaseFast`.

1. **Input Interface**:
   - Reads unquantized weights ($f64$), calibration input activations ($X_{\text{calib}}$ in $f64$), and target projections ($Y_{\text{target}}$ in $f64$) for matrix dimensions $[R, C]$.
   - $C$ must be a multiple of 8.
2. **Energy Metric & In-Register Projection**:
   ```zig
   inline fn evaluateLayerEnergy(
       y_true: @Vector(8, f64),
       x_act: @Vector(8, f64),
       w_cand: @Vector(8, i32),
       scale: f64,
   ) f64 {
       const w_f64: @Vector(8, f64) = @floatFromInt(w_cand);
       const vec_scale: @Vector(8, f64) = @splat(scale);
       const w_dequant = w_f64 * vec_scale;
       const y_approx = x_act * w_dequant; // Maps to vfmadd231pd FMA
       const delta = y_true - y_approx;
       const squared_loss = delta * delta; // Variance protection
       return @reduce(.Add, squared_loss);
   }
   ```
3. **Greedy Coordinate Search**:
   - For each 8-weight chunk, block scale is initialized: $\text{scale} = \max(|v|) / 7.0$.
   - Initial integer state is seeded via naive RTN: $\text{initial\_i4}_i = \text{clamp}(\text{round}(v_i / \text{scale}), -8, 7)$.
   - Evaluates baseline energy. Unrolls an 8-lane loop testing $w_i \pm 1$ per lane in the range $[-8, +7]$. Steps that lower the squared residual energy are greedily accepted.
4. **32-Bit Word Serialization**:
   - Packs 8 4-bit signed integers into a single 32-bit unsigned integer:
     ```zig
     var packed_word: u32 = 0;
     inline for (0..8) |i| {
         const bit_cast_u32 = @as(u32, @bitCast(optimized_lanes[i]));
         packed_word |= ((bit_cast_u32 & 0x0F) << (@as(u5, @intCast(i)) * 4));
     }
     @memcpy(packed_output[stream_idx .. stream_idx + 4], std.mem.asBytes(&packed_word));
     stream_idx += 4;
     ```
5. **Host Portability**:
   - On **Brandys (Ryzen 7 8700F / Zen 4)**: Compiled with `-Dcpu=native`. Zig lowers `@Vector(8, f64)` directly into native 512-bit `zmm` registers using EVEX `vfmadd231pd` instructions.
   - On **Pop (Coffee Lake)**: Compiled with default target. LLVM cleanly lowers `@Vector(8, f64)` into dual 256-bit `ymm` AVX2 instructions without generating illegal instruction traps (`SIGILL`).

---

### 3.3 Component 3: Autonomous Resumable Layer Orchestrator (`scripts/orchestrate_ebm_layers.py`)
Drives the 36-layer calibration down-walk.

1. **Calibration Data**:
   - Fixed sequence of 256 tokens consisting of deterministic logic questions, synthetic code syntax, and function definitions.
2. **Sequential Execution**:
   - For Layer $l \in [0 \dots 35]$:
     - Check `run/ebm_checkpoint.json`. If Layer $l$ is marked complete and `db/layers/layer_{l}.chpe` exists with valid size, advance to $l+1$.
     - If $l = 0$: $X_{\text{calib}}$ is the initial prompt embedding.
     - If $l > 0$: Propagate calibration tokens through quantized layers $0 \dots l-1$ using the slices stored in `db/layers/` to obtain corrupted input $\tilde{X}_l$.
     - Compute ground-truth projection $Y_l^*$ by passing tokens through the unquantized Oracle model.
     - Execute `ebm_layer_compiler` for Layer $l$, emitting `db/layers/layer_{l}.chpe`.
     - Update `run/ebm_checkpoint.json` atomically with execution duration, initial energy, and optimized energy.
     - Deallocate Layer $l$ unquantized weights from RAM before starting Layer $l+1$.

---

### 3.4 Component 4: Zero-Copy Sector Splice (`scripts/splice_ebm_chpe.py`)
Assembles the complete model upon completion of Layer 35.

1. **Header & Sector Law**:
   - The `.chpe` format consists of a 4,096-byte metadata header followed by sequential 16,384-byte sector tiles.
   - Each layer slice in `db/layers/layer_{l}.chpe` is an exact multiple of 16,384 bytes.
2. **Assembly via `sendfile`**:
   - The script creates `models/qwen2.5-3b-ebm.chpe`.
   - Writes the global `.chpe` header.
   - Iterates through `layer_0.chpe` to `layer_35.chpe`, invoking `os.sendfile()` to pipe data directly between file descriptors at kernel level.
   - Total runtime for 1.800 GB assembly: $< 1.0$ second.

---

### 3.5 Component 5: Model QA Acceptance Gate (`mill_qwen3b_fwd.py`)
Validates model correctness against the BF16 Oracle baseline.

1. **Acceptance Criteria**:
   - **Argmax Match**: Predicted token 0 must match the Oracle baseline bit-for-bit.
   - **Cosine Similarity**: Final layer activation similarity must exceed **0.95** (reversing the 0.2667 failure).
   - **RMS Error**: Root-Mean-Square error across output logits must drop below **1.0** (down from 4.333).
   - **Determinism**: Re-running the forward decode produces identical bitwise logits.

---

## 4. Cross-Host Workflow & Execution Architecture

### 4.1 Host Targets & ISA Scaling
1. **Host A (Baseline / AVX2 / ARM NEON)**:
   - Evaluates functional correctness, unit tests, and SMT/ATP verification gates.
2. **Host B (High-Throughput AVX-512 / Zen 4 / Neoverse-N1)**:
   - Runs full-velocity layer compilation with `@Vector(8, f64)` mapped directly to 512-bit vector registers.

### 4.2 Step-by-Step Execution Sequence
1. **Phase A (Verification & Functional Smoke Tests)**:
   - Run SMT2, Vampire, and Leo-III verification gates (`scripts/solve_ebm_hardware_gap.py`).
   - Verify bitvector pack/unpack roundtrips in Zig (`zig build test`).
   - Smoke test coordinate sweep on Layer 0 and Layer 1.
2. **Phase B (Full Model Down-Walk)**:
   - Build native optimized compiler: `zig build -Doptimize=ReleaseFast`.
   - Run `python3 scripts/orchestrate_ebm_layers.py`.
   - The runner executes all layers sequentially, persisting slices to `db/layers/`.
   - Assemble final artifact: `python3 scripts/splice_ebm_chpe.py`.
   - Verify final model against FP16/BF16 baseline (Cosine Similarity > 0.95, RMS Error < 1.0).

---

## 5. Risk Assessment & Mitigations

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| **Local Minima in Coordinate Search** | Residual error plateaus before reaching 0.95 Cosine Sim | Run 2 relaxation passes over the 8 lanes in `optimizeQwenRowChunkAvx512` if delta reduction is $< 30\%$. |
| **Calibration Prompt Overfitting** | Strong logic accuracy but degraded general text | Calibration set blends logic questions with general vocabulary sentences. |
| **System Crash / Out of Memory** | Lost computation during multi-hour compilation | Resumable JSON checkpointing and per-layer disk slices; only 1 layer active in RAM at a time. |
| **Cross-Host Compilation Failure** | Binary compiled on Brandys fails on Pop with `SIGILL` | Explicit build targets: `-Dcpu=native` only on Brandys; default host build on Pop. |
