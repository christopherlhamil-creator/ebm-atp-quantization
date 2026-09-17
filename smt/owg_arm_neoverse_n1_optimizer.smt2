; tests/smt/owg_arm_neoverse_n1_optimizer.smt2
; Z3 Formal SMT2 Model: ARMv8 Neoverse-N1 Microarchitecture Optimization for CHPE Inference
; Hardware profile derived from Phoronix Test Suite empirical measurements (t2a-standard-4)
;
; Hardware Parameters (Neoverse-N1):
; - Cores: 4
; - L1d Cache: 65,536 bytes (64 KiB per core) -> 49,152 B working budget (75% occupancy)
; - Cache Line: 64 bytes
; - SIMD: NEON 128-bit (16 bytes per vector)
; - Vector Registers: 32 (v0 - v31)
; - FMA Pipeline: 2 execution units, 4-cycle latency
;
; Models evaluated:
; - Qwen2.5-3B (Hidden: 2048, Intermediate: 11008, Heads: 16, KV Heads: 2, HeadDim: 128)
; - Universal Quant ladder: 2-bit, 4-bit, 8-bit, 16-bit (BF16)

(set-logic QF_NIA)

; === Hardware Constants ===
(define-fun hw_cores () Int 4)
(define-fun hw_cacheline () Int 64)
(define-fun hw_l1d_budget () Int 49152) ; 48 KiB budget out of 64 KiB
(define-fun hw_simd_bytes () Int 16)    ; 128-bit NEON
(define-fun hw_max_regs () Int 30)      ; 30 vector registers (leaving 2 scratch)
(define-fun hw_pipe_latency () Int 4)   ; 4 cycles

; === Model Constants (Qwen2.5-3B) ===
(define-fun qwen_hidden () Int 2048)
(define-fun qwen_intermediate () Int 11008)
(define-fun qwen_head_dim () Int 128)
(define-fun qwen_kv_heads () Int 2)
(define-fun kv_slot_bytes () Int 512)   ; Cactus FingerprintVector slot

; === Decision Variables for 4-bit GEMV ===
(declare-const cores_used_4bit Int)
(declare-const rows_per_core_4bit Int)
(declare-const tile_k_4bit Int)
(declare-const reg_acc_4bit Int)
(declare-const reg_act_4bit Int)
(declare-const reg_wt_4bit Int)
(declare-const unroll_4bit Int)
(declare-const working_set_l1_4bit Int)

; Constraints for 4-bit:
(assert (= cores_used_4bit hw_cores))
(assert (= (* rows_per_core_4bit cores_used_4bit) qwen_hidden))
; Cache line alignment on output buffer:
(assert (= (mod (* rows_per_core_4bit 4) hw_cacheline) 0))

; tile_k must be multiple of 32 (16 bytes NEON holding 32 4-bit weights):
(assert (> tile_k_4bit 0))
(assert (<= tile_k_4bit qwen_hidden))
(assert (= (mod tile_k_4bit 32) 0))

; Register constraints:
(assert (>= reg_acc_4bit 4))
(assert (>= reg_act_4bit 2))
(assert (>= reg_wt_4bit 2))
(assert (<= (+ reg_acc_4bit (+ reg_act_4bit reg_wt_4bit)) hw_max_regs))

; Unroll factor >= pipe latency to saturate dual FMA:
(assert (>= unroll_4bit hw_pipe_latency))
(assert (<= unroll_4bit 8))

; L1d working set constraint:
; Working set = (reg_acc_4bit * tile_k_4bit / 2) [weights] + (tile_k_4bit * 4) [act] + (reg_acc_4bit * 4) [accum] + 1024 [double-buf KV]
(assert (= working_set_l1_4bit 
   (+ (div (* reg_acc_4bit tile_k_4bit) 2)
   (+ (* tile_k_4bit 4)
   (+ (* reg_acc_4bit 4) 1024)))))

(assert (<= working_set_l1_4bit hw_l1d_budget))

; === Decision Variables for 2-bit GEMV ===
(declare-const tile_k_2bit Int)
(declare-const reg_acc_2bit Int)
(declare-const working_set_l1_2bit Int)

(assert (> tile_k_2bit 0))
(assert (<= tile_k_2bit qwen_hidden))
; 16 bytes NEON holds 64 2-bit weights:
(assert (= (mod tile_k_2bit 64) 0))
(assert (>= reg_acc_2bit 4))
(assert (<= (+ reg_acc_2bit 4) hw_max_regs))
(assert (= working_set_l1_2bit 
   (+ (div (* reg_acc_2bit tile_k_2bit) 4)
   (+ (* tile_k_2bit 4)
   (+ (* reg_acc_2bit 4) 1024)))))
(assert (<= working_set_l1_2bit hw_l1d_budget))

; === Decision Variables for 8-bit GEMV ===
(declare-const tile_k_8bit Int)
(declare-const reg_acc_8bit Int)
(declare-const working_set_l1_8bit Int)

(assert (> tile_k_8bit 0))
(assert (<= tile_k_8bit qwen_hidden))
; 16 bytes NEON holds 16 8-bit weights:
(assert (= (mod tile_k_8bit 16) 0))
(assert (>= reg_acc_8bit 4))
(assert (<= (+ reg_acc_8bit 4) hw_max_regs))
(assert (= working_set_l1_8bit 
   (+ (* reg_acc_8bit tile_k_8bit)
   (+ (* tile_k_8bit 4)
   (+ (* reg_acc_8bit 4) 1024)))))
(assert (<= working_set_l1_8bit hw_l1d_budget))

; === Decision Variables for Full BF16 GEMV ===
(declare-const tile_k_16bit Int)
(declare-const reg_acc_16bit Int)
(declare-const working_set_l1_16bit Int)

(assert (> tile_k_16bit 0))
(assert (<= tile_k_16bit qwen_hidden))
; 16 bytes NEON holds 8 BF16 weights:
(assert (= (mod tile_k_16bit 8) 0))
(assert (>= reg_acc_16bit 4))
(assert (<= (+ reg_acc_16bit 4) hw_max_regs))
(assert (= working_set_l1_16bit 
   (+ (* (* reg_acc_16bit tile_k_16bit) 2)
   (+ (* tile_k_16bit 4)
   (+ (* reg_acc_16bit 4) 1024)))))
(assert (<= working_set_l1_16bit hw_l1d_budget))

(check-sat)
(get-model)
