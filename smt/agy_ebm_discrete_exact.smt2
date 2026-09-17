; agy_ebm_discrete_exact.smt2
; Expected: sat
; Adversarial proof for Claim 5: EBM discrete bits against energy target from walk.
; Energy target e_target is an exact discrete integer from the walk.
; Z3 finds discrete bit configurations satisfying total_energy = e_target with ZERO rounding residual.
; Discrete bits b_i in {0, 1}. Not gradient descent. Not weight training.
(set-logic QF_BV)
(declare-const b0 (_ BitVec 1))
(declare-const b1 (_ BitVec 1))
(declare-const b2 (_ BitVec 1))
(declare-const b3 (_ BitVec 1))

; Discrete bit weights from walk (exact integer/BV weights)
(define-fun w0 () (_ BitVec 16) #x0003)
(define-fun w1 () (_ BitVec 16) #x0007)
(define-fun w2 () (_ BitVec 16) #x000e)
(define-fun w3 () (_ BitVec 16) #x001d)

; Couplings J_ij
(define-fun j01 () (_ BitVec 16) #x0005)
(define-fun j23 () (_ BitVec 16) #x000b)

; Energy target from walk (exact target 44 = 0x002c)
(define-fun e_target () (_ BitVec 16) #x002c)

; Linear energy contribution
(define-fun e_lin () (_ BitVec 16)
  (bvadd
    (ite (= b0 #b1) w0 #x0000)
    (ite (= b1 #b1) w1 #x0000)
    (ite (= b2 #b1) w2 #x0000)
    (ite (= b3 #b1) w3 #x0000)))

; Coupling energy contribution
(define-fun e_coup () (_ BitVec 16)
  (bvadd
    (ite (and (= b0 #b1) (= b1 #b1)) j01 #x0000)
    (ite (and (= b2 #b1) (= b3 #b1)) j23 #x0000)))

(define-fun total_energy () (_ BitVec 16)
  (bvadd e_lin e_coup))

; Assert exact match with target (zero rounding residual)
(assert (= total_energy e_target))

(check-sat)
