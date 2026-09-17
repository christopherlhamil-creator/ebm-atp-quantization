; Expected: sat
; Z3 integer inequalities. Radix 2^17. 1e-4 gate floors to 13.
; L5 Path B 1 ULP = 2^-12; 2^-12 * 2^17 = 32, which is E_drift inf (32 >= 13).
(set-logic QF_LIA)
(define-fun radix () Int 131072)
(define-fun drift_thresh () Int 13)
(define-fun l5_ulp_scaled () Int 32)
(assert (= radix 131072))
(assert (= (* 10000 drift_thresh) 130000))
(assert (< 130000 radix))
(assert (>= (* 10000 14) radix))
(assert (= l5_ulp_scaled 32))
(assert (>= l5_ulp_scaled drift_thresh))
(check-sat)
(get-value (radix drift_thresh l5_ulp_scaled))
