; Expected: unsat
; pack q_even low nibble, q_odd high. unpack recovers. mill_pack_w2_f64.py. bitwidth=4.
(set-logic QF_BV)
(declare-const q_even (_ BitVec 4))
(declare-const q_odd (_ BitVec 4))
(define-fun packed_byte () (_ BitVec 8) (concat q_odd q_even))
(define-fun u_even () (_ BitVec 4) ((_ extract 3 0) packed_byte))
(define-fun u_odd () (_ BitVec 4) ((_ extract 7 4) packed_byte))
(assert (or (distinct q_even u_even) (distinct q_odd u_odd)))
(check-sat)
