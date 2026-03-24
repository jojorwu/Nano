//! Centralized physics logic for the Spiking Neural Network engine.
//!
//! Numerical consistency between CPU and GPU backends is maintained through
//! fixed-point integer arithmetic.
//! The base scale is defined as SCALE (1024), representing 1.0 in floating-point math.

use crate::{IValue};

/// Efficient integer approximation of the sigmoidal gating function used for NMDA-like responses.
/// f(x) = SCALE / (1 + exp(-k*(x - theta)))
/// Active region is roughly [-512, 512] around the threshold.
#[inline]
pub fn sigmoid_gate_approx(input: IValue, theta: IValue) -> i32 {
    let diff = input - theta;
    if diff > 512 { return 1024; }
    let val = diff + 512;
    if val < 64 { return 64; } // Minimal leakage to prevent dead gradients
    val as i32
}

/// Applies axonal delay and dendritic gating to a synaptic weight.
#[inline]
pub fn apply_synaptic_gating(weight: IValue, gate: IValue) -> IValue {
    ((weight as i64 * gate as i64) >> 10) as i32
}

/// Calculates the decay factor for Leaky Integrate-and-Fire (LIF) dynamics,
/// modulated by local liquid current (LLIF).
#[inline]
pub fn calculate_dynamic_decay(base_decay: IValue, proximal: IValue, distal: IValue) -> IValue {
    let liquid_mod = ((proximal.abs() + distal.abs()) * 10) >> 10;
    (base_decay - liquid_mod).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_sigmoid_gate_bounds(input in -2000i32..2000, theta in -2000i32..2000) {
            let res = sigmoid_gate_approx(input, theta);
            assert!(res >= 64, "Result {} below minimum 64 for input {}, theta {}", res, input, theta);
            assert!(res <= 1024, "Result {} above maximum 1024 for input {}, theta {}", res, input, theta);
        }

        #[test]
        fn test_sigmoid_gate_monotonicity(input in -1000i32..1000, theta in -1000i32..1000) {
            let res1 = sigmoid_gate_approx(input, theta);
            let res2 = sigmoid_gate_approx(input + 10, theta);
            assert!(res2 >= res1, "Monotonicity violated: f({}) = {} > f({}) = {}", input, res1, input + 10, res2);
        }
    }
}
