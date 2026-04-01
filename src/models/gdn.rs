//! GatedDeltaNet recurrence and causal conv1d for Qwen3.5.
//!
//! The delta rule recurrence maintains a state matrix S of shape [B, Hv, Dv, Dk]:
//!   decay = exp(-exp(A_log) * softplus(a + dt_bias))
//!   S = decay * S + beta * (k^T @ v)   (outer product update)
//!   output = S @ q                       (state readout)

use mlx_rs::ops;
use mlx_rs::Array;

/// Causal conv1d update for decode (single token).
///
/// conv_state: [B, D, kernel_size-1] — sliding window of previous inputs
/// x: [B, 1, D] — current input
/// conv_weight: [D, 1, kernel_size] — depthwise conv weights
///
/// Returns (output, new_conv_state).
pub fn causal_conv1d_update(
    x: &Array,
    conv_state: &Array,
    conv_weight: &Array,
) -> (Array, Array) {
    let b = x.dim(0);
    let d = x.dim(2);

    // x: [B, 1, D] -> [B, D, 1]
    let x_t = ops::transpose(x, &[0, 2, 1][..]).unwrap();

    // Shift conv state: drop oldest, append new
    let kernel_size_minus_1 = conv_state.dim(2);
    let new_state = if kernel_size_minus_1 > 1 {
        let shifted = ops::slice(conv_state, &[0..i32::MAX, 0..i32::MAX, 1..i32::MAX]).unwrap();
        ops::concatenate(&[&shifted, &x_t], 2).unwrap()
    } else {
        x_t.clone()
    };

    // Compute conv: sum(state * weight) across kernel dim
    // conv_weight: [D, 1, kernel_size] -> [D, kernel_size] for element-wise
    let w = ops::reshape(conv_weight, &[d, kernel_size_minus_1 + 1]).unwrap();
    // state with current: [B, D, kernel_size]
    let full_state = ops::concatenate(&[conv_state, &x_t], 2).unwrap();
    // Element-wise multiply and sum across last dim
    let product = ops::multiply(&full_state, &w).unwrap();
    let output = ops::sum_axis(&product, &[-1][..], false).unwrap();
    // output: [B, D] -> [B, 1, D]
    let output = ops::expand_dims(&output, &[1][..]).unwrap();

    (output, new_state)
}

/// Gated delta rule recurrence for a single timestep.
///
/// q: [B, Hk, Dk]
/// k: [B, Hk, Dk]
/// v: [B, Hv, Dv]
/// beta: [B, Hk, Dk] — write gate
/// decay: [B, Hk, 1] — forget gate (scalar per head)
/// state: [B, Hv, Dv, Dk] — recurrent state matrix
///
/// Returns (output: [B, Hv, Dv], new_state: [B, Hv, Dv, Dk]).
pub fn gated_delta_update_single(
    q: &Array,
    k: &Array,
    v: &Array,
    beta: &Array,
    decay: &Array,
    state: &Array,
) -> (Array, Array) {
    let hk = k.dim(1);
    let hv = v.dim(1);

    // GQA: if hv > hk, we need to repeat k/q across value heads
    let heads_per_group = hv as i32 / hk as i32;

    // Expand k, q, beta, decay to match value heads if needed
    let (q, k, beta, decay) = if heads_per_group > 1 {
        (
            ops::repeat(&ops::expand_dims(q, &[2][..]).unwrap(), 2, heads_per_group as usize).unwrap(),
            ops::repeat(&ops::expand_dims(k, &[2][..]).unwrap(), 2, heads_per_group as usize).unwrap(),
            ops::repeat(&ops::expand_dims(beta, &[2][..]).unwrap(), 2, heads_per_group as usize).unwrap(),
            ops::repeat(&ops::expand_dims(decay, &[2][..]).unwrap(), 2, heads_per_group as usize).unwrap(),
        )
    } else {
        (q.clone(), k.clone(), beta.clone(), decay.clone())
    };

    // Now shapes: q,k,beta: [B, Hv, Dk], decay: [B, Hv, 1], v: [B, Hv, Dv]
    // state: [B, Hv, Dv, Dk]

    // Decay the state
    // decay: [B, Hv, 1] -> [B, Hv, 1, 1] for broadcasting with [B, Hv, Dv, Dk]
    let decay_expanded = ops::expand_dims(&decay, &[-1][..]).unwrap();
    let decayed_state = ops::multiply(state, &decay_expanded).unwrap();

    // Outer product update: beta * (v^T @ k) -> actually it's v * k^T
    // v: [B, Hv, Dv] -> [B, Hv, Dv, 1]
    // beta_k: [B, Hv, 1, Dk] = beta * k
    let v_col = ops::expand_dims(&v, &[-1][..]).unwrap();
    let beta_k = ops::multiply(&beta, &k).unwrap();
    let beta_k_row = ops::expand_dims(&beta_k, &[-2][..]).unwrap();
    let update = ops::multiply(&v_col, &beta_k_row).unwrap();

    // New state = decay * state + update
    let new_state = ops::add(&decayed_state, &update).unwrap();

    // Readout: output = state @ q
    // state: [B, Hv, Dv, Dk], q: [B, Hv, Dk] -> [B, Hv, Dk, 1]
    let q_col = ops::expand_dims(&q, &[-1][..]).unwrap();
    let output = ops::matmul(&new_state, &q_col).unwrap();
    // output: [B, Hv, Dv, 1] -> [B, Hv, Dv]
    let output = ops::squeeze(&output, &[-1][..]).unwrap();

    (output, new_state)
}

/// Compute decay gate from a, a_log, dt_bias.
/// decay = exp(-exp(A_log) * softplus(a + dt_bias))
pub fn compute_decay(a: &Array, a_log: &Array, dt_bias: &Array) -> Array {
    let a_shifted = ops::add(a, dt_bias).unwrap();
    let softplus_a = ops::log(&ops::add(&ops::exp(&a_shifted).unwrap(), &Array::from_float(1.0)).unwrap()).unwrap();
    let exp_a_log = ops::exp(a_log).unwrap();
    let exponent = ops::multiply(&exp_a_log, &softplus_a).unwrap();
    ops::exp(&ops::negative(&exponent).unwrap()).unwrap()
}
