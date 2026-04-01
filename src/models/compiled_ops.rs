//! Compiled MLX operations — fused closures for reduced GPU dispatch count.

use mlx_rs::Array;
use mlx_rs::error::Exception;
use std::sync::OnceLock;

/// C callback for fused beta+g computation
/// Inputs: [b_raw, a_raw, A_log, dt_bias]
/// Outputs: [beta, g]
///   beta = sigmoid(b_raw.f32).bf16
///   g = exp(-exp(A_log.f32) * softplus(a_raw + dt_bias))
unsafe extern "C" fn beta_g_fn(
    output: *mut mlx_sys::mlx_vector_array,
    inputs: mlx_sys::mlx_vector_array,
) -> std::os::raw::c_int {
    let mut b_raw = mlx_sys::mlx_array_new();
    let mut a_raw = mlx_sys::mlx_array_new();
    let mut a_log = mlx_sys::mlx_array_new();
    let mut dt_bias = mlx_sys::mlx_array_new();
    mlx_sys::mlx_vector_array_get(&mut b_raw, inputs, 0);
    mlx_sys::mlx_vector_array_get(&mut a_raw, inputs, 1);
    mlx_sys::mlx_vector_array_get(&mut a_log, inputs, 2);
    mlx_sys::mlx_vector_array_get(&mut dt_bias, inputs, 3);

    let b_raw = Array::from_ptr(b_raw);
    let a_raw = Array::from_ptr(a_raw);
    let a_log = Array::from_ptr(a_log);
    let dt_bias = Array::from_ptr(dt_bias);

    // beta = sigmoid(b_raw.f32).bf16
    let b_f32 = b_raw.as_type::<f32>().unwrap();
    let beta = mlx_rs::nn::sigmoid(b_f32).unwrap();
    let beta = beta.as_type::<half::bf16>().unwrap_or(beta);

    // g = exp(-exp(A_log.f32) * softplus(a_raw + dt_bias))
    let a_log_f32 = a_log.as_type::<f32>().unwrap();
    let sum = mlx_rs::ops::add(&a_raw, &dt_bias).unwrap();
    let exp_sum = mlx_rs::ops::exp(&sum).unwrap();
    let one = Array::from_slice(&[1.0f32], &[1]);
    let softplus = mlx_rs::ops::log(&mlx_rs::ops::add(&exp_sum, &one).unwrap()).unwrap();
    let exp_alog = mlx_rs::ops::exp(&a_log_f32).unwrap();
    let neg_product = mlx_rs::ops::negative(&exp_alog.multiply(softplus).unwrap()).unwrap();
    let g = mlx_rs::ops::exp(&neg_product).unwrap();

    *output = mlx_sys::mlx_vector_array_new();
    mlx_sys::mlx_vector_array_append_value(*output, beta.as_ptr());
    mlx_sys::mlx_vector_array_append_value(*output, g.as_ptr());
    0
}

struct ClosureHandle(mlx_sys::mlx_closure);
unsafe impl Send for ClosureHandle {}
unsafe impl Sync for ClosureHandle {}

static COMPILED_BETA_G: OnceLock<ClosureHandle> = OnceLock::new();

fn get_compiled_beta_g() -> mlx_sys::mlx_closure {
    COMPILED_BETA_G.get_or_init(|| unsafe {
        let raw_fn = mlx_sys::mlx_closure_new_func(Some(beta_g_fn));
        let mut compiled = mlx_sys::mlx_closure_new();
        mlx_sys::mlx_compile(&mut compiled, raw_fn, true); // shapeless=true
        mlx_sys::mlx_closure_free(raw_fn);
        ClosureHandle(compiled)
    }).0
}

/// Fused beta + g computation (single compiled dispatch for ~14 element-wise ops)
pub fn compiled_beta_g(b_raw: &Array, a_raw: &Array, a_log: &Array, dt_bias: &Array) -> Result<(Array, Array), Exception> {
    unsafe {
        let closure = get_compiled_beta_g();

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, b_raw.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, a_raw.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, a_log.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, dt_bias.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_closure_apply(&mut outputs, closure, inputs);
        mlx_sys::mlx_vector_array_free(inputs);

        if ret != 0 {
            return Err(Exception::custom("compiled beta_g failed"));
        }

        let mut beta = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut beta, outputs, 0);
        let mut g = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut g, outputs, 1);
        mlx_sys::mlx_vector_array_free(outputs);

        Ok((Array::from_ptr(beta), Array::from_ptr(g)))
    }
}

/// C callback for fused silu_multiply: silu(a) * b
/// Inputs: [a, b]
/// Outputs: [result]
unsafe extern "C" fn silu_mul_fn(
    output: *mut mlx_sys::mlx_vector_array,
    inputs: mlx_sys::mlx_vector_array,
) -> std::os::raw::c_int {
    let mut a = mlx_sys::mlx_array_new();
    let mut b = mlx_sys::mlx_array_new();
    mlx_sys::mlx_vector_array_get(&mut a, inputs, 0);
    mlx_sys::mlx_vector_array_get(&mut b, inputs, 1);
    let a = Array::from_ptr(a);
    let b = Array::from_ptr(b);
    let result = mlx_rs::nn::silu(a).unwrap().multiply(b).unwrap();
    *output = mlx_sys::mlx_vector_array_new();
    mlx_sys::mlx_vector_array_append_value(*output, result.as_ptr());
    0
}

static COMPILED_SILU_MUL: OnceLock<ClosureHandle> = OnceLock::new();

fn get_compiled_silu_mul() -> mlx_sys::mlx_closure {
    COMPILED_SILU_MUL.get_or_init(|| unsafe {
        let raw_fn = mlx_sys::mlx_closure_new_func(Some(silu_mul_fn));
        let mut compiled = mlx_sys::mlx_closure_new();
        mlx_sys::mlx_compile(&mut compiled, raw_fn, true);
        mlx_sys::mlx_closure_free(raw_fn);
        ClosureHandle(compiled)
    }).0
}

/// Fused silu(a) * b
pub fn compiled_silu_multiply(a: &Array, b: &Array) -> Result<Array, Exception> {
    unsafe {
        let closure = get_compiled_silu_mul();
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, a.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, b.as_ptr());
        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_closure_apply(&mut outputs, closure, inputs);
        mlx_sys::mlx_vector_array_free(inputs);
        if ret != 0 { return Err(Exception::custom("compiled silu_mul failed")); }
        let mut r = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut r, outputs, 0);
        mlx_sys::mlx_vector_array_free(outputs);
        Ok(Array::from_ptr(r))
    }
}

/// C callback for compiled sampler: categorical(logits * inv_temp)
/// Inputs: [logits, inv_temp]
/// Outputs: [token]
unsafe extern "C" fn sampler_fn(
    output: *mut mlx_sys::mlx_vector_array,
    inputs: mlx_sys::mlx_vector_array,
) -> std::os::raw::c_int {
    let mut logits = mlx_sys::mlx_array_new();
    let mut inv_temp = mlx_sys::mlx_array_new();
    mlx_sys::mlx_vector_array_get(&mut logits, inputs, 0);
    mlx_sys::mlx_vector_array_get(&mut inv_temp, inputs, 1);
    let logits = Array::from_ptr(logits);
    let inv_temp = Array::from_ptr(inv_temp);

    let scaled = logits.multiply(inv_temp).unwrap();
    let token = mlx_rs::random::categorical(&scaled, -1, None, None).unwrap();

    *output = mlx_sys::mlx_vector_array_new();
    mlx_sys::mlx_vector_array_append_value(*output, token.as_ptr());
    0
}

static COMPILED_SAMPLER: OnceLock<ClosureHandle> = OnceLock::new();

fn get_compiled_sampler() -> mlx_sys::mlx_closure {
    COMPILED_SAMPLER.get_or_init(|| unsafe {
        let raw_fn = mlx_sys::mlx_closure_new_func(Some(sampler_fn));
        let mut compiled = mlx_sys::mlx_closure_new();
        mlx_sys::mlx_compile(&mut compiled, raw_fn, true);
        mlx_sys::mlx_closure_free(raw_fn);
        ClosureHandle(compiled)
    }).0
}

/// Compiled sampler: categorical(logits * inv_temp)
pub fn compiled_sample(logits: &Array, inv_temp: &Array) -> Result<Array, Exception> {
    unsafe {
        let closure = get_compiled_sampler();
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, logits.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, inv_temp.as_ptr());
        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_closure_apply(&mut outputs, closure, inputs);
        mlx_sys::mlx_vector_array_free(inputs);
        if ret != 0 { return Err(Exception::custom("compiled sampler failed")); }
        let mut token = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut token, outputs, 0);
        mlx_sys::mlx_vector_array_free(outputs);
        Ok(Array::from_ptr(token))
    }
}

/// C callback for fused sigmoid_multiply: sigmoid(a) * b
unsafe extern "C" fn sigmoid_mul_fn(
    output: *mut mlx_sys::mlx_vector_array,
    inputs: mlx_sys::mlx_vector_array,
) -> std::os::raw::c_int {
    let mut a = mlx_sys::mlx_array_new();
    let mut b = mlx_sys::mlx_array_new();
    mlx_sys::mlx_vector_array_get(&mut a, inputs, 0);
    mlx_sys::mlx_vector_array_get(&mut b, inputs, 1);
    let a = Array::from_ptr(a);
    let b = Array::from_ptr(b);
    let result = mlx_rs::nn::sigmoid(a).unwrap().multiply(b).unwrap();
    *output = mlx_sys::mlx_vector_array_new();
    mlx_sys::mlx_vector_array_append_value(*output, result.as_ptr());
    0
}

static COMPILED_SIGMOID_MUL: OnceLock<ClosureHandle> = OnceLock::new();

fn get_compiled_sigmoid_mul() -> mlx_sys::mlx_closure {
    COMPILED_SIGMOID_MUL.get_or_init(|| unsafe {
        let raw_fn = mlx_sys::mlx_closure_new_func(Some(sigmoid_mul_fn));
        let mut compiled = mlx_sys::mlx_closure_new();
        mlx_sys::mlx_compile(&mut compiled, raw_fn, true);
        mlx_sys::mlx_closure_free(raw_fn);
        ClosureHandle(compiled)
    }).0
}

/// Fused sigmoid(a) * b
pub fn compiled_sigmoid_multiply(a: &Array, b: &Array) -> Result<Array, Exception> {
    unsafe {
        let closure = get_compiled_sigmoid_mul();
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, a.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, b.as_ptr());
        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_closure_apply(&mut outputs, closure, inputs);
        mlx_sys::mlx_vector_array_free(inputs);
        if ret != 0 { return Err(Exception::custom("compiled sigmoid_mul failed")); }
        let mut r = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut r, outputs, 0);
        mlx_sys::mlx_vector_array_free(outputs);
        Ok(Array::from_ptr(r))
    }
}

/// C callback for fused norm_gate: silu(z) * rms_norm(y, weight, eps)
/// Inputs: [y, z, weight]
/// Outputs: [gated]
unsafe extern "C" fn norm_gate_fn(
    output: *mut mlx_sys::mlx_vector_array,
    inputs: mlx_sys::mlx_vector_array,
) -> std::os::raw::c_int {
    let mut y = mlx_sys::mlx_array_new();
    let mut z = mlx_sys::mlx_array_new();
    let mut w = mlx_sys::mlx_array_new();
    mlx_sys::mlx_vector_array_get(&mut y, inputs, 0);
    mlx_sys::mlx_vector_array_get(&mut z, inputs, 1);
    mlx_sys::mlx_vector_array_get(&mut w, inputs, 2);
    let y = Array::from_ptr(y);
    let z = Array::from_ptr(z);
    let w = Array::from_ptr(w);

    let y_normed = mlx_rs::fast::rms_norm(&y, &w, 1e-6).unwrap();
    let gated = mlx_rs::nn::silu(z).unwrap().multiply(y_normed).unwrap();

    *output = mlx_sys::mlx_vector_array_new();
    mlx_sys::mlx_vector_array_append_value(*output, gated.as_ptr());
    0
}

static COMPILED_NORM_GATE: OnceLock<ClosureHandle> = OnceLock::new();

fn get_compiled_norm_gate() -> mlx_sys::mlx_closure {
    COMPILED_NORM_GATE.get_or_init(|| unsafe {
        let raw_fn = mlx_sys::mlx_closure_new_func(Some(norm_gate_fn));
        let mut compiled = mlx_sys::mlx_closure_new();
        mlx_sys::mlx_compile(&mut compiled, raw_fn, true);
        mlx_sys::mlx_closure_free(raw_fn);
        ClosureHandle(compiled)
    }).0
}

/// Fused norm_gate: silu(z) * rms_norm(y, weight, eps)
pub fn compiled_norm_gate(y: &Array, z: &Array, weight: &Array) -> Result<Array, Exception> {
    unsafe {
        let closure = get_compiled_norm_gate();
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, y.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, z.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, weight.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_closure_apply(&mut outputs, closure, inputs);
        mlx_sys::mlx_vector_array_free(inputs);

        if ret != 0 {
            return Err(Exception::custom("compiled norm_gate failed"));
        }

        let mut gated = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut gated, outputs, 0);
        mlx_sys::mlx_vector_array_free(outputs);
        Ok(Array::from_ptr(gated))
    }
}
