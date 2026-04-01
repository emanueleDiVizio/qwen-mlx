//! GatedDeltaNet Metal kernel — ported from Python mlx-lm.
//! Calls the MLX C API directly via mlx-sys FFI for the fused recurrence kernel.

use mlx_rs::Array;
use mlx_rs::error::Exception;
use std::ffi::CString;

/// The Metal kernel source for gated delta rule recurrence.
/// Template params: InT (input/output type), StT (state type), B, T, Hk, Hv, Dk, Dv
const GDN_KERNEL_SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"#;

fn make_cstring_vec(names: &[&str]) -> (Vec<CString>, Vec<*const std::os::raw::c_char>) {
    let cstrings: Vec<CString> = names.iter().map(|n| CString::new(*n).unwrap()).collect();
    let ptrs: Vec<*const std::os::raw::c_char> = cstrings.iter().map(|s| s.as_ptr()).collect();
    (cstrings, ptrs)
}

use std::sync::OnceLock;

struct KernelHandle(mlx_sys::mlx_fast_metal_kernel);
unsafe impl Send for KernelHandle {}
unsafe impl Sync for KernelHandle {}

struct ConfigHandle(mlx_sys::mlx_fast_metal_kernel_config);
unsafe impl Send for ConfigHandle {}
unsafe impl Sync for ConfigHandle {}

static CACHED_KERNEL: OnceLock<KernelHandle> = OnceLock::new();
static CACHED_CONFIG: OnceLock<ConfigHandle> = OnceLock::new();

fn get_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    CACHED_KERNEL.get_or_init(|| unsafe {
        let name = CString::new("gated_delta_step").unwrap();
        let source = CString::new(GDN_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();

        let (_in_cs, mut in_ptrs) = make_cstring_vec(&["q", "k", "v", "g", "beta", "state_in"]);
        let (_out_cs, mut out_ptrs) = make_cstring_vec(&["y", "state_out"]);

        let input_names = mlx_sys::mlx_vector_string_new_data(in_ptrs.as_mut_ptr(), in_ptrs.len());
        let output_names = mlx_sys::mlx_vector_string_new_data(out_ptrs.as_mut_ptr(), out_ptrs.len());

        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );

        mlx_sys::mlx_vector_string_free(input_names);
        mlx_sys::mlx_vector_string_free(output_names);

        KernelHandle(kernel)
    }).0
}

/// Get or create a cached kernel config for fixed dimensions.
/// The config is created once and reused for all decode calls with same dimensions.
fn get_or_create_config(
    batch: i32, seq_len: i32, hk: i32, hv: i32, dk: i32, dv: i32,
    input_dtype: mlx_sys::mlx_dtype,
    state_dtype: mlx_sys::mlx_dtype,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    // For the common decode case (B=1, T=1), use cached config
    if batch == 1 && seq_len == 1 {
        return CACHED_CONFIG.get_or_init(|| unsafe {
            let config = create_config(batch, seq_len, hk, hv, dk, dv, input_dtype, state_dtype);
            ConfigHandle(config)
        }).0;
    }
    // For prefill or other cases, create a fresh config
    unsafe { create_config(batch, seq_len, hk, hv, dk, dv, input_dtype, state_dtype) }
}

unsafe fn create_config(
    batch: i32, seq_len: i32, hk: i32, hv: i32, dk: i32, dv: i32,
    input_dtype: mlx_sys::mlx_dtype,
    state_dtype: mlx_sys::mlx_dtype,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    let config = mlx_sys::mlx_fast_metal_kernel_config_new();
    mlx_sys::mlx_fast_metal_kernel_config_set_init_value(config, 0.0);
    mlx_sys::mlx_fast_metal_kernel_config_set_verbose(config, false);

    let y_shape = [batch, seq_len, hv, dv];
    mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
        config, y_shape.as_ptr(), y_shape.len(), input_dtype,
    );

    let s_shape = [batch, hv, dv, dk];
    mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
        config, s_shape.as_ptr(), s_shape.len(), state_dtype,
    );

    mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * hv);
    mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1);

    let inp_t = CString::new("InT").unwrap();
    mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(config, inp_t.as_ptr(), input_dtype);
    let st_t = CString::new("StT").unwrap();
    mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(config, st_t.as_ptr(), state_dtype);

    for (n, val) in [("B", batch), ("T", seq_len), ("Hk", hk), ("Hv", hv), ("Dk", dk), ("Dv", dv)] {
        let cn = CString::new(n).unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, cn.as_ptr(), val);
    }

    config
}

pub fn gated_delta_kernel(
    q: &Array, k: &Array, v: &Array, g: &Array, beta: &Array, state: &Array,
    seq_len: i32, batch: i32, hk: i32, hv: i32, dk: i32, dv: i32,
) -> Result<(Array, Array), Exception> {
    let kernel = get_kernel();
    unsafe {
        let input_dtype = mlx_sys::mlx_array_dtype(q.as_ref().as_ptr());
        let state_dtype = mlx_sys::mlx_array_dtype(state.as_ref().as_ptr());
        let config = get_or_create_config(batch, seq_len, hk, hv, dk, dv, input_dtype, state_dtype);

        // Input arrays
        let inputs = mlx_sys::mlx_vector_array_new();
        for arr in [q, k, v, g, beta, state] {
            let mut tmp = mlx_sys::mlx_array_new();
            mlx_sys::mlx_array_set(&mut tmp, arr.as_ref().as_ptr());
            mlx_sys::mlx_vector_array_append_value(inputs, tmp);
            mlx_sys::mlx_array_free(tmp);
        }

        // Cached GPU stream
        struct StreamHandle(mlx_sys::mlx_stream);
        unsafe impl Send for StreamHandle {}
        unsafe impl Sync for StreamHandle {}
        static STREAM: OnceLock<StreamHandle> = OnceLock::new();
        let stream = STREAM.get_or_init(|| StreamHandle(mlx_sys::mlx_default_gpu_stream_new())).0;

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel, inputs, config, stream,
        );

        mlx_sys::mlx_vector_array_free(inputs);
        if batch != 1 || seq_len != 1 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
        }

        if ret != 0 {
            return Err(Exception::custom("GDN Metal kernel failed"));
        }

        let mut y_ptr = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut y_ptr, outputs, 0);
        let mut state_ptr = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut state_ptr, outputs, 1);
        mlx_sys::mlx_vector_array_free(outputs);

        Ok((Array::from_ptr(y_ptr), Array::from_ptr(state_ptr)))
    }
}
