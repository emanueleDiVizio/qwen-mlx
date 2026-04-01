//! Qwen3.5-35B-A3B: Hybrid GDN + Attention, MoE, built on mlx-rs.
//!
//! Extends the Qwen3 base with:
//! - GatedDeltaNet recurrent layers (30/40 layers)
//! - SparseMoeBlock with SwitchGLU experts + shared expert
//! - Output gating in full attention
//! - Partial rotary embedding (25% of head_dim)

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt},
    nn,
    ops::indexing::{IndexOp, NewAxis},
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

// Re-export from the Qwen3 base for cache and utilities
// TODO: integrate properly with mlx-lm utils

// ── Config ──

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub full_attention_interval: i32,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub max_position_embeddings: i32,
    // GDN
    #[serde(default)]
    pub linear_num_key_heads: i32,
    #[serde(default)]
    pub linear_num_value_heads: i32,
    #[serde(default)]
    pub linear_key_head_dim: i32,
    #[serde(default)]
    pub linear_value_head_dim: i32,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: i32,
    // MoE
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub moe_intermediate_size: i32,
    pub shared_expert_intermediate_size: i32,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_partial_rotary() -> f32 { 0.25 }
fn default_rope_theta() -> f32 { 10_000_000.0 }
fn default_conv_kernel() -> i32 { 4 }
fn default_true() -> bool { true }

impl Qwen35Config {
    pub fn is_linear_layer(&self, idx: i32) -> bool {
        (idx + 1) % self.full_attention_interval != 0
    }
}

// ── ExpertProj: a single projection's quantized weights ──

#[derive(Debug, ModuleParameters)]
pub struct ExpertProj {
    #[param] weight: mlx_rs::module::Param<Array>,
    #[param] scales: mlx_rs::module::Param<Array>,
    #[param] biases: mlx_rs::module::Param<Array>,
}

impl ExpertProj {
    pub fn new() -> Self {
        let z = || mlx_rs::module::Param::new(mlx_rs::Array::from_slice(&[0.0f32], &[1]));
        Self { weight: z(), scales: z(), biases: z() }
    }
}

impl mlx_rs::quantization::Quantizable for ExpertProj {
    type Quantized = Self;
    type QuantizationError = Exception;
    fn try_into_quantized(self, _: i32, _: i32) -> Result<Self, Exception> { Ok(self) }
}

// ── SwitchMlp: stacked expert weights ──

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct SwitchMlp {
    #[quantizable]
    #[param]
    gate_proj: ExpertProj,
    #[quantizable]
    #[param]
    up_proj: ExpertProj,
    #[quantizable]
    #[param]
    down_proj: ExpertProj,
}

impl SwitchMlp {
    pub fn new(_input_dims: i32, _hidden_dims: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: ExpertProj::new(),
            up_proj: ExpertProj::new(),
            down_proj: ExpertProj::new(),
        })
    }

    /// Forward for selected experts using gather_qmm (matches Python exactly)
    pub fn forward(&self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        let n = x.dim(0);
        let h = x.dim(-1);
        let x_exp = x.reshape(&[n, 1, 1, h])?;

        // Cached stream and null array for gather_qmm
        struct StreamHandle(mlx_sys::mlx_stream);
        unsafe impl Send for StreamHandle {}
        unsafe impl Sync for StreamHandle {}
        struct ArrayHandle(mlx_sys::mlx_array);
        unsafe impl Send for ArrayHandle {}
        unsafe impl Sync for ArrayHandle {}
        static STREAM: OnceLock<StreamHandle> = OnceLock::new();
        static NULL_LHS: OnceLock<ArrayHandle> = OnceLock::new();
        let stream = STREAM.get_or_init(|| unsafe { StreamHandle(mlx_sys::mlx_default_gpu_stream_new()) }).0;
        let null_lhs = NULL_LHS.get_or_init(|| unsafe { ArrayHandle(mlx_sys::mlx_array_new()) }).0;

        let gather_qmm = |input: &Array, w: &Array, s: &Array, b: &Array| -> Result<Array, Exception> {
            unsafe {
                let mut res = mlx_sys::mlx_array_new();
                let ret = mlx_sys::mlx_gather_qmm(
                    &mut res, input.as_ptr(), w.as_ptr(), s.as_ptr(), b.as_ptr(),
                    null_lhs, indices.as_ptr(), true, 64, 4, false, stream,
                );
                if ret != 0 { return Err(Exception::custom("gather_qmm failed")); }
                Ok(Array::from_ptr(res))
            }
        };

        let gate = gather_qmm(&x_exp, &self.gate_proj.weight, &self.gate_proj.scales, &self.gate_proj.biases)?;
        let up = gather_qmm(&x_exp, &self.up_proj.weight, &self.up_proj.scales, &self.up_proj.biases)?;
        let activated = super::compiled_ops::compiled_silu_multiply(&gate, &up)?;
        let out = gather_qmm(&activated, &self.down_proj.weight, &self.down_proj.scales, &self.down_proj.biases)?;
        Ok(out.squeeze_axes(&[-2])?)
    }
}

// ── Shared Expert (standard MLP) ──

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct SharedExpert {
    #[quantizable]
    #[param]
    gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    down_proj: MaybeQuantized<nn::Linear>,
    // Merged gate+up weights for single matmul dispatch
    merged_gate_up: Option<(Array, Array, Array)>,
}

impl SharedExpert {
    pub fn new(h: i32, intermediate: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: MaybeQuantized::Original(nn::LinearBuilder::new(h, intermediate).bias(false).build()?),
            up_proj: MaybeQuantized::Original(nn::LinearBuilder::new(h, intermediate).bias(false).build()?),
            down_proj: MaybeQuantized::Original(nn::LinearBuilder::new(intermediate, h).bias(false).build()?),
            merged_gate_up: None,
        })
    }
}

impl SharedExpert {
    /// Merged gate+up projection for single dispatch (set at load time)
    pub fn set_merged_gate_up(&mut self) {
        use mlx_rs::quantization::MaybeQuantized;
        let (g_w, g_s, g_b) = match &self.gate_proj {
            MaybeQuantized::Quantized(q) => (&*q.inner.weight, &*q.scales, &*q.biases),
            _ => return,
        };
        let (u_w, u_s, u_b) = match &self.up_proj {
            MaybeQuantized::Quantized(q) => (&*q.inner.weight, &*q.scales, &*q.biases),
            _ => return,
        };
        self.merged_gate_up = Some((
            mlx_rs::ops::concatenate_axis(&[g_w, u_w], 0).unwrap(),
            mlx_rs::ops::concatenate_axis(&[g_s, u_s], 0).unwrap(),
            mlx_rs::ops::concatenate_axis(&[g_b, u_b], 0).unwrap(),
        ));
    }
}

impl Module<&Array> for SharedExpert {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = super::compiled_ops::compiled_silu_multiply(&gate, &up)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

// ── SparseMoeBlock ──

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct SparseMoeBlock {
    num_experts_per_tok: i32,
    norm_topk_prob: bool,

    #[quantizable]
    #[param]
    gate: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    switch_mlp: SwitchMlp,

    #[quantizable]
    #[param]
    shared_expert: SharedExpert,

    #[quantizable]
    #[param]
    shared_expert_gate: MaybeQuantized<nn::Linear>,
}

impl SparseMoeBlock {
    pub fn new(cfg: &Qwen35Config) -> Result<Self, Exception> {
        let h = cfg.hidden_size;
        let moe_is = cfg.moe_intermediate_size;
        let shared_is = cfg.shared_expert_intermediate_size;

        Ok(Self {
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            gate: MaybeQuantized::Original(nn::LinearBuilder::new(h, cfg.num_experts).bias(false).build()?),
            switch_mlp: SwitchMlp::new(h, moe_is)?,
            shared_expert: SharedExpert::new(h, shared_is)?,
            shared_expert_gate: MaybeQuantized::Original(nn::LinearBuilder::new(h, 1).bias(false).build()?),
        })
    }
}

impl Module<&Array> for SparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let (b, s) = (shape[0], shape[1]);
        let x_flat = x.reshape(&[-1, shape[2]])?;

        // Route
        let router_logits = self.gate.forward(&x_flat)?;
        let routing_weights = mlx_rs::ops::softmax(&router_logits, false)?;

        // Top-k selection using argpartition (matches Python exactly)
        let k = self.num_experts_per_tok;
        let ne = routing_weights.dim(-1);
        // Python: inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        let partitioned = mlx_rs::ops::argpartition_axis(&routing_weights, -k, -1)?;
        let topk_indices = partitioned.index((.., (ne - k)..));
        let topk_weights = routing_weights.take_along_axis(&topk_indices, -1)?;

        let topk_weights = if self.norm_topk_prob {
            let wsum = topk_weights.sum_axis(-1, true)?;
            topk_weights.divide(&wsum)?
        } else {
            topk_weights
        };

        // Expert forward: [n_tokens, topk, hidden]
        let expert_out = self.switch_mlp.forward(&x_flat, &topk_indices)?;

        // Weighted sum over topk experts: [n, topk, hidden] * [n, topk, 1] -> sum -> [n, hidden]
        let topk_w_expanded = topk_weights.expand_dims(-1)?;
        let weighted = expert_out.multiply(topk_w_expanded)?;
        let moe_out = weighted.sum_axis(-2, false)?;

        // Shared expert
        let shared_out = self.shared_expert.forward(&x_flat)?;
        let shared_gated = nn::sigmoid(self.shared_expert_gate.forward(&x_flat)?)?.multiply(shared_out)?;

        // Combine
        let result = moe_out.add(shared_gated)?;
        result.reshape(&[b, s, -1])
    }

    fn training_mode(&mut self, _mode: bool) {}
}

// ── Full Attention (with output gate, partial rotary) ──

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35Attention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    rope_theta: f32,
    scale: f32,

    #[quantizable]
    #[param]
    q_proj: MaybeQuantized<nn::Linear>, // outputs 2 * n_heads * head_dim (queries + gate)
    #[quantizable]
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
}

impl Qwen35Attention {
    pub fn new(cfg: &Qwen35Config) -> Result<Self, Exception> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let scale = (hd as f32).sqrt().recip();

        // q_proj outputs 2× for output gating
        let q_proj = nn::LinearBuilder::new(h, 2 * nh * hd).bias(false).build()?;
        let k_proj = nn::LinearBuilder::new(h, nkv * hd).bias(false).build()?;
        let v_proj = nn::LinearBuilder::new(h, nkv * hd).bias(false).build()?;
        let o_proj = nn::LinearBuilder::new(nh * hd, h).bias(false).build()?;

        let q_norm = nn::RmsNormBuilder::new(hd).eps(cfg.rms_norm_eps).build()?;
        let k_norm = nn::RmsNormBuilder::new(hd).eps(cfg.rms_norm_eps).build()?;

        let rotary_dim = (hd as f32 * cfg.partial_rotary_factor) as i32;

        Ok(Self {
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            rotary_dim,
            rope_theta: cfg.rope_theta,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
        })
    }

    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        cache: &mut (Option<Array>, Option<Array>), // (cached_keys, cached_values)
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let (b, s) = (shape[0], shape[1]);
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim;

        // Q projection: split into queries + gate (interleaved per head, matching Python)
        let q_full = self.q_proj.forward(x)?;
        // Reshape to [B, S, nh, 2*hd] then split along last axis
        let q_reshaped = q_full.reshape(&[b, s, nh, 2 * hd])?;
        let queries = q_reshaped.index((.., .., .., ..hd)); // [B, S, nh, hd]
        let gate = q_reshaped.index((.., .., .., hd..)).reshape(&[b, s, -1])?; // [B, S, nh*hd]

        let queries = mlx_rs::fast::rms_norm(
            &queries.transpose_axes(&[0, 2, 1, 3])?,
            &*self.q_norm.weight, self.q_norm.eps as f32,
        )?;
        let mut keys = mlx_rs::fast::rms_norm(
            &self.k_proj.forward(x)?
                .reshape(&[b, s, nkv, hd])?
                .transpose_axes(&[0, 2, 1, 3])?,
            &*self.k_norm.weight, self.k_norm.eps as f32,
        )?;
        let mut values = self.v_proj.forward(x)?
            .reshape(&[b, s, nkv, hd])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Partial RoPE: apply to first rotary_dim dimensions of Q and K
        let offset = cache.0.as_ref().map(|k| k.dim(2) as i32).unwrap_or(0);
        let queries = mlx_rs::fast::rope(
            &queries, self.rotary_dim, false, self.rope_theta, 1.0, offset, None::<&Array>,
        )?;
        keys = mlx_rs::fast::rope(
            &keys, self.rotary_dim, false, self.rope_theta, 1.0, offset, None::<&Array>,
        )?;

        // KV cache: concatenate with previous keys/values
        if let Some(prev_k) = cache.0.take() {
            keys = mlx_rs::ops::concatenate_axis(&[&prev_k, &keys], 2)?;
        }
        if let Some(prev_v) = cache.1.take() {
            values = mlx_rs::ops::concatenate_axis(&[&prev_v, &values], 2)?;
        }

        // SDPA (decode: no mask needed; prefill: causal mask)
        let kv_len = keys.dim(2);
        let q_len = queries.dim(2);
        let attn = if q_len > 1 {
            // Causal mask for prefill
            let mask_offset = kv_len - q_len;
            let mut mask_data = vec![0.0f32; (q_len * kv_len) as usize];
            for i in 0..q_len as usize {
                for j in (i as i32 + mask_offset + 1) as usize..kv_len as usize {
                    mask_data[i * kv_len as usize + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Array::from_slice(&mask_data, &[q_len, kv_len])
                .as_type::<half::bf16>()?;
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, self.scale, &mask,
            )?
        } else {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, self.scale,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            )?
        };

        // Move keys/values into cache (no clone needed — SDPA already captured refs)
        cache.0 = Some(keys);
        cache.1 = Some(values);

        // Output gating: sigmoid(gate) * attention_output
        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, -1])?;
        let gated = attn.multiply(nn::sigmoid(gate)?)?;

        self.o_proj.forward(&gated)
    }
}

// ── GatedDeltaNet ──
// Weight names: linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, out_proj, norm, A_log, dt_bias}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct GatedDeltaNet {
    #[quantizable]
    #[param]
    in_proj_qkv: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    in_proj_z: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    in_proj_a: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    in_proj_b: MaybeQuantized<nn::Linear>,
    #[param]
    conv1d: nn::Conv1d,
    #[quantizable]
    #[param]
    out_proj: MaybeQuantized<nn::Linear>,
    #[param]
    norm: nn::RmsNorm, // RMSNormGated weight
    #[param]
    A_log: mlx_rs::module::Param<Array>, // [num_v_heads]
    #[param]
    dt_bias: mlx_rs::module::Param<Array>, // [num_v_heads]

    num_key_heads: i32,
    num_value_heads: i32,
    key_head_dim: i32,
    value_head_dim: i32,
    // Precomputed norm weights with scale factors baked in
    q_norm_weight: Array, // ones * inv_scale^2 = [1/dk; dk]
    k_norm_weight: Array, // ones * inv_scale = [1/sqrt(dk); dk]
    // Merged projection weights (concatenated at load time to reduce matmul count)
    merged_ab: Option<(Array, Array, Array)>,     // [2*Hv, hidden] for a+b
    merged_qkvz: Option<(Array, Array, Array)>,   // [qkv_dim+z_dim, hidden] for qkv+z
    // Precomputed conv weight in [1, K, D] layout for fast decode
    conv_weight_decode: Option<Array>,
}

impl GatedDeltaNet {
    pub fn new(cfg: &Qwen35Config) -> Result<Self, Exception> {
        let h = cfg.hidden_size;
        let hk = cfg.linear_num_key_heads;
        let hv = cfg.linear_num_value_heads;
        let dk = cfg.linear_key_head_dim;
        let dv = cfg.linear_value_head_dim;

        let qkv_dim = hk * dk + hk * dk + hv * dv; // q + k + v combined
        let z_dim = hv * dv;
        let a_dim = hv; // one scalar per value head (decay input)
        let b_dim = hv; // one scalar per value head (write gate beta)

        let conv_dim = hk * dk + hk * dk + hv * dv; // q + k + v go through conv
        let kernel = cfg.linear_conv_kernel_dim;

        let in_proj_qkv = nn::LinearBuilder::new(h, qkv_dim).bias(false).build()?;
        let in_proj_z = nn::LinearBuilder::new(h, z_dim).bias(false).build()?;
        let in_proj_a = nn::LinearBuilder::new(h, a_dim).bias(false).build()?;
        let in_proj_b = nn::LinearBuilder::new(h, b_dim).bias(false).build()?;

        let conv1d = nn::Conv1dBuilder::new(conv_dim, conv_dim, kernel)
            .groups(conv_dim)
            .bias(false)
            .build()?;

        let out_proj = nn::LinearBuilder::new(hv * dv, h).bias(false).build()?;
        let norm = nn::RmsNormBuilder::new(dv).eps(cfg.rms_norm_eps).build()?;

        let z = |n: i32| mlx_rs::module::Param::new(mlx_rs::Array::from_slice(&vec![0.0f32; n as usize], &[n]));

        Ok(Self {
            in_proj_qkv: MaybeQuantized::Original(in_proj_qkv),
            in_proj_z: MaybeQuantized::Original(in_proj_z),
            in_proj_a: MaybeQuantized::Original(in_proj_a),
            in_proj_b: MaybeQuantized::Original(in_proj_b),
            conv1d,
            out_proj: MaybeQuantized::Original(out_proj),
            norm,
            A_log: z(hv),
            dt_bias: z(hv),
            num_key_heads: hk,
            num_value_heads: hv,
            key_head_dim: dk,
            value_head_dim: dv,
            // Precompute: rms_norm(q, weight) with scale baked in
            q_norm_weight: Array::from_slice(&vec![half::bf16::from_f32((dk as f32).powi(-1)); dk as usize], &[dk]),
            k_norm_weight: Array::from_slice(&vec![half::bf16::from_f32((dk as f32).powf(-0.5)); dk as usize], &[dk]),
            merged_ab: None,
            merged_qkvz: None,
            conv_weight_decode: None,
        })
    }

    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        cache: &mut (Option<Array>, Option<Array>), // (conv_state, recurrent_state)
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let (b, s) = (shape[0], shape[1]);
        let hk = self.num_key_heads;
        let hv = self.num_value_heads;
        let dk = self.key_head_dim;
        let dv = self.value_head_dim;
        let v_per_group = hv / hk;

        // 1. Projections (merged for fewer matmul dispatches)
        let (qkv, z) = if let Some((ref w, ref s, ref bi)) = self.merged_qkvz {
            let qkvz = mlx_rs::ops::quantized_matmul(x, w, s, bi, true, 64, 4)?;
            let qkv_dim = hk * dk * 2 + hv * dv;
            (qkvz.index((.., .., ..qkv_dim)), qkvz.index((.., .., qkv_dim..)))
        } else {
            (self.in_proj_qkv.forward(x)?, self.in_proj_z.forward(x)?)
        };
        let (a_raw, b_raw) = if let Some((ref w, ref s, ref b)) = self.merged_ab {
            let ab = mlx_rs::ops::quantized_matmul(x, w, s, b, true, 64, 4)?;
            let hv = self.num_value_heads;
            (ab.index((.., .., hv..)), ab.index((.., .., ..hv)))  // a is second half, b is first
        } else {
            (self.in_proj_a.forward(x)?, self.in_proj_b.forward(x)?)
        };

        // 2. Causal conv1d on qkv
        let qk_dim = hk * dk;
        let conv_dim = qkv.dim(-1);
        let kernel_size = 4i32;

        let qkv_conv = if s == 1 {
            // Fast path for decode (S=1): direct dot product with conv state
            // conv_state: [B, 3, D], qkv: [B, 1, D]
            // padded = [state[0], state[1], state[2], qkv] → [B, 4, D]
            // output = sum_k(weight[ch, k] * padded[b, k, ch]) per channel
            let conv_state = match cache.0.take() {
                Some(cs) => cs,
                None => mlx_rs::ops::zeros::<half::bf16>(&[b, kernel_size - 1, conv_dim])?,
            };
            let padded = mlx_rs::ops::concatenate_axis(&[&conv_state, &qkv], 1)?;
            // Update state: shift window (drop oldest, keep last 3)
            cache.0 = Some(padded.index((.., 1.., ..)));

            // Use precomputed conv weight [1, K, D] (avoids reshape per call)
            let cw = self.conv_weight_decode.as_ref().unwrap_or_else(|| &self.conv1d.weight);
            let conv_out = padded.multiply(cw.clone())?.sum_axis(1, false)?  // [B, D]
                .reshape(&[b, 1, conv_dim])?;
            nn::silu(conv_out)?
        } else {
            // General path for prefill (S>1)
            let conv_state = match cache.0.take() {
                Some(cs) => cs,
                None => mlx_rs::ops::zeros::<half::bf16>(&[b, kernel_size - 1, conv_dim])?,
            };
            let padded = mlx_rs::ops::concatenate_axis(&[&conv_state, &qkv], 1)?;
            let total_len = padded.dim(1);
            cache.0 = Some(padded.index((.., (total_len - kernel_size + 1).., ..)));
            let conv_out = mlx_rs::ops::conv1d(&padded, &*self.conv1d.weight, 1, 0, 1, conv_dim)?;
            let conv_s = conv_out.dim(1);
            let conv_out = if conv_s > s { conv_out.index((.., (conv_s - s).., ..)) } else { conv_out };
            nn::silu(conv_out)?
        };

        // Split into q, k, v after conv
        let q = qkv_conv.index((.., .., ..qk_dim));
        let k = qkv_conv.index((.., .., qk_dim..2 * qk_dim));
        let v = qkv_conv.index((.., .., 2 * qk_dim..));

        // 3. Reshape to per-head
        let q = q.reshape(&[b, s, hk, dk])?;
        let k = k.reshape(&[b, s, hk, dk])?;
        let v = v.reshape(&[b, s, hv, dv])?;

        // RMS norm with scale baked into weight: q gets 1/dk, k gets 1/sqrt(dk)
        // Fused: rms_norm(q, weight=[1/dk; dk]) = rms_norm(q) * (1/dk) in one dispatch
        let q = mlx_rs::fast::rms_norm(&q, &self.q_norm_weight, 1e-6)?;
        let k = mlx_rs::fast::rms_norm(&k, &self.k_norm_weight, 1e-6)?;

        // 4. Fused beta+g via compiled closure (14 element-wise ops → 1 GPU dispatch)
        let (beta, g) = super::compiled_ops::compiled_beta_g(&b_raw, &a_raw, &*self.A_log, &*self.dt_bias)?;

        // 6. State init (F32 for numerical precision, matches Python)
        let state = match &cache.1 {
            Some(s) => s.clone(),
            None => mlx_rs::ops::zeros::<f32>(&[b, hv, dv, dk])?,
        };

        // 7. Fused Metal GDN kernel
        // Kernel expects: q,k [B,T,Hk,Dk], v [B,T,Hv,Dv], g [B,T,Hv], beta [B,T,Hv], state [B,Hv,Dv,Dk]
        // Kernel handles GQA internally (maps hv_idx → hk_idx)
        let (y, new_state) = super::metal_gdn::gated_delta_kernel(
            &q, &k, &v, &g, &beta, &state,
            s, b, hk, hv, dk, dv,
        )?;

        cache.1 = Some(new_state);

        // 8. RMSNormGated: silu(z) * rms_norm(y)
        let z = z.reshape(&[b * s, hv, dv])?;
        let y = y.reshape(&[b * s, hv, dv])?;
        let y_normed = mlx_rs::fast::rms_norm(&y, &*self.norm.weight, self.norm.eps as f32)?;
        let gated = nn::silu(z)?.multiply(y_normed)?;
        let gated = gated.reshape(&[b, s, hv * dv])?;

        // 9. Output projection
        self.out_proj.forward(&gated)
    }
}

// ── Decoder Layer ──

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    is_linear: bool,

    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,

    #[quantizable]
    #[param]
    self_attn: Option<Qwen35Attention>,
    #[quantizable]
    #[param]
    linear_attn: Option<GatedDeltaNet>,
    #[quantizable]
    #[param]
    mlp: SparseMoeBlock,
}

impl DecoderLayer {
    pub fn new(cfg: &Qwen35Config, layer_idx: i32) -> Result<Self, Exception> {
        let is_linear = cfg.is_linear_layer(layer_idx);

        let input_layernorm = nn::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;

        let (self_attn, gdn) = if is_linear {
            (None, Some(GatedDeltaNet::new(cfg)?))  // linear_attn
        } else {
            (Some(Qwen35Attention::new(cfg)?), None)
        };

        let mlp = SparseMoeBlock::new(cfg)?;

        Ok(Self {
            is_linear,
            input_layernorm,
            post_attention_layernorm,
            self_attn,
            linear_attn: gdn,
            mlp,
        })
    }

    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        cache: &mut (Option<Array>, Option<Array>),
    ) -> Result<Array, Exception> {
        // Direct fast::rms_norm (bypass nn::RmsNorm module dispatch overhead)
        let h = mlx_rs::fast::rms_norm(x, &*self.input_layernorm.weight, self.input_layernorm.eps as f32)?;

        let h = if let Some(attn) = &mut self.self_attn {
            attn.forward_with_cache(&h, cache)?
        } else if let Some(la) = &mut self.linear_attn {
            la.forward_with_cache(&h, cache)?
        } else {
            h
        };

        let h = x.add(h)?;
        let normed = mlx_rs::fast::rms_norm(&h, &*self.post_attention_layernorm.weight, self.post_attention_layernorm.eps as f32)?;
        let mlp_out = self.mlp.forward(&normed)?;
        h.add(mlp_out)
    }
}

// ── Inner Model (matches `language_model.model.` prefix) ──

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35Inner {
    #[quantizable]
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
}

// ── Language Model (matches `language_model.` prefix) ──

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35LanguageModel {
    #[quantizable]
    #[param]
    model: Qwen35Inner,
    #[quantizable]
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

// ── Full Model (top level, has language_model + vision_tower) ──

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35Model {
    #[quantizable]
    #[param]
    language_model: Qwen35LanguageModel,

    pub config: Qwen35Config,

    /// Raw safetensors weights for manual dequantize
    pub raw_weights: Option<std::collections::HashMap<String, Array>>,
    /// Pre-dequantized embedding table [vocab_size, hidden_size] BF16
    pub embed_dequant: Option<Array>,
}

impl Qwen35Model {
    pub fn new(cfg: Qwen35Config) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(cfg.vocab_size, cfg.hidden_size)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::new(&cfg, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let lm_head = if !cfg.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(cfg.hidden_size, cfg.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };

        let inner = Qwen35Inner {
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        };

        let language_model = Qwen35LanguageModel {
            model: inner,
            lm_head,
        };

        Ok(Self {
            language_model,
            config: cfg,
            raw_weights: None,
            embed_dequant: None,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        caches: &mut Vec<(Option<Array>, Option<Array>)>,
    ) -> Result<Array, Exception> {
        // Enable MLX compilation for fused element-wise ops
        // Enable compile for fused element-wise ops within compiled closures
        unsafe { mlx_sys::mlx_enable_compile(); }

        let lm = &mut self.language_model;
        let inner = &mut lm.model;

        // Embedding lookup: use pre-dequantized table if available (1 op vs 5)
        let mut h = if let Some(ref embed) = self.embed_dequant {
            let x = input_ids.flatten(None, None)?;
            let out = embed.index(&x);
            let ret_shape: Vec<i32> = input_ids.shape().iter().copied().chain(std::iter::once(-1)).collect();
            out.reshape(&ret_shape)?
        } else if let Some(ref rw) = self.raw_weights {
            let emb_w = rw.get("language_model.model.embed_tokens.weight").unwrap();
            let emb_s = rw.get("language_model.model.embed_tokens.scales").unwrap();
            let emb_b = rw.get("language_model.model.embed_tokens.biases").unwrap();
            let x = input_ids.flatten(None, None)?;
            let w = emb_w.index(&x);
            let s = emb_s.index(&x);
            let b = emb_b.index(&x);
            let out = mlx_rs::ops::dequantize(&w, &s, &b, 64, 4)?;
            let ret_shape: Vec<i32> = input_ids.shape().iter().copied().chain(std::iter::once(-1)).collect();
            out.reshape(&ret_shape)?
        } else {
            inner.embed_tokens.forward(input_ids)?
        };

        if caches.is_empty() {
            *caches = (0..inner.layers.len()).map(|_| (None, None)).collect();
        }

        for (layer, cache) in inner.layers.iter_mut().zip(caches.iter_mut()) {
            h = layer.forward_with_cache(&h, cache)?;
        }

        h = mlx_rs::fast::rms_norm(&h, &*inner.norm.weight, inner.norm.eps as f32)?;

        // Manual lm_head via raw dequantize
        if let Some(ref rw) = self.raw_weights {
            let w = rw.get("language_model.lm_head.weight").unwrap();
            let s = rw.get("language_model.lm_head.scales").unwrap();
            let b = rw.get("language_model.lm_head.biases").unwrap();
            return Ok(mlx_rs::ops::quantized_matmul(&h, w, s, b, true, 64, 4)?);
        }
        match lm.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&h),
            None => match &mut inner.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&h),
                MaybeQuantized::Quantized(q) => q.as_linear(&h),
            },
        }
    }
}

// ── Loading ──

pub fn load_config(model_dir: &Path) -> Result<Qwen35Config, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(model_dir.join("config.json"))?;
    let raw: serde_json::Value = serde_json::from_reader(file)?;

    // Config may have fields at top level or nested under text_config
    let cfg_value = if raw.get("text_config").is_some() {
        // Merge text_config into top level
        let mut merged = raw.get("text_config").unwrap().clone();
        if let (Some(m), Some(top)) = (merged.as_object_mut(), raw.as_object()) {
            for (k, v) in top {
                if k != "text_config" && k != "vision_config" {
                    m.entry(k.clone()).or_insert(v.clone());
                }
            }
        }
        merged
    } else {
        raw
    };

    Ok(serde_json::from_value(cfg_value)?)
}

pub fn load_model(model_dir: &Path) -> Result<Qwen35Model, Box<dyn std::error::Error>> {
    let cfg = load_config(model_dir)?;

    // Check if model is quantized
    let raw_config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let quant_cfg = raw_config.get("quantization");
    let (group_size, bits) = if let Some(q) = quant_cfg {
        (
            q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32,
            q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32,
        )
    } else {
        (64, 4)
    };

    let model = Qwen35Model::new(cfg)?;

    // Seed RNG for deterministic quantization init, then quantize
    let mut model = if quant_cfg.is_some() {
        mlx_rs::random::seed(42);
        mlx_rs::nn::quantize(model, group_size, bits)?
    } else {
        model
    };

    // Load weights from safetensors with key remapping.
    let mut all_tensors: std::collections::HashMap<String, Array>;
    {
        use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
        let index_path = model_dir.join("model.safetensors.index.json");
        let all_files: Vec<std::path::PathBuf> = if index_path.exists() {
            let index: serde_json::Value = serde_json::from_reader(std::fs::File::open(&index_path)?)?;
            let weight_map = index["weight_map"].as_object().unwrap();
            let files: HashSet<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();
            files.into_iter().map(|f| model_dir.join(f)).collect()
        } else {
            vec![model_dir.join("model.safetensors")]
        };

        // Load all safetensors
        all_tensors = std::collections::HashMap::new();
        for file in &all_files {
            all_tensors.extend(Array::load_safetensors(file)?);
        }

        // Verify raw dequantize matches Python
        if let (Some(w), Some(s), Some(b)) = (
            all_tensors.get("language_model.model.embed_tokens.weight"),
            all_tensors.get("language_model.model.embed_tokens.scales"),
            all_tensors.get("language_model.model.embed_tokens.biases"),
        ) {
            let tok1_w = w.index((1..2, ..));
            let tok1_s = s.index((1..2, ..));
            let tok1_b = b.index((1..2, ..));
            let deq = mlx_rs::ops::dequantize(&tok1_w, &tok1_s, &tok1_b, 64, 4).unwrap();
            let deq = deq.as_type::<f32>()?;
            deq.eval().ok();
            let d5 = deq.index((.., ..5));
            d5.eval().ok();
            // Python: [0.00238, 0.0000153, 0.00946, -0.00470, 0.00238]
        }

        let mut remapped: std::collections::HashMap<std::rc::Rc<str>, Array> = std::collections::HashMap::new();
        // Note: norm weights do NOT need +1 for this model (mlx-community 4-bit has pre-shifted weights)
        for (key, array) in &all_tensors {
            let array = array.clone();

            let needs_inner = key.ends_with(".weight")
                && !key.contains("switch_mlp")
                && !key.contains("layernorm")
                && !key.contains("norm.")
                && !key.contains("conv1d")
                && !key.contains("A_log")
                && !key.contains("dt_bias")
                && all_tensors.contains_key(&format!("{}.scales", &key[..key.len()-7]));

            if needs_inner {
                let prefix = &key[..key.len() - 7];
                remapped.insert(format!("{}.inner.weight", prefix).into(), array.clone());
            }
            remapped.insert(key.as_str().into(), array);
        }
        // Verify raw dequantize of embed token 1 BEFORE model update
        if let (Some(w), Some(s), Some(b)) = (
            all_tensors.get("language_model.model.embed_tokens.weight"),
            all_tensors.get("language_model.model.embed_tokens.scales"),
            all_tensors.get("language_model.model.embed_tokens.biases"),
        ) {
            let tok1_w = w.index((1..2, ..));
            let tok1_s = s.index((1..2, ..));
            let tok1_b = b.index((1..2, ..));
            let deq = mlx_rs::ops::dequantize(&tok1_w, &tok1_s, &tok1_b, 64, 4).unwrap();
            deq.eval().ok();
            let d5 = deq.index((.., ..5));
            d5.eval().ok();
            // Python: [0.00238, 0.0000153, 0.00946, -0.00470, 0.00238]
        }

        // First: update_flattened for ALL keys (inner.weight + scales + biases + non-quantized)
        model.update_flattened(remapped);

        // Then eval to materialize (like load_safetensors does)
        {
            use mlx_rs::module::ModuleParameters;
            model.eval()?;
        }

        // no extra debug
    }

    // model already quantized before loading

    // Merge a+b projection weights for each GDN layer (saves 1 matmul per layer)
    for layer in &mut model.language_model.model.layers {
        if let Some(ref gdn) = layer.linear_attn {
            // Get quantized weights from a and b projections
            use mlx_rs::quantization::MaybeQuantized;
            let (a_w, a_s, a_b) = match &gdn.in_proj_a {
                MaybeQuantized::Quantized(q) => (&*q.inner.weight, &*q.scales, &*q.biases),
                _ => continue,
            };
            let (b_w, b_s, b_b) = match &gdn.in_proj_b {
                MaybeQuantized::Quantized(q) => (&*q.inner.weight, &*q.scales, &*q.biases),
                _ => continue,
            };
            // Merge b+a weights
            let merged_w = mlx_rs::ops::concatenate_axis(&[b_w, a_w], 0).unwrap();
            let merged_s = mlx_rs::ops::concatenate_axis(&[b_s, a_s], 0).unwrap();
            let merged_b = mlx_rs::ops::concatenate_axis(&[b_b, a_b], 0).unwrap();

            // Merge qkv+z weights
            let (qkv_w, qkv_s, qkv_b) = match &gdn.in_proj_qkv {
                MaybeQuantized::Quantized(q) => (&*q.inner.weight, &*q.scales, &*q.biases),
                _ => continue,
            };
            let (z_w, z_s, z_b) = match &gdn.in_proj_z {
                MaybeQuantized::Quantized(q) => (&*q.inner.weight, &*q.scales, &*q.biases),
                _ => continue,
            };
            let merged_qkvz_w = mlx_rs::ops::concatenate_axis(&[qkv_w, z_w], 0).unwrap();
            let merged_qkvz_s = mlx_rs::ops::concatenate_axis(&[qkv_s, z_s], 0).unwrap();
            let merged_qkvz_b = mlx_rs::ops::concatenate_axis(&[qkv_b, z_b], 0).unwrap();

            if let Some(ref mut gdn) = layer.linear_attn {
                gdn.merged_ab = Some((merged_w, merged_s, merged_b));
                gdn.merged_qkvz = Some((merged_qkvz_w, merged_qkvz_s, merged_qkvz_b));
                // Precompute conv weight for decode: [D, K, 1] → [1, K, D]
                let cw = &*gdn.conv1d.weight;
                let d = cw.dim(0);
                let k = cw.dim(1);
                gdn.conv_weight_decode = Some(
                    cw.reshape(&[d, k]).unwrap()
                    .transpose_axes(&[1, 0]).unwrap()
                    .reshape(&[1, k, d]).unwrap()
                );
            }
        }
    }

    // Pre-dequantize embedding table (1 index op vs 4 ops per token)
    if let (Some(w), Some(s), Some(b)) = (
        all_tensors.get("language_model.model.embed_tokens.weight"),
        all_tensors.get("language_model.model.embed_tokens.scales"),
        all_tensors.get("language_model.model.embed_tokens.biases"),
    ) {
        let embed = mlx_rs::ops::dequantize(w, s, b, 64, 4)?;
        embed.eval()?;
        model.embed_dequant = Some(embed);
    }

    // Store raw weights for lm_head manual dequantize
    model.raw_weights = Some(all_tensors);

    Ok(model)
}
