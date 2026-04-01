//! Real MLX model executor for Qwen3.5 — implements ModelExecutor trait.
//!
//! Supports true batched forward: stacks B per-sequence states into [B, ...]
//! tensors, runs ONE forward pass, then unstacks back.
//!
//! GDN state: fixed-size [1, Hv, Dv, Dk] per sequence → easy to stack
//! Attention KV: variable-length → padded to max_len in batch

use std::path::Path;

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use super::executor::{ModelExecutor, BatchOutput};
use super::sequence::Sequence;
use crate::models::qwen3_5::{self, Qwen35Model};

/// Per-sequence model state: layer caches for GDN + attention.
pub struct Qwen35State {
    pub caches: Vec<(Option<Array>, Option<Array>)>,
}

/// MLX-backed executor for Qwen3.5-35B-A3B.
pub struct Qwen35Executor {
    model: Qwen35Model,
    inv_temp: Array,
    num_layers: usize,
}

impl Qwen35Executor {
    pub fn load(model_dir: &Path, temperature: f32) -> Result<Self, Box<dyn std::error::Error>> {
        let model = qwen3_5::load_model(model_dir)?;
        let num_layers = model.config.num_hidden_layers as usize;
        let inv_temp = Array::from_slice(&[1.0 / temperature], &[1]);
        Ok(Self { model, inv_temp, num_layers })
    }

    /// Stack optional arrays, padding any variable-length dimensions to max.
    /// Arrays with shape [1, X, Y, Z] where Y varies get padded to max Y.
    fn stack_optional_arrays(arrays: &[Option<&Array>]) -> Option<Array> {
        if !arrays.iter().all(|c| c.is_some()) {
            return None;
        }
        let arrs: Vec<&Array> = arrays.iter().map(|c| c.unwrap()).collect();
        if arrs.is_empty() { return None; }

        // Check if all shapes match (fast path)
        let shapes: Vec<Vec<i32>> = arrs.iter().map(|a| a.shape().to_vec()).collect();
        let all_same = shapes.windows(2).all(|w| w[0] == w[1]);

        if all_same {
            return Some(mlx_rs::ops::concatenate_axis(&arrs, 0).unwrap());
        }

        // Variable shapes (attention KV caches with different seq_len)
        // Find the dim that differs and pad with zeros
        let ndim = shapes[0].len();
        let mut max_shape = shapes[0].clone();
        for s in &shapes[1..] {
            for d in 0..ndim {
                max_shape[d] = max_shape[d].max(s[d]);
            }
        }

        // Pad each array by concatenating zeros along the variable dimension
        let padded: Vec<Array> = arrs.iter().zip(shapes.iter()).map(|(arr, shape)| {
            // Find which dim needs padding (usually dim 2 for KV [B, heads, seq_len, hd])
            let needs_pad: Vec<(usize, i32)> = (0..ndim)
                .filter(|&d| shape[d] < max_shape[d])
                .map(|d| (d, max_shape[d] - shape[d]))
                .collect();

            if needs_pad.is_empty() {
                (*arr).clone()
            } else {
                // Pad the first variable dimension (seq_len for KV cache)
                let (dim, pad_amount) = needs_pad[0];
                let mut pad_shape = shape.clone();
                pad_shape[dim] = pad_amount;
                let zeros = mlx_rs::ops::zeros_dtype(&pad_shape, arr.dtype()).unwrap();
                mlx_rs::ops::concatenate_axis(&[*arr, &zeros], dim as i32).unwrap()
            }
        }).collect();

        let refs: Vec<&Array> = padded.iter().collect();
        Some(mlx_rs::ops::concatenate_axis(&refs, 0).unwrap())
    }

    /// Stack B per-sequence caches into batched caches [B, ...] for each layer.
    fn stack_caches(&self, sequences: &mut [&mut Sequence]) -> Vec<(Option<Array>, Option<Array>)> {
        let b = sequences.len();
        let mut batched = Vec::with_capacity(self.num_layers);

        for layer_idx in 0..self.num_layers {
            // Collect this layer's cache from each sequence
            let cache_0s: Vec<Option<&Array>> = sequences.iter().map(|seq| {
                let state = seq.state.as_ref().unwrap().downcast_ref::<Qwen35State>().unwrap();
                state.caches[layer_idx].0.as_ref()
            }).collect();

            let cache_1s: Vec<Option<&Array>> = sequences.iter().map(|seq| {
                let state = seq.state.as_ref().unwrap().downcast_ref::<Qwen35State>().unwrap();
                state.caches[layer_idx].1.as_ref()
            }).collect();

            // Stack caches, padding variable-length dims (attention KV) to max length
            let stacked_0 = Self::stack_optional_arrays(&cache_0s);
            let stacked_1 = Self::stack_optional_arrays(&cache_1s);

            batched.push((stacked_0, stacked_1));
        }
        batched
    }

    /// Unstack batched caches [B, ...] back into per-sequence caches [1, ...].
    fn unstack_caches(&self, sequences: &mut [&mut Sequence], batched: Vec<(Option<Array>, Option<Array>)>) {
        let b = sequences.len() as i32;

        for (layer_idx, (cache_0, cache_1)) in batched.into_iter().enumerate() {
            for (seq_idx, seq) in sequences.iter_mut().enumerate() {
                let state = seq.state.as_mut().unwrap().downcast_mut::<Qwen35State>().unwrap();
                let i = seq_idx as i32;

                state.caches[layer_idx].0 = cache_0.as_ref().map(|c| c.index((i..i+1, .., ..)));
                state.caches[layer_idx].1 = cache_1.as_ref().map(|c| c.index((i..i+1, .., ..)));
            }
        }
    }
}

impl ModelExecutor for Qwen35Executor {
    fn prefill(&mut self, seq: &mut Sequence) -> Result<u32, Box<dyn std::error::Error>> {
        let seq_len = seq.prompt_tokens.len() as i32;
        let input = Array::from_slice(&seq.prompt_tokens, &[1, seq_len]);

        let mut caches: Vec<(Option<Array>, Option<Array>)> = Vec::new();
        let logits = self.model.forward(&input, &mut caches)?;

        let last = logits.index((.., -1, ..));
        let scaled = last.multiply(self.inv_temp.clone())?;
        let token_arr = mlx_rs::random::categorical(&scaled, -1, None, None)?;
        token_arr.eval()?;
        let token: u32 = token_arr.item();

        seq.state = Some(Box::new(Qwen35State { caches }));
        Ok(token)
    }

    fn decode_batch(&mut self, sequences: &mut [&mut Sequence]) -> Result<BatchOutput, Box<dyn std::error::Error>> {
        let b = sequences.len();

        if b == 1 {
            // Fast path: single sequence, no stacking overhead
            let seq = &mut sequences[0];
            let state = seq.state.as_mut().unwrap().downcast_mut::<Qwen35State>().unwrap();

            let input = Array::from_slice(&[seq.current_token as i32], &[1, 1]);
            let logits = self.model.forward(&input, &mut state.caches)?;
            let last = logits.reshape(&[1, -1])?;
            let scaled = last.multiply(self.inv_temp.clone())?;
            let token_arr = mlx_rs::random::categorical(&scaled, -1, None, None)?;
            token_arr.eval()?;
            let token: u32 = token_arr.item();

            return Ok(BatchOutput { tokens: vec![token] });
        }

        // True batched path: stack states → single forward → unstack
        // 1. Stack input tokens: [B, 1]
        let tokens: Vec<i32> = sequences.iter().map(|s| s.current_token as i32).collect();
        let input = Array::from_slice(&tokens, &[b as i32, 1]);

        // 2. Stack per-layer caches: [1,...] × B → [B,...]
        let mut batched_caches = self.stack_caches(sequences);

        // 3. Single batched forward
        let logits = self.model.forward(&input, &mut batched_caches)?;

        // 4. Sample each sequence: logits [B, 1, vocab] → [B, vocab]
        let logits_flat = logits.reshape(&[b as i32, -1])?;
        let scaled = logits_flat.multiply(self.inv_temp.clone())?;
        let token_arr = mlx_rs::random::categorical(&scaled, -1, None, None)?;
        token_arr.eval()?;

        // Extract per-sequence tokens
        let result_tokens: Vec<u32> = (0..b).map(|i| {
            let t = token_arr.index(i as i32);
            t.item()
        }).collect();

        // 5. Unstack caches back to per-sequence
        self.unstack_caches(sequences, batched_caches);

        Ok(BatchOutput { tokens: result_tokens })
    }

    fn max_batch_size(&self) -> usize { 16 }
}
