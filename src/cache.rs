//! Cache types for Qwen3.5 hybrid architecture.
//!
//! GDN layers use ArraysCache (conv state + recurrent state).
//! Full attention layers use KVCache (keys + values concatenation).

use mlx_rs::Array;

/// KV cache for full attention layers.
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Append new keys/values and return the full cache.
    pub fn update(&mut self, keys: &Array, values: &Array) -> (Array, Array) {
        let (k, v) = if let (Some(prev_k), Some(prev_v)) = (&self.keys, &self.values) {
            (
                mlx_rs::ops::concatenate(&[prev_k, keys], 2).unwrap(),
                mlx_rs::ops::concatenate(&[prev_v, values], 2).unwrap(),
            )
        } else {
            (keys.clone(), values.clone())
        };
        self.offset += keys.dim(2) as usize;
        self.keys = Some(k.clone());
        self.values = Some(v.clone());
        (k, v)
    }
}

/// Slot-based cache for GDN layers (conv state + recurrent state).
pub struct ArraysCache {
    slots: Vec<Option<Array>>,
}

impl ArraysCache {
    pub fn new(size: usize) -> Self {
        Self {
            slots: vec![None; size],
        }
    }

    pub fn get(&self, idx: usize) -> Option<&Array> {
        self.slots.get(idx).and_then(|s| s.as_ref())
    }

    pub fn set(&mut self, idx: usize, array: Array) {
        if idx < self.slots.len() {
            self.slots[idx] = Some(array);
        }
    }

    /// Returns the offset (number of tokens processed).
    /// For GDN, this is derived from the conv state width.
    pub fn offset(&self) -> usize {
        // Conv state at slot 0: shape [B, D, kernel_size-1]
        // After first token, offset = kernel_size - 1 + tokens_processed
        // We track this externally
        0
    }
}

/// Unified cache enum for the hybrid model.
pub enum LayerCache {
    Attention(KVCache),
    Gdn(ArraysCache),
}

impl LayerCache {
    pub fn as_kv(&mut self) -> &mut KVCache {
        match self {
            LayerCache::Attention(kv) => kv,
            _ => panic!("Expected KVCache"),
        }
    }

    pub fn as_gdn(&mut self) -> &mut ArraysCache {
        match self {
            LayerCache::Gdn(ac) => ac,
            _ => panic!("Expected ArraysCache"),
        }
    }
}
