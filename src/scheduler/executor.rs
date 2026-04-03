//! Model executor trait — decouples scheduling from model implementation.
//!
//! The executor handles:
//! - Prefill: process prompt tokens, initialize sequence state
//! - Decode: generate one token per sequence in a batch
//!
//! The scheduler never touches MLX arrays directly.

use super::sequence::{Sequence, SequenceId};

/// Output of a batched decode step.
pub struct BatchOutput {
    /// One sampled token per sequence in the batch (same order as input).
    pub tokens: Vec<u32>,
}

/// Trait for the model execution backend.
///
/// Implementations handle batching, GPU dispatch, and state management.
/// The scheduler calls these methods and handles sequencing/fairness.
pub trait ModelExecutor {
    /// Prefill a single sequence: process its prompt and initialize model state.
    fn prefill(&mut self, seq: &mut Sequence) -> Result<u32, Box<dyn std::error::Error>>;

    /// Generate exactly one token for each sequence in the batch.
    fn decode_batch(&mut self, sequences: &mut [&mut Sequence]) -> Result<BatchOutput, Box<dyn std::error::Error>>;

    /// Build graph + async eval for batch, but DON'T wait for results.
    /// Call `collect_batch()` later to get the tokens.
    /// Returns the number of sequences submitted.
    fn submit_batch(&mut self, sequences: &mut [&mut Sequence]) -> Result<usize, Box<dyn std::error::Error>> {
        // Default: just call decode_batch (no pipelining)
        let _ = sequences;
        Err("submit_batch not implemented".into())
    }

    /// Collect results from a previous `submit_batch()`.
    /// Blocks until GPU finishes.
    fn collect_batch(&mut self, sequences: &mut [&mut Sequence]) -> Result<BatchOutput, Box<dyn std::error::Error>> {
        let _ = sequences;
        Err("collect_batch not implemented".into())
    }

    /// Whether this executor supports submit/collect pipelining.
    fn supports_pipeline(&self) -> bool { false }

    /// Maximum batch size the executor supports.
    fn max_batch_size(&self) -> usize;
}
