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
    ///
    /// After prefill:
    /// - `seq.state` is populated with layer states
    /// - `seq.current_token` is set to the first generated token
    /// - `seq.status` transitions to `Decoding`
    ///
    /// Returns the first generated token.
    fn prefill(&mut self, seq: &mut Sequence) -> Result<u32, Box<dyn std::error::Error>>;

    /// Generate exactly one token for each sequence in the batch.
    ///
    /// All sequences must be in `Decoding` status.
    /// Each sequence's `current_token` is used as input, and its `state` is updated.
    ///
    /// For B sequences, this issues ONE batched forward pass (B tokens in parallel).
    fn decode_batch(&mut self, sequences: &mut [&mut Sequence]) -> Result<BatchOutput, Box<dyn std::error::Error>>;

    /// Maximum batch size the executor supports.
    fn max_batch_size(&self) -> usize;
}
