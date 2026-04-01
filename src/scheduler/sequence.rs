//! Sequence state for a single generation request.

use std::time::Instant;

/// Opaque sequence identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(pub u64);

/// Generation parameters per request.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub eos_tokens: Vec<u32>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            eos_tokens: vec![248056, 248057], // <|im_end|>, <|endoftext|>
        }
    }
}

/// Lifecycle of a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Waiting for prefill (prompt not yet processed).
    Waiting,
    /// Prefill in progress (processing prompt tokens).
    Prefilling,
    /// Actively generating tokens.
    Decoding,
    /// Generation complete (EOS or max tokens).
    Finished,
}

/// Per-sequence model state. Opaque to the scheduler — owned by the executor.
///
/// For Qwen3.5 GDN:
/// - 30 GDN layers: conv_state [K-1, conv_dim] + recurrent_state [Hv, Dv, Dk]  (~1MB/layer)
/// - 10 attention layers: KV cache [nkv, seq_len, hd]  (grows, but tiny with 2 KV heads)
///
/// Total: ~32MB per sequence (fixed GDN) + ~10KB * seq_len (attention, small)
///
/// Generic over the array type so scheduler logic can be tested without MLX.
/// Opaque model state — the executor downcasts to its concrete type.
/// Using Any so the scheduler doesn't depend on mlx-rs.
pub type ModelState = Box<dyn std::any::Any + Send>;

/// A single generation request being processed.
pub struct Sequence {
    pub id: SequenceId,
    pub status: SequenceStatus,
    pub params: GenerationParams,

    /// Model layer states (opaque — managed by executor, downcast by concrete impl).
    pub state: Option<ModelState>,

    /// Prompt token IDs (consumed during prefill).
    pub prompt_tokens: Vec<i32>,

    /// Generated tokens so far.
    pub output_tokens: Vec<u32>,

    /// Current token to feed into the model (for decode steps).
    pub current_token: u32,

    /// Channel to stream tokens back to the caller.
    pub token_sender: Option<tokio::sync::mpsc::UnboundedSender<SchedulerEvent>>,

    // --- Metrics ---
    pub created_at: Instant,
    pub first_token_at: Option<Instant>,
    pub tokens_generated: usize,
}

/// Events emitted per-sequence back to the caller.
#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    Token(u32),
    Finished { reason: FinishReason },
}

#[derive(Debug, Clone, Copy)]
pub enum FinishReason {
    Eos,
    MaxTokens,
    Error,
}

impl Sequence {
    pub fn new(
        id: SequenceId,
        prompt_tokens: Vec<i32>,
        params: GenerationParams,
        token_sender: Option<tokio::sync::mpsc::UnboundedSender<SchedulerEvent>>,
    ) -> Self {
        Self {
            id,
            status: SequenceStatus::Waiting,
            params,
            state: None, // initialized by executor during prefill
            prompt_tokens,
            output_tokens: Vec::new(),
            current_token: 0,
            token_sender,
            created_at: Instant::now(),
            first_token_at: None,
            tokens_generated: 0,
        }
    }

    /// Emit a token to the caller's stream.
    pub fn emit_token(&self, token: u32) {
        if let Some(ref tx) = self.token_sender {
            let _ = tx.send(SchedulerEvent::Token(token));
        }
    }

    /// Signal completion to the caller.
    pub fn emit_finished(&self, reason: FinishReason) {
        if let Some(ref tx) = self.token_sender {
            let _ = tx.send(SchedulerEvent::Finished { reason });
        }
    }

    /// Check if this sequence has hit a stop condition.
    pub fn should_stop(&self, token: u32) -> bool {
        self.params.eos_tokens.contains(&token)
            || self.tokens_generated >= self.params.max_tokens
    }

    /// Time to first token (None if not yet generated).
    pub fn ttft(&self) -> Option<std::time::Duration> {
        self.first_token_at.map(|t| t.duration_since(self.created_at))
    }
}
