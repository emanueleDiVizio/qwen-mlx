//! Token-level scheduler for concurrent multi-user inference.
//!
//! Designed for GDN architecture: fixed per-sequence state, cheap context switching,
//! constant per-token cost. Enables hundreds of concurrent users.
//!
//! # Architecture
//!
//! ```text
//! HTTP Request → enqueue(request) → waiting_queue
//!                                         ↓
//!                                   pick_batch(N)
//!                                         ↓
//!                              model.forward_batch(N)
//!                                         ↓
//!                              on_token_generated(seq, tok)
//!                                    ↓           ↓
//!                              requeue(seq)   finish(seq) → emit to caller
//! ```

mod sequence;
mod scheduler;
mod executor;
pub mod mlx_executor;

pub use sequence::{Sequence, SequenceId, SequenceStatus, GenerationParams, SchedulerEvent, FinishReason};
pub use scheduler::{TokenScheduler, SchedulerStats};
pub use executor::{ModelExecutor, BatchOutput};
