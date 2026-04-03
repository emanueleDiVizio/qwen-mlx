//! Token-level scheduler with continuous batching.
//!
//! Design principles:
//! - One token per sequence per iteration (not request-level batching)
//! - Exploit GDN's fixed-size state for cheap context switching
//! - Prefill gets priority over decode (minimize TTFT)
//! - No allocations on the hot path

use std::collections::VecDeque;
use std::time::Instant;

use super::executor::{ModelExecutor, BatchOutput};
use super::sequence::*;

/// Scheduler statistics.
#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    pub total_requests: u64,
    pub active_sequences: usize,
    pub waiting_sequences: usize,
    pub tokens_generated: u64,
    pub iterations: u64,
}

/// Token-level scheduler for concurrent generation.
///
/// # Scheduling policy (Phase 1: Round-robin)
///
/// Each iteration:
/// 1. Prefill any waiting sequences (up to `prefill_budget` per iteration)
/// 2. Pick up to `max_batch_size` active sequences
/// 3. Run one decode step for the batch
/// 4. Emit tokens, requeue or finish each sequence
///
/// GDN advantage: context switching between sequences is FREE (just swap state pointers).
/// No KV cache recomputation needed.
pub struct TokenScheduler {
    /// Sequences waiting for prefill (FIFO).
    waiting: VecDeque<Sequence>,

    /// Sequences actively decoding (round-robin pool).
    active: VecDeque<Sequence>,

    /// How many new sequences to prefill per scheduler iteration.
    /// Higher = lower TTFT but may delay active decode.
    prefill_budget: usize,

    /// Max sequences in a single decode batch.
    max_batch_size: usize,

    /// Next sequence ID.
    next_id: u64,

    /// Stats.
    pub stats: SchedulerStats,
}

impl TokenScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            active: VecDeque::new(),
            prefill_budget: 1, // prefill 1 new request per iteration
            max_batch_size,
            next_id: 1,
            stats: SchedulerStats::default(),
        }
    }

    /// Set how many waiting requests to prefill per iteration.
    pub fn set_prefill_budget(&mut self, budget: usize) {
        self.prefill_budget = budget;
    }

    /// Enqueue a new generation request. Returns the sequence ID.
    pub fn enqueue(
        &mut self,
        prompt_tokens: Vec<i32>,
        params: GenerationParams,
        token_sender: Option<tokio::sync::mpsc::UnboundedSender<SchedulerEvent>>,
    ) -> SequenceId {
        let id = SequenceId(self.next_id);
        self.next_id += 1;

        let seq = Sequence::new(id, prompt_tokens, params, token_sender);
        self.waiting.push_back(seq);
        self.stats.total_requests += 1;
        self.stats.waiting_sequences = self.waiting.len();
        id
    }

    /// Run one scheduler iteration: prefill waiting + decode active.
    ///
    /// This is the hot loop. Called repeatedly by the serving loop.
    /// Returns the number of tokens generated in this iteration.
    pub fn step(&mut self, executor: &mut dyn ModelExecutor) -> Result<usize, Box<dyn std::error::Error>> {
        // Phase 1: Prefill waiting sequences (limited by budget)
        let prefill_count = self.prefill_budget.min(self.waiting.len());
        for _ in 0..prefill_count {
            if let Some(mut seq) = self.waiting.pop_front() {
                seq.status = SequenceStatus::Prefilling;

                match executor.prefill(&mut seq) {
                    Ok(first_token) => {
                        seq.status = SequenceStatus::Decoding;
                        seq.current_token = first_token;
                        seq.first_token_at = Some(Instant::now());
                        seq.tokens_generated = 1;

                        // Emit first token
                        seq.emit_token(first_token);

                        if seq.should_stop(first_token) {
                            seq.status = SequenceStatus::Finished;
                            seq.emit_finished(FinishReason::Eos);
                        } else {
                            self.active.push_back(seq);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Prefill failed for {:?}: {}", seq.id, e);
                        seq.emit_finished(FinishReason::Error);
                    }
                }
            }
        }

        // Phase 2: Decode batch from active sequences
        let batch_size = self.max_batch_size.min(self.active.len());
        if batch_size == 0 {
            self.update_stats();
            return Ok(0);
        }

        // Take batch_size sequences from the front (round-robin)
        let mut batch: Vec<Sequence> = self.active.drain(..batch_size).collect();

        // Build mutable ref slice for executor
        let mut batch_refs: Vec<&mut Sequence> = batch.iter_mut().collect();
        let output = executor.decode_batch(&mut batch_refs)?;

        // Process results
        let mut tokens_emitted = 0;
        for (seq, &token) in batch.iter_mut().zip(output.tokens.iter()) {
            seq.current_token = token;
            seq.output_tokens.push(token);
            seq.tokens_generated += 1;
            tokens_emitted += 1;

            seq.emit_token(token);

            if seq.should_stop(token) {
                seq.status = SequenceStatus::Finished;
                let reason = if seq.params.eos_tokens.contains(&token) {
                    FinishReason::Eos
                } else {
                    FinishReason::MaxTokens
                };
                seq.emit_finished(reason);
            }
        }

        // Requeue unfinished sequences (at the back — round-robin fairness)
        for seq in batch {
            if seq.status != SequenceStatus::Finished {
                self.active.push_back(seq);
            }
        }

        self.stats.tokens_generated += tokens_emitted as u64;
        self.stats.iterations += 1;
        self.update_stats();
        Ok(tokens_emitted)
    }

    /// Take N sequences from the active queue (for external pipeline).
    pub fn take_batch(&mut self, n: usize) -> Vec<Sequence> {
        self.active.drain(..n.min(self.active.len())).collect()
    }

    /// Return a sequence to the active queue (after external processing).
    pub fn return_to_active(&mut self, seq: Sequence) {
        self.active.push_back(seq);
    }

    /// Number of active sequences.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Run prefill for waiting sequences (exposed for external pipeline loop).
    pub fn prefill_waiting(&mut self, executor: &mut dyn ModelExecutor) {
        let prefill_count = self.prefill_budget.min(self.waiting.len());
        for _ in 0..prefill_count {
            if let Some(mut seq) = self.waiting.pop_front() {
                seq.status = SequenceStatus::Prefilling;
                match executor.prefill(&mut seq) {
                    Ok(first_token) => {
                        seq.status = SequenceStatus::Decoding;
                        seq.current_token = first_token;
                        seq.first_token_at = Some(Instant::now());
                        seq.tokens_generated = 1;
                        seq.emit_token(first_token);
                        if seq.should_stop(first_token) {
                            seq.status = SequenceStatus::Finished;
                            seq.emit_finished(FinishReason::Eos);
                        } else {
                            self.active.push_back(seq);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Prefill failed for {:?}: {}", seq.id, e);
                        seq.emit_finished(FinishReason::Error);
                    }
                }
            }
        }
        self.update_stats();
    }

    /// Check if there's any work to do.
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.active.is_empty()
    }

    /// Number of active + waiting sequences.
    pub fn pending_count(&self) -> usize {
        self.waiting.len() + self.active.len()
    }

    fn update_stats(&mut self) {
        self.stats.active_sequences = self.active.len();
        self.stats.waiting_sequences = self.waiting.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock executor that returns deterministic tokens.
    struct MockExecutor {
        token_counter: u32,
    }

    impl MockExecutor {
        fn new() -> Self {
            Self { token_counter: 100 }
        }
    }

    impl ModelExecutor for MockExecutor {
        fn prefill(&mut self, seq: &mut Sequence) -> Result<u32, Box<dyn std::error::Error>> {
            // Initialize mock state
            seq.state = Some(Box::new(vec![0u8; 40]));
            self.token_counter += 1;
            Ok(self.token_counter)
        }

        fn decode_batch(&mut self, sequences: &mut [&mut Sequence]) -> Result<BatchOutput, Box<dyn std::error::Error>> {
            let tokens: Vec<u32> = sequences.iter().map(|_| {
                self.token_counter += 1;
                self.token_counter
            }).collect();
            Ok(BatchOutput { tokens })
        }

        fn max_batch_size(&self) -> usize { 8 }
    }

    #[test]
    fn test_single_sequence() {
        let mut sched = TokenScheduler::new(4);
        let mut exec = MockExecutor::new();

        let params = GenerationParams {
            max_tokens: 5,
            ..Default::default()
        };

        sched.enqueue(vec![1, 2, 3], params, None);
        assert_eq!(sched.pending_count(), 1);

        // Step 1: prefill (generates first token, moves to active)
        sched.step(&mut exec).unwrap();
        assert_eq!(sched.active.len(), 1);

        // Decode until done
        let mut total = 0;
        while sched.has_work() {
            total += sched.step(&mut exec).unwrap();
        }
        // prefill=1 token, decode=4 tokens, total=5=max_tokens
        assert!(total > 0);
        assert!(!sched.has_work());
    }

    #[test]
    fn test_interleaved_sequences() {
        let mut sched = TokenScheduler::new(4);
        let mut exec = MockExecutor::new();

        let params = GenerationParams {
            max_tokens: 3,
            ..Default::default()
        };

        sched.enqueue(vec![1], params.clone(), None);
        sched.enqueue(vec![2], params.clone(), None);
        sched.enqueue(vec![3], params, None);

        // Run until all done
        while sched.has_work() {
            sched.step(&mut exec).unwrap();
        }

        assert_eq!(sched.stats.total_requests, 3);
        // Each seq: 1 prefill token + 2 decode tokens = 3 total, 2 counted as decode
        assert_eq!(sched.stats.tokens_generated, 6); // 3 seqs × 2 decode tokens
    }

    #[test]
    fn test_round_robin_fairness() {
        let mut sched = TokenScheduler::new(1); // batch=1, strict round-robin
        let mut exec = MockExecutor::new();

        let params = GenerationParams {
            max_tokens: 100,
            ..Default::default()
        };

        sched.enqueue(vec![1], params.clone(), None);
        sched.enqueue(vec![2], params, None);

        // Prefill both
        sched.step(&mut exec).unwrap(); // prefill seq 1
        sched.step(&mut exec).unwrap(); // prefill seq 2

        // Now both active. With batch=1, they alternate:
        // decode seq 1, decode seq 2, decode seq 1, ...
        let id_0 = sched.active[0].id;
        let id_1 = sched.active[1].id;

        sched.step(&mut exec).unwrap();
        // After decode, seq at front was processed and moved to back
        assert_eq!(sched.active[0].id, id_1); // seq 2 is now front
        assert_eq!(sched.active[1].id, id_0); // seq 1 moved to back
    }
}
