//! Demo: multi-user concurrent token generation with the scheduler.
//!
//! Simulates N users sending prompts concurrently and shows
//! per-token interleaved generation with fairness.

use std::time::Instant;
use qwen_mlx::scheduler::*;

/// Mock executor for demo purposes.
/// In production, this wraps the actual Qwen3.5 model with batched forward.
struct DemoExecutor {
    counter: u32,
    /// Simulated tokens per sequence (for demo output)
    vocab: Vec<&'static str>,
}

impl DemoExecutor {
    fn new() -> Self {
        Self {
            counter: 0,
            vocab: vec![
                "The", " answer", " is", " that", " quantum", " computing",
                " uses", " qubits", " for", " parallel", " computation", ".",
                " This", " enables", " solving", " complex", " problems",
                " faster", " than", " classical", " computers", ".",
            ],
        }
    }
}

impl ModelExecutor for DemoExecutor {
    fn prefill(&mut self, seq: &mut Sequence) -> Result<u32, Box<dyn std::error::Error>> {
        // Simulate state initialization (~32MB per sequence for real model)
        seq.state = Some(Box::new(vec![0u8; 40]));
        self.counter += 1;
        Ok(self.counter + 100) // first token
    }

    fn decode_batch(&mut self, sequences: &mut [&mut Sequence]) -> Result<BatchOutput, Box<dyn std::error::Error>> {
        let tokens: Vec<u32> = sequences.iter().map(|seq| {
            // Deterministic token based on sequence progress
            let idx = seq.tokens_generated % self.vocab.len();
            (idx as u32) + 200
        }).collect();
        Ok(BatchOutput { tokens })
    }

    fn max_batch_size(&self) -> usize { 16 }
}

fn main() {
    println!("=== mlx-qwen Token Scheduler Demo ===\n");

    let mut scheduler = TokenScheduler::new(4); // max batch size 4
    scheduler.set_prefill_budget(2); // prefill up to 2 new requests per iteration

    let mut executor = DemoExecutor::new();

    // Simulate 5 concurrent users with different prompts
    let prompts = vec![
        ("Alice", vec![1, 2, 3, 4, 5]),
        ("Bob",   vec![10, 20, 30]),
        ("Carol", vec![100, 200]),
        ("Dave",  vec![1000]),
        ("Eve",   vec![2000, 3000, 4000, 5000, 6000]),
    ];

    // Stagger arrivals: enqueue first 3 immediately, rest after a few iterations
    let mut user_names: Vec<&str> = Vec::new();
    let mut seq_ids: Vec<SequenceId> = Vec::new();

    for (name, tokens) in &prompts[..3] {
        let params = GenerationParams {
            max_tokens: 10,
            ..Default::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let id = scheduler.enqueue(tokens.clone(), params, Some(tx));
        user_names.push(name);
        seq_ids.push(id);
        println!("  Enqueued {} (seq {:?}, {} prompt tokens)", name, id, tokens.len());
    }

    println!("\n--- Scheduling loop ---\n");

    let t0 = Instant::now();
    let mut iteration = 0;

    loop {
        // Stagger: add Dave at iteration 3, Eve at iteration 5
        if iteration == 3 {
            let params = GenerationParams { max_tokens: 8, ..Default::default() };
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let id = scheduler.enqueue(prompts[3].1.clone(), params, Some(tx));
            user_names.push(prompts[3].0);
            seq_ids.push(id);
            println!("  → Dave joins (seq {:?})", id);
        }
        if iteration == 5 {
            let params = GenerationParams { max_tokens: 12, ..Default::default() };
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let id = scheduler.enqueue(prompts[4].1.clone(), params, Some(tx));
            user_names.push(prompts[4].0);
            seq_ids.push(id);
            println!("  → Eve joins (seq {:?})", id);
        }

        if !scheduler.has_work() {
            break;
        }

        let tokens = scheduler.step(&mut executor).unwrap();
        let stats = &scheduler.stats;

        println!(
            "  iter {:2} | batch={} | waiting={} active={} | +{} tokens (total: {})",
            iteration,
            tokens,
            stats.waiting_sequences,
            stats.active_sequences,
            tokens,
            stats.tokens_generated,
        );

        iteration += 1;
        if iteration > 100 { break; } // safety
    }

    let elapsed = t0.elapsed();
    println!("\n--- Done ---");
    println!("  {} iterations in {:.1}ms", iteration, elapsed.as_secs_f64() * 1000.0);
    println!("  {} total tokens generated", scheduler.stats.tokens_generated);
    println!("  {} total requests served", scheduler.stats.total_requests);
    println!(
        "  {:.0} simulated tok/s",
        scheduler.stats.tokens_generated as f64 / elapsed.as_secs_f64()
    );
}
