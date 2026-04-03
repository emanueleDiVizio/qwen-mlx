//! Benchmark: multi-user concurrent inference with real model.
//!
//! Usage:
//!   multi_user_bench --model-dir <path> --users N [--tokens-per-user M]
//!   multi_user_bench --model-id <hf_model_id> [--token-source <source>] --users N [--tokens-per-user M]

use std::path::PathBuf;
use std::time::Instant;

use qwen_mlx::hf::{download_model, TokenSource};
use qwen_mlx::scheduler::*;
use qwen_mlx::scheduler::mlx_executor::Qwen35Executor;

fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    let model_dir_arg = args.iter().position(|a| a == "--model-dir")
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
    let model_id_arg = args.iter().position(|a| a == "--model-id")
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
    let token_source_arg = args.iter().position(|a| a == "--token-source")
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
        .unwrap_or("none");

    let model_dir = if let Some(dir) = model_dir_arg {
        PathBuf::from(dir)
    } else if let Some(model_id) = model_id_arg {
        let token_source = TokenSource::parse(token_source_arg)
            .expect("Invalid token source format");
        download_model(model_id, token_source, None)
            .expect("Failed to download model from Hugging Face")
    } else {
        eprintln!("Usage:");
        eprintln!("  multi_user_bench --model-dir <path> --users N [--tokens-per-user M]");
        eprintln!("  multi_user_bench --model-id <hf_model_id> [--token-source <source>] --users N");
        std::process::exit(1);
    };
    let num_users: usize = args.iter().position(|a| a == "--users")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let tokens_per_user: usize = args.iter().position(|a| a == "--tokens-per-user")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let batch_size: usize = args.iter().position(|a| a == "--batch")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(num_users);

    println!("=== Multi-User Inference Benchmark ===");
    println!("  Model: {:?}", model_dir);
    println!("  Users: {}", num_users);
    println!("  Tokens/user: {}", tokens_per_user);
    println!("  Batch size: {}", batch_size);

    // Load model
    println!("\nLoading model...");
    let t0 = Instant::now();
    let mut executor = Qwen35Executor::load(&model_dir, 0.7)
        .expect("Failed to load model");
    println!("  Model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Load tokenizer for prompt encoding
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .expect("Failed to load tokenizer");

    // Create scheduler
    let mut scheduler = TokenScheduler::new(batch_size);
    scheduler.set_prefill_budget(1); // prefill one at a time for fairness

    // Prompts for each user
    let prompts = [
        "What is quantum computing?",
        "Explain the theory of relativity.",
        "Write a poem about the ocean.",
        "What are the laws of thermodynamics?",
        "How does machine learning work?",
        "Describe the solar system.",
        "What is the meaning of life?",
        "Explain how computers work.",
        "What is consciousness?",
        "Describe the water cycle.",
        "How do neural networks learn?",
        "What is dark matter?",
        "Explain evolution by natural selection.",
        "What causes earthquakes?",
        "How does the internet work?",
        "What is string theory?",
    ];

    // Enqueue all users
    println!("\nEnqueuing {} users...", num_users);
    let mut receivers = Vec::new();

    for i in 0..num_users {
        let prompt = prompts[i % prompts.len()];
        let chat = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt);
        let encoding = tokenizer.encode(chat.as_str(), false).expect("Encode failed");
        let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();

        let params = GenerationParams {
            max_tokens: tokens_per_user,
            temperature: 0.7,
            ..Default::default()
        };

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let id = scheduler.enqueue(token_ids, params, Some(tx));
        receivers.push((id, prompt, rx));
        println!("  User {} (seq {:?}): \"{}\" ", i, id, prompt);
    }

    // Run scheduling loop
    println!("\n--- Generating ---\n");
    let t_start = Instant::now();
    let mut iteration = 0;

    while scheduler.has_work() {
        let tokens = scheduler.step(&mut executor).expect("Scheduler step failed");
        iteration += 1;

        if iteration % 10 == 0 || !scheduler.has_work() {
            let elapsed = t_start.elapsed().as_secs_f64();
            let stats = &scheduler.stats;
            let tps = stats.tokens_generated as f64 / elapsed;
            println!(
                "  iter {:4} | active={:2} waiting={:2} | {} total tokens | {:.1} tok/s aggregate",
                iteration, stats.active_sequences, stats.waiting_sequences,
                stats.tokens_generated, tps,
            );
        }
    }

    let total_time = t_start.elapsed();

    // Collect outputs
    println!("\n--- Results ---\n");
    for (id, prompt, mut rx) in receivers {
        let mut output_tokens = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                SchedulerEvent::Token(t) => output_tokens.push(t),
                SchedulerEvent::Finished { reason } => {
                    let text = tokenizer.decode(&output_tokens, true).unwrap_or_default();
                    let preview: String = text.chars().take(200).collect();
                    println!("  {:?} [{:?}] \"{}\":\n    {}...\n",
                        id, reason, prompt, preview);
                }
            }
        }
    }

    println!("\n--- Stats ---");
    println!("  Total time: {:.2}s", total_time.as_secs_f32());
    println!("  Total tokens: {}", scheduler.stats.tokens_generated);
    println!("  Iterations: {}", iteration);
    println!("  Aggregate throughput: {:.1} tok/s",
        scheduler.stats.tokens_generated as f64 / total_time.as_secs_f64());
    println!("  Per-user throughput: {:.1} tok/s",
        scheduler.stats.tokens_generated as f64 / total_time.as_secs_f64() / num_users as f64);
    println!("  Users served: {}", num_users);
}
