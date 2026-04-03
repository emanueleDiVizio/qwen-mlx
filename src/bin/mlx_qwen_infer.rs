//! MLX inference binary for Qwen3.5-35B-A3B on Apple Silicon.
//!
//! Usage:
//!   qwen_mlx_infer --model-dir <path> [--prompt "text"]
//!   qwen_mlx_infer --model-id <hf_model_id> [--token-source <source>] [--revision <rev>] [--prompt "text"]
//!
//! Token source formats (only needed for gated models):
//!   - "none" (default): No token (works for public models)
//!   - "cache": Read from ~/.cache/huggingface/token
//!   - "env:VAR_NAME": Read from environment variable
//!   - "literal:TOKEN": Use literal token value

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use mlx_rs::ops::indexing::IndexOp;
use qwen_mlx::hf::{download_model, TokenSource};
use qwen_mlx::models::qwen3_5;

fn main() {
    // Initialize tracing for download progress
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let model_dir_arg = args.iter().position(|a| a == "--model-dir")
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
    let model_id_arg = args.iter().position(|a| a == "--model-id")
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
    let token_source_arg = args.iter().position(|a| a == "--token-source")
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
        .unwrap_or("none");
    let revision_arg = args.iter().position(|a| a == "--revision")
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
    let prompt = args.iter().position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
        .unwrap_or("Hello! How are you today?");

    // Determine model directory: --model-dir takes precedence, otherwise download from HF
    let model_dir = if let Some(dir) = model_dir_arg {
        PathBuf::from(dir)
    } else if let Some(model_id) = model_id_arg {
        let token_source = TokenSource::parse(token_source_arg)
            .expect("Invalid token source format");
        download_model(model_id, token_source, revision_arg)
            .expect("Failed to download model from Hugging Face")
    } else {
        eprintln!("Error: Either --model-dir or --model-id must be provided");
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  qwen_mlx_infer --model-dir <path> [--prompt \"text\"]");
        eprintln!("  qwen_mlx_infer --model-id <hf_model_id> [--token-source <source>] [--revision <rev>] [--prompt \"text\"]");
        eprintln!();
        eprintln!("Token source formats (only needed for gated models):");
        eprintln!("  none          No token, for public models (default)");
        eprintln!("  cache         Read from ~/.cache/huggingface/token");
        eprintln!("  env:VAR_NAME  Read from environment variable");
        eprintln!("  literal:TOKEN Use literal token value");
        std::process::exit(1);
    };

    println!("Loading model from {:?}...", model_dir);
    let t0 = Instant::now();
    let mut model = qwen3_5::load_model(&model_dir).expect("Failed to load model");
    println!("Model loaded: {} layers in {:.1}s",
        model.config.num_hidden_layers, t0.elapsed().as_secs_f32());

    println!("Config: hidden_size={}, head_dim={}, vocab_size={}, linear_k_heads={}, linear_v_heads={}, linear_k_dim={}, linear_v_dim={}",
        model.config.hidden_size, model.config.head_dim, model.config.vocab_size,
        model.config.linear_num_key_heads, model.config.linear_num_value_heads,
        model.config.linear_key_head_dim, model.config.linear_value_head_dim);

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .expect("Failed to load tokenizer");

    // Wrap in Qwen chat template
    let chat_prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );
    println!("Prompt: {}", prompt);

    // Encode prompt (no special token addition — template has them)
    let encoding = tokenizer.encode(chat_prompt.as_str(), false).expect("Encode failed");
    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let seq_len = token_ids.len();
    let input = mlx_rs::Array::from_slice(&token_ids, &[1, seq_len as i32]);

    let mut caches: Vec<(Option<mlx_rs::Array>, Option<mlx_rs::Array>)> = Vec::new();
    let max_tokens = 200;
    let eos_tokens: &[u32] = &[248056, 248057]; // <|im_end|>, <|endoftext|>

    // Prefill
    print!("\n> ");
    std::io::stdout().flush().unwrap();

    let t_prefill = Instant::now();
    let logits = model.forward(&input, &mut caches).expect("Prefill failed");
    let last_logits = logits.index((.., -1, ..));
    let temp = 0.7f32;
    let inv_temp = mlx_rs::Array::from_slice(&[1.0 / temp], &[1]);
    let scaled = last_logits.multiply(inv_temp.clone()).unwrap();
    let mut next_token_arr = mlx_rs::random::categorical(&scaled, -1, None, None).unwrap();
    next_token_arr.eval().expect("eval failed");
    let prefill_time = t_prefill.elapsed();

    let mut current_token: u32 = next_token_arr.item();

    // Print first token
    let text = tokenizer.decode(&[current_token], true).unwrap_or_default();
    print!("{}", text);
    std::io::stdout().flush().unwrap();

    let mut n_generated = 1u32;

    // True double-buffered decode (matching Python mlx-lm exactly):
    // 1. Build graph N+1 using lazy output of step N
    // 2. Async-eval graph N+1
    // 3. Read result of step N (GPU was processing it during our graph building)
    let inv_temp = mlx_rs::Array::from_slice(&[1.0 / temp], &[1]);

    let async_eval_arr = |arr: &mlx_rs::Array| unsafe {
        let v = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(v, arr.as_ptr());
        mlx_sys::mlx_async_eval(v);
        mlx_sys::mlx_vector_array_free(v);
    };

    // Build step: forward + sample
    let do_step = |token: &mlx_rs::Array, model: &mut qwen_mlx::models::qwen3_5::Qwen35Model,
                   caches: &mut Vec<(Option<mlx_rs::Array>, Option<mlx_rs::Array>)>,
                   inv_temp: &mlx_rs::Array| -> mlx_rs::Array {
        let logits = model.forward(token, caches).expect("Decode failed");
        // For S=1 decode: logits is [1,1,V], reshape to [1,V] instead of index
        let l = logits.reshape(&[1, -1]).unwrap();
        let scaled = l.multiply(inv_temp.clone()).unwrap();
        mlx_rs::random::categorical(&scaled, -1, None, None).expect("sampling failed")
    };

    let t_decode = Instant::now();
    // Kick off first decode step
    let mut y = next_token_arr;
    let mut token_input = y.as_type::<i32>().unwrap().reshape(&[1, 1]).unwrap();
    let mut next_y = do_step(&token_input, &mut model, &mut caches, &inv_temp);
    async_eval_arr(&next_y);

    for n in 0..max_tokens {
        // Read PREVIOUS result (GPU was working on it during graph build)
        current_token = y.item();

        if eos_tokens.contains(&current_token) {
            break;
        }

        let text = tokenizer.decode(&[current_token], true).unwrap_or_default();
        print!("{}", text);
        std::io::stdout().flush().unwrap();
        n_generated += 1;

        // Swap: current becomes previous (clone for ref-counted array)
        y = next_y.clone();

        if n + 1 < max_tokens {
            // Build NEXT graph (lazy — using output of current step, not yet evaluated)
            token_input = y.as_type::<i32>().unwrap().reshape(&[1, 1]).unwrap();
            next_y = do_step(&token_input, &mut model, &mut caches, &inv_temp);
            async_eval_arr(&next_y);
        }

        if n_generated % 256 == 0 {
            unsafe { mlx_sys::mlx_clear_cache(); }
        }
    }
    let decode_time = t_decode.elapsed();
    println!("\n");

    println!("--- Stats ---");
    println!("Prefill: {} tokens in {:.3}s ({:.1} tok/s)",
        seq_len, prefill_time.as_secs_f32(),
        seq_len as f32 / prefill_time.as_secs_f32());
    if n_generated > 0 {
        println!("Decode: {} tokens in {:.3}s ({:.1} tok/s)",
            n_generated, decode_time.as_secs_f32(),
            n_generated as f32 / decode_time.as_secs_f32());
    }
}
