# qwen-mlx

A Rust inference engine for [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx) via [mlx-rs](https://github.com/oxideai/mlx-rs).

**112 tok/s single-user decode** (matching Python mlx-lm) and **200+ tok/s aggregate** with concurrent multi-user serving on M3 Ultra.

## Why this exists

Qwen3.5 uses a **GatedDeltaNet (GDN)** hybrid architecture: 30 recurrent layers + 10 attention layers + MoE. The recurrent layers maintain a fixed-size state (~1MB/layer) instead of a growing KV cache. This makes multi-user serving radically cheaper than traditional transformers:

| | Transformer (KV cache) | GDN (recurrent state) |
|---|---|---|
| Per-user memory at 1K tokens | ~50MB | ~32MB (fixed) |
| Per-user memory at 10K tokens | ~500MB | ~32MB (fixed) |
| Context switch cost | Reload KV cache | Swap state pointer |
| Max concurrent users (96GB) | ~20-30 | Hundreds |

This engine exploits that property with a token-level scheduler that interleaves users at every decode step.

## Performance

Measured on Mac Studio M3 Ultra (96GB), Qwen3.5-35B-A3B 4-bit quantized:

### Single-user decode

| Engine | Decode tok/s | Prefill tok/s |
|--------|-------------|---------------|
| **mlx-qwen (Rust)** | **112** | 125 |
| mlx-lm (Python) | 113 | 256 |
| mistral.rs (candle) | ~15 | — |
| llama.cpp (GGUF Q8) | 73 | 771 |

### Multi-user concurrent (token-level scheduling)

| Concurrent users | Aggregate tok/s | Per-user tok/s | Throughput scaling |
|-----------------|----------------|----------------|-------------------|
| 1 | 74 | 74 | 1.0x |
| 2 | 107 | 54 | 1.4x |
| 4 | 198 | 50 | 2.7x |
| 8 | 203 | 25 | 2.7x |

All outputs verified coherent — correct answers, proper markdown, working code generation.

## Architecture

```
src/
  models/
    qwen3_5.rs       — Full Qwen3.5-35B-A3B model (40 layers, GDN + Attention + MoE)
    metal_gdn.rs     — Fused Metal kernel for GDN recurrence (via mlx-sys FFI)
    compiled_ops.rs   — Compiled MLX closures for fused element-wise ops
  scheduler/
    mod.rs           — Token-level continuous batching scheduler
    sequence.rs      — Per-user sequence state and lifecycle
    scheduler.rs     — Round-robin scheduling with prefill priority
    executor.rs      — ModelExecutor trait (decouples scheduling from model)
    mlx_executor.rs  — Real MLX executor with batched B>1 forward
  bin/
    mlx_qwen_infer.rs   — Single-user inference CLI
    multi_user_bench.rs  — Multi-user concurrent benchmark
    scheduler_demo.rs    — Scheduler demo with mock executor
```

### Key optimizations

**Correctness fixes** (the model produced garbage until these were found):
- `nn::Conv1d` stale weight bug in mlx-rs — bypassed with direct `ops::conv1d` API
- Conv state initialization gave wrong shape for S=1
- Attention Q/gate split was sequential instead of interleaved per head
- `categorical()` expects logits, not probabilities (was double-softmaxing)

**Performance** (27 -> 112 tok/s single-user):
- Fused Metal GDN recurrence kernel with F32 state accumulation
- Fast S=1 decode conv path (direct multiply+sum, skip conv1d dispatch)
- True double-buffered async eval (overlap graph building with GPU compute)
- Compiled MLX closures for `silu*multiply` and `beta+g` (fuse ~14 ops each)
- Merged qkv+z and a+b projections into single matmul dispatches
- Precomputed norm weights with scale factors baked in
- Cached Metal kernel configs, GPU streams, conv weight layouts
- Bypass `nn::RmsNorm` module dispatch with direct `fast::rms_norm`

**Multi-user** (token-level scheduling):
- Batched B>1 forward: stack per-sequence states, single GPU dispatch, unstack
- Attention KV caches zero-padded to max length in batch
- Round-robin fairness with configurable prefill budget
- Streaming token emission via tokio channels

## Usage

### Single-user inference

```bash
cargo build --release --bin mlx_qwen_infer
./target/release/mlx_qwen_infer \
  --model-dir /path/to/Qwen3.5-35B-A3B-4bit \
  --prompt "What is quantum computing?"
```

### Multi-user benchmark

```bash
cargo build --release --bin multi_user_bench
./target/release/multi_user_bench \
  --model-dir /path/to/Qwen3.5-35B-A3B-4bit \
  --users 4 \
  --tokens-per-user 200 \
  --batch 4
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust toolchain
- cmake (for mlx-sys build): `brew install cmake`
- Model: [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit)

The 4-bit quantized model requires ~19GB of memory. The M3 Ultra 96GB can comfortably run 8+ concurrent users.

## The GDN serving advantage

Traditional LLM serving systems (vLLM, TGI, SGLang) are built around managing the transformer KV cache — PagedAttention, prefix caching, memory fragmentation, cache eviction policies. This is necessary because the KV cache grows linearly with sequence length and dominates GPU memory.

GDN eliminates this problem entirely. The recurrent state is **fixed-size** regardless of sequence length:

```
GDN state per user:
  30 layers x (conv_state [3, 8192] + recurrent_state [32, 128, 128])
  = 30 x (~50KB + ~1MB)
  = ~32MB total (constant, never grows)

Attention KV cache per user (only 10 layers):
  10 layers x 2 heads x seq_len x 256 dims x 2 bytes
  = ~10KB per token of context
```

At 1000 tokens of context, a GDN user needs ~42MB vs ~50MB+ for a pure attention model of similar size. At 10K tokens, GDN is still ~132MB while attention would be ~500MB+.

This means:
- **No memory fragmentation** — all states are the same size
- **No PagedAttention needed** — just swap fixed-size buffers
- **No cache eviction** — every user fits
- **Trivial scheduling** — pick N users, stack states, forward, unstack

The token-level scheduler in this repo is ~200 lines of Rust. A comparable vLLM scheduler is thousands of lines of Python managing paged memory blocks.

## License

MIT
