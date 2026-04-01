# Building a 112 tok/s Rust Inference Engine for Qwen3.5 on Apple Silicon

## TL;DR

I built a Rust inference engine for Qwen3.5-35B-A3B that matches Python mlx-lm speed (112 vs 113 tok/s) on M3 Ultra, with multi-user continuous batching reaching 200+ tok/s aggregate. The GDN (GatedDeltaNet) architecture eliminates the KV cache scaling problem, making multi-user serving trivially efficient.

## Background

Qwen3.5-35B-A3B is a 35B parameter Mixture-of-Experts model that uses a novel GatedDeltaNet (GDN) architecture for 30 of its 40 layers, with standard attention only every 4th layer. GDN replaces the attention mechanism with a linear recurrence that maintains a fixed-size state matrix, eliminating the O(seq_len) KV cache that dominates transformer serving costs.

I set out to build a Rust inference engine on Apple's MLX framework to:
1. Match or exceed Python mlx-lm performance
2. Exploit GDN's fixed-state property for efficient multi-user serving
3. Create an open-source, production-ready inference backend

## The journey: garbage to 112 tok/s

### Phase 1: Making it work (garbage output)

The initial implementation produced completely wrong output. Four bugs, each individually catastrophic:

**Bug 1: `nn::Conv1d` stale weights.** The mlx-rs `nn::Conv1d` module ignores weight updates from `update_flattened()`. After loading safetensors weights, the conv1d forward pass still used the initial random weights. The conv output was essentially the identity function instead of the actual convolution.

*Fix:* Bypass `nn::Conv1d` entirely — call `mlx_rs::ops::conv1d()` directly with the weight parameter.

**Bug 2: Conv state initialization.** `zeros_like(qkv_cat.index((.., ..kernel_size-1, ..)))` was meant to create a `[B, 3, D]` zero tensor for the conv sliding window. But when `S=1`, the slice `..3` on a length-1 axis returns only 1 element, giving `[B, 1, D]`. The subsequent conv1d processed a 2-wide window instead of 4-wide, producing wrong values.

*Fix:* Explicit `zeros::<bf16>([B, kernel_size-1, conv_dim])`.

**Bug 3: Attention Q/gate split.** Qwen3.5's attention uses output gating: the Q projection outputs `2 * num_heads * head_dim` values, split into queries and gates. The weight layout is **interleaved per head**: `[head0_q, head0_gate, head1_q, head1_gate, ...]`. My code split sequentially: first half as queries, second half as gates. Every attention layer read the wrong weights.

*Fix:* `reshape(B, S, nh, 2*hd)` then split along the last axis.

**Bug 4: Double softmax.** MLX's `categorical()` function expects **logits** (it applies softmax internally). I passed `softmax(logits)`, which got softmax'd again, producing a nearly uniform distribution over all 248K tokens — hence the random multilingual garbage.

*Fix:* Pass scaled logits directly to `categorical()`.

### Phase 2: Making it fast (27 -> 112 tok/s)

With correct output at 27 tok/s, I applied a series of optimizations:

**Fused Metal GDN kernel (27 -> 46 tok/s, +70%).** The ops-based recurrence dispatched ~12 separate GPU operations per timestep per layer. I ported Python's Metal kernel that fuses the entire recurrence (state decay, kv_mem, delta update, output projection) into a single dispatch with SIMD parallelism across the key dimension.

Key fix vs the original kernel attempt: added `StT` template parameter for F32 state output (the kernel accumulates in F32 but reads/writes BF16 for I/O) and fixed threadgroup to `(32, 4, 1)` matching Python.

**F32 state accumulation (46 -> 74 tok/s, +61%).** The recurrent state was stored as BF16 and converted to F32 for each Metal kernel invocation, then back. Keeping state in F32 permanently eliminated two dtype conversion operations per layer.

**Fast S=1 conv path (74 -> 88 tok/s, +19%).** For decode (single token), the conv1d reduces to a dot product: `sum(weight * state_window)` per channel. Instead of dispatching a full conv1d operation, I reshape the weight to `[1, K, D]`, multiply element-wise with the padded window `[B, K, D]`, and sum along the kernel dimension. This replaces 1 conv1d dispatch with 2 simple element-wise ops that MLX can fuse.

**True double-buffered async eval (88 -> 103 tok/s, +17%).** Python's mlx-lm achieves high throughput by overlapping GPU compute with CPU graph building:

```python
next_y = _step(y)           # Build next graph (CPU, ~1.4ms)
mx.async_eval(next_y)       # Submit to GPU
yield y.item()              # Read PREVIOUS result
y = next_y                  # Swap
```

The key insight: `_step(y)` takes a **lazy MLX array** `y` as input, so the entire next-step graph is built before `y` is evaluated. The GPU processes step N while the CPU builds step N+1.

My initial attempt passed `current_token` (a Rust integer) to the next forward call, requiring the previous eval to complete first. Passing the lazy `token_arr` (MLX array) instead enabled true overlap.

**Compiled fused closures (103 -> 112 tok/s, +9%).** MLX's `@mx.compile` fuses element-wise ops into single GPU dispatches. I implemented this via `mlx_sys::mlx_compile` with C callback closures:

- `compiled_silu_multiply`: fuses `silu(a) * b` (used 80x per token in MoE gate+up projections)
- `compiled_beta_g`: fuses sigmoid + compute_g (~14 ops -> 1 dispatch, used 30x per token)

The closures are compiled once with `shapeless=true` and cached via `OnceLock`.

Additional micro-optimizations:
- Precomputed norm weights with scale factors baked in (1 dispatch instead of 3)
- Merged qkv+z and a+b projections into single quantized matmuls
- Cached Metal kernel configs, GPU streams, conv weight layouts
- Bypassed `nn::RmsNorm` module dispatch with direct `fast::rms_norm`
- Disabled MLX compile mode (paradoxically added overhead for this graph structure)

### Phase 3: Multi-user serving (200+ tok/s aggregate)

The token-level scheduler processes one token per user per iteration:

```
loop {
    batch = scheduler.pick_batch(max_batch_size);
    tokens = executor.decode_batch(batch);
    for (seq, token) in batch.zip(tokens) {
        seq.emit_token(token);
        if seq.should_stop(token) { finish(seq); }
        else { requeue(seq); }
    }
}
```

**Batched forward.** The executor stacks per-sequence states into `[B, ...]` tensors:
- GDN conv/recurrent states: fixed-size, directly concatenated along batch dim
- Attention KV caches: variable-length, zero-padded to max length in batch

A single `model.forward([B, 1], batched_caches)` processes all B users in one GPU dispatch. Per-sequence tokens are sampled from the `[B, vocab_size]` logits.

## The GDN serving insight

The key architectural insight is that GDN's fixed-size state makes multi-user serving qualitatively different from transformer serving:

**No PagedAttention needed.** vLLM's core innovation is managing KV cache memory with virtual memory-like paging. GDN states are all the same size — there's nothing to page.

**No memory fragmentation.** Transformer KV caches grow at different rates (different sequence lengths), causing fragmentation. GDN states are fixed-size buffers — allocation is trivial.

**No cache eviction.** When GPU memory fills up, transformer systems must evict KV caches and recompute them later. GDN states fit in ~32MB per user — a 96GB machine can hold thousands.

**Trivial scheduling.** The scheduler is ~200 lines of Rust. Pick N users, stack their states, forward, unstack. No block tables, no prefix trees, no defragmentation.

The result: multi-user throughput scales almost linearly to the GPU's compute limit, not its memory limit.

## Numbers

**Single-user:** 112 tok/s decode on M3 Ultra, within 1% of Python mlx-lm (113 tok/s).

**Multi-user:** 200+ tok/s aggregate with 8 concurrent users, all producing coherent output.

**Model quality:** Qwen3.5-35B-A3B produces high-quality output with proper reasoning (`<think>` tokens), correct factual answers, working code, and well-structured markdown.

## What's next

- HTTP server with OpenAI-compatible API for production deployment
- True B>1 batched matmuls (current batching stacks states but doesn't saturate GPU at small B)
- Speculative decoding with Qwen3.5-9B as draft model
- Multi-node inference dispatch via load-balancing router
