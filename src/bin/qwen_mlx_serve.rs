//! OpenAI-compatible HTTP server for qwen-mlx.
//!
//! Architecture:
//!   HTTP handlers → enqueue request → [channel] → scheduler loop → [channel] → SSE stream
//!
//! One background thread owns the scheduler + executor (no mutex contention).
//! HTTP handlers communicate via channels only.
//!
//! Usage: qwen_mlx_serve --model-dir <path> [--port 8191] [--max-batch 8]

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use qwen_mlx::scheduler::*;
use qwen_mlx::scheduler::mlx_executor::Qwen35Executor;

// --- OpenAI types ---

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temp")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    model: String,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    content: String,
}

fn default_max_tokens() -> usize { 2048 }
fn default_temp() -> f32 { 0.7 }

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<Delta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<Delta>,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

// --- Scheduler communication ---

/// Request from HTTP handler to scheduler thread.
struct InferRequest {
    token_ids: Vec<i32>,
    params: GenerationParams,
    /// Scheduler sends tokens back through this channel.
    token_tx: mpsc::UnboundedSender<SchedulerEvent>,
}

// --- App state ---

struct AppState {
    /// Send requests to the scheduler thread.
    request_tx: mpsc::UnboundedSender<InferRequest>,
    tokenizer: tokenizers::Tokenizer,
}

// --- Scheduler thread ---
/// Single thread that owns the model + scheduler. No mutexes.
fn run_scheduler_loop(
    mut executor: Qwen35Executor,
    max_batch: usize,
    mut request_rx: mpsc::UnboundedReceiver<InferRequest>,
) {
    let mut scheduler = TokenScheduler::new(max_batch);
    scheduler.set_prefill_budget(2);

    // Pipeline: alternate between submit (GPU starts) and collect (GPU done)
    // Between submit and collect, we drain HTTP requests — overlapping CPU+GPU.
    let mut gpu_inflight = false;
    let mut inflight_batch: Vec<Sequence> = Vec::new();

    loop {
        // Drain incoming requests
        while let Ok(req) = request_rx.try_recv() {
            scheduler.enqueue(req.token_ids, req.params, Some(req.token_tx));
        }

        // Collect previous GPU results if inflight
        if gpu_inflight {
            let mut batch_refs: Vec<&mut Sequence> = inflight_batch.iter_mut().collect();
            match executor.collect_batch(&mut batch_refs) {
                Ok(output) => {
                    for (seq, &token) in inflight_batch.iter_mut().zip(output.tokens.iter()) {
                        seq.current_token = token;
                        seq.output_tokens.push(token);
                        seq.tokens_generated += 1;
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
                    scheduler.stats.tokens_generated += inflight_batch.len() as u64;
                    scheduler.stats.iterations += 1;
                }
                Err(e) => eprintln!("Collect error: {}", e),
            }
            // Requeue unfinished
            for seq in inflight_batch.drain(..) {
                if seq.status != SequenceStatus::Finished {
                    scheduler.return_to_active(seq);
                }
            }
            gpu_inflight = false;
        }

        // Prefill any waiting sequences
        scheduler.prefill_waiting(&mut executor);

        // Submit next batch if we have active sequences
        let batch_size = max_batch.min(scheduler.active_count());
        if batch_size > 0 {
            inflight_batch = scheduler.take_batch(batch_size);
            let mut batch_refs: Vec<&mut Sequence> = inflight_batch.iter_mut().collect();
            match executor.submit_batch(&mut batch_refs) {
                Ok(_) => { gpu_inflight = true; }
                Err(e) => {
                    eprintln!("Submit error: {}", e);
                    // Return sequences on error
                    for seq in inflight_batch.drain(..) {
                        scheduler.return_to_active(seq);
                    }
                }
            }
        } else if !scheduler.has_work() && !gpu_inflight {
            // Nothing to do — wait for next request
            match request_rx.blocking_recv() {
                Some(req) => {
                    scheduler.enqueue(req.token_ids, req.params, Some(req.token_tx));
                }
                None => break,
            }
        }
    }
}

// --- Handlers ---

async fn health() -> &'static str { "ok" }

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Sse<impl Stream<Item = Result<sse::Event, std::convert::Infallible>>>, StatusCode> {
    // Build prompt
    let mut prompt = String::new();
    for msg in &req.messages {
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    prompt.push_str("<|im_start|>assistant\n");

    // Tokenize
    let encoding = state.tokenizer.encode(prompt.as_str(), false)
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();

    let params = GenerationParams {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        ..Default::default()
    };

    // Create token channel
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();

    // Send to scheduler thread (non-blocking, no mutex)
    state.request_tx.send(InferRequest {
        token_ids,
        params,
        token_tx,
    }).map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;

    let req_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    // Stream tokens back as SSE
    let tokenizer = state.tokenizer.clone();
    let stream = async_stream::stream! {
        // Role event
        yield Ok(make_sse_event(&req_id, Some("assistant"), None, None));

        // Token events
        while let Some(event) = token_rx.recv().await {
            match event {
                SchedulerEvent::Token(token_id) => {
                    let text = tokenizer.decode(&[token_id], true).unwrap_or_default();
                    if !text.is_empty() {
                        yield Ok(make_sse_event(&req_id, None, Some(&text), None));
                    }
                }
                SchedulerEvent::Finished { reason } => {
                    let reason_str = match reason {
                        FinishReason::Eos => "stop",
                        FinishReason::MaxTokens => "length",
                        FinishReason::Error => "error",
                    };
                    yield Ok(make_sse_event(&req_id, None, None, Some(reason_str)));
                    yield Ok(sse::Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    };

    Ok(Sse::new(stream))
}

fn make_sse_event(id: &str, role: Option<&str>, content: Option<&str>, finish: Option<&str>) -> sse::Event {
    sse::Event::default().data(serde_json::to_string(&ChatResponse {
        id: id.to_string(),
        object: "chat.completion.chunk".into(),
        choices: vec![Choice {
            index: 0,
            delta: if role.is_some() || content.is_some() {
                Some(Delta {
                    role: role.map(String::from),
                    content: content.map(String::from),
                })
            } else { None },
            message: None,
            finish_reason: finish.map(String::from),
        }],
    }).unwrap())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args.iter().position(|a| a == "--model-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Usage: qwen_mlx_serve --model-dir <path> [--port 8191] [--max-batch 8]");
            std::process::exit(1);
        });
    let port: u16 = args.iter().position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8191);
    let max_batch: usize = args.iter().position(|a| a == "--max-batch")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    eprintln!("Loading model from {:?}...", model_dir);
    let executor = Qwen35Executor::load(&model_dir, 0.7)
        .expect("Failed to load model");
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .expect("Failed to load tokenizer");
    eprintln!("Model loaded.");

    // Channel: HTTP handlers → scheduler thread
    let (request_tx, request_rx) = mpsc::unbounded_channel();

    // Scheduler runs on its own OS thread (owns GPU, no async overhead)
    std::thread::spawn(move || {
        run_scheduler_loop(executor, max_batch, request_rx);
    });

    let state = Arc::new(AppState { request_tx, tokenizer });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    eprintln!("Serving on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
