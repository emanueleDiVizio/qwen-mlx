//! Hugging Face model downloading utilities.
//!
//! Uses the `hf-hub` crate to download models from the Hugging Face Hub.

use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::Cache;

/// Token source for Hugging Face authentication.
#[derive(Debug, Clone)]
pub enum TokenSource {
    /// Read token from ~/.cache/huggingface/token (default)
    CacheToken,
    /// Read token from environment variable
    EnvVar(String),
    /// Use literal token value
    Literal(String),
    /// No token (for public/non-gated models)
    None,
}

impl TokenSource {
    /// Parse token source from string.
    ///
    /// Formats:
    /// - "cache" -> CacheToken
    /// - "env:VAR_NAME" -> EnvVar("VAR_NAME")
    /// - "literal:TOKEN" -> Literal("TOKEN")
    /// - "none" -> None
    pub fn parse(s: &str) -> Result<Self> {
        if s == "cache" {
            Ok(TokenSource::CacheToken)
        } else if s == "none" {
            Ok(TokenSource::None)
        } else if let Some(var) = s.strip_prefix("env:") {
            Ok(TokenSource::EnvVar(var.to_string()))
        } else if let Some(token) = s.strip_prefix("literal:") {
            Ok(TokenSource::Literal(token.to_string()))
        } else {
            anyhow::bail!(
                "Invalid token source '{}'. Expected: cache, none, env:VAR_NAME, or literal:TOKEN",
                s
            )
        }
    }

    /// Resolve the token value.
    fn resolve(&self) -> Result<Option<String>> {
        match self {
            TokenSource::CacheToken => {
                let cache = Cache::default();
                Ok(cache.token())
            }
            TokenSource::EnvVar(var) => {
                let token = std::env::var(var)
                    .with_context(|| format!("Environment variable '{}' not set", var))?;
                Ok(Some(token))
            }
            TokenSource::Literal(token) => Ok(Some(token.clone())),
            TokenSource::None => Ok(None),
        }
    }
}

impl Default for TokenSource {
    fn default() -> Self {
        TokenSource::CacheToken
    }
}

/// Download a model from Hugging Face Hub.
///
/// Returns the path to the directory containing the downloaded model files.
///
/// # Arguments
/// * `model_id` - The Hugging Face model ID (e.g., "Qwen/Qwen3.5-35B-A3B-MLX-4bit")
/// * `token_source` - Authentication token source
/// * `revision` - Git revision (branch, tag, or commit). Defaults to "main".
pub fn download_model(
    model_id: &str,
    token_source: TokenSource,
    revision: Option<&str>,
) -> Result<PathBuf> {
    let token = token_source.resolve()?;
    let revision = revision.unwrap_or("main");

    tracing::info!("Downloading model {} (revision: {})", model_id, revision);

    // Build API with optional token
    let api = if let Some(token) = token {
        ApiBuilder::new().with_token(Some(token)).build()?
    } else {
        Api::new()?
    };

    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    // Download required files
    // config.json is always needed
    let config_path = repo
        .get("config.json")
        .context("Failed to download config.json")?;
    tracing::info!("Downloaded config.json");

    // tokenizer.json is needed for inference
    let _ = repo
        .get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;
    tracing::info!("Downloaded tokenizer.json");

    // Check if model uses sharded weights
    let index_result = repo.get("model.safetensors.index.json");

    if let Ok(index_path) = index_result {
        // Sharded model: parse index and download all shard files
        tracing::info!("Downloaded model.safetensors.index.json (sharded model)");

        let index: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&index_path)?)?;

        if let Some(weight_map) = index["weight_map"].as_object() {
            // Collect unique shard files
            let mut files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            files.sort();
            files.dedup();

            tracing::info!("Downloading {} shard files...", files.len());

            for file in &files {
                let _ = repo
                    .get(file)
                    .with_context(|| format!("Failed to download {}", file))?;
                tracing::info!("Downloaded {}", file);
            }
        }
    } else {
        // Single safetensors file
        let _ = repo
            .get("model.safetensors")
            .context("Failed to download model.safetensors")?;
        tracing::info!("Downloaded model.safetensors");
    }

    // Return the directory containing the model files
    // hf-hub stores files in a snapshot directory
    let model_dir = config_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Could not determine model directory"))?
        .to_path_buf();

    tracing::info!("Model downloaded to {:?}", model_dir);

    Ok(model_dir)
}
