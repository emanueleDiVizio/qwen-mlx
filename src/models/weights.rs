//! Weight loading for Qwen3.5 from safetensors or MLX quantized format.

use std::collections::HashMap;
use std::path::Path;
use mlx_rs::Array;
use anyhow::Result;

/// Load safetensors weight files into a flat name->Array map.
pub fn load_safetensors(model_dir: &Path) -> Result<HashMap<String, Array>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let mut weights = HashMap::new();

    if index_path.exists() {
        // Sharded model: read index, load each unique file
        let index: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&index_path)?)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("No weight_map in index"))?;

        // Collect unique files
        let mut files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        files.sort();
        files.dedup();

        for file in &files {
            let path = model_dir.join(file);
            tracing::info!("Loading {}", file);
            let file_weights = mlx_rs::io::safetensors::load(&path)?;
            weights.extend(file_weights);
        }
    } else {
        // Single file
        let path = model_dir.join("model.safetensors");
        weights = mlx_rs::io::safetensors::load(&path)?;
    }

    Ok(weights)
}

/// Load MLX quantized model (config.json + weights in npz/safetensors).
pub fn load_mlx_quantized(model_dir: &Path) -> Result<HashMap<String, Array>> {
    // MLX community models use safetensors format with quantized values
    load_safetensors(model_dir)
}

/// Get a weight by name, with optional prefix stripping.
pub fn get_weight(
    weights: &HashMap<String, Array>,
    name: &str,
    prefix: &str,
) -> Result<Array> {
    let full_name = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", prefix, name)
    };

    weights
        .get(&full_name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Weight not found: {}", full_name))
}
