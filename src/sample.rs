//! Token sampling utilities.

use mlx_rs::Array;

/// Sample a token from logits.
pub fn sample_token(logits: &Array, temperature: f32) -> u32 {
    if temperature <= 0.0 {
        // Greedy: argmax
        let token = mlx_rs::ops::argmax(logits, -1, false).unwrap();
        token.eval().unwrap();
        token.item::<u32>()
    } else {
        // Temperature sampling
        let scaled = mlx_rs::ops::divide(logits, &Array::from_float(temperature)).unwrap();
        let probs = mlx_rs::ops::softmax(&scaled, &[-1][..]).unwrap();
        let token = mlx_rs::random::categorical(&probs, -1, None).unwrap();
        token.eval().unwrap();
        token.item::<u32>()
    }
}
