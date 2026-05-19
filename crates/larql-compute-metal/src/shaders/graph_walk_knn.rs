//! Graph walk KNN: score residual against gate vectors for a single layer.
//!
//! Input:  residual vector [hidden_size], gate matrix [num_features, hidden_size]
//! Output: top-K feature indices + scores
//!
//! This is the gate KNN step of the graph walk.
//! At d=2560, K=8192 features: one dot product sweep per layer.
//! CPU may be fast enough (~0.1ms), but GPU enables sub-0.01ms.
//!
//! ## Retention rationale (ADR-017)
//!
//! **Status**: experimental, diag-only. The shader is built into
//! `all_shaders()` so `metal/diag/shader_bench.rs` can benchmark
//! it, but the production graph-walk path in `larql-inference`
//! runs the gate-KNN scoring step on CPU.
//!
//! Kept on disk because gate-KNN on GPU is a plausible future win
//! when (a) the larql-inference walk runs all-GPU rather than the
//! current hybrid CPU+GPU split, or (b) a feature-set with N≫10K
//! pushes the CPU dot-product sweep above the GPU dispatch overhead.
//!
//! **Removal trigger**: if a year passes with no diag bench result
//! showing >2× CPU on a representative N, demote.

pub const SHADER: &str = r#"
// Gate KNN: compute dot products between a query vector and all gate vectors,
// then select top-K. Each thread computes one dot product.
//
// Grid: (num_features, 1, 1).
// After dispatch, CPU selects top-K from the scores buffer.
kernel void gate_knn_score(
    device const float* query   [[buffer(0)]],  // [hidden_size]
    device const float* gates   [[buffer(1)]],  // [num_features, hidden_size]
    device float*       scores  [[buffer(2)]],  // [num_features] — output dot products
    constant uint&      hidden  [[buffer(3)]],  // hidden_size
    constant uint&      n_feat  [[buffer(4)]],  // num_features
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_feat) return;

    // Dot product: query · gates[tid]
    float dot = 0.0f;
    uint base = tid * hidden;
    for (uint i = 0; i < hidden; i++) {
        dot += query[i] * gates[base + i];
    }
    scores[tid] = dot;
}

// Fused gate KNN with Q8 quantized gate vectors.
// Each gate vector is stored as (scale: float, data: [hidden_size] int8).
// Dot product: sum(query[i] * (data[i] * scale)).
kernel void gate_knn_score_q8(
    device const float* query   [[buffer(0)]],  // [hidden_size] f32
    device const uchar* gates   [[buffer(1)]],  // packed: [n_feat × (4 + hidden)] — scale(f32) + data(int8)
    device float*       scores  [[buffer(2)]],  // [n_feat]
    constant uint&      hidden  [[buffer(3)]],
    constant uint&      n_feat  [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_feat) return;

    uint stride = 4 + hidden;  // 4 bytes scale + hidden bytes data
    uint base = tid * stride;

    // Read scale (first 4 bytes as float)
    float scale = *reinterpret_cast<device const float*>(gates + base);

    // Dot product with dequantized int8
    float dot = 0.0f;
    device const char* data = reinterpret_cast<device const char*>(gates + base + 4);
    for (uint i = 0; i < hidden; i++) {
        dot += query[i] * (float(data[i]) * scale);
    }
    scores[tid] = dot;
}
"#;
