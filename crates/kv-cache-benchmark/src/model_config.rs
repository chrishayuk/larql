/// Model configuration for KV cache benchmarking.
///
/// Dimensions are taken from real model architectures. The benchmark uses
/// these to generate correctly-shaped synthetic vectors and compute
/// analytical memory formulas.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelConfig {
    pub name: &'static str,
    pub layers: usize,
    pub kv_heads: usize,
    pub q_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
}

impl ModelConfig {
    /// Gemma 3-4B (the model we run)
    pub fn gemma_4b() -> Self {
        Self {
            name: "Gemma 3-4B",
            layers: 34,
            kv_heads: 2,
            q_heads: 10,
            head_dim: 256,
            hidden_dim: 2560,
            intermediate_dim: 10240,
            vocab_size: 262144,
        }
    }

    /// Llama 3.1 8B
    ///
    /// Llama 3 8B and 3.1 8B use the same KV-cache dimensions. Keep the
    /// older `llama_8b` constructor as an alias for existing callers.
    pub fn llama_31_8b() -> Self {
        Self {
            name: "Llama 3.1 8B",
            layers: 32,
            kv_heads: 8,
            q_heads: 32,
            head_dim: 128,
            hidden_dim: 4096,
            intermediate_dim: 14336,
            vocab_size: 128256,
        }
    }

    /// Llama 3 / 3.1 8B.
    pub fn llama_8b() -> Self {
        Self::llama_31_8b()
    }

    /// Llama 3 70B (config-level only, not running the full model)
    pub fn llama_70b() -> Self {
        Self {
            name: "Llama 3 70B",
            layers: 80,
            kv_heads: 8,
            q_heads: 64,
            head_dim: 128,
            hidden_dim: 8192,
            intermediate_dim: 28672,
            vocab_size: 128256,
        }
    }

    /// Bytes per token for standard FP16 KV cache:
    /// layers × 2 (K+V) × kv_heads × head_dim × 2 (fp16 bytes)
    pub fn kv_bytes_per_token(&self) -> usize {
        self.layers * 2 * self.kv_heads * self.head_dim * 2
    }

    /// Total KV cache memory at a given sequence length.
    pub fn kv_memory(&self, seq_len: usize) -> usize {
        seq_len * self.kv_bytes_per_token()
    }

    /// Dimension of a single K or V vector (per head).
    pub fn kv_dim(&self) -> usize {
        self.head_dim
    }

    /// All standard benchmark models.
    pub fn all() -> Vec<Self> {
        vec![Self::gemma_4b(), Self::llama_31_8b(), Self::llama_70b()]
    }
}
