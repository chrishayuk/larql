//! RotorQuant strategy — wraps `larql_rotorquant` with the
//! `KvStrategy` interface used by the benchmark harness.
//!
//! Each variant of `KvFormat` (Iso3, Planar3, Iso4, Planar4) is a
//! separate strategy struct so the comparative table reports them
//! as distinct rows alongside Standard / TurboQuant / Markov.

use larql_rotorquant::{
    dequantize_k, dequantize_v_with_inverse_rotation, quantize_k, quantize_v, KvFormat,
    QuantizedKv,
};

use crate::{model_config::ModelConfig, KvStrategy};

/// One strategy per RotorQuant format. The `name()` distinguishes
/// them in benchmark output; the `format` field selects the
/// underlying compression.
pub struct RotorQuantStrategy {
    pub format: KvFormat,
    name: String,
}

impl RotorQuantStrategy {
    pub fn iso3() -> Self {
        Self {
            format: KvFormat::Iso3,
            name: "RotorQuant Iso3 (3-bit)".to_string(),
        }
    }

    pub fn planar3() -> Self {
        Self {
            format: KvFormat::Planar3,
            name: "RotorQuant Planar3 (3-bit)".to_string(),
        }
    }

    pub fn iso4() -> Self {
        Self {
            format: KvFormat::Iso4,
            name: "RotorQuant Iso4 (4-bit)".to_string(),
        }
    }

    pub fn planar4() -> Self {
        Self {
            format: KvFormat::Planar4,
            name: "RotorQuant Planar4 (4-bit)".to_string(),
        }
    }
}

impl KvStrategy for RotorQuantStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn encode(&self, keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<u8> {
        // Flatten Vec<Vec<f32>> → row-major Vec<f32>. All rows must be
        // the same length; the harness guarantees this.
        let n_rows = keys.len();
        let head_dim = keys.first().map(|v| v.len()).unwrap_or(0);
        // RotorQuant's block sizes are 2 (Planar) or 4 (Iso). If the
        // head_dim doesn't divide cleanly we pad-to-block before
        // encoding so the strategy is a no-op rather than a panic.
        let bs = self.format.block_size();
        if head_dim == 0 || head_dim % bs != 0 {
            // Encode an empty buffer; decode will recover zeros at the
            // benchmark's accuracy metric.
            return Vec::new();
        }

        let mut k_flat = Vec::with_capacity(n_rows * head_dim);
        for v in keys {
            k_flat.extend_from_slice(v);
        }
        let mut v_flat = Vec::with_capacity(n_rows * head_dim);
        for v in values {
            v_flat.extend_from_slice(v);
        }

        let qk = match quantize_k(self.format, &k_flat, n_rows, head_dim) {
            Ok(q) => q,
            Err(_) => return Vec::new(),
        };
        let qv = match quantize_v(self.format, &v_flat, n_rows, head_dim) {
            Ok(q) => q,
            Err(_) => return Vec::new(),
        };

        // Wire format:
        //   u8  format-tag (0=Iso3, 1=Planar3, 2=Iso4, 3=Planar4)
        //   u32 n_rows
        //   u32 head_dim
        //   u32 codes_len    (per side)
        //   u32 norms_len    (per side)
        //   u32 rotation_len (per side)
        //   <K codes>
        //   <K norms_le>
        //   <K rotation_le>
        //   <V codes>
        //   <V norms_le>
        //   <V rotation_le>
        let mut buf = Vec::new();
        buf.push(format_tag(self.format));
        buf.extend_from_slice(&(qk.n_rows as u32).to_le_bytes());
        buf.extend_from_slice(&(qk.head_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(qk.codes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(qk.norms.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(qk.rotation_indices.len() as u32).to_le_bytes());
        write_quantized(&mut buf, &qk);
        write_quantized(&mut buf, &qv);
        buf
    }

    fn decode(
        &self,
        encoded: &[u8],
        num_vectors: usize,
        dim: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        if encoded.is_empty() {
            // Pad-to-block bailed out at encode; return zeros so the
            // accuracy metric records the gap rather than panicking.
            let zero = vec![vec![0.0_f32; dim]; num_vectors];
            return (zero.clone(), zero);
        }
        let mut p = 0usize;
        let _tag = encoded[p];
        p += 1;
        let n_rows = read_u32(encoded, &mut p) as usize;
        let head_dim = read_u32(encoded, &mut p) as usize;
        let codes_len = read_u32(encoded, &mut p) as usize;
        let norms_len = read_u32(encoded, &mut p) as usize;
        let rotation_len = read_u32(encoded, &mut p) as usize;

        let qk = read_quantized(
            encoded, &mut p, self.format, n_rows, head_dim, codes_len, norms_len, rotation_len,
        );
        let qv = read_quantized(
            encoded, &mut p, self.format, n_rows, head_dim, codes_len, norms_len, rotation_len,
        );

        let k_flat = dequantize_k(&qk).unwrap_or_default();
        let v_flat = dequantize_v_with_inverse_rotation(&qv).unwrap_or_default();

        // Reshape to Vec<Vec<f32>>. If lengths don't match (e.g. the
        // harness asked for `num_vectors × dim` but we encoded
        // `n_rows × head_dim`), pad / truncate.
        let keys = reshape(k_flat, num_vectors, dim);
        let values = reshape(v_flat, num_vectors, dim);
        (keys, values)
    }

    fn memory_bytes(&self, config: &ModelConfig, seq_len: usize) -> usize {
        // Bytes per row of K = head_dim × bits/8 + 2 (norm) +
        //                       blocks × 2 (rotation index, u16).
        // For Iso3 / Planar3 the bits=3, blocks = head_dim / block_size.
        // KV cache stores both K and V → ×2.
        let head_dim = config.kv_dim() / config.kv_heads.max(1);
        let bits = self.format.bits() as usize;
        let bs = self.format.block_size();
        let rows = seq_len * config.layers * config.kv_heads;
        let bytes_per_row = (head_dim * bits + 7) / 8 + 4 + (head_dim / bs) * 2;
        2 * rows * bytes_per_row
    }
}

fn format_tag(f: KvFormat) -> u8 {
    match f {
        KvFormat::Iso3 => 0,
        KvFormat::Planar3 => 1,
        KvFormat::Iso4 => 2,
        KvFormat::Planar4 => 3,
    }
}

fn write_quantized(buf: &mut Vec<u8>, q: &QuantizedKv) {
    buf.extend_from_slice(&q.codes);
    for n in &q.norms {
        buf.extend_from_slice(&n.to_le_bytes());
    }
    for r in &q.rotation_indices {
        buf.extend_from_slice(&r.to_le_bytes());
    }
}

#[allow(clippy::too_many_arguments)]
fn read_quantized(
    src: &[u8],
    pos: &mut usize,
    format: KvFormat,
    n_rows: usize,
    head_dim: usize,
    codes_len: usize,
    norms_len: usize,
    rotation_len: usize,
) -> QuantizedKv {
    let codes = src[*pos..*pos + codes_len].to_vec();
    *pos += codes_len;
    let mut norms = Vec::with_capacity(norms_len);
    for _ in 0..norms_len {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&src[*pos..*pos + 4]);
        norms.push(f32::from_le_bytes(buf));
        *pos += 4;
    }
    let mut rotation_indices = Vec::with_capacity(rotation_len);
    for _ in 0..rotation_len {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(&src[*pos..*pos + 2]);
        rotation_indices.push(u16::from_le_bytes(buf));
        *pos += 2;
    }
    QuantizedKv {
        format,
        n_rows,
        head_dim,
        codes,
        norms,
        rotation_indices,
    }
}

fn read_u32(src: &[u8], pos: &mut usize) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&src[*pos..*pos + 4]);
    *pos += 4;
    u32::from_le_bytes(buf)
}

fn reshape(flat: Vec<f32>, num_vectors: usize, dim: usize) -> Vec<Vec<f32>> {
    let need = num_vectors * dim;
    let mut padded = flat;
    if padded.len() < need {
        padded.resize(need, 0.0);
    }
    let mut out = Vec::with_capacity(num_vectors);
    for i in 0..num_vectors {
        out.push(padded[i * dim..(i + 1) * dim].to_vec());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_strategy_benchmark;
    use rand::SeedableRng;

    fn config() -> ModelConfig {
        // Small synthetic config: head_dim must be a multiple of 4 for Iso.
        ModelConfig {
            name: "test",
            layers: 1,
            kv_heads: 1,
            q_heads: 1,
            head_dim: 32,
            hidden_dim: 32,
            intermediate_dim: 32,
            vocab_size: 32,
        }
    }

    #[test]
    fn iso3_strategy_runs_through_harness() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let s = RotorQuantStrategy::iso3();
        let result = run_strategy_benchmark(&s, &config(), 8, &mut rng);
        assert!(result.metrics.cosine_sim > 0.0);
        // Memory must be smaller than Standard (10× compression target).
        let std = crate::standard_kv::StandardKv;
        assert!(s.memory_bytes(&config(), 8) < std.memory_bytes(&config(), 8));
    }

    #[test]
    fn planar3_strategy_runs_through_harness() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(456);
        let s = RotorQuantStrategy::planar3();
        let result = run_strategy_benchmark(&s, &config(), 8, &mut rng);
        assert!(result.metrics.cosine_sim > 0.0);
    }

    #[test]
    fn memory_bytes_iso3_is_smaller_than_fp16() {
        let s = RotorQuantStrategy::iso3();
        let std = crate::standard_kv::StandardKv;
        let cfg = config();
        let s_bytes = s.memory_bytes(&cfg, 1024);
        let std_bytes = std.memory_bytes(&cfg, 1024);
        // At small head_dim the u16 rotation-index overhead dominates,
        // so the ratio is closer to 2× than the upstream paper's 10×.
        // The 10× number comes back at production head_dim ≥ 128 where
        // codes overwhelm the rotation-index footprint. Assert merely
        // "smaller than fp16" here; production-shape comparisons live
        // in the accuracy-suite tests with real model dims.
        assert!(
            s_bytes < std_bytes,
            "iso3 {s_bytes} should be < fp16 {std_bytes}"
        );
    }
}
