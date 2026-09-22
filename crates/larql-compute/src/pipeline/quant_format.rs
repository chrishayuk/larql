/// Bytes per Q4_KF pre-baked super-block. Q4_KF keeps the 256-element
/// Q4_K block shape but expands packed scale/min metadata for faster decode.
pub const Q4_KF_BLOCK_BYTES: usize = 160;

/// Quantization format for a weight tensor.
/// Names match GGUF conventions (Q4_K, Q6_K, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    Q4_0,  // 18 bytes per 32 values (one f16 scale)
    Q4_K,  // 144 bytes per 256 values (GGUF-canonical, Ollama-compatible)
    Q4_KF, // 160 bytes per 256 values (pre-baked half scales — fast decode)
    /// 176 bytes per 256 values — Q4_K's affine layout plus a 32-byte
    /// high-bit plane. **CPU only**: there is no Q5_K Metal shader, so
    /// `MetalBackend::supports_quant` reports `false` and the Metal decode
    /// paths reject it rather than mis-routing its 176-byte stride into a
    /// Q4_K (144) or Q6_K (210) kernel.
    Q5_K,
    Q6_K, // 210 bytes per 256 values (6-bit with sub-block scales)
    Q8_0, // int8 values + separate f32 scales
    BF16, // raw bfloat16 (2 bytes per value, no quantization scales)
    F16,  // raw float16  (2 bytes per value)
    F32,  // raw float32  (4 bytes per value)
    /// BitNet 1.58-bit ternary (GGML I2_S, type 36): 4 trits/byte packed
    /// row-major (`cols/4` bytes per row) plus a separate per-channel f32
    /// scale array. Unlike the block-quant formats, the weight is NOT a
    /// flat `&[u8]` block stream — it is carried by
    /// [`crate::cpu::ops::ternary_matvec::BitLinearWeight`] (bytes + scales)
    /// and served by the dedicated `ternary_matvec` dispatch, not the
    /// block-quant `quant_matvec` path (which has no per-channel-scale input).
    I2S,
}

impl QuantFormat {
    /// Packed block geometry as `(elements_per_block, bytes_per_block)`.
    ///
    /// This is the compute-side mirror of the GGML layout constants used by
    /// the quantizers. Callers that need byte offsets should ask the format
    /// instead of spelling `256 * 144` or `32 * 18` locally.
    pub fn packed_block_layout(self) -> Option<(usize, usize)> {
        use larql_models::quant::ggml;

        match self {
            Self::Q4_0 => Some((ggml::Q4_0_BLOCK_ELEMS, ggml::Q4_0_BLOCK_BYTES)),
            Self::Q4_K => Some((ggml::Q4_K_BLOCK_ELEMS, ggml::Q4_K_BLOCK_BYTES)),
            Self::Q4_KF => Some((ggml::Q4_K_BLOCK_ELEMS, Q4_KF_BLOCK_BYTES)),
            Self::Q5_K => Some((ggml::Q5_K_BLOCK_ELEMS, ggml::Q5_K_BLOCK_BYTES)),
            Self::Q6_K => Some((ggml::Q6_K_BLOCK_ELEMS, ggml::Q6_K_BLOCK_BYTES)),
            _ => None,
        }
    }

    /// Byte length for a packed row-major matrix with `rows * cols` values.
    ///
    /// Current interleaved FFN fallback stores each matrix contiguously, so
    /// this intentionally preserves the historical flat packing calculation.
    /// Manifest-aware paths should prefer recorded offsets and lengths.
    pub fn packed_matrix_bytes(self, rows: usize, cols: usize) -> Option<usize> {
        let elems = rows.checked_mul(cols)?;
        let (block_elems, block_bytes) = self.packed_block_layout()?;
        Some(elems.div_ceil(block_elems) * block_bytes)
    }

    /// Whether this format uses the GGUF k-quant 256-element super-block
    /// layout that flows through the dedicated Q4_K / Q4_KF / Q6_K matvec
    /// dispatchers (vs the legacy block-32 Q4_0 / Q8_0 path). Used to gate
    /// the "skip Q8 quantize" fast path in `residual_norm` and FFN routing.
    ///
    /// Adding a future k-quant format extends this one method, not the ~10
    /// OR-chains it currently replaces (Q5_K did exactly that on
    /// 2026-08-01). Roadmap #7 (`FormatRoute` enum) is the fuller version
    /// of this idea; this helper is the contained step that addresses the
    /// user-visible code-duplication cost without rippling through 49 files.
    ///
    /// This is a claim about **layout**, not about kernel availability: it
    /// says the format's weights are 256-element super-blocks consumed from
    /// f32 input, not that every backend can serve it. Q5_K is in the
    /// family but has no Metal shader — ask [`Self::has_metal_kernel`]
    /// before routing on a GPU backend.
    pub fn is_kquant_family(self) -> bool {
        matches!(self, Self::Q4_K | Self::Q4_KF | Self::Q5_K | Self::Q6_K)
    }

    /// Whether a Metal shader exists for this format's weight layout.
    ///
    /// Separate from [`Self::is_kquant_family`] because the two diverged
    /// when Q5_K landed CPU-only. The Metal decode paths gate on
    /// `is_kquant_family` to choose between the k-quant and legacy-Q8
    /// routes, and **both** of those routes read a wrong byte stride for
    /// Q5_K (144 or 210 vs its 176), so a format that is structurally a
    /// k-quant but has no shader must be rejected up front rather than
    /// silently decoded as garbage. Same bug class as the pre-2026-05-09
    /// Q8_0-through-`q4_matvec` routing.
    pub fn has_metal_kernel(self) -> bool {
        !matches!(self, Self::Q5_K)
    }

    /// Whether this format uses the llama.cpp-exact "Q4_KF" pre-baked
    /// half-scale fast path (`q4kf_proj` shader). Distinct from the
    /// canonical `Q4_K` GGUF layout used by Ollama extracts.
    pub fn is_q4kf(self) -> bool {
        matches!(self, Self::Q4_KF)
    }

    /// Whether this format uses the legacy block-32 Q8 dispatch path
    /// (`q4_matvec` / `q8_matvec` against pre-quantised Q8 input). The
    /// inverse of [`Self::is_kquant_family`] for the dense matvec dispatch
    /// (the float-input `BF16` / `F16` / `F32` branches don't run on
    /// these dispatchers, so `is_legacy_q8` covers exactly the rest).
    pub fn is_legacy_q8(self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q8_0)
    }

    /// Parse a GGUF-convention registry tag (`"Q4_K"`, `"Q6_K"`, …) into a
    /// `QuantFormat`. The canonical inverse of the names the extractor and
    /// weight manifests record; `None` for any tag with no compute mapping.
    ///
    /// This is the contained version of Roadmap #7's `from_registry_tag`:
    /// it lets the string-keyed matvec dispatchers (`q4k_q8k_matvec_parallel`,
    /// `kquant_forward::cached`) ask the format for its packed layout instead
    /// of re-spelling `(cols/256)*144` locally, without changing their `&str`
    /// call-site signatures.
    pub fn from_registry_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "Q4_0" => Self::Q4_0,
            "Q4_K" => Self::Q4_K,
            "Q4_KF" => Self::Q4_KF,
            "Q5_K" => Self::Q5_K,
            "Q6_K" => Self::Q6_K,
            "Q8_0" => Self::Q8_0,
            "BF16" => Self::BF16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            // BitNet ternary (GGML type 36). The vindex bitnet sidecar tags
            // its I2_S weight stream with this so the registry recognises it.
            "I2_S" => Self::I2S,
            _ => return None,
        })
    }

    /// Inverse of [`Self::from_registry_tag`] — the canonical registry tag
    /// string for this format. `from_registry_tag(f.registry_tag()) == Some(f)`
    /// for every variant. Used by writers that record the per-tensor format
    /// tag into the weight manifest / index.
    pub fn registry_tag(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_K => "Q4_K",
            Self::Q4_KF => "Q4_KF",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::BF16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::I2S => "I2_S",
        }
    }

    /// Whether this format is BitNet ternary (I2_S). Served by the dedicated
    /// `ternary_matvec` path with a [`crate::cpu::ops::ternary_matvec::BitLinearWeight`],
    /// never the block-quant `quant_matvec` dispatch.
    pub fn is_ternary(self) -> bool {
        matches!(self, Self::I2S)
    }
}

/// A quantized weight matrix — raw bytes with format tag.
#[derive(Clone, Copy)]
pub struct QuantWeight<'a> {
    pub data: &'a [u8],
    pub scales: Option<&'a [f32]>, // only for Q8_0 (separate scale array)
    pub format: QuantFormat,
}

impl Default for QuantWeight<'_> {
    fn default() -> Self {
        Self {
            data: &[],
            scales: None,
            format: QuantFormat::Q4_0,
        }
    }
}
