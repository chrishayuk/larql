/// On-device / on-disk format for a quantised K or V tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFormat {
    /// 3-bit Planar — Givens rotation per coordinate pair, 8-codeword
    /// Lloyd-Max codebook.
    Planar3,
    /// 4-bit Planar — same rotation, 16-codeword codebook.
    Planar4,
    /// 3-bit Iso — quaternion rotation per 4-coordinate group, 8-codeword.
    Iso3,
    /// 4-bit Iso — same rotation, 16-codeword.
    Iso4,
}

impl KvFormat {
    pub fn block_size(self) -> usize {
        match self {
            KvFormat::Planar3 | KvFormat::Planar4 => 2,
            KvFormat::Iso3 | KvFormat::Iso4 => 4,
        }
    }

    pub fn bits(self) -> u8 {
        match self {
            KvFormat::Planar3 | KvFormat::Iso3 => 3,
            KvFormat::Planar4 | KvFormat::Iso4 => 4,
        }
    }

    pub fn codebook_size(self) -> usize {
        1 << self.bits()
    }

    /// Number of pre-tabulated rotations from which `rotation_indices`
    /// picks. We use 8 for `Planar*` (16 angles, 8 distinct mod-π),
    /// 16 for `Iso*` (small quaternion lookup).
    pub fn rotation_count(self) -> usize {
        match self {
            KvFormat::Planar3 | KvFormat::Planar4 => 8,
            KvFormat::Iso3 | KvFormat::Iso4 => 16,
        }
    }
}

/// Quantised K or V buffer plus everything needed for round-trip.
#[derive(Clone, Debug)]
pub struct QuantizedKv {
    pub format: KvFormat,
    pub n_rows: usize,
    pub head_dim: usize,
    /// Packed codes — `bits` per code, `n_rows * head_dim` codes total.
    /// Layout: row-major, codes packed LSB-first within each byte.
    pub codes: Vec<u8>,
    /// One f32 per row — the L2 norm of the row before quantisation.
    pub norms: Vec<f32>,
    /// One u16 per (row, block) — indexes into the format's rotation
    /// table (`KvFormat::rotation_count()`). Required for V's inverse.
    pub rotation_indices: Vec<u16>,
}

impl QuantizedKv {
    pub fn n_blocks_per_row(&self) -> usize {
        self.head_dim / self.format.block_size()
    }
}

/// Reusable shape/capacity state for RotorQuant operations.
///
/// The CPU reference path does not need temporary buffers today, but
/// CUDA quantization does. Keeping the scratch object in the safe API
/// lets inference/session code allocate once per format and reject
/// shape mismatches before launching kernels.
#[derive(Clone, Debug)]
pub struct KvScratch {
    format: KvFormat,
    max_seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    workspace: Vec<f32>,
}

impl KvScratch {
    pub fn new(
        format: KvFormat,
        max_seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> Result<Self, crate::RotorQuantError> {
        if max_seq_len == 0 {
            return Err(crate::RotorQuantError::InvalidScratch(
                "max_seq_len must be non-zero".to_string(),
            ));
        }
        if num_heads == 0 {
            return Err(crate::RotorQuantError::InvalidScratch(
                "num_heads must be non-zero".to_string(),
            ));
        }
        if head_dim == 0 {
            return Err(crate::RotorQuantError::InvalidScratch(
                "head_dim must be non-zero".to_string(),
            ));
        }
        if !head_dim.is_multiple_of(format.block_size()) {
            return Err(crate::RotorQuantError::HeadDimNotDivisible {
                format,
                head_dim,
                block_size: format.block_size(),
            });
        }

        Ok(Self {
            format,
            max_seq_len,
            head_dim,
            num_heads,
            workspace: Vec::new(),
        })
    }

    pub fn format(&self) -> KvFormat {
        self.format
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn max_rows(&self) -> usize {
        self.max_seq_len * self.num_heads
    }

    pub(crate) fn validate(
        &mut self,
        format: KvFormat,
        n_rows: usize,
        head_dim: usize,
    ) -> Result<(), crate::RotorQuantError> {
        if self.format != format {
            return Err(crate::RotorQuantError::ScratchFormatMismatch {
                requested: format,
                scratch: self.format,
            });
        }
        if self.head_dim != head_dim {
            return Err(crate::RotorQuantError::ScratchHeadDimMismatch {
                requested: head_dim,
                scratch: self.head_dim,
            });
        }
        let capacity = self.max_rows();
        if n_rows > capacity {
            return Err(crate::RotorQuantError::ScratchCapacityExceeded {
                requested: n_rows,
                capacity,
            });
        }

        let needed = n_rows * head_dim;
        if self.workspace.len() < needed {
            self.workspace.resize(needed, 0.0);
        }
        Ok(())
    }
}
