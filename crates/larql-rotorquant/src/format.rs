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
