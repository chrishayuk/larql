//! What the harness can see of a provider's storage beyond the trait:
//! matrices (canonical/v1), encoded rows and the decode scratch (codec/v1,
//! CONTINUATION-CODEC-1 C3). Read by pointer, never from a provider's own
//! report. Split from `measured.rs` (800-line rule).

use larql_kv::{CanonicalKvState, CodecKvState, CodecRecentKvState, WindowKvState};
use larql_vindex::format::vindex3::opplan::exec::kv::{ContinuationProvider, RowKvState};

/// What the harness can see of a provider's storage beyond the trait.
pub trait Inspect: ContinuationProvider {
    /// The K and V matrix data pointers of `layer`, for a provider whose
    /// authority is a matrix.
    fn matrix_ptrs(&self, layer: usize) -> Option<(usize, usize)>;
    /// Rows held in `layer`'s matrix.
    fn matrix_rows(&self, layer: usize) -> Option<usize>;
    /// For a provider holding ENCODED rows (CODEC-1): the layer's row
    /// list, the first position it holds, and bytes per encoded row.
    fn code_list(&self, layer: usize) -> Option<CodeList>;
    /// The encoded-row allocations for absolute positions `range`.
    fn code_rows(&self, layer: usize, range: std::ops::Range<usize>) -> Vec<usize>;
    /// The decode scratch (K, V) allocations, for a provider that has one.
    fn scratch_ptrs(&self) -> Option<[usize; 2]>;
    /// For a provider holding codes AND exact K rows (CODEC-3 recon):
    /// one layer's three row lists and every row allocation they hold.
    fn mixed_layout(&self, _layer: usize) -> Option<MixedLayout> {
        None
    }
}

/// [`Inspect::mixed_layout`]: one layer of codec-recent/v1, read by
/// pointer. Positions `base..end` are held; K is exact from `exact_start`.
#[derive(Clone, Debug)]
pub struct MixedLayout {
    pub base: usize,
    pub exact_start: usize,
    pub end: usize,
    /// Rows held in each list: V codes, K codes, exact K.
    pub counts: [usize; 3],
    /// The three lists' own allocations (V codes, K codes, exact K).
    pub lists: [usize; 3],
    /// Every row allocation the lists hold.
    pub rows: Vec<usize>,
    /// Bytes of one tensor's encoded row.
    pub half_bytes: usize,
}

/// [`Measured::mixed_residency`]'s tally: append-born live bytes against
/// what a window of `w` declares, by class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixedResidency {
    pub append_born_live: usize,
    /// Declared: V codes + aged K codes + exact K rows + the lists.
    pub expected: usize,
    pub v_code_bytes: usize,
    pub k_code_bytes: usize,
    pub k_exact_bytes: usize,
    pub list_bytes: usize,
    /// Layers whose exact-K count is not min(w, held), or whose exact
    /// range does not end at the newest row.
    pub window_mismatches: usize,
    pub strays: usize,
    pub scratch_bytes: usize,
    pub scratch_bound: usize,
}

impl MixedResidency {
    pub fn holds(&self) -> bool {
        self.append_born_live == self.expected
            && self.window_mismatches == 0
            && self.strays == 0
            && self.scratch_bytes <= self.scratch_bound
    }

    /// Bytes held for rows, lists excluded.
    pub fn row_bytes(&self) -> usize {
        self.v_code_bytes + self.k_code_bytes + self.k_exact_bytes
    }
}

/// [`Measured::codec_residency`]'s tally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodecResidency {
    /// Live bytes born inside append scopes.
    pub append_born_live: usize,
    /// Declared: held rows × encoded row bytes, plus the row lists.
    pub expected: usize,
    /// Append-born live allocations that are neither an encoded row nor a
    /// row list — a second cache, an f32 copy, an undeclared buffer.
    pub strays: usize,
    pub scratch_bytes: usize,
    pub scratch_bound: usize,
}

impl CodecResidency {
    pub fn holds(&self) -> bool {
        self.append_born_live == self.expected
            && self.strays == 0
            && self.scratch_bytes <= self.scratch_bound
    }
}

/// [`Inspect::code_list`]: one layer's encoded storage.
#[derive(Clone, Copy, Debug)]
pub struct CodeList {
    pub list: usize,
    pub base: usize,
    pub end: usize,
    pub row_bytes: usize,
}

impl Inspect for RowKvState {
    fn matrix_ptrs(&self, _: usize) -> Option<(usize, usize)> {
        None
    }
    fn matrix_rows(&self, _: usize) -> Option<usize> {
        None
    }
    fn code_list(&self, _: usize) -> Option<CodeList> {
        None
    }
    fn code_rows(&self, _: usize, _: std::ops::Range<usize>) -> Vec<usize> {
        Vec::new()
    }
    fn scratch_ptrs(&self) -> Option<[usize; 2]> {
        None
    }
}

/// window/v1 holds adopted rows, no matrix (measurement-only impl).
impl Inspect for WindowKvState {
    fn matrix_ptrs(&self, _: usize) -> Option<(usize, usize)> {
        None
    }
    fn matrix_rows(&self, _: usize) -> Option<usize> {
        None
    }
    fn code_list(&self, _: usize) -> Option<CodeList> {
        None
    }
    fn code_rows(&self, _: usize, _: std::ops::Range<usize>) -> Vec<usize> {
        Vec::new()
    }
    fn scratch_ptrs(&self) -> Option<[usize; 2]> {
        None
    }
}

impl Inspect for CanonicalKvState {
    fn matrix_ptrs(&self, layer: usize) -> Option<(usize, usize)> {
        let (k, v) = self.cache().get_layer(layer)?;
        Some((k.as_ptr() as usize, v.as_ptr() as usize))
    }
    fn matrix_rows(&self, layer: usize) -> Option<usize> {
        self.cache().get_layer(layer).map(|(k, _)| k.shape()[0])
    }
    fn code_list(&self, _: usize) -> Option<CodeList> {
        None
    }
    fn code_rows(&self, _: usize, _: std::ops::Range<usize>) -> Vec<usize> {
        Vec::new()
    }
    fn scratch_ptrs(&self) -> Option<[usize; 2]> {
        None
    }
}

/// codec/v1 (CODEC-1 C3): encoded rows, one allocation per position, in a
/// row list; plus one decode scratch. Read by pointer, never by report.
impl Inspect for CodecKvState {
    fn matrix_ptrs(&self, _: usize) -> Option<(usize, usize)> {
        None
    }
    fn matrix_rows(&self, _: usize) -> Option<usize> {
        None
    }
    fn code_list(&self, layer: usize) -> Option<CodeList> {
        let rows = self.encoded_rows(layer);
        let base = self.rows_base(layer);
        Some(CodeList {
            list: rows.as_ptr() as usize,
            base,
            end: base + rows.len(),
            row_bytes: self.encoded_row_bytes(layer),
        })
    }
    fn code_rows(&self, layer: usize, range: std::ops::Range<usize>) -> Vec<usize> {
        let rows = self.encoded_rows(layer);
        let base = self.rows_base(layer);
        range
            .filter_map(|p| p.checked_sub(base).and_then(|i| rows.get(i)))
            .map(|r| r.as_ptr() as usize)
            .collect()
    }
    fn scratch_ptrs(&self) -> Option<[usize; 2]> {
        let (k, v) = self.scratch();
        Some([k.as_ptr() as usize, v.as_ptr() as usize])
    }
}

/// codec-recent/v1 (CODEC-3 recon): V codes, aged K codes and exact K
/// rows, one allocation per row in three lists; plus codec/v1's scratch.
impl Inspect for CodecRecentKvState {
    fn matrix_ptrs(&self, _: usize) -> Option<(usize, usize)> {
        None
    }
    fn matrix_rows(&self, _: usize) -> Option<usize> {
        None
    }
    fn code_list(&self, _: usize) -> Option<CodeList> {
        None
    }
    fn code_rows(&self, _: usize, _: std::ops::Range<usize>) -> Vec<usize> {
        Vec::new()
    }
    fn scratch_ptrs(&self) -> Option<[usize; 2]> {
        let (k, v) = self.scratch();
        Some([k.as_ptr() as usize, v.as_ptr() as usize])
    }
    fn mixed_layout(&self, layer: usize) -> Option<MixedLayout> {
        let (v, k, exact) = (
            self.encoded_values(layer),
            self.encoded_keys(layer),
            self.exact_keys(layer),
        );
        let base = self.rows_base(layer);
        Some(MixedLayout {
            base,
            exact_start: self.exact_start(layer),
            end: base + v.len(),
            counts: [v.len(), k.len(), exact.len()],
            lists: [
                v.as_ptr() as usize,
                k.as_ptr() as usize,
                exact.as_ptr() as usize,
            ],
            rows: v
                .iter()
                .chain(k)
                .map(|r| r.as_ptr() as usize)
                .chain(exact.iter().map(|r| r.as_ptr() as usize))
                .collect(),
            half_bytes: self.encoded_half_bytes(layer),
        })
    }
}
