//! `codec/v1`: the plan-required K/V range, held compressed
//! (CONTINUATION-CODEC-1).
//!
//! Retention is `window/v1`'s: each layer holds exactly the positions the
//! plan's [`HistoryRange`] says a later step can still read, and frees the
//! rest. What changes is the bytes per position. Each appended K and V row
//! is encoded ONCE, per head, by the TurboQuant codec (WHT + Lloyd-Max,
//! 3 or 4 bits) and the backend's f32 row is dropped. A stored position is
//! never decoded and re-encoded — the codec's reconstruction norm is below
//! one, so re-encoding would decay it.
//!
//! Reads go through [`KvState::prepare_layer`]: it decodes the layer's
//! retained range into ONE provider-owned f32 scratch, reused across
//! layers, which [`KvState::rows`] lends as a contiguous view. The view's
//! end is what was decoded, so a read the hook did not precede is refused
//! by the step, never served stale.
//!
//! APPROXIMATE by declaration: output is not bit-identical to `row/v1`.
//! The representation and scratch were decided in CODEC-1 C2 before this
//! code; see `docs/represent/forecasts/continuation-codec-1-notes.json`.
//!
//! [`HistoryRange`]: larql_vindex::format::vindex3::opplan::exec::kv::HistoryRange

use larql_vindex::format::vindex3::opplan::exec::continuation::{
    LatentKvRows, LayerContinuationGeometry, RecurrentState,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    BoxedContinuation, ContinuationFactory, ContinuationRegion,
};
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, KvState, LayerKvGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::kv_view::KvView;

use crate::engines::turbo_quant::{codebooks, TurboQuant};

/// [`CodecKvState`]'s family. Revision 1: TurboQuant per head, the
/// plan-required range, one decode scratch.
pub const IDENTITY_FAMILY: &str = "codec";
pub const IDENTITY_REVISION: u32 = 1;

/// The one configuration key, and the widths it accepts.
pub const BITS_KEY: &str = "bits";
pub const SUPPORTED_BITS: [u8; 2] = [3, 4];

/// The name its refusals carry.
const PROVIDER_NAME: &str = "CodecKvState";

/// Why a layer's shape cannot be held, or `None` when it can: the codec's
/// rotation needs a power-of-two block, and blocks are whole heads.
pub(super) fn unsupported(geometry: &LayerKvGeometry) -> Option<String> {
    let LayerKvGeometry {
        kv_dim, head_dim, ..
    } = *geometry;
    if head_dim == 0 || !head_dim.is_power_of_two() {
        return Some(format!(
            "head_dim {head_dim} is not a power of two, which the codec's rotation requires"
        ));
    }
    if !kv_dim.is_multiple_of(head_dim) {
        return Some(format!(
            "kv_dim {kv_dim} is not a whole number of {head_dim}-wide heads"
        ));
    }
    None
}

/// One layer: its geometry, the first position still held, and one
/// encoded allocation per held position (K blocks, then V blocks).
struct Layer {
    geometry: LayerKvGeometry,
    base: usize,
    rows: Vec<Vec<u8>>,
}

impl Layer {
    fn new(geometry: LayerKvGeometry) -> Self {
        Self {
            geometry,
            base: 0,
            rows: Vec::new(),
        }
    }

    fn end(&self) -> usize {
        self.base + self.rows.len()
    }

    fn heads(&self) -> usize {
        self.geometry.kv_dim / self.geometry.head_dim
    }

    /// Free every position below the plan's floor for a step that has
    /// just appended `position` — each position's own allocation.
    fn release_below(&mut self, position: usize) {
        let floor = self.geometry.history.required_start(position);
        let unreachable = floor.saturating_sub(self.base).min(self.rows.len());
        self.rows.drain(..unreachable);
        self.base += unreachable;
    }
}

/// The one decoded layer: which layer, the range it covers, and its K
/// and V as contiguous rows.
#[derive(Default)]
struct Scratch {
    layer: Option<usize>,
    base: usize,
    end: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    indices: Vec<u8>,
}

/// Continuation state holding the plan-required K/V range, TurboQuant-
/// compressed. KV-only: a plan needing recurrent or latent state is
/// refused at selection.
pub struct CodecKvState {
    codec: TurboQuant,
    layers: Vec<Layer>,
    position: usize,
    scratch: Scratch,
    encode_f32: Vec<f32>,
    encode_u8: Vec<u8>,
}

impl CodecKvState {
    /// A fresh state at `bits` (3 or 4 — the factory has checked).
    pub fn new(bits: u8) -> Self {
        // The codec's constant table is a process-wide static built on first
        // use; build it here, so no append ever allocates it.
        let _ = codebooks::unit_codebook(bits);
        Self {
            codec: TurboQuant::new(bits),
            layers: Vec::new(),
            position: 0,
            scratch: Scratch::default(),
            encode_f32: Vec::new(),
            encode_u8: Vec::new(),
        }
    }

    pub fn identity() -> ContinuationIdentity {
        ContinuationIdentity::new(IDENTITY_FAMILY, IDENTITY_REVISION)
    }

    /// Encoded bytes one position occupies on `layer`: K then V, each
    /// `heads` blocks of the codec's norm plus packed indices.
    pub fn encoded_row_bytes(&self, layer: usize) -> usize {
        let l = &self.layers[layer];
        2 * l.heads() * self.codec.bytes_per_vector(l.geometry.head_dim)
    }

    /// The encoded allocations `layer` holds, oldest first — one per held
    /// position. An inspection surface: memory accounting reads storage
    /// by allocation, never from a byte count the provider reports.
    pub fn encoded_rows(&self, layer: usize) -> &[Vec<u8>] {
        &self.layers[layer].rows
    }

    /// The first position `layer` still holds. Inspection only.
    pub fn rows_base(&self, layer: usize) -> usize {
        self.layers[layer].base
    }

    /// The decode scratch (K, V) as allocated — capacity included, since
    /// that is what is resident. Inspection only.
    pub fn scratch(&self) -> (&Vec<f32>, &Vec<f32>) {
        (&self.scratch.keys, &self.scratch.values)
    }

    /// Encode one f32 row, head by head, onto `out`.
    fn encode_row(&mut self, row: &[f32], head_dim: usize, out: &mut Vec<u8>) {
        for head in row.chunks_exact(head_dim) {
            self.codec
                .encode_vector_into(head, out, &mut self.encode_f32, &mut self.encode_u8);
        }
    }
}

impl KvState for CodecKvState {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        if self.layers.is_empty() {
            for (layer, geometry) in layers.iter().enumerate() {
                if let Some(reason) = unsupported(geometry) {
                    panic!("{PROVIDER_NAME} was prepared for layer {layer}, which {reason}");
                }
            }
            self.layers = layers.iter().copied().map(Layer::new).collect();
            // The codec's per-head workspace is sized here, at announcement,
            // so no append allocates anything but the row it stores.
            let widest = layers.iter().map(|g| g.head_dim).max().unwrap_or(0);
            self.encode_f32.reserve_exact(widest);
            self.encode_f32.resize(widest, 0.0);
            self.encode_u8.reserve_exact(widest);
            self.scratch.indices.reserve_exact(widest);
            return;
        }
        // A held state is being resumed: it must be state for a program of
        // this shape. Reshaping it silently would continue a different
        // conversation.
        let held: Vec<LayerKvGeometry> = self.layers.iter().map(|l| l.geometry).collect();
        assert_eq!(
            held, layers,
            "resumed codec state was prepared for a different program geometry"
        );
    }

    /// Refuses, by name and before any row exists, a layer shape the
    /// codec cannot hold; otherwise the KV-only default.
    fn prepare_continuation(
        &mut self,
        layers: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        for (layer, geometry) in layers.iter().enumerate() {
            if let Some(reason) = geometry.kv().and_then(unsupported) {
                return Err(ContinuationError::GeometryUnsupported {
                    provider: PROVIDER_NAME,
                    layer,
                    reason,
                });
            }
        }
        let kv: Vec<LayerKvGeometry> = layers.iter().filter_map(|g| g.kv().cloned()).collect();
        if kv.len() != layers.len() {
            let layer = layers.iter().position(|g| g.kv().is_none()).unwrap_or(0);
            return Err(match layers.get(layer) {
                Some(LayerContinuationGeometry::LatentKv(_)) => {
                    ContinuationError::LatentUnsupported {
                        provider: PROVIDER_NAME,
                        layer,
                    }
                }
                _ => ContinuationError::RecurrentUnsupported {
                    provider: PROVIDER_NAME,
                    layer,
                },
            });
        }
        self.prepare(&kv);
        Ok(())
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        let geometry = self.layers[layer].geometry;
        let kv_dim = geometry.kv_dim;
        assert_eq!(
            key.len(),
            kv_dim,
            "K row at layer {layer} is {} wide; the plan says {kv_dim}",
            key.len()
        );
        assert_eq!(
            value.len(),
            kv_dim,
            "V row at layer {layer} is {} wide; the plan says {kv_dim}",
            value.len()
        );
        let row_bytes = self.encoded_row_bytes(layer);
        let mut encoded = Vec::with_capacity(row_bytes);
        self.encode_row(&key, geometry.head_dim, &mut encoded);
        self.encode_row(&value, geometry.head_dim, &mut encoded);
        // Blocks are read back at the declared stride: an encoder that
        // wrote any other length would be decoded misaligned.
        assert_eq!(
            encoded.len(),
            row_bytes,
            "the codec wrote {} bytes for a row it declares as {row_bytes}",
            encoded.len()
        );
        let l = &mut self.layers[layer];
        let position = l.end();
        l.rows.push(encoded);
        l.release_below(position);
    }

    fn prepare_layer(&mut self, layer: usize) {
        let l = &self.layers[layer];
        let LayerKvGeometry {
            kv_dim, head_dim, ..
        } = l.geometry;
        let block = self.codec.bytes_per_vector(head_dim);
        let half = l.heads() * block;
        let held = l.rows.len();
        let s = &mut self.scratch;
        // Exact growth: the scratch's capacity is the largest range decoded
        // so far, never an amortised doubling past it.
        let len = held * kv_dim;
        for buffer in [&mut s.keys, &mut s.values] {
            buffer.reserve_exact(len.saturating_sub(buffer.len()));
            buffer.resize(len, 0.0);
        }
        for (row, encoded) in l.rows.iter().enumerate() {
            let (k_codes, v_codes) = encoded.split_at(half);
            let span = row * kv_dim..(row + 1) * kv_dim;
            for (codes, out) in [
                (k_codes, &mut s.keys[span.clone()]),
                (v_codes, &mut s.values[span]),
            ] {
                for (block_codes, head) in codes
                    .chunks_exact(block)
                    .zip(out.chunks_exact_mut(head_dim))
                {
                    self.codec
                        .decode_block_into(block_codes, head, &mut s.indices);
                }
            }
        }
        s.layer = Some(layer);
        s.base = l.base;
        s.end = l.end();
    }

    fn rows(&self, layer: usize) -> KvView<'_> {
        let s = &self.scratch;
        if s.layer != Some(layer) {
            return KvView::empty();
        }
        let kv_dim = self.layers[layer].geometry.kv_dim;
        let len = (s.end - s.base) * kv_dim;
        KvView::contiguous(s.base, kv_dim, &s.keys[..len], &s.values[..len])
            .expect("the scratch holds whole rows of this layer's width")
    }

    fn position(&self) -> usize {
        self.position
    }

    fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        Err(ContinuationError::RecurrentUnsupported {
            provider: PROVIDER_NAME,
            layer,
        })
    }

    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        Err(ContinuationError::LatentUnsupported {
            provider: PROVIDER_NAME,
            layer,
        })
    }
}

/// Builds [`CodecKvState`]: the K/V region only, configured by exactly
/// one key, `bits`, which must be named — there is no default width.
#[derive(Debug, Clone, Copy, Default)]
pub struct CodecFactory;

impl ContinuationFactory for CodecFactory {
    fn identity(&self) -> ContinuationIdentity {
        CodecKvState::identity()
    }

    fn regions(&self) -> &[ContinuationRegion] {
        &[ContinuationRegion::Kv]
    }

    fn validate_config(&self, config: &ContinuationConfig) -> Result<(), String> {
        if let Some(other) = config.keys().find(|k| *k != BITS_KEY) {
            return Err(format!("takes only `{BITS_KEY}`; given `{other}`"));
        }
        match config.get(BITS_KEY) {
            None => Err(format!(
                "requires `{BITS_KEY}` (one of {SUPPORTED_BITS:?}); there is no default width"
            )),
            Some(bits) => match bits.parse::<u8>() {
                Ok(b) if SUPPORTED_BITS.contains(&b) => Ok(()),
                _ => Err(format!(
                    "`{BITS_KEY}` must be one of {SUPPORTED_BITS:?}; given `{bits}`"
                )),
            },
        }
    }

    fn build(&self, config: &ContinuationConfig) -> BoxedContinuation {
        let bits = config
            .get(BITS_KEY)
            .and_then(|b| b.parse::<u8>().ok())
            .filter(|b| SUPPORTED_BITS.contains(b))
            .expect("build is reached only with a config validate_config accepted");
        Box::new(CodecKvState::new(bits))
    }
}
