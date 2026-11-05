//! `codec-recent/v1`: `codec/v1` with the newest `w` K rows of every layer
//! held exact (CONTINUATION-CODEC-3 reconnaissance).
//!
//! CODEC-MAP-1 located codec/v1's failing tail in recent K: keeping the
//! newest 256 K rows exact removed 94% of CH's worst-5% tail, against the
//! simulator. This is the real representation that map simulated.
//!
//! Retention is `codec/v1`'s (and so `window/v1`'s): each layer holds the
//! plan-required range and frees the rest. Within it:
//! - every V row is encoded once, at append, exactly as codec/v1 does;
//! - the newest `w` K rows are an exact f32 copy, one allocation each;
//! - a K row is encoded once, when an append pushes it out of the window.
//!
//! The codec is deterministic, so a K row encoded when it ages out holds
//! the bytes codec/v1 would have written at append. Age is relative to the
//! layer's newest held row at the read, as in the simulator's map
//! `all:v:all|all:k:older<w>`; reads are therefore bit-identical to it.
//!
//! The exact row is copied, not adopted: the backend's row is freed inside
//! `append` like codec/v1's, so everything the provider holds was born
//! inside its own appends, where residency accounting can see it.
//!
//! APPROXIMATE by declaration, as codec/v1.

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

use super::codec::{unsupported, BITS_KEY, SUPPORTED_BITS};
use crate::engines::turbo_quant::{codebooks, TurboQuant};

/// [`CodecRecentKvState`]'s family. Revision 1: codec/v1's encoding and
/// retention, with an exact window of the newest K rows.
pub const IDENTITY_FAMILY: &str = "codec-recent";
pub const IDENTITY_REVISION: u32 = 1;

/// The window key: how many of each layer's newest K rows stay exact.
pub const EXACT_RECENT_K_KEY: &str = "exact_recent_k";

/// The name its refusals carry.
const PROVIDER_NAME: &str = "CodecRecentKvState";

/// One layer. Positions `base..end` are held; V is encoded for all of
/// them, K is encoded for `base..exact_start` and exact for the rest.
struct Layer {
    geometry: LayerKvGeometry,
    base: usize,
    v_codes: Vec<Vec<u8>>,
    k_codes: Vec<Vec<u8>>,
    k_exact: Vec<Vec<f32>>,
}

impl Layer {
    fn new(geometry: LayerKvGeometry) -> Self {
        Self {
            geometry,
            base: 0,
            v_codes: Vec::new(),
            k_codes: Vec::new(),
            k_exact: Vec::new(),
        }
    }

    fn end(&self) -> usize {
        self.base + self.v_codes.len()
    }

    fn exact_start(&self) -> usize {
        self.end() - self.k_exact.len()
    }

    fn heads(&self) -> usize {
        self.geometry.kv_dim / self.geometry.head_dim
    }

    /// Free every position below the plan's floor for a step that has
    /// just appended `position`: its V codes, and its K codes or exact
    /// row, whichever it holds.
    fn release_below(&mut self, position: usize) {
        let floor = self.geometry.history.required_start(position);
        let unreachable = floor.saturating_sub(self.base).min(self.v_codes.len());
        self.v_codes.drain(..unreachable);
        let coded = unreachable.min(self.k_codes.len());
        self.k_codes.drain(..coded);
        self.k_exact.drain(..unreachable - coded);
        self.base += unreachable;
    }
}

/// The one decoded layer, as codec/v1's: which layer, the range it covers,
/// and its K and V as contiguous rows.
#[derive(Default)]
struct Scratch {
    layer: Option<usize>,
    base: usize,
    end: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    indices: Vec<u8>,
}

/// The codec and the per-head workspace it encodes through.
struct Encoder {
    codec: TurboQuant,
    f32s: Vec<f32>,
    u8s: Vec<u8>,
}

impl Encoder {
    /// Encode one f32 row, head by head, into a new allocation of exactly
    /// `bytes`.
    fn encode(&mut self, row: &[f32], head_dim: usize, bytes: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(bytes);
        for head in row.chunks_exact(head_dim) {
            self.codec
                .encode_vector_into(head, &mut out, &mut self.f32s, &mut self.u8s);
        }
        // Blocks are read back at the declared stride: an encoder that
        // wrote any other length would be decoded misaligned.
        assert_eq!(
            out.len(),
            bytes,
            "the codec wrote {} bytes for a row it declares as {bytes}",
            out.len()
        );
        out
    }
}

/// Continuation state holding the plan-required K/V range: V and older K
/// TurboQuant-compressed, the newest `w` K rows exact. KV-only.
pub struct CodecRecentKvState {
    encoder: Encoder,
    window: usize,
    layers: Vec<Layer>,
    position: usize,
    scratch: Scratch,
}

impl CodecRecentKvState {
    /// A fresh state at `bits` (3 or 4) keeping the newest `window` K rows
    /// of each layer exact. `window` 0 holds what codec/v1 holds; the
    /// factory refuses it, since that is codec/v1's identity.
    pub fn new(bits: u8, window: usize) -> Self {
        // The codec's constant table is a process-wide static built on first
        // use; build it here, so no append ever allocates it.
        let _ = codebooks::unit_codebook(bits);
        Self {
            encoder: Encoder {
                codec: TurboQuant::new(bits),
                f32s: Vec::new(),
                u8s: Vec::new(),
            },
            window,
            layers: Vec::new(),
            position: 0,
            scratch: Scratch::default(),
        }
    }

    pub fn identity() -> ContinuationIdentity {
        ContinuationIdentity::new(IDENTITY_FAMILY, IDENTITY_REVISION)
    }

    /// Encoded bytes of one tensor's row (K or V) on `layer`: `heads`
    /// blocks of the codec's norm plus packed indices.
    pub fn encoded_half_bytes(&self, layer: usize) -> usize {
        let l = &self.layers[layer];
        l.heads() * self.encoder.codec.bytes_per_vector(l.geometry.head_dim)
    }

    /// The first position `layer` still holds. Inspection only.
    pub fn rows_base(&self, layer: usize) -> usize {
        self.layers[layer].base
    }

    /// The first position whose K is held exact. Inspection only.
    pub fn exact_start(&self, layer: usize) -> usize {
        self.layers[layer].exact_start()
    }

    /// The encoded V allocations `layer` holds, oldest first — one per
    /// held position. An inspection surface: memory accounting reads
    /// storage by allocation, never from a byte count the provider reports.
    pub fn encoded_values(&self, layer: usize) -> &[Vec<u8>] {
        &self.layers[layer].v_codes
    }

    /// The encoded K allocations `layer` holds, for `base..exact_start`.
    pub fn encoded_keys(&self, layer: usize) -> &[Vec<u8>] {
        &self.layers[layer].k_codes
    }

    /// The exact K rows `layer` holds, for `exact_start..end`.
    pub fn exact_keys(&self, layer: usize) -> &[Vec<f32>] {
        &self.layers[layer].k_exact
    }

    /// The decode scratch (K, V) as allocated — capacity included, since
    /// that is what is resident. Inspection only.
    pub fn scratch(&self) -> (&Vec<f32>, &Vec<f32>) {
        (&self.scratch.keys, &self.scratch.values)
    }
}

impl KvState for CodecRecentKvState {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        if self.layers.is_empty() {
            for (layer, geometry) in layers.iter().enumerate() {
                if let Some(reason) = unsupported(geometry) {
                    panic!("{PROVIDER_NAME} was prepared for layer {layer}, which {reason}");
                }
            }
            self.layers = layers.iter().copied().map(Layer::new).collect();
            // The codec's per-head workspace is sized here, at announcement,
            // so no append allocates anything but the rows it stores.
            let widest = layers.iter().map(|g| g.head_dim).max().unwrap_or(0);
            self.encoder.f32s.reserve_exact(widest);
            self.encoder.f32s.resize(widest, 0.0);
            self.encoder.u8s.reserve_exact(widest);
            self.scratch.indices.reserve_exact(widest);
            return;
        }
        // A held state is being resumed: it must be state for a program of
        // this shape. Reshaping it silently would continue a different
        // conversation.
        let held: Vec<LayerKvGeometry> = self.layers.iter().map(|l| l.geometry).collect();
        assert_eq!(
            held, layers,
            "resumed codec-recent state was prepared for a different program geometry"
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
        let LayerKvGeometry {
            kv_dim, head_dim, ..
        } = geometry;
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
        let half = self.encoded_half_bytes(layer);
        let v_code = self.encoder.encode(&value, head_dim, half);
        let l = &mut self.layers[layer];
        let position = l.end();
        l.v_codes.push(v_code);
        // An exact copy at exactly the row's width: the backend's row may
        // carry spare capacity, and residency is declared per row.
        let mut exact = Vec::with_capacity(kv_dim);
        exact.extend_from_slice(&key);
        l.k_exact.push(exact);
        // At most one row ages per append; a front drain of a list of at
        // most w + 1 row handles, as codec/v1 drains its own.
        while l.k_exact.len() > self.window {
            let aged = l.k_exact.remove(0);
            l.k_codes.push(self.encoder.encode(&aged, head_dim, half));
        }
        l.release_below(position);
    }

    fn prepare_layer(&mut self, layer: usize) {
        let l = &self.layers[layer];
        let LayerKvGeometry {
            kv_dim, head_dim, ..
        } = l.geometry;
        let block = self.encoder.codec.bytes_per_vector(head_dim);
        let held = l.v_codes.len();
        let s = &mut self.scratch;
        // Exact growth: the scratch's capacity is the largest range decoded
        // so far, never an amortised doubling past it.
        let len = held * kv_dim;
        for buffer in [&mut s.keys, &mut s.values] {
            buffer.reserve_exact(len.saturating_sub(buffer.len()));
            buffer.resize(len, 0.0);
        }
        let decode = |codes: &[u8], out: &mut [f32], indices: &mut Vec<u8>| {
            for (block_codes, head) in codes
                .chunks_exact(block)
                .zip(out.chunks_exact_mut(head_dim))
            {
                self.encoder
                    .codec
                    .decode_block_into(block_codes, head, indices);
            }
        };
        for (row, codes) in l.v_codes.iter().enumerate() {
            decode(
                codes,
                &mut s.values[row * kv_dim..(row + 1) * kv_dim],
                &mut s.indices,
            );
        }
        for (row, codes) in l.k_codes.iter().enumerate() {
            decode(
                codes,
                &mut s.keys[row * kv_dim..(row + 1) * kv_dim],
                &mut s.indices,
            );
        }
        let coded = l.k_codes.len();
        for (i, exact) in l.k_exact.iter().enumerate() {
            let row = coded + i;
            s.keys[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(exact);
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

/// Builds [`CodecRecentKvState`]: the K/V region only, configured by
/// exactly two keys, `bits` and `exact_recent_k`, both named — there is no
/// default width and no default window.
#[derive(Debug, Clone, Copy, Default)]
pub struct CodecRecentFactory;

/// `exact_recent_k` as a window of at least one row, or why not.
fn parse_window(raw: &str) -> Result<usize, String> {
    match raw.parse::<usize>() {
        Ok(0) => Err(format!(
            "`{EXACT_RECENT_K_KEY}` must be at least 1; a window of 0 is codec/v1"
        )),
        Ok(w) => Ok(w),
        Err(_) => Err(format!(
            "`{EXACT_RECENT_K_KEY}` must be a whole number of rows; given `{raw}`"
        )),
    }
}

impl ContinuationFactory for CodecRecentFactory {
    fn identity(&self) -> ContinuationIdentity {
        CodecRecentKvState::identity()
    }

    fn regions(&self) -> &[ContinuationRegion] {
        &[ContinuationRegion::Kv]
    }

    fn validate_config(&self, config: &ContinuationConfig) -> Result<(), String> {
        if let Some(other) = config
            .keys()
            .find(|k| *k != BITS_KEY && *k != EXACT_RECENT_K_KEY)
        {
            return Err(format!(
                "takes only `{BITS_KEY}` and `{EXACT_RECENT_K_KEY}`; given `{other}`"
            ));
        }
        match config.get(BITS_KEY) {
            None => {
                return Err(format!(
                    "requires `{BITS_KEY}` (one of {SUPPORTED_BITS:?}); there is no default width"
                ))
            }
            Some(bits) => match bits.parse::<u8>() {
                Ok(b) if SUPPORTED_BITS.contains(&b) => {}
                _ => {
                    return Err(format!(
                        "`{BITS_KEY}` must be one of {SUPPORTED_BITS:?}; given `{bits}`"
                    ))
                }
            },
        }
        match config.get(EXACT_RECENT_K_KEY) {
            None => Err(format!(
                "requires `{EXACT_RECENT_K_KEY}` (rows of K held exact); there is no default window"
            )),
            Some(raw) => parse_window(raw).map(|_| ()),
        }
    }

    fn build(&self, config: &ContinuationConfig) -> BoxedContinuation {
        let bits = config
            .get(BITS_KEY)
            .and_then(|b| b.parse::<u8>().ok())
            .filter(|b| SUPPORTED_BITS.contains(b))
            .expect("build is reached only with a config validate_config accepted");
        let window = config
            .get(EXACT_RECENT_K_KEY)
            .and_then(|w| parse_window(w).ok())
            .expect("build is reached only with a config validate_config accepted");
        Box::new(CodecRecentKvState::new(bits, window))
    }
}
