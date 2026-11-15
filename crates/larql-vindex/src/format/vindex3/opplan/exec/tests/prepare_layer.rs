//! CONTINUATION-CODEC-1 C1: `prepare_layer` precedes every read.
//!
//! A provider that lends ONLY what its last `prepare_layer` materialised —
//! copying the layer's held range into one scratch, the way a compressed
//! provider decodes into one — is exact only if the executor prepares
//! every layer before every read. Two claims:
//!
//! - through batched prefill, resumed prefill and decode it is
//!   bit-identical to row/v1, so no read site skips the hook;
//! - a wrapper that skips the hook is refused by name at the step, before
//!   any kernel — a missed hook cannot become attention over stale rows.

use super::golden::{G_LAYERS, G_TOKENS};
use crate::format::vindex3::opplan::exec::continuation::{LatentKvRows, RecurrentState};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::{
    ContinuationError, KvState, LayerKvGeometry, RowKvState,
};
use crate::format::vindex3::opplan::exec::kv_view::KvView;
use crate::format::vindex3::opplan::exec::prefill_plan;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

/// The prompt's first positions go in by batched prefill (position 0),
/// the rest by resumed prefill (per-position reads), then decode.
const BATCHED_PREFIX: usize = 3;
const DECODE_TOKENS: [u32; 4] = [1, 2, 3, 4];

/// One layer's range, materialised as contiguous rows.
struct Scratch {
    layer: usize,
    base: usize,
    width: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// Holds rows in a [`RowKvState`] but lends only its scratch: the layer
/// last prepared, as it stood when prepared.
#[derive(Default)]
struct Materialising {
    inner: RowKvState,
    widths: Vec<usize>,
    scratch: Option<Scratch>,
    prepared: usize,
}

impl KvState for Materialising {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.widths = layers.iter().map(|g| g.kv_dim).collect();
        self.inner.prepare(layers)
    }
    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.inner.append(layer, key, value)
    }
    fn rows(&self, layer: usize) -> KvView<'_> {
        match &self.scratch {
            Some(s) if s.layer == layer => KvView::contiguous(s.base, s.width, &s.keys, &s.values)
                .expect("the scratch is whole rows of its layer's width"),
            _ => KvView::empty(),
        }
    }
    fn prepare_layer(&mut self, layer: usize) {
        let view = self.inner.rows(layer);
        let (keys, values) = view.to_owned_rows();
        self.scratch = Some(Scratch {
            layer,
            base: view.base(),
            width: self.widths[layer],
            keys: keys.concat(),
            values: values.concat(),
        });
        self.prepared += 1;
    }
    fn position(&self) -> usize {
        self.inner.position()
    }
    fn set_position(&mut self, position: usize) {
        self.inner.set_position(position)
    }
    fn invalidate(&mut self, reason: &str) {
        self.inner.invalidate(reason)
    }
    fn invalidated(&self) -> Option<&str> {
        self.inner.invalidated()
    }
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.inner.recurrent_state(layer)
    }
    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.inner.latent_state(layer)
    }
}

/// Forwards everything to [`Materialising`] except the hook.
#[derive(Default)]
struct SkipsTheHook(Materialising);

impl KvState for SkipsTheHook {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.0.prepare(layers)
    }
    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.0.append(layer, key, value)
    }
    fn rows(&self, layer: usize) -> KvView<'_> {
        self.0.rows(layer)
    }
    fn prepare_layer(&mut self, _layer: usize) {}
    fn position(&self) -> usize {
        self.0.position()
    }
    fn set_position(&mut self, position: usize) {
        self.0.set_position(position)
    }
    fn invalidate(&mut self, reason: &str) {
        self.0.invalidate(reason)
    }
    fn invalidated(&self) -> Option<&str> {
        self.0.invalidated()
    }
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.0.recurrent_state(layer)
    }
    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.0.latent_state(layer)
    }
}

/// Batched prefill, resumed prefill, then decode over `kv`; every logit
/// row the journey produces, in order.
fn journey(kv: &mut dyn KvState) -> Result<Vec<Option<Vec<f32>>>, crate::error::VindexError> {
    let (_c, plan, store) = super::decode::fixture();
    let backend = ReferenceBackend::new();
    let mut logits = Vec::new();
    let (head, tail) = G_TOKENS.split_at(BATCHED_PREFIX);
    logits.push(prefill_plan(&plan, &store, head, &backend, &mut *kv)?.logits);
    logits.push(prefill_plan(&plan, &store, tail, &backend, &mut *kv)?.logits);
    let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, kv)?;
    for token in DECODE_TOKENS {
        logits.push(session.step(token)?.logits);
    }
    Ok(logits)
}

#[test]
fn a_provider_lending_only_prepared_rows_is_exact_through_every_phase() {
    let mut row = RowKvState::default();
    let expected = journey(&mut row).unwrap();
    let mut materialising = Materialising::default();
    let got = journey(&mut materialising).unwrap();
    assert_eq!(got, expected, "a read site skipped prepare_layer");
    assert!(
        materialising.prepared
            >= (G_TOKENS.len() - BATCHED_PREFIX + DECODE_TOKENS.len()) * G_LAYERS,
        "every resumed and decoded position prepares every layer: {} calls",
        materialising.prepared
    );
}

#[test]
fn a_wrapper_that_skips_the_hook_is_refused_by_name_at_the_step() {
    let mut skips = SkipsTheHook::default();
    let refusal = journey(&mut skips).unwrap_err().to_string();
    assert!(
        refusal.contains("continuation view holds positions"),
        "a stale read must be refused by the step's range check: {refusal}"
    );
}
