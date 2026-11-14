//! Every `StandardEngine` entry point that carries a caller's FFN into the
//! dispatch ring, and one way to drive them all.
//!
//! Enumerated rather than written out per test so the gate can sweep them
//! exhaustively — the failure it guards against is precisely one path being
//! missed, which is what an ad-hoc list of hand-written cases produces.

use larql_inference::ffn::{FfnBackend, MoeFfn};
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{AsyncComputeBackend, KvEngine};
use larql_kv::StandardEngine;
use ndarray::Array2;

use crate::routes::ExecutingRoute;

/// Prompt for every entry point. Three tokens: enough that prefill caches a
/// distinguishable number of rows, small enough to stay fast.
pub const PROMPT: [u32; 3] = [0, 1, 2];
/// The token a decode step is driven with.
pub const NEXT_TOKEN: u32 = 3;

/// Synthetic hidden-state values for the `from_hidden` entry points. Any finite
/// spread works; these are a deterministic ramp so a failure reproduces.
const HIDDEN_STRIDE_ROW: usize = 7;
const HIDDEN_MODULUS: usize = 13;
const HIDDEN_SCALE: f32 = 0.01;
const HIDDEN_OFFSET: f32 = -0.06;

/// A `StandardEngine` route that reaches the widened dispatch helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    /// `prefill` over the sync slot → `kv_prefill_via_dispatch`.
    PrefillSync,
    /// `prefill` over the async slot → `kv_prefill_via_dispatch_async`.
    PrefillAsync,
    /// `prefill_from_hidden` → `kv_prefill_from_hidden_via_dispatch`.
    PrefillFromHidden,
    /// `prefill_from_hidden` over the async slot.
    PrefillFromHiddenAsync,
    /// `prefill_resident` → the same `do_prefill` body with `index: Some`.
    PrefillResident,
    /// `decode_step` over the sync slot → `kv_decode_step_via_dispatch`.
    DecodeSync,
    /// `decode_step` over the async slot.
    DecodeAsync,
    /// `decode_step_resident` → the same `do_decode_step` body.
    DecodeResident,
}

impl Entry {
    pub const ALL: [Self; 8] = [
        Self::PrefillSync,
        Self::PrefillAsync,
        Self::PrefillFromHidden,
        Self::PrefillFromHiddenAsync,
        Self::PrefillResident,
        Self::DecodeSync,
        Self::DecodeAsync,
        Self::DecodeResident,
    ];

    fn is_decode(self) -> bool {
        matches!(
            self,
            Self::DecodeSync | Self::DecodeAsync | Self::DecodeResident
        )
    }

    fn uses_async_slot(self) -> bool {
        matches!(
            self,
            Self::PrefillAsync | Self::PrefillFromHiddenAsync | Self::DecodeAsync
        )
    }

    fn engine(self) -> StandardEngine {
        if self.uses_async_slot() {
            let backend: Box<dyn AsyncComputeBackend> = Box::new(larql_compute::CpuBackend);
            StandardEngine::with_async_backend(None, backend)
        } else {
            StandardEngine::new(None)
        }
    }
}

/// A structurally valid index for the resident entry points.
///
/// The CPU backend ignores it with the Q4K-direct flags off (their default in
/// tests), so it exists only to satisfy the signature. The resident routes
/// differ from the plain ones exactly by threading `Some(index)` through
/// `do_prefill` / `do_decode_step`, which is what is under test.
pub fn empty_index(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    larql_vindex::VectorIndex::new(
        vec![None; weights.num_layers],
        vec![None; weights.num_layers],
        weights.num_layers,
        weights.hidden_size,
    )
}

fn synthetic_hidden(weights: &ModelWeights) -> Array2<f32> {
    Array2::from_shape_fn((PROMPT.len(), weights.hidden_size), |(row, col)| {
        ((row * HIDDEN_STRIDE_ROW + col) % HIDDEN_MODULUS) as f32 * HIDDEN_SCALE + HIDDEN_OFFSET
    })
}

/// Drive one entry point with `ffn`.
///
/// Decode entries prefill first with a route that executes, so a refusal
/// observed afterwards can only have come from the decode step itself.
pub fn drive(
    entry: Entry,
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
) -> Result<Array2<f32>, EngineError> {
    let index = empty_index(weights);
    let mut engine = entry.engine();

    if entry.is_decode() {
        let clean = MoeFfn::strict(weights, &ExecutingRoute);
        engine
            .prefill(weights, &clean, &PROMPT)
            .expect("setup prefill must succeed before the decode step under test");
    }

    match entry {
        Entry::PrefillSync | Entry::PrefillAsync => engine.prefill(weights, ffn, &PROMPT),
        Entry::PrefillFromHidden | Entry::PrefillFromHiddenAsync => {
            engine.prefill_from_hidden(weights, ffn, &synthetic_hidden(weights))
        }
        Entry::PrefillResident => engine.prefill_resident(weights, ffn, &index, &PROMPT),
        Entry::DecodeSync | Entry::DecodeAsync => engine.decode_step(weights, ffn, NEXT_TOKEN),
        Entry::DecodeResident => engine.decode_step_resident(weights, ffn, &index, NEXT_TOKEN),
    }
}
