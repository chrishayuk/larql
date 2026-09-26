//! CONTINUATION-VIEW-1 V4 (B, S3): `ExactRetention`, a PROOF provider that
//! keeps only the rows the plan's retention authority still requires.
//!
//! Not WINDOW-1's shipped provider: a test target, one allocation per row
//! (it adopts the backend's rows, as row/v1 does), dropping every row below
//! `HistoryRange::required_start(p)` after appending position `p`. It reads
//! each layer's `HistoryRange` from the plan geometry and never interprets a
//! `window` itself. `arm_violation` seeds S3: the next append on layer 0
//! drops one row the plan still requires.

use larql_vindex::format::vindex3::opplan::exec::continuation::{LatentKvRows, RecurrentState};
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, ContinuationProvider, HistoryRange, LayerKvGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::kv_view::KvView;

use super::measured::Inspect;

const NAME: &str = "ExactRetention";

struct Layer {
    history: HistoryRange,
    /// The first position still held.
    base: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

/// What one append left behind, checked against the plan.
#[derive(Clone, Debug)]
pub struct RetentionCheck {
    pub position: usize,
    pub base: usize,
    pub end: usize,
    pub required_start: usize,
    pub dropped: usize,
}

impl RetentionCheck {
    /// B's first two conditions for this append.
    pub fn holds(&self) -> bool {
        self.base == self.required_start && self.end == self.position + 1
    }
}

pub struct ExactRetention {
    layers: Vec<Layer>,
    position: usize,
    armed: bool,
    pub checks: Vec<RetentionCheck>,
}

/// Checks reserved up front: the provider must not allocate for its own
/// record inside an append, or the harness would rightly see an allocation
/// no storage accounts for (gemma3-4b's journey appends ≈38,000 times).
const CHECK_CAPACITY: usize = 1 << 17;

impl Default for ExactRetention {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            position: 0,
            armed: false,
            checks: Vec::with_capacity(CHECK_CAPACITY),
        }
    }
}

impl ExactRetention {
    /// S3: the next append on layer 0 drops one row the next step needs.
    pub fn arm_violation(&mut self) {
        self.armed = true;
    }

    /// Rows physically held on `layer`.
    pub fn resident(&self, layer: usize) -> usize {
        self.layers[layer].keys.len()
    }
}

impl Inspect for ExactRetention {
    fn matrix_ptrs(&self, _: usize) -> Option<(usize, usize)> {
        None
    }
    fn matrix_rows(&self, _: usize) -> Option<usize> {
        None
    }
}

impl ContinuationProvider for ExactRetention {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        if self.layers.is_empty() {
            self.layers = layers
                .iter()
                .map(|g| Layer {
                    history: g.history,
                    base: 0,
                    keys: Vec::new(),
                    values: Vec::new(),
                })
                .collect();
        }
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        let l = &mut self.layers[layer];
        let position = l.base + l.keys.len();
        l.keys.push(key);
        l.values.push(value);
        let required_start = l.history.required_start(position);
        let floor = if self.armed && layer == 0 {
            self.armed = false;
            // One past what the NEXT step (position + 1) needs: a row the
            // plan still requires, deliberately dropped.
            l.history.required_start(position + 1) + 1
        } else {
            required_start
        };
        let dropped = floor.saturating_sub(l.base).min(l.keys.len());
        l.keys.drain(..dropped);
        l.values.drain(..dropped);
        l.base += dropped;
        self.checks.push(RetentionCheck {
            position,
            base: l.base,
            end: l.base + l.keys.len(),
            required_start,
            dropped,
        });
    }

    fn rows(&self, layer: usize) -> KvView<'_> {
        let l = &self.layers[layer];
        KvView::rows_from(l.base, &l.keys, &l.values).expect("base never exceeds end")
    }

    fn position(&self) -> usize {
        self.position
    }

    fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        Err(ContinuationError::RecurrentUnsupported {
            provider: NAME,
            layer,
        })
    }

    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        Err(ContinuationError::LatentUnsupported {
            provider: NAME,
            layer,
        })
    }
}
