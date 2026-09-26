//! `window/v1`: continuation state bounded by the attention horizon
//! (CONTINUATION-WINDOW-1).
//!
//! Each layer holds exactly the rows the plan's retention authority
//! ([`HistoryRange`]) says a later step can still read, and releases the
//! rest. A sliding layer of window `w` holds `w` rows however long the
//! conversation grows; a full-span layer holds everything, exactly as
//! `row/v1` does. The rows are the backend's own allocations, adopted as
//! they are appended — never copied — and a row that falls below the
//! plan's floor is freed as it is drained. Rows are lent through a
//! [`KvView`] over absolute positions `[base, end)`, so every executor path
//! reads them by position without knowing where the physical history
//! starts.
//!
//! The representation (adopted rows with a draining front) was chosen in
//! WINDOW-1's W1 before this code was written; see
//! `docs/represent/forecasts/continuation-window-1-notes.json`.

use larql_vindex::format::vindex3::opplan::exec::continuation::{LatentKvRows, RecurrentState};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    BoxedContinuation, ContinuationFactory, ContinuationRegion,
};
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, KvState, LayerKvGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::kv_view::KvView;

/// [`WindowKvState`]'s family. Revision 1 retains exactly the plan's
/// required range; it moves when the same appended history would be served
/// back differently.
pub const IDENTITY_FAMILY: &str = "window";
pub const IDENTITY_REVISION: u32 = 1;

/// The name its refusals carry.
const PROVIDER_NAME: &str = "WindowKvState";

/// One layer: its geometry (the retention authority included), the first
/// position still held, and the adopted rows from there on.
struct Layer {
    geometry: LayerKvGeometry,
    base: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl Layer {
    fn new(geometry: LayerKvGeometry) -> Self {
        Self {
            geometry,
            base: 0,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Release every row below the plan's floor for a step that has just
    /// appended `position`.
    fn release_below(&mut self, position: usize) {
        let floor = self.geometry.history.required_start(position);
        let unreachable = floor.saturating_sub(self.base).min(self.keys.len());
        // Draining drops each row's own allocation: nothing below the floor
        // stays allocated anywhere this provider owns.
        self.keys.drain(..unreachable);
        self.values.drain(..unreachable);
        self.base += unreachable;
    }
}

/// Continuation state holding exactly the plan-required K/V range.
///
/// KV-only: it declares the [`ContinuationRegion::Kv`] region and no
/// other, so a plan that needs recurrent or latent state is refused at
/// selection rather than half-served.
#[derive(Default)]
pub struct WindowKvState {
    layers: Vec<Layer>,
    position: usize,
}

impl WindowKvState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn identity() -> ContinuationIdentity {
        ContinuationIdentity::new(IDENTITY_FAMILY, IDENTITY_REVISION)
    }
}

impl KvState for WindowKvState {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        if self.layers.is_empty() {
            self.layers = layers.iter().copied().map(Layer::new).collect();
            return;
        }
        // A held state is being resumed: it must be state for a program of
        // this shape, retention authority included. Reshaping it silently
        // would continue a different conversation.
        let held: Vec<LayerKvGeometry> = self.layers.iter().map(|l| l.geometry).collect();
        assert_eq!(
            held, layers,
            "resumed window state was prepared for a different program geometry"
        );
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        let l = &mut self.layers[layer];
        let kv_dim = l.geometry.kv_dim;
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
        let position = l.base + l.keys.len();
        l.keys.push(key);
        l.values.push(value);
        l.release_below(position);
    }

    fn rows(&self, layer: usize) -> KvView<'_> {
        let l = &self.layers[layer];
        KvView::rows_from(l.base, &l.keys, &l.values).expect("a layer's base never exceeds its end")
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

/// Builds [`WindowKvState`]: the K/V region only, no options.
#[derive(Debug, Clone, Copy, Default)]
pub struct WindowFactory;

impl ContinuationFactory for WindowFactory {
    fn identity(&self) -> ContinuationIdentity {
        WindowKvState::identity()
    }

    fn regions(&self) -> &[ContinuationRegion] {
        &[ContinuationRegion::Kv]
    }

    fn build(&self, _config: &ContinuationConfig) -> BoxedContinuation {
        Box::new(WindowKvState::new())
    }
}
