//! The external continuation provider: `hostile-test-provider/v77`.
//!
//! Deliberately boring (frozen in C5's design): row semantics, KV only,
//! its own storage type and its own position counter. It borrows only
//! exported API — the provider trait and its geometry and error types —
//! and says so in its refusals by name. A `corrupt` switch builds the
//! sibling the "numbers are its own" control needs: identical except
//! that it moves one stored value.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use larql_vindex::format::vindex3::opplan::exec::continuation::{LatentKvRows, RecurrentState};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    BoxedContinuation, ContinuationFactory, ContinuationRegion,
};
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, ContinuationProvider, LayerKvGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::kv_view::KvView;

/// The family no shipped source names. The genericity scan looks for it.
pub const HOSTILE_FAMILY: &str = "hostile-test-provider";
/// The revision the journey registers.
pub const HOSTILE_REVISION: u32 = 77;
/// The one option it accepts. Recorded, never acted on: its only job is
/// to make configuration a second authority.
pub const BITS_OPTION: &str = "bits";
/// The name its refusals carry.
const PROVIDER_NAME: &str = "HostileRows";

pub fn hostile_identity(revision: u32) -> ContinuationIdentity {
    ContinuationIdentity::new(HOSTILE_FAMILY, revision)
}

/// One layer's rows, in the order they were appended.
#[derive(Default)]
struct Layer {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

/// The provider's state for one conversation.
pub struct HostileRows {
    layers: Vec<Layer>,
    position: usize,
    /// Move the first stored key by one ULP (the sibling provider).
    corrupt: bool,
}

impl HostileRows {
    fn new(corrupt: bool) -> Self {
        Self {
            layers: Vec::new(),
            position: 0,
            corrupt,
        }
    }
}

impl ContinuationProvider for HostileRows {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        // An announcement, not a reset: a resumed state keeps its rows.
        if self.layers.len() < layers.len() {
            self.layers.resize_with(layers.len(), Layer::default);
        }
    }

    fn append(&mut self, layer: usize, mut key: Vec<f32>, value: Vec<f32>) {
        if self.corrupt {
            key[0] = f32::from_bits(key[0].to_bits() + 1);
            self.corrupt = false;
        }
        let rows = &mut self.layers[layer];
        rows.keys.push(key);
        rows.values.push(value);
    }

    fn rows(&self, layer: usize) -> KvView<'_> {
        let rows = &self.layers[layer];
        KvView::over_rows(&rows.keys, &rows.values)
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

/// Registers `hostile-test-provider/v{revision}`, counting every build.
pub struct HostileFactory {
    pub revision: u32,
    pub corrupt: bool,
    pub builds: Arc<AtomicUsize>,
}

impl HostileFactory {
    pub fn new(revision: u32) -> Self {
        Self {
            revision,
            corrupt: false,
            builds: Arc::default(),
        }
    }

    /// The sibling that moves one stored value.
    pub fn corrupting(revision: u32) -> Self {
        Self {
            corrupt: true,
            ..Self::new(revision)
        }
    }

    pub fn builds(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.builds)
    }

    /// A provider built directly, bypassing the registry — only the F5
    /// control does this, to show the late refusal path exists.
    pub fn build_unselected() -> BoxedContinuation {
        Box::new(HostileRows::new(false))
    }
}

impl ContinuationFactory for HostileFactory {
    fn identity(&self) -> ContinuationIdentity {
        hostile_identity(self.revision)
    }

    fn regions(&self) -> &[ContinuationRegion] {
        &[ContinuationRegion::Kv]
    }

    fn validate_config(&self, config: &ContinuationConfig) -> Result<(), String> {
        match config.keys().find(|key| *key != BITS_OPTION) {
            Some(key) => Err(format!(
                "unknown option `{key}`; takes only `{BITS_OPTION}`"
            )),
            None => Ok(()),
        }
    }

    fn build(&self, _config: &ContinuationConfig) -> BoxedContinuation {
        self.builds.fetch_add(1, Ordering::SeqCst);
        Box::new(HostileRows::new(self.corrupt))
    }
}
