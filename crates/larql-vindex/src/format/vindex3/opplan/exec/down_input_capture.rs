//! A positioned capture of the dense FFN's `down_proj` input (SG-EDIT-1 I5,
//! item E).
//!
//! [`StepObserver::ffn_down_input`] hands an observer the intermediate the
//! executing backend actually fed to `down_proj`, but carries no position:
//! the executor steps one position at a time and leaves the position to the
//! observer. This capture takes it from the step's own `Embedded` event —
//! the first event every step fires — so an address names a
//! `(layer, position)` pair rather than "whatever this layer saw last".
//!
//! Asking for the down input routes the FFN through the backend's observed
//! entry point instead of its plain one. The acceptance test for this
//! module is therefore that a captured run and an uncaptured run are bit
//! for bit identical; nothing here may assume it.

use std::collections::{BTreeMap, BTreeSet};

use super::intervene::vector_sha256;
use super::observe::{StepEvent, StepObserver};

/// The `down_proj` inputs captured at declared `(layer, position)` pairs.
#[derive(Debug, Default)]
pub struct DownInputCapture {
    wanted: BTreeSet<(usize, usize)>,
    layers: BTreeSet<usize>,
    position: Option<usize>,
    captured: BTreeMap<(usize, usize), Vec<f32>>,
}

impl DownInputCapture {
    /// Capture at these `(layer, position)` pairs.
    pub fn at(addresses: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let wanted: BTreeSet<(usize, usize)> = addresses.into_iter().collect();
        let layers = wanted.iter().map(|&(layer, _)| layer).collect();
        Self {
            wanted,
            layers,
            position: None,
            captured: BTreeMap::new(),
        }
    }

    /// The captured input at an address, if the run reached it.
    pub fn get(&self, layer: usize, position: usize) -> Option<&[f32]> {
        self.captured.get(&(layer, position)).map(Vec::as_slice)
    }

    /// The captured input's hash ([`vector_sha256`]), if the run reached it.
    pub fn sha256(&self, layer: usize, position: usize) -> Option<String> {
        self.get(layer, position).map(vector_sha256)
    }

    /// How many declared addresses were captured.
    pub fn captured(&self) -> usize {
        self.captured.len()
    }

    /// Declared addresses the run has not reached.
    pub fn missing(&self) -> Vec<(usize, usize)> {
        self.wanted
            .iter()
            .filter(|key| !self.captured.contains_key(key))
            .copied()
            .collect()
    }

    /// Feed one step event. A caller composing this into its own observer
    /// forwards every event here.
    pub fn observe_event(&mut self, event: &StepEvent) {
        if let StepEvent::Embedded { position } = event {
            self.position = Some(*position);
        }
    }

    /// Whether the executor should hand this layer's down input over.
    pub fn wants_layer(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    /// Feed one down input. Only a declared `(layer, position)` is kept.
    ///
    /// # Panics
    ///
    /// When a down input arrives before any `Embedded` event: the executor
    /// fires `Embedded` first on every step, so this is a caller that did
    /// not forward events, and a position guessed here would be a wrong
    /// address rather than a missing one.
    pub fn observe_down_input(&mut self, layer: usize, values: &[f32]) {
        let position = self
            .position
            .expect("a down input arrived before the step's `Embedded` event was forwarded");
        let key = (layer, position);
        if self.wanted.contains(&key) {
            self.captured.insert(key, values.to_vec());
        }
    }
}

impl StepObserver for DownInputCapture {
    fn event(&mut self, event: StepEvent) {
        self.observe_event(&event);
    }

    fn wants_ffn_down_input(&self, layer: usize) -> bool {
        self.wants_layer(layer)
    }

    fn ffn_down_input(&mut self, layer: usize, values: &[f32]) {
        self.observe_down_input(layer, values);
    }
}
