//! Logical diff for VINDEX3 containers (V3-LQL-3D): report **model
//! facts**, not files.
//!
//! Three layers, mirroring the format's object → representation →
//! segment authority:
//!
//! - **semantic**: knowledge edges (the L0 store), feature-slot value
//!   changes (gate row / up row / down column, at slot granularity),
//!   and programme metadata;
//! - **representation**: effective-tensor changes that are not
//!   feature-slots (attention, norms, embedding, head), plus
//!   dtype/shape/structure notes;
//! - **physical** (subordinate): segment hashes — how the meaning
//!   happens to be stored.
//!
//! Each side is a container root plus OPTIONAL overlay state, and the
//! comparison always reads **effective** values through the same
//! [`OperandSource`] execution uses. That is what makes the diff see
//! meaning rather than storage: a container-plus-overlay and the
//! clean container COMPILE bakes from it are semantically identical
//! here, while the physical layer is allowed to disagree.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::encode::segment::read_segment_header;
use super::index::Vindex3Index;
use super::inspect::inspect_container;
use super::opplan::exec::operands::{OperandOverrides, OperandSource, OperandStore};
use super::opplan::{plan_component_ops, ComponentOpPlan, LayerFfn, OperandRef};
use crate::error::VindexError;
use crate::format::filenames::{INDEX_JSON, KNN_STORE_BIN};
use crate::patch::knn_store::KnnStore;

/// One side of a diff: a container root, its plan, and its effective
/// operand state (base representation + optional overlay).
pub struct DiffSide {
    pub root: PathBuf,
    pub model: String,
    pub plan: ComponentOpPlan,
    index: Vindex3Index,
    store: OperandStore,
    overrides: OperandOverrides,
    pub knn: KnnStore,
}

impl DiffSide {
    /// Open a container from disk: base representation, `knn_store.bin`
    /// when the container carries one, no overlay.
    pub fn open(root: &Path, component: &str) -> Result<Self, VindexError> {
        let inspection = inspect_container(root, false)?;
        let outcome = plan_component_ops(&inspection, root, component)?;
        let plan = outcome
            .plan
            .ok_or_else(|| VindexError::Parse(format!("component `{component}` has no plan")))?;
        let store = OperandStore::open(root, &inspection)?;
        let raw_index = std::fs::read_to_string(root.join(INDEX_JSON))?;
        let index: Vindex3Index = serde_json::from_str(&raw_index)
            .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
        let knn_path = root.join(KNN_STORE_BIN);
        let knn = if knn_path.exists() {
            KnnStore::load(&knn_path).map_err(VindexError::Parse)?
        } else {
            KnnStore::default()
        };
        Ok(Self {
            root: root.to_path_buf(),
            model: index.model.clone(),
            plan,
            index,
            store,
            overrides: OperandOverrides::default(),
            knn,
        })
    }

    /// Layer overlay state onto this side: its operand edits become
    /// part of the side's EFFECTIVE values, its KNN store replaces the
    /// container's. This is what `DIFF … CURRENT` means on V3 — the
    /// session's model state, not merely the bound path.
    pub fn with_overlay(mut self, overrides: OperandOverrides, knn: KnnStore) -> Self {
        self.overrides = overrides;
        self.knn = knn;
        self
    }

    fn source(&self) -> OperandSource<'_> {
        OperandSource::overlaid(&self.store, &self.overrides)
    }
}

/// A knowledge edge, as the L0 store speaks it.
pub type KnowledgeEdge = (String, String, String);

/// One feature slot whose effective values differ.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SlotDiff {
    pub layer: usize,
    pub feature: usize,
    pub gate_changed: bool,
    pub up_changed: bool,
    pub down_changed: bool,
}

/// The logical report: model facts that differ.
#[derive(Debug, Default)]
pub struct SemanticDiff {
    pub knowledge_added: Vec<KnowledgeEdge>,
    pub knowledge_removed: Vec<KnowledgeEdge>,
    pub features: Vec<SlotDiff>,
    /// Programme/metadata differences (model name, layer count,
    /// structure), human-readable.
    pub metadata: Vec<String>,
    /// Representation-level changes that are not feature slots:
    /// `object/tensor` names whose effective values differ.
    pub changed_tensors: Vec<String>,
}

impl SemanticDiff {
    pub fn is_empty(&self) -> bool {
        self.knowledge_added.is_empty()
            && self.knowledge_removed.is_empty()
            && self.features.is_empty()
            && self.metadata.is_empty()
            && self.changed_tensors.is_empty()
    }
}

/// The subordinate physical report: how storage differs.
#[derive(Debug, Default)]
pub struct PhysicalDiff {
    /// `(representation id, sha256 in A, sha256 in B)` for segments
    /// whose bytes differ.
    pub changed_segments: Vec<(String, String, String)>,
    pub only_in_a: Vec<String>,
    pub only_in_b: Vec<String>,
}

/// Which FFN role a tensor plays, for slot-granular interpretation.
#[derive(Clone, Copy)]
enum FfnRole {
    /// Rows are features.
    GateRows(usize),
    UpRows(usize),
    /// Columns are features.
    DownCols(usize),
}

fn ffn_roles(plan: &ComponentOpPlan) -> BTreeMap<(String, String), FfnRole> {
    let mut roles = BTreeMap::new();
    for (layer_index, layer) in plan.layers.iter().enumerate() {
        if let Some(LayerFfn::Dense(ffn)) = &layer.ffn {
            let gate = ffn.gate.as_ref().unwrap_or(&ffn.up);
            roles.insert(
                (gate.object.clone(), gate.tensor.clone()),
                FfnRole::GateRows(layer_index),
            );
            roles.insert(
                (ffn.up.object.clone(), ffn.up.tensor.clone()),
                FfnRole::UpRows(layer_index),
            );
            roles.insert(
                (ffn.down.object.clone(), ffn.down.tensor.clone()),
                FfnRole::DownCols(layer_index),
            );
        }
    }
    roles
}

fn knowledge_set(knn: &KnnStore) -> BTreeSet<KnowledgeEdge> {
    knn.entries()
        .values()
        .flatten()
        .map(|e| (e.entity.clone(), e.relation.clone(), e.target_token.clone()))
        .collect()
}

/// The logical diff: what model facts differ between the two sides'
/// EFFECTIVE states.
pub fn semantic_diff(a: &DiffSide, b: &DiffSide) -> Result<SemanticDiff, VindexError> {
    let mut diff = SemanticDiff::default();

    if a.model != b.model {
        diff.metadata
            .push(format!("model: {} → {}", a.model, b.model));
    }
    if a.plan.layers.len() != b.plan.layers.len() {
        diff.metadata.push(format!(
            "layers: {} → {}",
            a.plan.layers.len(),
            b.plan.layers.len()
        ));
    }

    // Knowledge edges (the L0 store).
    let (ka, kb) = (knowledge_set(&a.knn), knowledge_set(&b.knn));
    diff.knowledge_added = kb.difference(&ka).cloned().collect();
    diff.knowledge_removed = ka.difference(&kb).cloned().collect();

    // Tensor universe: every object/tensor named by either side's
    // segment headers, compared as EFFECTIVE values.
    let roles = ffn_roles(&a.plan);
    let mut slots: BTreeMap<(usize, usize), SlotDiff> = BTreeMap::new();

    let tensors_of =
        |side: &DiffSide| -> Result<BTreeMap<(String, String), OperandRef>, VindexError> {
            let mut map = BTreeMap::new();
            for entry in side.index.representations.values() {
                let (header, _) = read_segment_header(&side.root.join(&entry.segment))?;
                for t in header.tensors {
                    map.insert(
                        (entry.object.clone(), t.name.clone()),
                        OperandRef {
                            object: entry.object.clone(),
                            tensor: t.name,
                            dtype: t.dtype,
                            shape: t.shape,
                        },
                    );
                }
            }
            Ok(map)
        };
    let ta = tensors_of(a)?;
    let tb = tensors_of(b)?;

    for (key, ref_a) in &ta {
        let Some(ref_b) = tb.get(key) else {
            diff.metadata
                .push(format!("only in A: {}/{}", key.0, key.1));
            continue;
        };
        if ref_a.shape != ref_b.shape {
            diff.metadata.push(format!(
                "shape: {}/{} {:?} → {:?}",
                key.0, key.1, ref_a.shape, ref_b.shape
            ));
            continue;
        }
        let va = a.source().load(ref_a)?;
        let vb = b.source().load(ref_b)?;
        if va == vb {
            continue;
        }
        match roles.get(key) {
            Some(&FfnRole::GateRows(layer)) => {
                mark_row_changes(&mut slots, layer, ref_a, &va, &vb, |d| {
                    d.gate_changed = true;
                });
            }
            Some(&FfnRole::UpRows(layer)) => {
                mark_row_changes(&mut slots, layer, ref_a, &va, &vb, |d| d.up_changed = true);
            }
            Some(&FfnRole::DownCols(layer)) => {
                mark_col_changes(&mut slots, layer, ref_a, &va, &vb);
            }
            None => diff.changed_tensors.push(format!("{}/{}", key.0, key.1)),
        }
    }
    for key in tb.keys() {
        if !ta.contains_key(key) {
            diff.metadata
                .push(format!("only in B: {}/{}", key.0, key.1));
        }
    }

    diff.features = slots.into_values().collect();
    Ok(diff)
}

fn mark_row_changes(
    slots: &mut BTreeMap<(usize, usize), SlotDiff>,
    layer: usize,
    operand: &OperandRef,
    va: &[f32],
    vb: &[f32],
    mark: impl Fn(&mut SlotDiff),
) {
    let cols = operand.shape[1];
    for feature in 0..operand.shape[0] {
        if va[feature * cols..(feature + 1) * cols] != vb[feature * cols..(feature + 1) * cols] {
            mark(slots.entry((layer, feature)).or_insert(SlotDiff {
                layer,
                feature,
                gate_changed: false,
                up_changed: false,
                down_changed: false,
            }));
        }
    }
}

fn mark_col_changes(
    slots: &mut BTreeMap<(usize, usize), SlotDiff>,
    layer: usize,
    operand: &OperandRef,
    va: &[f32],
    vb: &[f32],
) {
    let (rows, cols) = (operand.shape[0], operand.shape[1]);
    for feature in 0..cols {
        let changed = (0..rows).any(|r| va[r * cols + feature] != vb[r * cols + feature]);
        if changed {
            slots
                .entry((layer, feature))
                .or_insert(SlotDiff {
                    layer,
                    feature,
                    gate_changed: false,
                    up_changed: false,
                    down_changed: false,
                })
                .down_changed = true;
        }
    }
}

impl DiffSide {
    /// The side's container index — the physical layer's authority.
    pub fn index(&self) -> &Vindex3Index {
        &self.index
    }
}

/// The subordinate physical report: segment-level differences between
/// two container indexes (overlays have no physical form).
pub fn physical_diff(a: &Vindex3Index, b: &Vindex3Index) -> PhysicalDiff {
    let mut out = PhysicalDiff::default();
    for (rep_id, ea) in &a.representations {
        match b.representations.get(rep_id) {
            Some(eb) if ea.segment_sha256 != eb.segment_sha256 => out.changed_segments.push((
                rep_id.clone(),
                ea.segment_sha256.clone(),
                eb.segment_sha256.clone(),
            )),
            Some(_) => {}
            None => out.only_in_a.push(rep_id.clone()),
        }
    }
    for rep_id in b.representations.keys() {
        if !a.representations.contains_key(rep_id) {
            out.only_in_b.push(rep_id.clone());
        }
    }
    out
}
