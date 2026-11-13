//! One query head's sources, captured and replayed (SG-EDIT-1 I5, item B).
//!
//! No production intervention edits K/V rows or the KV cache, and the cache
//! holds rows per KV head, so the query heads of a group read identical V.
//! The H1-only counterfactual is built instead from what the head record
//! already carries — the conditioned query, the KV head's key and value
//! rows over `source_start..=position`, and the natural `ctx_h` — by
//! rerunning that ONE query head's scores, softmax and weighted-V sum
//! ([`replay_softmax_source_head`]) with chosen value rows replaced, and
//! applying the result with a head `Replace` on `ctx_h`. The executor then
//! runs the gate, `o_proj`, the post-attention norm over every head, the
//! residual scale and the add itself: the replay never projects a head
//! delta under a stale norm scale.
//!
//! What this is, and is not: only the replaced head sees the new rows. The
//! cache, the other query heads of the KV group and later positions are
//! untouched, so the result is an attention computation no edit to the
//! shared V could produce. It isolates one head's response; it does not
//! simulate a payload edit.
//!
//! The acceptance laws (`tests/source_head.rs`), on the PRODUCTION backend,
//! whose softmax the replay shares (the reference backend transcribes its
//! own and is only close): a replay with the natural
//! rows reproduces the captured `ctx_h`, and a `Replace` with that replay
//! leaves the whole run — every carrier write and the logits — unchanged.

use std::collections::{BTreeMap, BTreeSet};

use super::super::ComponentOpPlan;
use super::head_replay::replay_softmax_source_head;
use super::intervene::{vector_sha256, VectorProvenance};
use super::intervene_heads::{HeadAddress, HeadIntervention};
use super::observe::{AttentionHeadRecord, StepEvent, StepObserver};
use crate::error::VindexError;

/// One query head's sources at one position, as the kernel held them.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceHead {
    pub layer: usize,
    pub head: usize,
    pub position: usize,
    /// The first source position the head could read.
    pub source_start: usize,
    /// The conditioned query slice the head scored with.
    pub query: Vec<f32>,
    /// The KV head's conditioned key rows, one per source position.
    pub keys: Vec<Vec<f32>>,
    /// The KV head's value rows, one per source position.
    pub values: Vec<Vec<f32>>,
    /// The natural `ctx_h` the kernel produced.
    pub ctx: Vec<f32>,
}

impl SourceHead {
    fn from_record(layer: usize, record: &AttentionHeadRecord<'_>) -> Self {
        Self {
            layer,
            head: record.head,
            position: record.position,
            source_start: record.source_start,
            query: record.query.to_vec(),
            keys: record.source_keys.iter().map(|k| k.to_vec()).collect(),
            values: record.source_values.iter().map(|v| v.to_vec()).collect(),
            ctx: record.values.to_vec(),
        }
    }

    /// The absolute source positions this head read, ascending.
    pub fn sources(&self) -> std::ops::RangeInclusive<usize> {
        self.source_start..=self.position
    }

    /// Rerun the head with the captured rows.
    pub fn replay(&self, plan: &ComponentOpPlan) -> Result<Vec<f32>, VindexError> {
        self.replay_with_values(plan, &BTreeMap::new())
    }

    /// Rerun the head with the value rows at the named absolute source
    /// positions replaced. Every named position must be a source of this
    /// head, and every row `head_dim` wide.
    pub fn replay_with_values(
        &self,
        plan: &ComponentOpPlan,
        replacements: &BTreeMap<usize, Vec<f32>>,
    ) -> Result<Vec<f32>, VindexError> {
        let op = plan
            .layers
            .get(self.layer)
            .and_then(|layer| layer.attention.softmax())
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "source-head replay: layer {} has no softmax attention",
                    self.layer
                ))
            })?;
        if op.sinks.is_some() {
            return Err(VindexError::Parse(format!(
                "source-head replay: layer {} declares attention sinks, which the replay does \
                 not model",
                self.layer
            )));
        }
        let mut values = self.values.clone();
        for (&source, row) in replacements {
            if !self.sources().contains(&source) {
                return Err(VindexError::Parse(format!(
                    "source-head replay: position {source} is not a source of layer {} head {} \
                     at position {} (sources {:?})",
                    self.layer,
                    self.head,
                    self.position,
                    self.sources()
                )));
            }
            if row.len() != op.head_dim {
                return Err(VindexError::Parse(format!(
                    "source-head replay: replacement row at position {source} has width {} on a \
                     head of width {}",
                    row.len(),
                    op.head_dim
                )));
            }
            values[source - self.source_start] = row.clone();
        }
        let replay = replay_softmax_source_head(
            &self.query,
            &self.keys,
            &values,
            op.score_scale,
            op.logit_softcapping,
        )?;
        Ok(replay.values)
    }

    /// A head `Replace` of this head's `ctx_h` at its own position, with
    /// `ctx` — normally a replay's output. The vector's provenance is
    /// literal; the caller records which rows were replaced.
    pub fn replace_with(&self, ctx: Vec<f32>) -> Result<HeadIntervention, VindexError> {
        let sha256 = vector_sha256(&ctx);
        HeadIntervention::replace(
            HeadAddress::new(self.layer, self.head, [self.position])?,
            ctx,
            VectorProvenance::Literal { sha256 },
        )
    }
}

/// Captures [`SourceHead`]s at declared `(layer, head, position)` addresses.
#[derive(Debug, Default)]
pub struct SourceHeadCapture {
    wanted: BTreeSet<(usize, usize, usize)>,
    captured: BTreeMap<(usize, usize, usize), SourceHead>,
}

impl SourceHeadCapture {
    pub fn at(addresses: impl IntoIterator<Item = (usize, usize, usize)>) -> Self {
        Self {
            wanted: addresses.into_iter().collect(),
            captured: BTreeMap::new(),
        }
    }

    pub fn get(&self, layer: usize, head: usize, position: usize) -> Option<&SourceHead> {
        self.captured.get(&(layer, head, position))
    }

    pub fn captured(&self) -> usize {
        self.captured.len()
    }

    /// Whether any declared address sits at this layer and position.
    pub fn wants_at(&self, layer: usize, position: usize) -> bool {
        self.wanted
            .iter()
            .any(|&(l, _, p)| l == layer && p == position)
    }

    /// Feed one head record; keeps it at a declared address. Call from
    /// [`StepObserver::attention_head`].
    pub fn observe(&mut self, layer: usize, record: &AttentionHeadRecord<'_>) {
        let key = (layer, record.head, record.position);
        if self.wanted.contains(&key) {
            self.captured
                .insert(key, SourceHead::from_record(layer, record));
        }
    }
}

impl StepObserver for SourceHeadCapture {
    fn event(&mut self, _event: StepEvent) {}

    fn wants_attention_heads(&self) -> bool {
        !self.wanted.is_empty()
    }

    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        self.wants_at(layer, position)
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        self.observe(layer, &record);
    }
}
