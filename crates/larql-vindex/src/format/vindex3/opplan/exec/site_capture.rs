//! PREC-DIAG-1 E0a: sub-layer capture on the cached decode step, as
//! copies, bounded before it allocates.
//!
//! A [`SiteCapture`] is a [`StepObserver`]. It subscribes to the step's
//! existing observation points — it never re-enacts a sublayer — and
//! copies the values at the requested positions into buffers sized and
//! budgeted when the capture is built. Everything it reads is borrowed
//! from the executor; nothing it does is read back, so the step computes
//! exactly what the unobserved step computes (pinned by
//! `tests/site_capture.rs`).
//!
//! Three rules make a capture a record rather than a sample:
//!
//! - **Bounded first.** [`SiteCapture::new`] computes the bytes the
//!   capture will hold and refuses with [`CaptureFault::ResourceLimit`]
//!   before allocating anything when that exceeds the budget.
//! - **Shape-checked.** Every write is checked against the width the plan
//!   declared for that site and layer; a mismatch is a fault, never a
//!   truncation.
//! - **Complete or refused.** [`SiteCapture::finish`] refuses unless every
//!   requested `(site, layer, position)` was written exactly once. A
//!   site the executor never reached is a fault, not an omission.
//!
//! Scope: single-stream components whose every executed layer is softmax
//! attention, with no layer scale — the Gemma 3 text stacks PREC-DIAG-1
//! measures. Anything else is refused at [`CaptureRequest::for_plan`].

use std::collections::BTreeMap;
use std::fmt;

use super::observe::{
    AttentionHeadRecord, CarrierWriteRecord, InputSite, StepEvent, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::ComponentOpPlan;

/// Bytes per captured value: every site is captured as the f32 the
/// executor held.
pub const BYTES_PER_VALUE: u64 = 4;

/// One capture site, in forward order within a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CaptureSite {
    /// The `[hidden]` vector entering layer 0: the embedding after its
    /// scale (and norm, where the image declares one).
    Embedding,
    /// The normalised residual entering attention (input to q, k, v).
    AttentionInput,
    /// Q after any q-norm and query scale, before position encoding.
    QueryUnpositioned,
    /// Q after position encoding — what the scores read.
    QueryPositioned,
    /// This position's K after any k-norm, before position encoding.
    KeyUnpositioned,
    /// This position's K after position encoding — what the cache holds.
    KeyPositioned,
    /// The attention branch after the output projection, before any
    /// post-attention norm.
    AttentionOutput,
    /// What the attention site added to the residual: after any
    /// post-attention norm and residual-delta scale.
    AttentionDelta,
    /// The normalised residual entering the FFN.
    FfnInput,
    /// The FFN's output before any post-FFN norm.
    FfnOutput,
    /// What the FFN site added to the residual: after any post-FFN norm
    /// and residual-delta scale.
    FfnDelta,
    /// The residual after the layer's FFN write — the layer output on a
    /// component without a layer scale.
    Residual,
    /// The vector the output head reads: the final-normed carrier.
    HeadInput,
}

impl CaptureSite {
    /// Every site, in forward order.
    pub const ALL: [CaptureSite; 13] = [
        CaptureSite::Embedding,
        CaptureSite::AttentionInput,
        CaptureSite::QueryUnpositioned,
        CaptureSite::QueryPositioned,
        CaptureSite::KeyUnpositioned,
        CaptureSite::KeyPositioned,
        CaptureSite::AttentionOutput,
        CaptureSite::AttentionDelta,
        CaptureSite::FfnInput,
        CaptureSite::FfnOutput,
        CaptureSite::FfnDelta,
        CaptureSite::Residual,
        CaptureSite::HeadInput,
    ];

    /// The site's file-safe name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::AttentionInput => "attention_input",
            Self::QueryUnpositioned => "query_unpositioned",
            Self::QueryPositioned => "query_positioned",
            Self::KeyUnpositioned => "key_unpositioned",
            Self::KeyPositioned => "key_positioned",
            Self::AttentionOutput => "attention_output",
            Self::AttentionDelta => "attention_delta",
            Self::FfnInput => "ffn_input",
            Self::FfnOutput => "ffn_output",
            Self::FfnDelta => "ffn_delta",
            Self::Residual => "residual",
            Self::HeadInput => "head_input",
        }
    }

    /// Whether the site is taken once per step (not per layer).
    pub fn once_per_step(self) -> bool {
        matches!(self, Self::Embedding | Self::HeadInput)
    }

    /// Width of the site at one layer.
    fn width(self, hidden: usize, layer: &LayerShape) -> usize {
        match self {
            Self::QueryUnpositioned | Self::QueryPositioned => layer.num_q_heads * layer.head_dim,
            Self::KeyUnpositioned | Self::KeyPositioned => layer.num_kv_heads * layer.head_dim,
            _ => hidden,
        }
    }

    /// Head-sized chunks one position of this site is written in: the
    /// q/k sites arrive one head record at a time, the rest whole.
    fn chunks(self, layer: &LayerShape) -> usize {
        match self {
            Self::QueryUnpositioned | Self::QueryPositioned => layer.num_q_heads,
            Self::KeyUnpositioned | Self::KeyPositioned => layer.num_kv_heads,
            _ => 1,
        }
    }
}

/// One executed layer's attention geometry, read from the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerShape {
    pub layer: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Why a capture was refused or failed. `ResourceLimit` is the one a
/// caller records as `resource_limit`; every other fault is a refusal of
/// the request or a defect in what the executor delivered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureFault {
    /// The capture would hold `required` bytes against a budget of
    /// `budget`. Raised before anything is allocated.
    ResourceLimit { required: u64, budget: u64 },
    /// The request itself is not capturable (empty, out of range,
    /// duplicated, or a plan outside E0a's scope).
    Request(String),
    /// A write arrived with the wrong width.
    Shape {
        site: CaptureSite,
        layer: Option<usize>,
        expected: usize,
        got: usize,
    },
    /// The same `(site, layer, position, chunk)` arrived twice with
    /// different values.
    Conflict {
        site: CaptureSite,
        layer: Option<usize>,
        position: usize,
    },
    /// The executor delivered a value E0a cannot capture faithfully.
    Unsupported(String),
    /// `count` requested slots were never written; `first` names one.
    Incomplete {
        count: usize,
        first: (CaptureSite, Option<usize>, usize),
    },
}

impl CaptureFault {
    pub fn is_resource_limit(&self) -> bool {
        matches!(self, Self::ResourceLimit { .. })
    }
}

impl fmt::Display for CaptureFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceLimit { required, budget } => write!(
                f,
                "resource_limit: the capture needs {required} bytes against a budget of {budget}"
            ),
            Self::Request(why) => write!(f, "capture request refused: {why}"),
            Self::Shape {
                site,
                layer,
                expected,
                got,
            } => write!(
                f,
                "capture shape fault at {} layer {layer:?}: expected width {expected}, got {got}",
                site.name()
            ),
            Self::Conflict {
                site,
                layer,
                position,
            } => write!(
                f,
                "capture conflict at {} layer {layer:?} position {position}: written twice \
                 with different values",
                site.name()
            ),
            Self::Unsupported(why) => write!(f, "capture unsupported: {why}"),
            Self::Incomplete { count, first } => write!(
                f,
                "capture incomplete: {count} requested slots never written (first: {} layer \
                 {:?} position {})",
                first.0.name(),
                first.1,
                first.2
            ),
        }
    }
}

impl std::error::Error for CaptureFault {}

/// What to capture: positions, geometry and the byte budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureRequest {
    /// Strictly increasing, each `< sequence_len`.
    pub positions: Vec<usize>,
    pub sequence_len: usize,
    pub hidden: usize,
    pub layers: Vec<LayerShape>,
    pub budget_bytes: u64,
}

impl CaptureRequest {
    /// A request over `plan`'s executed `layers` (indices into
    /// `plan.layers`), refused when the plan is outside E0a's scope or
    /// the positions are not a strictly increasing set inside the
    /// sequence.
    pub fn for_plan(
        plan: &ComponentOpPlan,
        layers: std::ops::Range<usize>,
        hidden: usize,
        positions: Vec<usize>,
        sequence_len: usize,
        budget_bytes: u64,
    ) -> Result<Self, CaptureFault> {
        if !plan.residual_topology.is_single_stream() || plan.attention_residual_exit.is_some() {
            return Err(CaptureFault::Request(
                "E0a captures single-stream components only".to_string(),
            ));
        }
        if layers.end > plan.layers.len() {
            return Err(CaptureFault::Request(format!(
                "executed layers {layers:?} exceed the plan's {}",
                plan.layers.len()
            )));
        }
        let mut shapes = Vec::with_capacity(layers.len());
        for index in layers {
            let op = plan.layers[index].attention.softmax().ok_or_else(|| {
                CaptureFault::Request(format!(
                    "layer {index} is not softmax attention; E0a captures softmax stacks only"
                ))
            })?;
            shapes.push(LayerShape {
                layer: index,
                num_q_heads: op.num_q_heads,
                num_kv_heads: op.num_kv_heads,
                head_dim: op.head_dim,
            });
        }
        let request = Self {
            positions,
            sequence_len,
            hidden,
            layers: shapes,
            budget_bytes,
        };
        request.validate()?;
        Ok(request)
    }

    fn validate(&self) -> Result<(), CaptureFault> {
        if self.positions.is_empty() {
            return Err(CaptureFault::Request("no positions requested".to_string()));
        }
        if !self.positions.windows(2).all(|w| w[0] < w[1]) {
            return Err(CaptureFault::Request(
                "positions must be strictly increasing (no duplicates)".to_string(),
            ));
        }
        if let Some(&last) = self.positions.last() {
            if last >= self.sequence_len {
                return Err(CaptureFault::Request(format!(
                    "position {last} is outside a sequence of {}",
                    self.sequence_len
                )));
            }
        }
        if self.layers.is_empty() {
            return Err(CaptureFault::Request("no executed layers".to_string()));
        }
        Ok(())
    }

    /// Every slot the capture will hold, in forward order: the
    /// once-per-step sites with `layer = None`, the rest per layer.
    fn slots(&self) -> Vec<(CaptureSite, Option<usize>, usize, usize)> {
        let mut out = Vec::new();
        for site in CaptureSite::ALL {
            if site.once_per_step() {
                out.push((site, None, self.hidden, 1));
            } else {
                for shape in &self.layers {
                    out.push((
                        site,
                        Some(shape.layer),
                        site.width(self.hidden, shape),
                        site.chunks(shape),
                    ));
                }
            }
        }
        out
    }

    /// The bytes the capture holds (and writes): every slot's
    /// `positions × width` f32 values. Saturating, so an absurd request
    /// is refused by the budget rather than wrapping.
    pub fn required_bytes(&self) -> u64 {
        let positions = self.positions.len() as u64;
        self.slots().iter().fold(0u64, |acc, (_, _, width, _)| {
            acc.saturating_add(
                positions
                    .saturating_mul(*width as u64)
                    .saturating_mul(BYTES_PER_VALUE),
            )
        })
    }
}

/// One site at one layer (or once per step), every requested position.
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedPlane {
    pub site: CaptureSite,
    /// `None` for a once-per-step site.
    pub layer: Option<usize>,
    pub width: usize,
    /// `[positions, width]` row-major.
    pub values: Vec<f32>,
}

/// A finished, complete capture.
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedSites {
    pub request: CaptureRequest,
    pub planes: Vec<CapturedPlane>,
}

struct Slot {
    plane: CapturedPlane,
    chunks: usize,
    /// `[positions, chunks]`: whether each head chunk was written.
    written: Vec<bool>,
}

/// The capture observer. Build with [`SiteCapture::new`], pass to
/// `DecodeSession::step_observed` for every token, then [`finish`].
///
/// [`finish`]: SiteCapture::finish
pub struct SiteCapture {
    request: CaptureRequest,
    slots: Vec<Slot>,
    index: BTreeMap<(CaptureSite, Option<usize>), usize>,
    /// Position of the step in flight, from `Embedded`.
    position: Option<usize>,
    /// The first fault seen; observation cannot return errors, so the
    /// first one is held and surfaced by `finish`.
    fault: Option<CaptureFault>,
}

impl SiteCapture {
    /// Build the capture, refusing with [`CaptureFault::ResourceLimit`]
    /// before allocating when it would exceed the budget.
    pub fn new(request: CaptureRequest) -> Result<Self, CaptureFault> {
        request.validate()?;
        let required = request.required_bytes();
        if required > request.budget_bytes {
            return Err(CaptureFault::ResourceLimit {
                required,
                budget: request.budget_bytes,
            });
        }
        let positions = request.positions.len();
        let mut slots = Vec::new();
        let mut index = BTreeMap::new();
        for (site, layer, width, chunks) in request.slots() {
            index.insert((site, layer), slots.len());
            slots.push(Slot {
                plane: CapturedPlane {
                    site,
                    layer,
                    width,
                    values: vec![0.0; positions * width],
                },
                chunks,
                written: vec![false; positions * chunks],
            });
        }
        Ok(Self {
            request,
            slots,
            index,
            position: None,
            fault: None,
        })
    }

    fn fail(&mut self, fault: CaptureFault) {
        if self.fault.is_none() {
            self.fault = Some(fault);
        }
    }

    fn row_of(&self, position: usize) -> Option<usize> {
        self.request.positions.binary_search(&position).ok()
    }

    /// Copy `values` into chunk `chunk` of `(site, layer)` at `position`,
    /// checking width and single assignment.
    fn put(
        &mut self,
        site: CaptureSite,
        layer: Option<usize>,
        position: usize,
        chunk: usize,
        values: &[f32],
    ) {
        let Some(row) = self.row_of(position) else {
            return;
        };
        let Some(&slot_index) = self.index.get(&(site, layer)) else {
            // A layer outside the request (a slice executing more than
            // was asked for) is not captured; that is not a fault.
            return;
        };
        let slot = &mut self.slots[slot_index];
        let width = slot.plane.width;
        let chunk_width = width / slot.chunks;
        if values.len() != chunk_width || chunk >= slot.chunks {
            let fault = CaptureFault::Shape {
                site,
                layer,
                expected: chunk_width,
                got: values.len(),
            };
            self.fail(fault);
            return;
        }
        let flag = row * slot.chunks + chunk;
        let start = row * width + chunk * chunk_width;
        let target = &mut slot.plane.values[start..start + chunk_width];
        if slot.written[flag] {
            // GQA hands one KV head to several query heads: the same
            // values again are expected; different values are a fault.
            if target
                .iter()
                .zip(values)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                self.fail(CaptureFault::Conflict {
                    site,
                    layer,
                    position,
                });
            }
            return;
        }
        target.copy_from_slice(values);
        slot.written[flag] = true;
    }

    fn current(&mut self) -> Option<usize> {
        if self.position.is_none() {
            self.fail(CaptureFault::Unsupported(
                "an operand arrived before the step's position was known".to_string(),
            ));
        }
        self.position
    }

    /// The capture, if every requested slot was written exactly as
    /// declared; otherwise the first fault.
    pub fn finish(self) -> Result<CapturedSites, CaptureFault> {
        if let Some(fault) = self.fault {
            return Err(fault);
        }
        let mut missing = 0usize;
        let mut first = None;
        for slot in &self.slots {
            for (flag, written) in slot.written.iter().enumerate() {
                if !written {
                    missing += 1;
                    first.get_or_insert((
                        slot.plane.site,
                        slot.plane.layer,
                        self.request.positions[flag / slot.chunks],
                    ));
                }
            }
        }
        if let Some(first) = first {
            return Err(CaptureFault::Incomplete {
                count: missing,
                first,
            });
        }
        Ok(CapturedSites {
            request: self.request,
            planes: self.slots.into_iter().map(|slot| slot.plane).collect(),
        })
    }
}

impl StepObserver for SiteCapture {
    fn event(&mut self, event: StepEvent) {
        if let StepEvent::Embedded { position } = event {
            self.position = Some(position);
        }
    }

    fn entering_carrier(&mut self, position: usize, values: &[f32]) {
        self.position = Some(position);
        self.put(CaptureSite::Embedding, None, position, 0, values);
    }

    fn operand_input(&mut self, layer: usize, site: InputSite, values: &[f32]) {
        let Some(position) = self.current() else {
            return;
        };
        let site = match site {
            InputSite::Attention => CaptureSite::AttentionInput,
            InputSite::Ffn => CaptureSite::FfnInput,
            InputSite::FfnOutput => CaptureSite::FfnOutput,
        };
        self.put(site, Some(layer), position, 0, values);
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.layer_scale.is_some() {
            self.fail(CaptureFault::Unsupported(format!(
                "layer {} carries a layer scale; the residual site would not be the layer output",
                record.layer
            )));
            return;
        }
        match record.site {
            SublayerSite::Attention => self.put(
                CaptureSite::AttentionDelta,
                Some(record.layer),
                record.position,
                0,
                record.delta,
            ),
            SublayerSite::Ffn => {
                self.put(
                    CaptureSite::FfnDelta,
                    Some(record.layer),
                    record.position,
                    0,
                    record.delta,
                );
                self.put(
                    CaptureSite::Residual,
                    Some(record.layer),
                    record.position,
                    0,
                    record.after,
                );
            }
        }
    }

    /// Asked so the per-head path is admitted; the per-position filter
    /// below is what routes only the requested positions through it.
    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        self.index
            .contains_key(&(CaptureSite::QueryPositioned, Some(layer)))
            && self.row_of(position).is_some()
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        let position = record.position;
        let layer = Some(layer);
        self.put(
            CaptureSite::QueryPositioned,
            layer,
            position,
            record.head,
            record.query,
        );
        // The fresh key is the last attended source: `source_keys` runs
        // `source_start..=position`.
        match position
            .checked_sub(record.source_start)
            .and_then(|offset| record.source_keys.get(offset))
        {
            Some(key) => self.put(
                CaptureSite::KeyPositioned,
                layer,
                position,
                record.kv_head,
                key,
            ),
            None => self.fail(CaptureFault::Unsupported(format!(
                "head {} at position {position} carried no key for its own position",
                record.head
            ))),
        }
        match (record.query_unpositioned, record.key_unpositioned) {
            (Some(q), Some(k)) => {
                self.put(
                    CaptureSite::QueryUnpositioned,
                    layer,
                    position,
                    record.head,
                    q,
                );
                self.put(
                    CaptureSite::KeyUnpositioned,
                    layer,
                    position,
                    record.kv_head,
                    k,
                );
            }
            _ => self.fail(CaptureFault::Unsupported(
                "the backend's head record carried no pre-rotation q/k".to_string(),
            )),
        }
    }

    fn attention_output(&mut self, layer: usize, position: usize, values: &[f32]) {
        self.put(
            CaptureSite::AttentionOutput,
            Some(layer),
            position,
            0,
            values,
        );
    }

    fn head_input(&mut self, position: usize, values: &[f32]) {
        self.put(CaptureSite::HeadInput, None, position, 0, values);
    }
}
