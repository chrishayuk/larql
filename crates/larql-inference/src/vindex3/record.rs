//! V3-STREAM-1: the run record (`docs/v3-stream-1-run-record.md`).
//!
//! One runner-side consumer turns a tokenwise VINDEX3 run into a
//! LOSSLESS, sequenced, provenance-bearing record that replays to the
//! same events; a separate LIVE TAP may lose events into a bounded
//! channel without ever stalling the executor or touching the record.
//!
//! The executor allocates no identity: `run_id`, the model name, the
//! component and the prompt come from the caller, the provenance from
//! the session's own prepared image, and every event's sequence number
//! and run-relative timestamp from this recorder — assigned BEFORE the
//! live fan-out, so a dropped live event still has an identity.
//!
//! The record's event vocabulary is the runner's, mirrored from the
//! executor's rather than borrowed: the executor's enum is
//! non-exhaustive and carries no serialisation, and a wire schema must
//! not move whenever the executor learns a new event. A variant this
//! module does not know is recorded as [`EventKind::Unknown`] with the
//! executor's own debug spelling — recorded, never skipped.

use std::io::{BufRead, BufReader, Write as _};
use std::path::Path;
use std::sync::mpsc::{SyncSender, TrySendError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use larql_vindex::format::vindex3::opplan::exec::intervene::{InterventionKind, InterventionPlan};
use larql_vindex::format::vindex3::opplan::exec::intervene_heads::{
    HeadInterventionKind, HeadInterventionPlan,
};
use larql_vindex::format::vindex3::opplan::exec::observe::{
    CarrierForm, CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use larql_vindex::format::vindex3::opplan::exec::observe_heads::{HeadReader, HEAD_SUM_METHOD};
use larql_vindex::format::vindex3::opplan::exec::observe_lens::{LensReader, LENS_METHOD};
use larql_vindex::format::vindex3::opplan::exec::observe_stats::StatsObserver;
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::opplan::exec::provenance::{ExecutionProvenance, RunProvenance};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The schema every record written by this module names.
pub const RECORD_SCHEMA: &str = "larql.run-record.v1";

/// Who ran what: caller-supplied, never executor-allocated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunIdentity {
    pub run_id: String,
    /// The container's self-declared model name.
    pub model: String,
    pub component: String,
    /// The prompt as token ids, in order.
    pub tokens: Vec<u32>,
    pub schema: String,
    /// Wall-clock start, metadata only; never an ordering key.
    pub started_unix_ms: u64,
    /// V3-INTERVENE-1: the hash of the intervention declaration this run
    /// executed under; `None` for an unintervened run. Part of the
    /// identity, so a record from an intervened run is never read as a
    /// baseline of the same prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention_sha256: Option<String>,
    /// V3-INTERVENE-2: the hash of the head-intervention declaration this
    /// run executed under; `None` for a run with none. Independent of
    /// [`Self::intervention_sha256`] — a declaration may carry both
    /// carrier and head entries, and each gets its own hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_intervention_sha256: Option<String>,
}

impl RunIdentity {
    pub fn new(run_id: &str, model: &str, component: &str, tokens: &[u32]) -> Self {
        Self {
            run_id: run_id.to_string(),
            model: model.to_string(),
            component: component.to_string(),
            tokens: tokens.to_vec(),
            schema: RECORD_SCHEMA.to_string(),
            started_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
                .unwrap_or(0),
            intervention_sha256: None,
            head_intervention_sha256: None,
        }
    }
}

/// The runner's spelling of a sublayer site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Site {
    Attention,
    Ffn,
}

impl From<SublayerSite> for Site {
    fn from(site: SublayerSite) -> Self {
        match site {
            SublayerSite::Attention => Self::Attention,
            SublayerSite::Ffn => Self::Ffn,
        }
    }
}

/// The runner's spelling of a carrier form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Carrier {
    Single,
    Bundle,
    History,
}

impl From<CarrierForm> for Carrier {
    fn from(form: CarrierForm) -> Self {
        match form {
            CarrierForm::Single => Self::Single,
            CarrierForm::Bundle => Self::Bundle,
            CarrierForm::History => Self::History,
        }
    }
}

/// The runner's spelling of an intervention kind (V3-INTERVENE-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Zero,
    Add,
    Replace,
}

impl From<InterventionKind> for Kind {
    fn from(kind: InterventionKind) -> Self {
        match kind {
            InterventionKind::Zero => Self::Zero,
            InterventionKind::Add => Self::Add,
            InterventionKind::Replace => Self::Replace,
        }
    }
}

/// The runner's spelling of a head intervention kind (V3-INTERVENE-2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadKind {
    Zero,
    Scale,
    Replace,
}

impl From<HeadInterventionKind> for HeadKind {
    fn from(kind: HeadInterventionKind) -> Self {
        match kind {
            HeadInterventionKind::Zero => Self::Zero,
            HeadInterventionKind::Scale => Self::Scale,
            HeadInterventionKind::Replace => Self::Replace,
        }
    }
}

/// What one recorded event says.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    /// The token was embedded; the position is the envelope's.
    Embedded,
    /// The carrier entering layer 0: its width, not its values.
    EnteringCarrier {
        hidden: usize,
    },
    /// The stats the observer computed AT the write, while the vectors
    /// were still borrowed. Precedes the structural write event, as the
    /// executor fires them.
    CarrierStats {
        layer: usize,
        site: Site,
        norm: f64,
        delta_norm: f64,
        layer_scale: Option<f32>,
        projection: Vec<f32>,
        probe: Vec<f32>,
    },
    CarrierWrite {
        layer: usize,
        site: Site,
        carrier: Carrier,
    },
    AttentionDone {
        layer: usize,
    },
    FfnDone {
        layer: usize,
    },
    Logits {
        vocab: usize,
    },
    /// What the model's own head says at an armed site (V3-LENS-1):
    /// the image's final norm and output head applied to the layer
    /// output, then a full log-softmax. One head pass per readout.
    Readout {
        layer: usize,
        site: Site,
        method: String,
        tokens: Vec<TokenStanding>,
        top: Vec<TopStanding>,
    },
    /// V3-HEAD-OBS-1: this layer's attention emitted one head record per
    /// query head on this step, before its write (structural).
    HeadsObserved {
        layer: usize,
        heads: usize,
    },
    /// V3-HEAD-OBS-1: this layer's attention has no softmax heads, so the
    /// write above it has no head decomposition; the receipt names the
    /// layer as uncovered.
    HeadsUncovered {
        layer: usize,
    },
    /// V3-HEAD-OBS-1, stats level: one query head's child of the
    /// attention write — `‖c′_h‖`, its projection on the run's basis, the
    /// KV head, the top source positions with their weights, and the
    /// sink mass. Keyed beneath the site's own write, which stays
    /// authoritative.
    HeadWrite {
        layer: usize,
        head: usize,
        kv_head: usize,
        norm: f64,
        projection: Vec<f32>,
        sources: Vec<SourceStanding>,
        sink: f32,
    },
    /// V3-HEAD-OBS-1: the head-sum law's measured residual at one
    /// attention write, `‖Σ_h c′_h + bias′ − delta‖ / ‖delta‖`, by the
    /// named method.
    HeadSum {
        layer: usize,
        method: String,
        heads: usize,
        residual: f64,
    },
    /// V3-INTERVENE-1: an intervention fired on this site's write at the
    /// envelope's position. Precedes the write's stats and structural
    /// event, as the executor fires it.
    Intervened {
        layer: usize,
        site: Site,
        intervention: Kind,
    },
    /// V3-INTERVENE-2: a head intervention fired on this head's `ctx_h`,
    /// inside the attention kernel, at the envelope's position. Precedes
    /// the layer's `HeadsObserved`/`HeadWrite`/`HeadSum` and its
    /// attention `CarrierWrite`.
    HeadIntervened {
        layer: usize,
        head: usize,
        intervention: HeadKind,
    },
    /// V3-INTERVENE-2, J5: the shortcut gap for one `zero` head firing —
    /// `‖delta_real − (delta_base − c′_h)‖ / ‖delta_base‖`, where
    /// `delta_base` and `c′_h` are reconstructed from this SAME run's
    /// pre-intervention head records (the head-sum law's own terms, not
    /// a second run). Requires the head reader armed and retaining
    /// children; recorded once per `zero` firing, after that write's
    /// `HeadSum`.
    HeadInterventionGap {
        layer: usize,
        head: usize,
        gap: f64,
    },
    /// An executor event this schema has no spelling for yet. Recorded
    /// with the executor's debug form so nothing is lost.
    Unknown {
        debug: String,
    },
}

/// One source position's attention weight in a [`EventKind::HeadWrite`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceStanding {
    pub position: usize,
    pub weight: f32,
}

/// One declared token's standing in a readout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenStanding {
    pub id: u32,
    pub logprob: f64,
    pub rank: usize,
}

/// One of the top ids in a readout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopStanding {
    pub id: u32,
    pub logprob: f64,
}

/// One event with its run-scoped identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedEvent {
    /// Strictly increasing by one from zero within a run.
    pub sequence: u64,
    /// Nanoseconds since the recorder was armed, from a monotonic clock.
    pub timestamp_ns: u64,
    /// The absolute position the event belongs to (the last `Embedded`).
    pub position: usize,
    #[serde(flatten)]
    pub event: EventKind,
}

/// What the live tap could not deliver — kept OUTSIDE the channel that
/// overflowed, so a full channel cannot lose the count of its own loss.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropLedger {
    pub dropped: u64,
    pub first_dropped_sequence: Option<u64>,
    pub last_dropped_sequence: Option<u64>,
}

impl DropLedger {
    fn note(&mut self, sequence: u64) {
        self.dropped += 1;
        self.first_dropped_sequence.get_or_insert(sequence);
        self.last_dropped_sequence = Some(sequence);
    }
}

/// The lossy consumer: a bounded, non-blocking channel and its ledger.
pub struct LiveTap {
    sender: SyncSender<RecordedEvent>,
    ledger: DropLedger,
}

impl LiveTap {
    pub fn new(sender: SyncSender<RecordedEvent>) -> Self {
        Self {
            sender,
            ledger: DropLedger::default(),
        }
    }

    /// Never blocks: a full or disconnected channel drops the event and
    /// the ledger records which one.
    fn offer(&mut self, event: &RecordedEvent) {
        match self.sender.try_send(event.clone()) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.ledger.note(event.sequence);
            }
        }
    }

    pub fn ledger(&self) -> &DropLedger {
        &self.ledger
    }
}

/// The receipt: describes the event log without being part of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Receipt {
    pub events: u64,
    pub last_sequence: Option<u64>,
    /// SHA-256 over the exact bytes of the event lines as written, each
    /// followed by a newline; the receipt line is not covered.
    pub log_sha256: String,
    pub provenance_fingerprint: String,
    /// Whether the run reached its end; a record without this is a valid
    /// prefix, never a complete run.
    pub complete: bool,
    /// The live tap's loss, if a tap was attached; zero otherwise.
    pub live_dropped: u64,
    /// Head passes the lens performed — its price — zero without a lens.
    #[serde(default)]
    pub head_passes: u64,
    /// The lens's first failure, if it had one; the readouts stop there.
    #[serde(default)]
    pub lens_failure: Option<String>,
    /// V3-HEAD-OBS-1: how many head records the executor handed the head
    /// reader; zero when none was armed.
    #[serde(default)]
    pub head_records: u64,
    /// V3-HEAD-OBS-1: the executed layers whose attention has no softmax
    /// heads, named once each, ascending; empty when heads were not
    /// armed or every layer was covered.
    #[serde(default)]
    pub head_layers_uncovered: Vec<usize>,
    /// V3-HEAD-OBS-1: why the head reader stopped, if it did; the record
    /// stays complete without its rows from that point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_failure: Option<String>,
    /// V3-INTERVENE-1: how many interventions the run declared.
    #[serde(default)]
    pub interventions_declared: u64,
    /// V3-INTERVENE-1: how many firings the run recorded.
    #[serde(default)]
    pub interventions_applied: u64,
    /// V3-INTERVENE-1: a declared address the run never reached, named
    /// at the end of the run. A declared intervention that never fired
    /// is a refusal on the receipt, never a silent no-op.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention_refusal: Option<String>,
    /// V3-INTERVENE-2: how many head interventions the run declared.
    #[serde(default)]
    pub head_interventions_declared: u64,
    /// V3-INTERVENE-2: how many head firings the run recorded.
    #[serde(default)]
    pub head_interventions_applied: u64,
    /// V3-INTERVENE-2: a declared head address the run never reached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_intervention_refusal: Option<String>,
}

/// A finished record: header, events, receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub identity: RunIdentity,
    /// The `RunProvenance` as the executor serialised it.
    pub provenance: serde_json::Value,
    pub provenance_fingerprint: String,
    pub events: Vec<RecordedEvent>,
    pub receipt: Receipt,
}

/// The lossless consumer.
pub struct RunRecorder<'a> {
    identity: RunIdentity,
    provenance: RunProvenance,
    stats: Option<StatsObserver>,
    lens: Option<Box<dyn LensReader + 'a>>,
    /// V3-HEAD-OBS-1: the head reader, when armed; the recorder asks the
    /// executor for heads exactly when one is attached.
    heads: Option<Box<dyn HeadReader + 'a>>,
    /// Layers the executor named as uncovered, ascending, once each.
    uncovered: Vec<usize>,
    armed: Instant,
    position: usize,
    events: Vec<RecordedEvent>,
    live: Option<LiveTap>,
    complete: bool,
    /// V3-INTERVENE-1: the declaration this run executes under, if any.
    interventions: Option<InterventionPlan>,
    /// Firings recorded so far.
    applied: u64,
    /// One past the highest position an event carried, so the receipt
    /// can name declared addresses the run never reached.
    positions_executed: usize,
    /// V3-INTERVENE-2: the head declaration this run executes under.
    head_interventions: Option<HeadInterventionPlan>,
    /// Head firings recorded so far.
    head_applied: u64,
    /// `zero` head firings not yet matched to their write's decomposition
    /// (J5): `(layer, head, position)`, cleared as `carrier_write`
    /// consumes them.
    pending_head_zero: Vec<(usize, usize, usize)>,
}

impl<'a> RunRecorder<'a> {
    /// Arm a recorder with an explicit provenance.
    pub fn arm(
        identity: RunIdentity,
        provenance: RunProvenance,
        stats: Option<StatsObserver>,
    ) -> Self {
        Self {
            identity,
            provenance,
            stats,
            lens: None,
            heads: None,
            uncovered: Vec::new(),
            armed: Instant::now(),
            position: 0,
            events: Vec::new(),
            live: None,
            complete: false,
            interventions: None,
            applied: 0,
            positions_executed: 0,
            head_interventions: None,
            head_applied: 0,
            pending_head_zero: Vec::new(),
        }
    }

    /// Declare the interventions this run executes under (V3-INTERVENE-1).
    /// The declaration's hash joins the run identity; the receipt counts
    /// firings against declarations and names any address never reached.
    pub fn with_interventions(mut self, interventions: &InterventionPlan) -> Self {
        self.identity.intervention_sha256 = interventions.declaration_sha256();
        self.interventions = Some(interventions.clone());
        self
    }

    /// Declare the head interventions this run executes under
    /// (V3-INTERVENE-2). The declaration's hash joins the run identity;
    /// the receipt counts firings against declarations and names any
    /// address never reached. A `zero` firing's shortcut gap (J5) is
    /// computed only when the head reader is ALSO armed and retaining
    /// children ([`HeadReader`]/[`super::HeadStats::retaining_children`]).
    pub fn with_head_interventions(mut self, head_interventions: &HeadInterventionPlan) -> Self {
        self.identity.head_intervention_sha256 = head_interventions.declaration_sha256();
        self.head_interventions = Some(head_interventions.clone());
        self
    }

    /// Arm a recorder over a prepared image: the provenance is read off
    /// the image and the stats observer, exactly as the executor states
    /// them.
    pub fn for_image(
        identity: RunIdentity,
        image: &PreparedOperands,
        stats: Option<StatsObserver>,
    ) -> Self {
        let provenance = RunProvenance::new(ExecutionProvenance::of(image), stats.as_ref());
        Self::arm(identity, provenance, stats)
    }

    /// Arm a logit lens (V3-LENS-1): its readouts become events in the
    /// same sequence, and its head passes go on the receipt.
    pub fn with_lens(mut self, lens: Box<dyn LensReader + 'a>) -> Self {
        self.lens = Some(lens);
        self
    }

    /// Arm a head reader (V3-HEAD-OBS-1): the recorder then asks the
    /// executor for per-head records, hands them to the reader, and at
    /// every attention write records the reader's stats-level rows and
    /// the head-sum residual as events in the same sequence; the count,
    /// the uncovered layers and any failure go on the receipt.
    pub fn with_heads(mut self, heads: Box<dyn HeadReader + 'a>) -> Self {
        self.heads = Some(heads);
        self
    }

    /// Attach the live tap. Events already recorded are not replayed
    /// into it.
    pub fn with_live_tap(mut self, sender: SyncSender<RecordedEvent>) -> Self {
        self.live = Some(LiveTap::new(sender));
        self
    }

    pub fn events(&self) -> &[RecordedEvent] {
        &self.events
    }

    pub fn drop_ledger(&self) -> Option<&DropLedger> {
        self.live.as_ref().map(LiveTap::ledger)
    }

    pub fn provenance(&self) -> &RunProvenance {
        &self.provenance
    }

    /// Mark the run as having reached its end.
    pub fn complete(&mut self) {
        self.complete = true;
    }

    fn push(&mut self, event: EventKind) {
        let recorded = RecordedEvent {
            sequence: u64::try_from(self.events.len()).expect("sequence fits"),
            timestamp_ns: u64::try_from(self.armed.elapsed().as_nanos()).unwrap_or(u64::MAX),
            position: self.position,
            event,
        };
        if let Some(tap) = &mut self.live {
            tap.offer(&recorded);
        }
        self.events.push(recorded);
    }

    /// Seal the record: the receipt hashes the event lines exactly as
    /// [`RunRecord::write_jsonl`] writes them.
    pub fn finish(self) -> RunRecord {
        let lines = event_lines(&self.events);
        let live_dropped = self.live.as_ref().map(|t| t.ledger.dropped).unwrap_or(0);
        let head_passes = self
            .lens
            .as_ref()
            .map(|l| u64::try_from(l.head_passes()).expect("count fits"))
            .unwrap_or(0);
        let lens_failure = self
            .lens
            .as_ref()
            .and_then(|l| l.failure().map(ToString::to_string));
        let head_records = self
            .heads
            .as_ref()
            .map(|h| u64::try_from(h.records()).expect("count fits"))
            .unwrap_or(0);
        let head_failure = self
            .heads
            .as_ref()
            .and_then(|h| h.failure().map(ToString::to_string));
        let head_layers_uncovered = self.uncovered.clone();
        let fingerprint = self.provenance.fingerprint();
        let interventions_declared = self
            .interventions
            .as_ref()
            .map(|p| u64::try_from(p.declared()).expect("count fits"))
            .unwrap_or(0);
        // A declared address the run never reached is named on the
        // receipt, but only once the run is complete: an incomplete
        // record is a valid prefix, and a prefix has not failed to reach
        // anything yet.
        let intervention_refusal = match (&self.interventions, self.complete) {
            (Some(plan), true) => {
                let unreached = plan.unreached(self.positions_executed);
                (!unreached.is_empty()).then(|| {
                    let named: Vec<String> = unreached
                        .iter()
                        .map(|u| format!("layer {} {:?} position {}", u.layer, u.site, u.position))
                        .collect();
                    format!(
                        "declared intervention never fired: the run executed {} position(s) and \
                         did not reach {}",
                        self.positions_executed,
                        named.join(", ")
                    )
                })
            }
            _ => None,
        };
        let head_interventions_declared = self
            .head_interventions
            .as_ref()
            .map(|p| u64::try_from(p.declared()).expect("count fits"))
            .unwrap_or(0);
        let head_intervention_refusal = match (&self.head_interventions, self.complete) {
            (Some(plan), true) => {
                let unreached = plan.unreached(self.positions_executed);
                (!unreached.is_empty()).then(|| {
                    let named: Vec<String> = unreached
                        .iter()
                        .map(|u| {
                            format!("layer {} head {} position {}", u.layer, u.head, u.position)
                        })
                        .collect();
                    format!(
                        "declared head intervention never fired: the run executed {} \
                         position(s) and did not reach {}",
                        self.positions_executed,
                        named.join(", ")
                    )
                })
            }
            _ => None,
        };
        RunRecord {
            identity: self.identity,
            provenance: serde_json::to_value(&self.provenance).expect("provenance serialises"),
            provenance_fingerprint: fingerprint.clone(),
            receipt: Receipt {
                events: u64::try_from(self.events.len()).expect("count fits"),
                last_sequence: self.events.last().map(|e| e.sequence),
                log_sha256: hash_lines(&lines),
                provenance_fingerprint: fingerprint,
                complete: self.complete,
                live_dropped,
                head_passes,
                lens_failure,
                head_records,
                head_layers_uncovered,
                head_failure,
                interventions_declared,
                interventions_applied: self.applied,
                intervention_refusal,
                head_interventions_declared,
                head_interventions_applied: self.head_applied,
                head_intervention_refusal,
            },
            events: self.events,
        }
    }
}

impl StepObserver for RunRecorder<'_> {
    fn event(&mut self, event: StepEvent) {
        let kind = match event {
            StepEvent::Embedded { position } => {
                self.position = position;
                self.positions_executed = self.positions_executed.max(position + 1);
                EventKind::Embedded
            }
            StepEvent::AttentionDone { layer } => EventKind::AttentionDone { layer },
            StepEvent::FfnDone { layer } => EventKind::FfnDone { layer },
            StepEvent::Logits { vocab } => EventKind::Logits { vocab },
            StepEvent::HeadsObserved { layer, heads } => EventKind::HeadsObserved { layer, heads },
            StepEvent::HeadsUncovered { layer } => {
                if let Err(at) = self.uncovered.binary_search(&layer) {
                    self.uncovered.insert(at, layer);
                }
                EventKind::HeadsUncovered { layer }
            }
            StepEvent::Intervened { layer, site, kind } => {
                self.applied += 1;
                EventKind::Intervened {
                    layer,
                    site: site.into(),
                    intervention: kind.into(),
                }
            }
            StepEvent::HeadIntervened { layer, head, kind } => {
                self.head_applied += 1;
                if kind == HeadInterventionKind::Zero {
                    self.pending_head_zero.push((layer, head, self.position));
                }
                EventKind::HeadIntervened {
                    layer,
                    head,
                    intervention: kind.into(),
                }
            }
            StepEvent::CarrierWrite {
                layer,
                site,
                carrier,
            } => EventKind::CarrierWrite {
                layer,
                site: site.into(),
                carrier: carrier.into(),
            },
            other => EventKind::Unknown {
                debug: format!("{other:?}"),
            },
        };
        self.push(kind);
    }

    fn entering_carrier(&mut self, position: usize, values: &[f32]) {
        self.position = position;
        self.push(EventKind::EnteringCarrier {
            hidden: values.len(),
        });
    }

    fn wants_attention_heads(&self) -> bool {
        self.heads.is_some()
    }

    fn attention_head(
        &mut self,
        layer: usize,
        record: larql_vindex::format::vindex3::opplan::exec::observe::AttentionHeadRecord<'_>,
    ) {
        if let Some(heads) = &mut self.heads {
            heads.attention_head(layer, record);
        }
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        // V3-HEAD-OBS-1: the write's head decomposition, keyed beneath the
        // write and recorded before its stats and structural event.
        let head_write = match &mut self.heads {
            Some(heads) => heads.finish_write(&record),
            None => None,
        };
        if let Some(write) = head_write {
            // V3-INTERVENE-2, J5: a `zero` firing at THIS write's
            // (layer, position) gets its shortcut gap now — the one
            // moment `write.sum` (delta_base) and `write.children`
            // (c′_h, from the SAME run's pre-intervention records) are
            // both in hand, beside the write's own `delta` (delta_real).
            let (matched, remaining): (Vec<_>, Vec<_>) = self
                .pending_head_zero
                .drain(..)
                .partition(|&(layer, _, position)| {
                    layer == write.layer && position == record.position
                });
            self.pending_head_zero = remaining;
            for (layer, head, _position) in matched {
                if let Some(children) = &write.children {
                    if let Some(i) = write.rows.iter().position(|row| row.head == head) {
                        let base_norm: f64 = write
                            .sum
                            .iter()
                            .map(|v| f64::from(*v).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let shortcut: Vec<f64> = write
                            .sum
                            .iter()
                            .zip(&children[i])
                            .map(|(s, c)| f64::from(*s) - f64::from(*c))
                            .collect();
                        let err: f64 = shortcut
                            .iter()
                            .zip(record.delta)
                            .map(|(s, d)| (s - f64::from(*d)).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let gap = if base_norm > 0.0 {
                            err / base_norm
                        } else {
                            err
                        };
                        self.push(EventKind::HeadInterventionGap { layer, head, gap });
                    }
                }
            }
            self.push(EventKind::HeadSum {
                layer: write.layer,
                method: HEAD_SUM_METHOD.to_string(),
                heads: write.rows.len(),
                residual: write.residual,
            });
            for row in write.rows {
                self.push(EventKind::HeadWrite {
                    layer: write.layer,
                    head: row.head,
                    kv_head: row.kv_head,
                    norm: row.norm,
                    projection: row.projection,
                    sources: row
                        .sources
                        .into_iter()
                        .map(|(position, weight)| SourceStanding { position, weight })
                        .collect(),
                    sink: row.sink,
                });
            }
        }
        if let Some(stats) = &mut self.stats {
            stats.carrier_write(record);
            let row = stats
                .rows
                .pop()
                .expect("the stats observer appends one row per write");
            self.push(EventKind::CarrierStats {
                layer: row.layer,
                site: row.site.into(),
                norm: row.norm,
                delta_norm: row.delta_norm,
                layer_scale: row.layer_scale,
                projection: row.projection,
                probe: row.probe,
            });
        }
        let readout = match &mut self.lens {
            Some(lens) => lens.read(record),
            None => None,
        };
        if let Some(readout) = readout {
            self.push(EventKind::Readout {
                layer: readout.layer,
                site: readout.site.into(),
                method: LENS_METHOD.to_string(),
                tokens: readout
                    .tokens
                    .into_iter()
                    .map(|t| TokenStanding {
                        id: t.id,
                        logprob: t.logprob,
                        rank: t.rank,
                    })
                    .collect(),
                top: readout
                    .top
                    .into_iter()
                    .map(|(id, logprob)| TopStanding { id, logprob })
                    .collect(),
            });
        }
    }
}

/// Why a record could not be read back.
#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("record line {line}: {source}")]
    Parse {
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("record has no header line")]
    MissingHeader,
    #[error("record has no receipt line; it is a prefix, not a run")]
    MissingReceipt,
    #[error("the receipt's log hash {expected} does not match the event lines ({actual})")]
    HashMismatch { expected: String, actual: String },
    #[error("the receipt counts {expected} events but {actual} lines were read")]
    CountMismatch { expected: u64, actual: u64 },
}

#[derive(Serialize, Deserialize)]
struct Header {
    identity: RunIdentity,
    provenance: serde_json::Value,
    provenance_fingerprint: String,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Line {
    Header { header: Header },
    Receipt { receipt: Receipt },
    Event(RecordedEvent),
}

/// The event lines exactly as written and hashed.
pub fn event_lines(events: &[RecordedEvent]) -> Vec<String> {
    events
        .iter()
        .map(|e| serde_json::to_string(e).expect("an event serialises"))
        .collect()
}

/// SHA-256 over each line followed by a newline, as lowercase hex.
pub fn hash_lines(lines: &[String]) -> String {
    let mut hasher = Sha256::new();
    for line in lines {
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

impl RunRecord {
    /// One JSON object per line: the header, every event, the receipt.
    pub fn write_jsonl(&self, path: &Path) -> Result<(), RecordError> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        let header = Line::Header {
            header: Header {
                identity: self.identity.clone(),
                provenance: self.provenance.clone(),
                provenance_fingerprint: self.provenance_fingerprint.clone(),
            },
        };
        writeln!(
            out,
            "{}",
            serde_json::to_string(&header).expect("header serialises")
        )?;
        for line in event_lines(&self.events) {
            writeln!(out, "{line}")?;
        }
        let receipt = Line::Receipt {
            receipt: self.receipt.clone(),
        };
        writeln!(
            out,
            "{}",
            serde_json::to_string(&receipt).expect("receipt serialises")
        )?;
        out.flush()?;
        Ok(())
    }

    /// Read a record back and VERIFY it: the receipt's hash must match
    /// the event lines and its count must match the lines read. A
    /// record that fails either is refused, not returned as a prefix.
    pub fn read_jsonl(path: &Path) -> Result<Self, RecordError> {
        let reader = BufReader::new(std::fs::File::open(path)?);
        let mut header: Option<Header> = None;
        let mut receipt: Option<Receipt> = None;
        let mut events = Vec::new();
        let mut lines = Vec::new();
        for (index, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let parsed: Line =
                serde_json::from_str(&line).map_err(|source| RecordError::Parse {
                    line: index + 1,
                    source,
                })?;
            match parsed {
                Line::Header { header: h } => header = Some(h),
                Line::Receipt { receipt: r } => receipt = Some(r),
                Line::Event(event) => {
                    lines.push(line);
                    events.push(event);
                }
            }
        }
        let header = header.ok_or(RecordError::MissingHeader)?;
        let receipt = receipt.ok_or(RecordError::MissingReceipt)?;
        let actual = hash_lines(&lines);
        if actual != receipt.log_sha256 {
            return Err(RecordError::HashMismatch {
                expected: receipt.log_sha256,
                actual,
            });
        }
        let count = u64::try_from(events.len()).expect("count fits");
        if count != receipt.events {
            return Err(RecordError::CountMismatch {
                expected: receipt.events,
                actual: count,
            });
        }
        Ok(Self {
            identity: header.identity,
            provenance: header.provenance,
            provenance_fingerprint: header.provenance_fingerprint,
            events,
            receipt,
        })
    }
}
