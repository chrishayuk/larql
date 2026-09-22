//! ATTR-1D sealed production witness (frozen contract
//! `sha256:4f48973807db0695d5f30b83a00c73637afbd0dc77d116b5a1f9eb6ec913431c`,
//! `bench/attr-1d/attr1d-contract.json`).
//!
//! One non-intervened production CPU execution at zero-based L24 attention,
//! final prompt position, on the lexicographically smallest train
//! TransitionIdentity with all three prompt families in the frozen GW-0
//! census (`bench/gw0/gemma3-4b-it-phase1/`). This is a disposable witness
//! runner, not a generic attribution CLI: it exists to seal one witness and
//! nothing downstream reads its code.
//!
//! Standalone by design: it does not depend on `observatory_record.rs`'s
//! shared aggregator or its `use super::*` glob, so it can be built, tested
//! and committed independently of the other in-progress GW-rung runners that
//! aggregator dispatches to.
use larql_inference::vindex3::attribution::{
    describe_attention_support, AttentionHeadEvidence, AttentionSiteEvidence,
    AttentionSourceEvidence, AttributionReader, PostAttentionTransform,
    PreparedAttentionProjection, PromptRoleMap, SupportCoordinate, SupportCoordinateContext,
    SupportObservationIdentity, TokenSpan, ATTENTION_PROBABILITY_MAX_ABSOLUTE_ERROR,
    ATTR1D_CONTRACT_IDENTITY, HEAD_SUM_MAX_RELATIVE_L2, PROJECTION_RECONSTRUCTION_SCALE,
    SOURCE_SPLIT_MAX_RELATIVE_L2,
};
use larql_inference::vindex3::{
    CarrierWriteRecord, ExecutionProvenance, StepEvent, StepObserver, SublayerSite,
};
use larql_models::config::NormType;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    head_replay::replay_attention_heads,
    kv::RowKvState,
    observe::{AttentionHeadRecord, NoopObserver},
    operands::OperandStore,
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
};
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const LAYER: usize = 24;
const FAMILIES: [&str; 3] = ["canonical", "alternate", "question"];

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn atomic_json(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path)?;
    Ok(())
}

#[derive(Default)]
struct HeadCapture {
    kv_head: usize,
    source_start: usize,
    weights: Vec<f32>,
    mixed: Vec<f32>,
    source_values: Vec<Vec<f32>>,
}

struct L24Site {
    target_position: usize,
    heads: BTreeMap<usize, HeadCapture>,
    recorded_delta: Option<Vec<f32>>,
    error: Option<String>,
}

impl L24Site {
    fn new(target_position: usize) -> Self {
        Self {
            target_position,
            heads: BTreeMap::new(),
            recorded_delta: None,
            error: None,
        }
    }
}

impl StepObserver for L24Site {
    fn event(&mut self, _event: StepEvent) {}

    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        layer == LAYER && position == self.target_position
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if layer != LAYER || record.position != self.target_position {
            return;
        }
        if self.heads.contains_key(&record.head) {
            self.error = Some(format!("duplicate L24 head {} capture", record.head));
            return;
        }
        if record.weights.len() != record.source_values.len() {
            self.error = Some(format!(
                "L24 head {} weights/source-values length mismatch",
                record.head
            ));
            return;
        }
        self.heads.insert(
            record.head,
            HeadCapture {
                kv_head: record.kv_head,
                source_start: record.source_start,
                weights: record.weights.to_vec(),
                mixed: record.values.to_vec(),
                source_values: record.source_values.to_vec(),
            },
        );
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.layer == LAYER
            && record.site == SublayerSite::Attention
            && record.position == self.target_position
        {
            if self.recorded_delta.is_some() {
                self.error = Some("duplicate L24 carrier write".into());
                return;
            }
            self.recorded_delta = Some(record.delta.to_vec());
        }
    }
}

/// Adapter onto the exact prepared L24 image. `project_head` reuses the
/// already-frozen `replay_attention_heads` per-head isolate-and-project
/// path (`head_replay.rs`) rather than reaching into `PreparedAttention`
/// internals: it is called once per (head, source-or-mixed) input with all
/// other heads zeroed, and only the requested head's contribution is read
/// back. `entering_carrier` in that call is a throwaway zero vector — this
/// adapter never reads `carrier_after`/`applied_delta` from it, only the
/// per-head raw contribution, which does not depend on the carrier.
struct RealProjection<'a> {
    plan: &'a ComponentOpPlan,
    ops: &'a PreparedOperands,
    backend: &'a ProductionBackend,
    store: &'a OperandStore,
    layer: usize,
    num_heads: usize,
    head_dim: usize,
}

impl PreparedAttentionProjection for RealProjection<'_> {
    fn hidden_size(&self) -> usize {
        self.ops.hidden()
    }

    fn project_head(
        &self,
        query_head: usize,
        values: &[f32],
    ) -> std::result::Result<Vec<f32>, String> {
        if values.len() != self.head_dim {
            return Err(format!(
                "head {query_head} projection input width {} does not match head_dim {}",
                values.len(),
                self.head_dim
            ));
        }
        let mut head_values = vec![vec![0.0f32; self.head_dim]; self.num_heads];
        head_values[query_head] = values.to_vec();
        let zero_carrier = vec![0.0f32; self.hidden_size()];
        let replay = replay_attention_heads(
            self.plan,
            self.ops,
            self.backend,
            self.layer,
            &zero_carrier,
            &head_values,
        )
        .map_err(|error| error.to_string())?;
        replay
            .contributions
            .get(query_head)
            .cloned()
            .ok_or_else(|| format!("replay produced no contribution for head {query_head}"))
    }

    fn output_bias(&self) -> std::result::Result<Option<Vec<f32>>, String> {
        Ok(None)
    }

    fn post_attention_transform(&self) -> std::result::Result<PostAttentionTransform, String> {
        let layer_plan = self
            .plan
            .layers
            .get(self.layer)
            .ok_or_else(|| format!("layer {} is outside the plan", self.layer))?;
        let norm_op = layer_plan
            .post_attention_norm
            .as_ref()
            .ok_or("layer has no post-attention norm")?;
        if norm_op.kind != NormType::RmsNorm {
            return Err("post-attention norm is not RmsNorm".into());
        }
        let weights = self
            .store
            .load(&norm_op.weight)
            .map_err(|error| error.to_string())?;
        Ok(PostAttentionTransform::RmsNorm {
            epsilon: norm_op.eps,
            weight_offset: norm_op.weight_offset,
            weights,
            residual_scale: layer_plan.residual_scale.unwrap_or(1.0),
        })
    }
}

fn spans_from_positions(positions: &[Value]) -> Result<Vec<TokenSpan>> {
    positions
        .iter()
        .map(|value| {
            let position = value.as_u64().ok_or("bad role position")? as usize;
            Ok(TokenSpan::new(position, position + 1)?)
        })
        .collect()
}

/// `attr1d_witness CONTAINER INPUT_MANIFEST SOURCE_ROLES OUTPUT_DIR`
fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err(
            "Usage: attr1d_witness CONTAINER INPUT_MANIFEST SOURCE_ROLES OUTPUT_DIR".into(),
        );
    }
    let container = Path::new(&args[0]);
    let input_manifest_path = Path::new(&args[1]);
    let source_roles_path = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    std::fs::create_dir_all(output)?;
    let result_path = output.join("attr1d-witness.json");
    if result_path.exists() {
        return Err(format!("ATTR-1D witness is already sealed at {}", output.display()).into());
    }

    // --- mechanical witness row selection over the frozen GW-0 census ---
    let input_manifest: Value = serde_json::from_slice(&std::fs::read(input_manifest_path)?)?;
    if input_manifest["schema"] != "larql.gw0.input-manifest.v1" {
        return Err("not the frozen GW-0 input manifest".into());
    }
    let input_rows_path = input_manifest_path
        .parent()
        .ok_or("input manifest has no parent")?
        .join(
            input_manifest["rows"]["path"]
                .as_str()
                .ok_or("input rows path missing")?,
        );

    let mut rows: Vec<Value> = Vec::new();
    for line in BufReader::new(File::open(&input_rows_path)?).lines() {
        rows.push(serde_json::from_str(&line?)?);
    }

    let mut groups: BTreeMap<(String, String, String), Vec<Value>> = BTreeMap::new();
    for row in &rows {
        let edge = &row["semantic_edge"];
        if edge["status"] != "positive" {
            return Err("GW-0 row is not a positive semantic edge".into());
        }
        let key = (
            edge["relation"]
                .as_str()
                .ok_or("relation missing")?
                .to_string(),
            edge["subject"]
                .as_str()
                .ok_or("subject missing")?
                .to_string(),
            edge["target"].as_str().ok_or("target missing")?.to_string(),
        );
        groups.entry(key).or_default().push(row.clone());
    }

    let declared_families: BTreeSet<&str> = FAMILIES.iter().copied().collect();
    let mut winner: Option<(String, String, String)> = None;
    for (key, members) in &groups {
        let families: BTreeSet<&str> = members
            .iter()
            .map(|row| {
                row["semantic_edge"]["prompt_semantic_family"]
                    .as_str()
                    .unwrap_or("")
            })
            .collect();
        if families != declared_families {
            continue;
        }
        let splits: BTreeSet<&str> = members
            .iter()
            .map(|row| row["split"].as_str().unwrap_or(""))
            .collect();
        if splits != BTreeSet::from(["train"]) {
            continue;
        }
        winner = Some(key.clone());
        break; // BTreeMap iterates keys in ascending lexicographic order.
    }
    let winner = winner.ok_or("no train TransitionIdentity has all three prompt families")?;
    let group = &groups[&winner];
    let canonical_rows: Vec<&Value> = group
        .iter()
        .filter(|row| row["semantic_edge"]["prompt_semantic_family"] == "canonical")
        .collect();
    if canonical_rows.len() != 1 {
        return Err("winning TransitionIdentity does not have exactly one canonical row".into());
    }
    let row = canonical_rows[0];
    let edge_id = row["edge_id"]
        .as_str()
        .ok_or("edge_id missing")?
        .to_string();
    let token_ids: Vec<u32> = row["prompt"]["token_ids"]
        .as_array()
        .ok_or("prompt token IDs missing")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .map(|v| v as u32)
                .ok_or("bad prompt token ID")
        })
        .collect::<std::result::Result<_, _>>()?;
    let target_position = row["capture_request"]["position"]
        .as_u64()
        .ok_or("capture position missing")? as usize;
    if target_position + 1 != token_ids.len() {
        return Err("winning row's capture position is not the final prompt position".into());
    }
    let target_token_ids: Vec<u32> = row["semantic_edge"]["target_token_ids"]
        .as_array()
        .ok_or("target token IDs missing")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .map(|v| v as u32)
                .ok_or("bad target token ID")
        })
        .collect::<std::result::Result<_, _>>()?;
    let target_token_id = *target_token_ids
        .first()
        .ok_or("winning row has no target token")?;

    eprintln!(
        "ATTR-1D witness row: {} / {} / {} ({})",
        winner.0, winner.1, winner.2, edge_id
    );

    // --- frozen prompt-structure manifest (GW-KEY-1 source roles) ---
    let mut role_row: Option<Value> = None;
    for line in BufReader::new(File::open(source_roles_path)?).lines() {
        let value: Value = serde_json::from_str(&line?)?;
        if value["edge_id"] == Value::String(edge_id.clone()) {
            role_row = Some(value);
            break;
        }
    }
    let role_row = role_row.ok_or("no GW-KEY-1 source-role row for the winning edge")?;
    if role_row["capture_position"].as_u64() != Some(target_position as u64) {
        return Err("GW-KEY-1 source-role capture position does not match the GW-0 row".into());
    }
    if role_row["token_count"].as_u64() != Some(token_ids.len() as u64) {
        return Err("GW-KEY-1 source-role token count does not match the GW-0 row".into());
    }
    let roles = &role_row["roles"];
    let bos_spans = spans_from_positions(
        roles["bos_system"]
            .as_array()
            .ok_or("bos_system roles missing")?,
    )?;
    let relation_spans = spans_from_positions(
        roles["relation_query"]
            .as_array()
            .ok_or("relation_query roles missing")?,
    )?;
    let entity_spans = spans_from_positions(
        roles["subject_entity"]
            .as_array()
            .ok_or("subject_entity roles missing")?,
    )?;
    let role_map = PromptRoleMap::new(
        token_ids.len(),
        bos_spans,
        relation_spans,
        entity_spans,
        target_position,
    )?;

    // --- prepare the production image and check the frozen witness site ---
    eprintln!("Preparing the frozen ATTR-1D production image...");
    let inspection = inspect_container(container, true)?;
    if !inspection.is_coherent() {
        return Err(format!("container defects: {:?}", inspection.defects).into());
    }
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !outcome.closed() {
        return Err(format!("plan refused: {:?}", outcome.defects).into());
    }
    let plan = outcome.plan.ok_or("closed target plan absent")?;
    let plan_identity = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    let container_identity = format!(
        "sha256:{}",
        sha(&serde_json::to_vec(
            &json!({"index": inspection.index, "graph": inspection.graph})
        )?)
    );

    let layer_plan = plan.layers.get(LAYER).ok_or("model has no L24")?;
    let op = layer_plan.attention.softmax().ok_or("L24 is not softmax")?;
    if op.output_gate.is_some() {
        return Err("L24 declares an output gate; ATTR-1D witness v1 refuses gated heads".into());
    }
    if op.o_bias.is_some() {
        return Err("L24 declares an output bias; ATTR-1D witness v1 refuses biased heads".into());
    }
    if op.sinks.is_some() {
        return Err(
            "L24 declares attention sinks; ATTR-1D witness v1 refuses sink attention".into(),
        );
    }
    let post_norm_op = layer_plan
        .post_attention_norm
        .as_ref()
        .ok_or("L24 has no post-attention norm")?;
    if post_norm_op.kind != NormType::RmsNorm {
        return Err("L24 post-attention norm is not RmsNorm".into());
    }
    let num_heads = op.num_q_heads;
    let head_dim = op.head_dim;

    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;

    // --- reader: the real output-head row for the target's first token,
    // dequantised from the exact prepared representation, never a
    // normalized readout (readout_carrier_selected applies the final norm
    // and softcap, which a linear reader row must not carry) ---
    let selected = ops.select_output_head(&[target_token_id])?;
    let reader_identity = format!(
        "selected_token_row/v1+identity/v1:token_{target_token_id}:{}",
        selected.representation()
    );
    let reader_values = selected.row_f32(0)?;
    let reader = AttributionReader::new(reader_identity.clone(), reader_values)?;

    // --- the real non-intervened production CPU execution ---
    let mut site = L24Site::new(target_position);
    let mut observed_logits = Vec::with_capacity(token_ids.len());
    {
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
        for &token in &token_ids {
            observed_logits.push(
                session
                    .step_observed(token, &mut site)?
                    .logits
                    .ok_or("missing observed logits")?,
            );
        }
    }
    if let Some(error) = site.error.take() {
        return Err(error.into());
    }

    let mut bit_mismatches = 0usize;
    {
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
        for (&token, observed) in token_ids.iter().zip(&observed_logits) {
            let control = session
                .step_observed(token, &mut NoopObserver)?
                .logits
                .ok_or("missing control logits")?;
            if observed.len() != control.len() {
                return Err("observed/control vocabulary width mismatch".into());
            }
            bit_mismatches += observed
                .iter()
                .zip(&control)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
    }
    if bit_mismatches != 0 {
        return Err("ATTR-1D witness observation changed production logits".into());
    }

    if site.heads.len() != num_heads {
        return Err(format!(
            "expected {num_heads} L24 heads, captured {}",
            site.heads.len()
        )
        .into());
    }
    let recorded_delta = site
        .recorded_delta
        .ok_or("missing L24 recorded carrier delta")?;

    let mut heads = Vec::with_capacity(num_heads);
    let mut max_attention_probability_gap_observed = 0.0f64;
    for head in 0..num_heads {
        let capture = site
            .heads
            .get(&head)
            .ok_or_else(|| format!("missing L24 head {head}"))?;
        if capture.mixed.len() != head_dim {
            return Err(format!("L24 head {head} mixed-value width is not {head_dim}").into());
        }
        let probability_sum: f64 = capture
            .weights
            .iter()
            .map(|weight| f64::from(*weight))
            .sum();
        max_attention_probability_gap_observed = f64::max(
            max_attention_probability_gap_observed,
            (probability_sum - 1.0).abs(),
        );
        if (probability_sum - 1.0).abs() > 1e-3 {
            return Err(format!(
                "L24 head {head} attention probabilities sum to {probability_sum}, not ~1.0 \
                 — the assumed sink=0.0 is false for this op"
            )
            .into());
        }
        let sources = capture
            .weights
            .iter()
            .zip(&capture.source_values)
            .enumerate()
            .map(|(offset, (&weight, values))| AttentionSourceEvidence {
                position: capture.source_start + offset,
                attention_probability: weight,
                values: values.clone(),
            })
            .collect();
        heads.push(AttentionHeadEvidence {
            query_head: head,
            kv_head: capture.kv_head,
            sink: 0.0,
            mixed_values: capture.mixed.clone(),
            gate: None,
            sources,
        });
    }

    let evidence = AttentionSiteEvidence {
        expected_query_heads: num_heads,
        visible_source_range: TokenSpan::new(0, target_position + 1)?,
        recorded_delta,
        heads,
    };

    let projection = RealProjection {
        plan: &plan,
        ops: &ops,
        backend: &backend,
        store: &store,
        layer: LAYER,
        num_heads,
        head_dim,
    };
    let readers = [reader];

    eprintln!("Running the ATTR-1D deterministic transform...");
    let first = describe_attention_support(&evidence, &projection, &readers, &role_map)?;
    let second = describe_attention_support(&evidence, &projection, &readers, &role_map)?;
    let first_bytes = serde_json::to_vec(&first)?;
    let second_bytes = serde_json::to_vec(&second)?;
    let replay_pass = first_bytes == second_bytes;
    let numerical_gates_pass = first.passes_numerical_gates();
    let overall_pass = numerical_gates_pass && replay_pass;

    let coordinate_context = SupportCoordinateContext::new(
        container_identity.clone(),
        plan_identity.clone(),
        "target".to_string(),
        format!("sha256:{}", sha(&serde_json::to_vec(&token_ids)?)),
        target_position,
        LAYER,
    )?;
    let coordinate = SupportCoordinate::site(coordinate_context)?;
    let coordinate_id = coordinate.identity()?;
    let execution_identity = format!(
        "sha256:{}",
        sha(&serde_json::to_vec(
            &json!({"edge_id": edge_id, "token_ids": token_ids})
        )?)
    );
    let provenance_fingerprint = ExecutionProvenance::of(&ops).fingerprint();
    let observation_receipt_digest = format!(
        "sha256:{}",
        sha(&serde_json::to_vec(&json!({
            "observed_terminal_logits": observed_logits.last(),
        }))?)
    );
    let observation = SupportObservationIdentity::new(
        &coordinate,
        execution_identity,
        provenance_fingerprint,
        plan_identity.clone(),
        observation_receipt_digest,
        0,
    )?;
    let observation_id = observation.identity()?;
    let role_map_identity = role_map.identity()?;

    let max_source_reconstruction_relative_l2 = first
        .heads
        .iter()
        .map(|head| head.source_reconstruction_relative_l2)
        .fold(0.0f64, f64::max);
    let max_attention_probability_error = first
        .heads
        .iter()
        .map(|head| head.attention_probability_error)
        .fold(0.0f64, f64::max);
    let projection_reconstruction_pass = first
        .projection_reconstruction
        .iter()
        .all(|check| check.absolute_error <= check.permitted_error);
    let head_sum_pass = first.head_reconstruction_relative_l2 <= HEAD_SUM_MAX_RELATIVE_L2;
    let source_split_pass = max_source_reconstruction_relative_l2 <= SOURCE_SPLIT_MAX_RELATIVE_L2;
    let attention_probability_pass =
        max_attention_probability_error <= ATTENTION_PROBABILITY_MAX_ABSOLUTE_ERROR;

    let manifest = json!({
        "schema": "larql.attr1d.witness-result.v1",
        "attr1d_contract_identity": ATTR1D_CONTRACT_IDENTITY,
        "status": if overall_pass { "witness_pass" } else { "witness_fail" },
        "winning_transition_identity": {
            "relation": winner.0,
            "subject": winner.1,
            "target": winner.2,
        },
        "edge_id": edge_id,
        "prompt_text": row["prompt"]["text"],
        "container_identity": container_identity,
        "plan_identity": plan_identity,
        "support_coordinate_id": coordinate_id,
        "support_coordinate": coordinate,
        "support_observation_id": observation_id,
        "support_observation": observation,
        "reader_identity": reader_identity,
        "role_map_identity": role_map_identity,
        "architecture_facts": {
            "attention_sinks_declared_in_plan": false,
            "attention_sinks_checked": "L24 softmax op's `sinks: Option<SinkOp>` field was checked before this run and the witness refuses if it is Some; this is an observed fact about the prepared plan, not an assumed default",
            "sink_value_used": 0.0,
            "max_attention_probability_gap_observed": max_attention_probability_gap_observed,
            "note": "every head's Σ attention_probability (+ sink=0.0) landed within max_attention_probability_gap_observed of 1.0, independently corroborating the declared absence of an attention sink for this op",
        },
        "gates": {
            "head_sum_max_relative_l2": {
                "threshold": HEAD_SUM_MAX_RELATIVE_L2,
                "measured": first.head_reconstruction_relative_l2,
                "pass": head_sum_pass,
            },
            "source_split_max_relative_l2": {
                "threshold": SOURCE_SPLIT_MAX_RELATIVE_L2,
                "measured_max_over_heads": max_source_reconstruction_relative_l2,
                "pass": source_split_pass,
            },
            "attention_probability_max_absolute_error": {
                "threshold": ATTENTION_PROBABILITY_MAX_ABSOLUTE_ERROR,
                "measured_max_over_heads": max_attention_probability_error,
                "pass": attention_probability_pass,
            },
            "projection_reconstruction_scale": {
                "threshold_scale": PROJECTION_RECONSTRUCTION_SCALE,
                "per_reader": &first.projection_reconstruction,
                "pass": projection_reconstruction_pass,
            },
            "coverage": {
                "pass": true,
                "note": "implied by Ok(..) from describe_attention_support, which refuses on missing/duplicate coordinates",
            },
            "refusal_accounting": {
                "pass": true,
                "note": "no coverage refusal, missing sidecar, truncated source surface, unstable denominator or unsupported norm occurred on this witness row",
            },
            "deterministic_replay": {
                "pass": replay_pass,
                "note": "two independent calls to describe_attention_support on the same sealed evidence serialize to byte-identical canonical JSON",
            },
            "backend_agreement": {
                "pass": true,
                "note": "vacuous: only one backend/execution (non-intervened production CPU) is defined here, so there is no second realization to compare",
            },
            "numerical_summary": numerical_gates_pass,
        },
        "overall_pass": overall_pass,
        "descriptive_support": first,
    });
    atomic_json(&result_path, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    if !overall_pass {
        return Err("ATTR-1D witness FAILED — see sealed result for the failing gate".into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    run(&args)
}
