//! Oracle capture over a reusable VINDEX3 decode session.
//!
//! Every query is an independent sequence (`DecodeSession::reset` clears KV)
//! while model operands stay resident. Only the final fixture token is
//! observed: it carries the complete query context and yields one residual key
//! per selected layer. The ordinary decode traversal still computes every
//! operation; observers are read-only.

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;

use larql_models::config::Activation;
use larql_vindex::error::VindexError;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::{
    FfnBlockContributionCall, FfnCall, PlanBackend, ProjectCall, WeightSlice,
};
use larql_vindex::format::vindex3::opplan::exec::decode::{DecodeObserver, DecodeSession};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use serde::{Deserialize, Serialize};

use super::format::{
    CaptureManifest, ObjectDescriptor, SampleDescriptor, CONTRIBUTION_SEMANTICS_V1, SCHEMA_V1,
};
use super::{CaptureArgs, CaptureBackend, ContributionEngine, SeamKind};

const RESIDUALS_FILE: &str = "residuals.f32";
const CONTRIBUTIONS_FILE: &str = "contributions.f32";
const SEAMS_FILE: &str = "seams.f32";
const CHECKPOINT_FILE: &str = "capture-state.json";
const PROGRESS_FILE: &str = "progress.jsonl";
const CHECKPOINT_SCHEMA_V1: &str = "larql.mad-v3-knn.checkpoint.v1";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct QueryFixture {
    id: String,
    semantic_id: String,
    #[serde(default)]
    operation_id: Option<String>,
    wording_id: String,
    split: String,
    #[serde(default)]
    group_id: Option<String>,
    #[serde(default)]
    regime: Option<String>,
    #[serde(default)]
    alias_id: Option<String>,
    #[serde(default)]
    layout_id: Option<String>,
    token_ids: Vec<u32>,
    #[serde(default)]
    expected_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct AnswerOutcome {
    expected_token_ids: Option<Vec<u32>>,
    generated_token_ids: Option<Vec<u32>>,
    answer_exact: Option<bool>,
    answer_token_accuracy: Option<f64>,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct CaptureCheckpoint {
    schema: String,
    model: String,
    backend: String,
    hidden_size: usize,
    residual_layers: Vec<usize>,
    objects: Vec<(String, usize, u64)>,
    contribution_engine: String,
    block_channels: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    block_channel_schemas: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    seam_layers: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    seam_kinds: Vec<String>,
    queries: Vec<QueryFixture>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProgressRow {
    sample_id: String,
    answer: AnswerOutcome,
}

#[derive(Clone, Copy)]
struct Block {
    object: usize,
    start: usize,
    end: usize,
}

struct BlockSchema {
    block_channels: usize,
    blocks: Vec<Block>,
}

struct OracleObserver<'a, B> {
    backend: &'a B,
    residual_layers: HashSet<usize>,
    seam_layers: HashSet<usize>,
    seam_kinds: HashSet<SeamKind>,
    blocks: &'a HashMap<usize, Vec<BlockSchema>>,
    residuals: HashMap<usize, Vec<f32>>,
    seams: HashMap<(usize, SeamKind), Vec<f32>>,
    contributions: Vec<f32>,
    contribution_engine: ContributionEngine,
}

impl<'a, B: PlanBackend> OracleObserver<'a, B> {
    fn new(
        backend: &'a B,
        residual_layers: &[usize],
        seam_layers: &[usize],
        seam_kinds: &[SeamKind],
        blocks: &'a HashMap<usize, Vec<BlockSchema>>,
        object_count: usize,
        contribution_engine: ContributionEngine,
    ) -> Self {
        Self {
            backend,
            residual_layers: residual_layers.iter().copied().collect(),
            seam_layers: seam_layers.iter().copied().collect(),
            seam_kinds: seam_kinds.iter().copied().collect(),
            blocks,
            residuals: HashMap::new(),
            seams: HashMap::new(),
            contributions: vec![0.0; object_count],
            contribution_engine,
        }
    }

    fn record_seam(&mut self, layer: usize, kind: SeamKind, values: &[f32]) {
        if self.seam_layers.contains(&layer) && self.seam_kinds.contains(&kind) {
            self.seams.insert((layer, kind), values.to_vec());
        }
    }
}

impl<B: PlanBackend> DecodeObserver for OracleObserver<'_, B> {
    fn layer_input(&mut self, layer: usize, residual: &[f32]) -> Result<(), VindexError> {
        if self.residual_layers.contains(&layer) {
            self.residuals.insert(layer, residual.to_vec());
        }
        Ok(())
    }

    fn attention_input(&mut self, layer: usize, input: &[f32]) -> Result<(), VindexError> {
        self.record_seam(layer, SeamKind::AttentionInput, input);
        Ok(())
    }

    fn attention_output(&mut self, layer: usize, output: &[f32]) -> Result<(), VindexError> {
        self.record_seam(layer, SeamKind::AttentionOutput, output);
        Ok(())
    }

    fn post_attention(&mut self, layer: usize, residual: &[f32]) -> Result<(), VindexError> {
        self.record_seam(layer, SeamKind::PostAttention, residual);
        Ok(())
    }

    fn ffn_call(&mut self, layer: usize, call: &FfnCall<'_>) -> Result<(), VindexError> {
        self.record_seam(layer, SeamKind::FfnInput, call.x);
        let Some(blocks) = self.blocks.get(&layer) else {
            return Ok(());
        };
        let up = self.backend.project(ProjectCall {
            weight: call.up,
            out_dim: call.intermediate,
            in_dim: call.hidden,
            x: call.x,
        })?;
        let inner = match call.gate {
            Some(gate_weight) => {
                if call.activation != Activation::Silu {
                    return Err(VindexError::Parse(format!(
                        "MAD-V3 contribution capture has only judged gated SiLU, got {:?}",
                        call.activation
                    )));
                }
                let gate = self.backend.project(ProjectCall {
                    weight: gate_weight,
                    out_dim: call.intermediate,
                    in_dim: call.hidden,
                    x: call.x,
                })?;
                larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate, &up)
            }
            None => {
                if call.activation != Activation::Silu {
                    return Err(VindexError::Parse(format!(
                        "MAD-V3 contribution capture has only judged ungated SiLU, got {:?}",
                        call.activation
                    )));
                }
                up.iter()
                    .map(|&value| larql_compute::cpu::ops::geglu::silu(value))
                    .collect()
            }
        };
        if self.contribution_engine == ContributionEngine::Fast {
            for schema in blocks {
                let values = self
                    .backend
                    .ffn_block_contributions(FfnBlockContributionCall {
                        down: call.down,
                        inner: &inner,
                        hidden: call.hidden,
                        intermediate: call.intermediate,
                        block_channels: schema.block_channels,
                    })?
                    .ok_or_else(|| {
                        VindexError::Parse(format!(
                            "backend `{}` has no accelerated f16 FFN-block contribution kernel",
                            self.backend.name()
                        ))
                    })?;
                if values.len() != schema.blocks.len() {
                    return Err(VindexError::Parse(format!(
                        "layer {layer} schema {}: accelerated observer returned {} blocks, expected {}",
                        schema.block_channels,
                        values.len(),
                        schema.blocks.len()
                    )));
                }
                for (block, value) in schema.blocks.iter().zip(values) {
                    if !value.is_finite() || value < 0.0 {
                        return Err(VindexError::Parse(format!(
                            "layer {layer} schema {} block {}..{} accelerated contribution is invalid: {value}",
                            schema.block_channels, block.start, block.end
                        )));
                    }
                    self.contributions[block.object] = value;
                }
            }
            return Ok(());
        }

        for schema in blocks {
            for block in &schema.blocks {
                let mut norm2 = 0.0f64;
                for output in 0..call.hidden {
                    // Preserve channel order within a block. Blocks partition
                    // each exact schema; no execution value is replaced.
                    let mut contribution = 0.0f32;
                    for (channel, &activation) in
                        inner.iter().enumerate().take(block.end).skip(block.start)
                    {
                        let index = output * call.intermediate + channel;
                        contribution += weight_value(call.down, index)? * activation;
                    }
                    norm2 += f64::from(contribution) * f64::from(contribution);
                }
                let value = norm2 as f32;
                if !value.is_finite() {
                    return Err(VindexError::Parse(format!(
                        "layer {layer} schema {} block {}..{} contribution overflowed f32",
                        schema.block_channels, block.start, block.end
                    )));
                }
                self.contributions[block.object] = value;
            }
        }
        Ok(())
    }

    fn ffn_output(&mut self, layer: usize, output: &[f32]) -> Result<(), VindexError> {
        self.record_seam(layer, SeamKind::FfnOutput, output);
        Ok(())
    }
}

pub fn run_capture(args: CaptureArgs) -> Result<(), Box<dyn std::error::Error>> {
    prepare_output(&args.output, args.resume)?;
    let mut queries = read_queries(&args.queries)?;
    if let Some(limit) = args.limit {
        queries.truncate(limit);
    }
    validate_queries(&queries)?;
    if args.block_channels == 0 {
        return Err("--block-channels must be positive".into());
    }
    let mut block_channel_schemas = if args.page_channels.is_empty() {
        vec![args.block_channels]
    } else {
        args.page_channels.clone()
    };
    if block_channel_schemas.contains(&0) {
        return Err("--page-channels values must be positive".into());
    }
    block_channel_schemas.sort_unstable();
    if block_channel_schemas
        .windows(2)
        .any(|pair| pair[0] == pair[1])
    {
        return Err("--page-channels values must be unique".into());
    }
    let checkpoint_block_channel_schemas = if args.page_channels.is_empty() {
        Vec::new()
    } else {
        block_channel_schemas.clone()
    };

    let inspection = inspect_container(&args.container, false)?;
    let outcome = plan_component_ops(&inspection, &args.container, &args.component)?;
    if !outcome.defects.is_empty() {
        for defect in &outcome.defects {
            eprintln!("defect: {defect}");
        }
        return Err(format!(
            "component `{}` does not close: {} defect(s)",
            args.component,
            outcome.defects.len()
        )
        .into());
    }
    let plan = outcome
        .plan
        .ok_or_else(|| format!("component `{}` produced no plan", args.component))?;
    let residual_layers = parse_layers(&args.residual_layers, plan.layers.len())?;
    let seam_layers = match (&args.seam_layers, args.seam_kinds.is_empty()) {
        (None, true) => Vec::new(),
        (Some(spec), false) => parse_layers(spec, plan.layers.len())?,
        (Some(_), true) => return Err("--seam-layers requires --seam-kinds".into()),
        (None, false) => return Err("--seam-kinds requires --seam-layers".into()),
    };
    let unique_seam_kinds: HashSet<_> = args.seam_kinds.iter().copied().collect();
    if unique_seam_kinds.len() != args.seam_kinds.len() {
        return Err("--seam-kinds values must be unique".into());
    }
    let seam_kinds: Vec<_> = SeamKind::ORDERED
        .into_iter()
        .filter(|kind| unique_seam_kinds.contains(kind))
        .collect();
    let contribution_layers = match &args.contribution_layers {
        Some(spec) => parse_layers(spec, plan.layers.len())?,
        None => (0..plan.layers.len()).collect(),
    };
    let (objects, blocks) = object_table(&plan, &contribution_layers, &block_channel_schemas)?;
    let store = OperandStore::open(&args.container, &inspection)?;

    match args.backend {
        CaptureBackend::Reference => capture_with_backend(
            &ReferenceBackend::new(),
            &plan,
            &store,
            &queries,
            &residual_layers,
            &seam_layers,
            &seam_kinds,
            &objects,
            &blocks,
            &args.output,
            &inspection.index.model,
            args.contribution_engine,
            args.block_channels,
            &checkpoint_block_channel_schemas,
            args.resume,
        )?,
        CaptureBackend::Production => capture_with_backend(
            &ProductionBackend::new(),
            &plan,
            &store,
            &queries,
            &residual_layers,
            &seam_layers,
            &seam_kinds,
            &objects,
            &blocks,
            &args.output,
            &inspection.index.model,
            args.contribution_engine,
            args.block_channels,
            &checkpoint_block_channel_schemas,
            args.resume,
        )?,
        #[cfg(feature = "gpu")]
        CaptureBackend::Metal => {
            use larql_vindex::format::vindex3::opplan::exec::backend::WeightFormat;
            use larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend;
            let gpu = larql_compute_metal::MetalBackend::new()
                .ok_or("no Metal device available for --backend metal")?;
            let backend = DevicePlanBackend::new(gpu, "mad-v3-metal-f16", WeightFormat::F16);
            capture_with_backend(
                &backend,
                &plan,
                &store,
                &queries,
                &residual_layers,
                &seam_layers,
                &seam_kinds,
                &objects,
                &blocks,
                &args.output,
                &inspection.index.model,
                args.contribution_engine,
                args.block_channels,
                &checkpoint_block_channel_schemas,
                args.resume,
            )?
        }
    }
    Ok(())
}

fn weight_value(weight: WeightSlice<'_>, index: usize) -> Result<f32, VindexError> {
    match weight {
        WeightSlice::F32(values) => values.get(index).copied().ok_or_else(|| {
            VindexError::Parse(format!("f32 FFN weight index {index} is out of range"))
        }),
        WeightSlice::F16(bytes) => {
            let offset = index
                .checked_mul(2)
                .ok_or_else(|| VindexError::Parse("f16 FFN weight offset overflow".to_string()))?;
            let raw = bytes.get(offset..offset + 2).ok_or_else(|| {
                VindexError::Parse(format!("f16 FFN weight index {index} is out of range"))
            })?;
            Ok(larql_models::quant::half::f16_to_f32(u16::from_le_bytes([
                raw[0], raw[1],
            ])))
        }
        WeightSlice::Mxfp4 { .. } | WeightSlice::Nvfp4 { .. } => Err(VindexError::Parse(
            "MAD-V3 block attribution is not yet judged for packed FP4 weights".to_string(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_with_backend<B: PlanBackend>(
    backend: &B,
    plan: &ComponentOpPlan,
    store: &OperandStore,
    queries: &[QueryFixture],
    residual_layers: &[usize],
    seam_layers: &[usize],
    seam_kinds: &[SeamKind],
    objects: &[ObjectDescriptor],
    blocks: &HashMap<usize, Vec<BlockSchema>>,
    output: &Path,
    model: &str,
    contribution_engine: ContributionEngine,
    block_channels: usize,
    block_channel_schemas: &[usize],
    resume: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let hidden_size = plan
        .embedding
        .as_ref()
        .ok_or("plan has no embedding")?
        .table
        .shape[1];
    let checkpoint = CaptureCheckpoint {
        schema: CHECKPOINT_SCHEMA_V1.into(),
        model: model.to_string(),
        backend: backend.name().to_string(),
        hidden_size,
        residual_layers: residual_layers.to_vec(),
        objects: objects
            .iter()
            .map(|object| (object.id.clone(), object.layer, object.byte_count))
            .collect(),
        contribution_engine: contribution_engine_name(contribution_engine).into(),
        block_channels,
        block_channel_schemas: block_channel_schemas.to_vec(),
        seam_layers: seam_layers.to_vec(),
        seam_kinds: seam_kinds
            .iter()
            .map(|kind| kind.as_str().to_string())
            .collect(),
        queries: queries.to_vec(),
    };
    let residual_stride = hidden_size
        .checked_mul(residual_layers.len())
        .and_then(|values| values.checked_mul(4))
        .ok_or("residual row size overflow")?;
    let contribution_stride = objects
        .len()
        .checked_mul(4)
        .ok_or("contribution row size overflow")?;
    let seam_stride = hidden_size
        .checked_mul(seam_layers.len())
        .and_then(|values| values.checked_mul(seam_kinds.len()))
        .and_then(|values| values.checked_mul(4))
        .ok_or("seam row size overflow")?;
    let (start, mut answer_outcomes, mut progress_writer) = open_checkpoint(
        output,
        &checkpoint,
        residual_stride,
        contribution_stride,
        seam_stride,
        resume,
    )?;
    let mut residual_writer = append_writer(output.join(RESIDUALS_FILE))?;
    let mut contribution_writer = append_writer(output.join(CONTRIBUTIONS_FILE))?;
    let mut seam_writer = if seam_stride > 0 {
        Some(append_writer(output.join(SEAMS_FILE))?)
    } else {
        None
    };
    if start > 0 {
        eprintln!("resuming at sample {start}/{}", queries.len());
    }
    if start < queries.len() {
        eprintln!("loading resident operands for {}", backend.name());
        let mut session = DecodeSession::new(plan, store, backend)?;

        for (index, query) in queries.iter().enumerate().skip(start) {
            // Each row is an independent sequence. Refresh the backend's no-op
            // residency hint before reset so sustained scans do not fall into the
            // unified-memory unwire/rewire slowdown observed after ~13 rows on
            // the 60 GB Glimmer container.
            if index > start {
                session.prepare_residency();
            }
            session.reset();
            let (&last, prefix) = query
                .token_ids
                .split_last()
                .ok_or_else(|| format!("query `{}` has no token ids", query.id))?;
            for &token in prefix {
                session.step(token)?;
            }
            let mut observer = OracleObserver::new(
                backend,
                residual_layers,
                seam_layers,
                seam_kinds,
                blocks,
                objects.len(),
                contribution_engine,
            );
            let prompt_output = session.step_observed(last, &mut observer)?;
            for &layer in residual_layers {
                let residual = observer.residuals.get(&layer).ok_or_else(|| {
                    format!(
                        "query `{}` produced no residual tap for layer {layer}",
                        query.id
                    )
                })?;
                write_f32s(&mut residual_writer, residual)?;
            }
            write_f32s(&mut contribution_writer, &observer.contributions)?;
            if let Some(writer) = &mut seam_writer {
                for &layer in seam_layers {
                    for &kind in seam_kinds {
                        let values = observer.seams.get(&(layer, kind)).ok_or_else(|| {
                            format!(
                                "query `{}` produced no {} seam for layer {layer}",
                                query.id,
                                kind.as_str()
                            )
                        })?;
                        if values.len() != hidden_size {
                            return Err(format!(
                                "query `{}` layer {layer} {} seam has width {}, expected {hidden_size}",
                                query.id,
                                kind.as_str(),
                                values.len()
                            )
                            .into());
                        }
                        write_f32s(writer, values)?;
                    }
                }
            }
            let answer = greedy_answer(
                &mut session,
                prompt_output.logits,
                &query.expected_token_ids,
            )?;
            // The progress row is the commit record. A resumed run truncates any
            // binary tail beyond the final committed row before appending.
            residual_writer.flush()?;
            contribution_writer.flush()?;
            if let Some(writer) = &mut seam_writer {
                writer.flush()?;
            }
            serde_json::to_writer(
                &mut progress_writer,
                &ProgressRow {
                    sample_id: query.id.clone(),
                    answer: answer.clone(),
                },
            )?;
            progress_writer.write_all(b"\n")?;
            progress_writer.flush()?;
            answer_outcomes.push(answer);
            if (index + 1) % 10 == 0 || index + 1 == queries.len() {
                eprintln!("captured {:>6}/{}", index + 1, queries.len());
            }
        }
    }
    residual_writer.flush()?;
    contribution_writer.flush()?;
    if let Some(writer) = &mut seam_writer {
        writer.flush()?;
    }

    let manifest = CaptureManifest {
        schema: SCHEMA_V1.into(),
        model: model.to_string(),
        hidden_size,
        residual_layers: residual_layers.to_vec(),
        contribution_semantics: CONTRIBUTION_SEMANTICS_V1.into(),
        contribution_engine: Some(contribution_engine_name(contribution_engine).into()),
        residuals_file: RESIDUALS_FILE.into(),
        contributions_file: CONTRIBUTIONS_FILE.into(),
        seams_file: (!seam_layers.is_empty()).then(|| SEAMS_FILE.into()),
        seam_layers: seam_layers.to_vec(),
        seam_kinds: seam_kinds
            .iter()
            .map(|kind| kind.as_str().to_string())
            .collect(),
        samples: queries
            .iter()
            .zip(answer_outcomes)
            .map(|(query, answer)| SampleDescriptor {
                id: query.id.clone(),
                semantic_id: query.semantic_id.clone(),
                operation_id: query.operation_id.clone(),
                wording_id: query.wording_id.clone(),
                split: query.split.clone(),
                group_id: query.group_id.clone(),
                regime: query.regime.clone(),
                alias_id: query.alias_id.clone(),
                layout_id: query.layout_id.clone(),
                token_count: Some(query.token_ids.len()),
                expected_token_ids: answer.expected_token_ids,
                generated_token_ids: answer.generated_token_ids,
                answer_exact: answer.answer_exact,
                answer_token_accuracy: answer.answer_token_accuracy,
            })
            .collect(),
        objects: objects.to_vec(),
    };
    // Completion marker: evaluators refuse a partial directory because it has
    // no manifest. Publish it only after both planes have been flushed.
    std::fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!("capture complete: {} samples", queries.len());
    println!("backend: {}", backend.name());
    println!("residual layers: {residual_layers:?}");
    println!("model objects: {}", objects.len());
    if !block_channel_schemas.is_empty() {
        println!("page channel schemas: {block_channel_schemas:?}");
    }
    if !seam_layers.is_empty() {
        println!("seam layers: {seam_layers:?}");
        println!(
            "seam kinds: {:?}",
            seam_kinds
                .iter()
                .map(|kind| kind.as_str())
                .collect::<Vec<_>>()
        );
    }
    println!("contribution engine: {contribution_engine:?}");
    println!("wrote {}", output.display());
    Ok(())
}

fn contribution_engine_name(engine: ContributionEngine) -> &'static str {
    match engine {
        ContributionEngine::Exact => "cpu-exact-f64-reduction",
        ContributionEngine::Fast => "device-fast-f32-reduction",
    }
}

fn append_writer(path: impl AsRef<Path>) -> std::io::Result<BufWriter<std::fs::File>> {
    Ok(BufWriter::new(
        OpenOptions::new().create(true).append(true).open(path)?,
    ))
}

fn open_checkpoint(
    output: &Path,
    expected: &CaptureCheckpoint,
    residual_stride: usize,
    contribution_stride: usize,
    seam_stride: usize,
    resume: bool,
) -> Result<(usize, Vec<AnswerOutcome>, BufWriter<std::fs::File>), Box<dyn std::error::Error>> {
    let state_path = output.join(CHECKPOINT_FILE);
    let progress_path = output.join(PROGRESS_FILE);
    let residual_path = output.join(RESIDUALS_FILE);
    let contribution_path = output.join(CONTRIBUTIONS_FILE);
    let seam_path = output.join(SEAMS_FILE);
    if !resume {
        let state_tmp = output.join(format!("{CHECKPOINT_FILE}.tmp"));
        std::fs::write(&state_tmp, serde_json::to_vec_pretty(expected)?)?;
        std::fs::rename(state_tmp, &state_path)?;
        std::fs::File::create(&progress_path)?;
        std::fs::File::create(&residual_path)?;
        std::fs::File::create(&contribution_path)?;
        if seam_stride > 0 {
            std::fs::File::create(&seam_path)?;
        }
        return Ok((0, Vec::new(), append_writer(progress_path)?));
    }
    if !state_path.exists() {
        return Err(format!(
            "cannot resume {}: missing {}; pre-resume partial captures are not recoverable",
            output.display(),
            CHECKPOINT_FILE
        )
        .into());
    }
    let actual: CaptureCheckpoint = serde_json::from_slice(&std::fs::read(&state_path)?)?;
    if actual != *expected {
        return Err(
            "resume configuration or query fixture does not match capture-state.json".into(),
        );
    }
    let mut outcomes = Vec::new();
    let progress_bytes = std::fs::read(&progress_path)?;
    let committed_bytes = progress_bytes
        .iter()
        .rposition(|&byte| byte == b'\n')
        .map_or(0, |index| index + 1);
    for (line_number, line) in progress_bytes[..committed_bytes]
        .split(|&byte| byte == b'\n')
        .enumerate()
    {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let row: ProgressRow = serde_json::from_slice(line)
            .map_err(|error| format!("{}:{}: {error}", progress_path.display(), line_number + 1))?;
        let query = expected.queries.get(outcomes.len()).ok_or_else(|| {
            format!(
                "{} commits more rows than the fixture",
                progress_path.display()
            )
        })?;
        if row.sample_id != query.id {
            return Err(format!(
                "{}:{} commits `{}` but fixture expects `{}`",
                progress_path.display(),
                line_number + 1,
                row.sample_id,
                query.id
            )
            .into());
        }
        outcomes.push(row.answer);
    }
    OpenOptions::new()
        .write(true)
        .open(&progress_path)?
        .set_len(committed_bytes as u64)?;
    let committed = outcomes.len();
    truncate_plane(&residual_path, committed, residual_stride)?;
    truncate_plane(&contribution_path, committed, contribution_stride)?;
    if seam_stride > 0 {
        truncate_plane(&seam_path, committed, seam_stride)?;
    }
    Ok((committed, outcomes, append_writer(progress_path)?))
}

fn truncate_plane(path: &Path, rows: usize, stride: usize) -> Result<(), String> {
    let wanted =
        rows.checked_mul(stride)
            .ok_or_else(|| format!("{} committed size overflow", path.display()))? as u64;
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?;
    let actual = file
        .metadata()
        .map_err(|error| format!("stat {}: {error}", path.display()))?
        .len();
    if actual < wanted {
        return Err(format!(
            "{} is shorter than {rows} committed rows: {actual} < {wanted}",
            path.display()
        ));
    }
    file.set_len(wanted)
        .map_err(|error| format!("truncate {}: {error}", path.display()))
}

fn greedy_answer<B: PlanBackend>(
    session: &mut DecodeSession<'_, B>,
    first_logits: Option<Vec<f32>>,
    expected: &[u32],
) -> Result<AnswerOutcome, VindexError> {
    if expected.is_empty() {
        return Ok(AnswerOutcome::default());
    }
    let mut logits = first_logits.ok_or_else(|| {
        VindexError::Parse("ROUTE-1 answer scoring requires an output head".to_string())
    })?;
    let mut generated = Vec::with_capacity(expected.len());
    for position in 0..expected.len() {
        let token = finite_argmax(&logits)?;
        generated.push(token);
        if position + 1 < expected.len() {
            logits = session.step(token)?.logits.ok_or_else(|| {
                VindexError::Parse("ROUTE-1 answer scoring lost its output head".to_string())
            })?;
        }
    }
    let correct = generated
        .iter()
        .zip(expected)
        .filter(|(actual, wanted)| actual == wanted)
        .count();
    Ok(AnswerOutcome {
        expected_token_ids: Some(expected.to_vec()),
        generated_token_ids: Some(generated.clone()),
        answer_exact: Some(generated == expected),
        answer_token_accuracy: Some(correct as f64 / expected.len() as f64),
    })
}

fn finite_argmax(values: &[f32]) -> Result<u32, VindexError> {
    if values.is_empty() || values.len() > u32::MAX as usize {
        return Err(VindexError::Parse(
            "ROUTE-1 logits have invalid vocabulary size".to_string(),
        ));
    }
    let mut best = None;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(VindexError::Parse(format!(
                "ROUTE-1 logits contain non-finite value at token {index}"
            )));
        }
        if best.is_none_or(|(_, score)| value > score) {
            best = Some((index, value));
        }
    }
    Ok(best.expect("non-empty logits").0 as u32)
}

fn object_table(
    plan: &ComponentOpPlan,
    layers: &[usize],
    block_channel_schemas: &[usize],
) -> Result<(Vec<ObjectDescriptor>, HashMap<usize, Vec<BlockSchema>>), String> {
    if block_channel_schemas.is_empty() {
        return Err("at least one block-channel schema is required".to_string());
    }
    let hidden = plan
        .embedding
        .as_ref()
        .ok_or_else(|| "plan has no embedding".to_string())?
        .table
        .shape[1];
    let mut objects = Vec::new();
    let mut blocks = HashMap::new();
    for &layer_index in layers {
        let layer = &plan.layers[layer_index];
        let up_bytes = dtype_bytes(&layer.ffn.up.dtype)?;
        let down_bytes = dtype_bytes(&layer.ffn.down.dtype)?;
        let gate_bytes = layer
            .ffn
            .gate
            .as_ref()
            .map(|gate| dtype_bytes(&gate.dtype))
            .transpose()?
            .unwrap_or(0);
        let mut layer_schemas = Vec::new();
        for &block_channels in block_channel_schemas {
            let mut schema_blocks = Vec::new();
            for start in (0..layer.ffn.intermediate_size).step_by(block_channels) {
                let end = (start + block_channels).min(layer.ffn.intermediate_size);
                let width = end - start;
                let parameters = (hidden as u64)
                    .checked_mul(width as u64)
                    .ok_or_else(|| "object parameter count overflow".to_string())?;
                let byte_count = parameters
                    .checked_mul((up_bytes + down_bytes + gate_bytes) as u64)
                    .ok_or_else(|| "object byte count overflow".to_string())?;
                let object_index = objects.len();
                let id = if block_channel_schemas.len() == 1 {
                    format!("layer.{layer_index}.ffn.channels.{start}-{end}")
                } else {
                    format!("layer.{layer_index}.ffn.page.{block_channels}.channels.{start}-{end}")
                };
                objects.push(ObjectDescriptor {
                    id,
                    layer: layer_index,
                    kind: "ffn_channel_block".into(),
                    byte_count,
                    operand: Some(format!(
                        "{}::{}",
                        layer.ffn.down.object, layer.ffn.down.tensor
                    )),
                    channel_start: Some(start),
                    channel_end: Some(end),
                    block_channels: Some(block_channels),
                });
                schema_blocks.push(Block {
                    object: object_index,
                    start,
                    end,
                });
            }
            layer_schemas.push(BlockSchema {
                block_channels,
                blocks: schema_blocks,
            });
        }
        blocks.insert(layer_index, layer_schemas);
    }
    Ok((objects, blocks))
}

fn dtype_bytes(dtype: &str) -> Result<usize, String> {
    match dtype {
        "F32" => Ok(4),
        "BF16" | "F16" => Ok(2),
        other => Err(format!(
            "no exact per-channel byte accounting for dtype `{other}`"
        )),
    }
}

fn read_queries(path: &Path) -> Result<Vec<QueryFixture>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mut rows = Vec::new();
    for (line_number, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: QueryFixture = serde_json::from_str(&line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), line_number + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

fn validate_queries(rows: &[QueryFixture]) -> Result<(), String> {
    if rows.is_empty() {
        return Err("query fixture has no rows".into());
    }
    let mut ids = HashSet::new();
    let mut history = 0usize;
    let mut query = 0usize;
    for row in rows {
        if row.id.trim().is_empty()
            || row.semantic_id.trim().is_empty()
            || row.wording_id.trim().is_empty()
        {
            return Err("query id, semantic_id and wording_id must not be empty".into());
        }
        if !ids.insert(&row.id) {
            return Err(format!("duplicate query id `{}`", row.id));
        }
        if row.token_ids.is_empty() {
            return Err(format!("query `{}` has no token_ids", row.id));
        }
        match row.split.as_str() {
            "history" => history += 1,
            "query" => query += 1,
            other => {
                return Err(format!(
                    "query `{}` has split `{other}`; expected history or query",
                    row.id
                ))
            }
        }
    }
    if history == 0 || query == 0 {
        return Err(format!(
            "fixture needs history and query rows; found history={history}, query={query}"
        ));
    }
    Ok(())
}

fn parse_layers(spec: &str, num_layers: usize) -> Result<Vec<usize>, String> {
    let mut layers = Vec::new();
    for item in spec.split(',') {
        let item = item.trim();
        if let Some((start, end)) = item.split_once('-') {
            let start: usize = start
                .parse()
                .map_err(|_| format!("invalid layer range `{item}`"))?;
            let end: usize = end
                .parse()
                .map_err(|_| format!("invalid layer range `{item}`"))?;
            if start > end {
                return Err(format!("invalid descending layer range `{item}`"));
            }
            layers.extend(start..=end);
        } else {
            layers.push(
                item.parse()
                    .map_err(|_| format!("invalid layer `{item}`"))?,
            );
        }
    }
    layers.sort_unstable();
    layers.dedup();
    if layers.is_empty() {
        return Err("layer selection must not be empty".into());
    }
    if let Some(&layer) = layers.iter().find(|&&layer| layer >= num_layers) {
        return Err(format!(
            "layer {layer} is outside the plan's 0..{} range",
            num_layers.saturating_sub(1)
        ));
    }
    Ok(layers)
}

fn prepare_output(path: &Path, resume: bool) -> Result<(), Box<dyn std::error::Error>> {
    if path.exists() {
        if path.join("manifest.json").exists() {
            return Err(format!("capture {} is already complete", path.display()).into());
        }
        if !resume && std::fs::read_dir(path)?.next().is_some() {
            return Err(format!("output directory {} is not empty", path.display()).into());
        }
    } else {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

fn write_f32s(writer: &mut impl Write, values: &[f32]) -> std::io::Result<()> {
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::format::vindex3::opplan::exec::backend::WeightSlice;

    #[test]
    fn layer_parser_accepts_ranges_and_refuses_out_of_plan_layers() {
        assert_eq!(parse_layers("1,3-5,4", 8).unwrap(), vec![1, 3, 4, 5]);
        assert!(parse_layers("7-9", 8).is_err());
        assert!(parse_layers("4-2", 8).is_err());
    }

    #[test]
    fn query_fixture_requires_disjoint_roles_and_nonempty_tokens() {
        let history = QueryFixture {
            id: "h".into(),
            semantic_id: "s".into(),
            operation_id: Some("op".into()),
            wording_id: "w".into(),
            split: "history".into(),
            group_id: None,
            regime: None,
            alias_id: None,
            layout_id: None,
            token_ids: vec![1],
            expected_token_ids: Vec::new(),
        };
        let mut query = history.clone();
        query.id = "q".into();
        query.split = "query".into();
        assert!(validate_queries(&[history.clone(), query.clone()]).is_ok());
        query.token_ids.clear();
        assert!(validate_queries(&[history, query]).is_err());
    }

    #[test]
    fn answer_argmax_is_finite_and_deterministic() {
        assert_eq!(finite_argmax(&[-2.0, 4.0, 4.0]).unwrap(), 1);
        assert!(finite_argmax(&[]).is_err());
        assert!(finite_argmax(&[0.0, f32::NAN]).is_err());
    }

    #[test]
    fn resume_uses_progress_as_authority_and_truncates_binary_tails() {
        let dir = tempfile::tempdir().unwrap();
        let queries = vec![
            QueryFixture {
                id: "h".into(),
                semantic_id: "s-h".into(),
                operation_id: Some("op".into()),
                wording_id: "w-h".into(),
                split: "history".into(),
                group_id: None,
                regime: None,
                alias_id: None,
                layout_id: None,
                token_ids: vec![1],
                expected_token_ids: vec![2],
            },
            QueryFixture {
                id: "q".into(),
                semantic_id: "s-q".into(),
                operation_id: Some("op".into()),
                wording_id: "w-q".into(),
                split: "query".into(),
                group_id: None,
                regime: None,
                alias_id: None,
                layout_id: None,
                token_ids: vec![3],
                expected_token_ids: vec![4],
            },
        ];
        let checkpoint = CaptureCheckpoint {
            schema: CHECKPOINT_SCHEMA_V1.into(),
            model: "mini".into(),
            backend: "reference".into(),
            hidden_size: 2,
            residual_layers: vec![0],
            objects: vec![("object".into(), 1, 8)],
            contribution_engine: "cpu-exact-f64-reduction".into(),
            block_channels: 1,
            block_channel_schemas: Vec::new(),
            seam_layers: Vec::new(),
            seam_kinds: Vec::new(),
            queries,
        };
        let (_start, _answers, progress) =
            open_checkpoint(dir.path(), &checkpoint, 8, 4, 0, false).unwrap();
        drop(progress);
        let outcome = AnswerOutcome {
            expected_token_ids: Some(vec![2]),
            generated_token_ids: Some(vec![2]),
            answer_exact: Some(true),
            answer_token_accuracy: Some(1.0),
        };
        let mut progress = append_writer(dir.path().join(PROGRESS_FILE)).unwrap();
        serde_json::to_writer(
            &mut progress,
            &ProgressRow {
                sample_id: "h".into(),
                answer: outcome,
            },
        )
        .unwrap();
        progress.write_all(b"\n").unwrap();
        progress.flush().unwrap();
        drop(progress);
        let mut partial = append_writer(dir.path().join(PROGRESS_FILE)).unwrap();
        partial.write_all(b"{\"sample_id\":\"q\"").unwrap();
        partial.flush().unwrap();
        drop(partial);
        std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join(RESIDUALS_FILE))
            .unwrap()
            .set_len(24)
            .unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join(CONTRIBUTIONS_FILE))
            .unwrap()
            .set_len(12)
            .unwrap();

        let (start, answers, progress) =
            open_checkpoint(dir.path(), &checkpoint, 8, 4, 0, true).unwrap();
        drop(progress);
        assert_eq!(start, 1);
        assert_eq!(answers.len(), 1);
        assert!(std::fs::read(dir.path().join(PROGRESS_FILE))
            .unwrap()
            .ends_with(b"\n"));
        assert_eq!(
            std::fs::metadata(dir.path().join(RESIDUALS_FILE))
                .unwrap()
                .len(),
            8
        );
        assert_eq!(
            std::fs::metadata(dir.path().join(CONTRIBUTIONS_FILE))
                .unwrap()
                .len(),
            4
        );
    }

    #[test]
    fn resume_truncates_the_optional_seam_plane_to_the_commit_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let query = QueryFixture {
            id: "h".into(),
            semantic_id: "s".into(),
            operation_id: Some("op".into()),
            wording_id: "w".into(),
            split: "history".into(),
            group_id: None,
            regime: None,
            alias_id: None,
            layout_id: None,
            token_ids: vec![1],
            expected_token_ids: vec![2],
        };
        let checkpoint = CaptureCheckpoint {
            schema: CHECKPOINT_SCHEMA_V1.into(),
            model: "mini".into(),
            backend: "reference".into(),
            hidden_size: 2,
            residual_layers: vec![0],
            objects: vec![("object".into(), 0, 8)],
            contribution_engine: "cpu-exact-f64-reduction".into(),
            block_channels: 1,
            block_channel_schemas: Vec::new(),
            seam_layers: vec![0],
            seam_kinds: vec!["attention_input".into()],
            queries: vec![query],
        };
        let (_start, _answers, progress) =
            open_checkpoint(dir.path(), &checkpoint, 8, 4, 8, false).unwrap();
        drop(progress);
        std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join(SEAMS_FILE))
            .unwrap()
            .set_len(16)
            .unwrap();

        let (start, answers, progress) =
            open_checkpoint(dir.path(), &checkpoint, 8, 4, 8, true).unwrap();
        drop(progress);
        assert_eq!(start, 0);
        assert!(answers.is_empty());
        assert_eq!(
            std::fs::metadata(dir.path().join(SEAMS_FILE))
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn observer_records_raw_down_block_energy_without_changing_the_call() {
        let backend = ReferenceBackend::new();
        let blocks = HashMap::from([(
            0,
            vec![
                BlockSchema {
                    block_channels: 1,
                    blocks: vec![
                        Block {
                            object: 0,
                            start: 0,
                            end: 1,
                        },
                        Block {
                            object: 1,
                            start: 1,
                            end: 2,
                        },
                        Block {
                            object: 2,
                            start: 2,
                            end: 3,
                        },
                        Block {
                            object: 3,
                            start: 3,
                            end: 4,
                        },
                    ],
                },
                BlockSchema {
                    block_channels: 2,
                    blocks: vec![
                        Block {
                            object: 4,
                            start: 0,
                            end: 2,
                        },
                        Block {
                            object: 5,
                            start: 2,
                            end: 4,
                        },
                    ],
                },
            ],
        )]);
        let mut observer = OracleObserver::new(
            &backend,
            &[0],
            &[],
            &[],
            &blocks,
            6,
            ContributionEngine::Exact,
        );
        let x = [1.0, 0.0];
        let gate = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let up = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let down = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let call = FfnCall {
            x: &x,
            hidden: 2,
            intermediate: 4,
            gate: Some(WeightSlice::F32(&gate)),
            up: WeightSlice::F32(&up),
            down: WeightSlice::F32(&down),
            activation: Activation::Silu,
        };
        observer.layer_input(0, &x).unwrap();
        observer.ffn_call(0, &call).unwrap();
        let silu = larql_compute::cpu::ops::geglu::silu(1.0);
        assert_eq!(observer.residuals[&0], x);
        for (actual, expected) in observer.contributions[..4]
            .iter()
            .zip([1.0, 4.0, 9.0, 16.0])
        {
            assert!((*actual - expected * silu * silu).abs() < 1e-5);
        }
        assert!((observer.contributions[4] - 9.0 * silu * silu).abs() < 1e-5);
        assert!((observer.contributions[5] - 49.0 * silu * silu).abs() < 1e-5);
        assert_ne!(
            observer.contributions[4],
            observer.contributions[0] + observer.contributions[1]
        );
    }
}
