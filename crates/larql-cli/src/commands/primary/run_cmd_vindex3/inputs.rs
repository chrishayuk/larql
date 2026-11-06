//! Image inputs and explicit continuation providers for the V3 run surface.
use super::*;
use larql_compute::forward::{EmbeddingChunk, PositionScheme};
use larql_inference::vindex3::input::{CachedInputSession, InputPosition, ReplaySession};
use larql_inference::vindex3::LogitsSession;

use crate::commands::primary::continuation::{select_in, REPLAY_ENGINE};

pub(super) fn generate<B: PlanBackend>(
    model: &ResidentModel<'_, B>,
    ids: &[u32],
    out: &mut dyn Write,
    status: &mut dyn Write,
) -> Result<Vec<u32>, BoxErr> {
    let inputs = if model.args.image.is_empty() {
        ids.iter().copied().map(InputPosition::Token).collect()
    } else {
        image_inputs(model, ids)?
    };
    if !model.args.v3_ffn_shards.is_empty() {
        let continuation = select_in(
            model.continuations,
            model.plan,
            &model.continuation_choice(),
        )?;
        let mode = format!(
            "local-kv-remote-ffn over {}",
            continuation.authority().identity
        );
        let mut session = larql_inference::vindex3::dense_ffn::DenseFfnSession::new(
            model.plan,
            model.ops,
            model.backend,
            &continuation,
        )?;
        let logits = session.extend_inputs(&inputs)?;
        emit(model, &mut session, logits, ids, out, status, &mode)
    } else if !model.args.v3_shards.is_empty() {
        use larql_inference::vindex3::distributed::{artifact_identity, DistributedSession};
        let token = model
            .args
            .v3_shard_token_env
            .as_deref()
            .map(std::env::var)
            .transpose()?;
        let transport = larql_router::vindex3::HttpLayerShards::connect(
            &model.args.v3_shards,
            token.as_deref(),
        )?;
        let identity = artifact_identity(model.container, model.plan)?;
        let mut session = DistributedSession::new(
            model.plan,
            model.ops,
            model.backend,
            &identity,
            model.store.into(),
            transport,
        )?;
        let logits = session.extend_inputs(&inputs)?;
        emit(
            model,
            &mut session,
            logits,
            ids,
            out,
            status,
            "distributed-prefix",
        )
    } else if model.args.kv_cache == KvCacheKind::None
        || model.args.engine.as_deref() == Some(REPLAY_ENGINE)
    {
        let continuation = select_in(
            model.continuations,
            model.plan,
            &model.continuation_choice(),
        )?;
        let mode = format!("{REPLAY_ENGINE} over {}", continuation.authority().identity);
        let mut session = ReplaySession::new(model.plan, model.ops, model.backend, continuation);
        let logits = session.extend_inputs(&inputs)?;
        emit(model, &mut session, logits, ids, out, status, &mode)
    } else {
        let continuation = select_in(
            model.continuations,
            model.plan,
            &model.continuation_choice(),
        )?;
        let mut state = continuation.build();
        let mut session =
            CachedInputSession::new(model.plan, model.ops, model.backend, &mut *state)?;
        let logits = session.extend_inputs(&inputs)?;
        emit(
            model,
            &mut session,
            logits,
            ids,
            out,
            status,
            &continuation.authority().identity.to_string(),
        )
    }
}

pub(super) fn emit<B: PlanBackend, S: LogitsSession>(
    model: &ResidentModel<'_, B>,
    session: &mut S,
    logits: Vec<f32>,
    ids: &[u32],
    out: &mut dyn Write,
    status: &mut dyn Write,
    mode: &str,
) -> Result<Vec<u32>, BoxErr> {
    let mut detok = Detokenizer::new(model.tokenizer);
    detok.seed(ids);
    let prompt_positions = session.position();
    let mut logits = logits;
    let mut generated = Vec::new();
    for index in 0..model.args.max_tokens {
        let (id, _) = super::super::vindex3_cmd::decode::argmax(&logits)
            .ok_or("empty or non-finite logits")?;
        let id = id as u32;
        if model.eos.eos_token_ids.contains(&id) {
            break;
        }
        let delta = detok.push(id);
        if model.eos.is_eos_with_tokenizer(id, &delta, model.tokenizer) {
            break;
        }
        out.write_all(delta.as_bytes())?;
        out.flush()?;
        generated.push(id);
        if index + 1 < model.args.max_tokens {
            logits = session.step(id)?;
        }
    }
    writeln!(out)?;
    if model.args.emit_ids {
        writeln!(status, "[{}] prompt ids: {:?}", model.engine, ids)?;
        writeln!(status, "[{}] generated ids: {:?}", model.engine, generated)?;
    }
    if model.args.verbose {
        writeln!(
            status,
            "[{}] continuation={mode}; prompt positions={}",
            model.engine, prompt_positions
        )?;
    }
    Ok(generated)
}

pub(super) fn image_inputs<B: PlanBackend>(
    model: &ResidentModel<'_, B>,
    ids: &[u32],
) -> Result<Vec<InputPosition>, BoxErr> {
    if model.args.metal {
        return Err("V3 image input currently requires CPU execution".into());
    }
    let dir = model
        .args
        .mm_weights
        .as_deref()
        .ok_or("--image requires --mm-weights pointing to the vision/projector checkpoint")?;
    let arch = larql_models::detect::detect_architecture(dir)?;
    // The vision source must describe the same model as the container,
    // at the same width. Which protocol it speaks is read from its own
    // declaration below, never from its family name.
    if arch.family() != model.family || arch.config().hidden_size != model.ops.hidden() {
        return Err(format!(
            "V3 image input requires a vision source of the container's own family \
             (`{}`) with matching LM hidden width; the source is `{}`",
            model.family,
            arch.family()
        )
        .into());
    }
    if model.ops.carries_hyper_connection() || model.ops.carries_attention_residual() {
        return Err("V3 image input requires a single residual stream".into());
    }
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let vision = larql_models::encoders::vision_tower::VisionConfig::from_json(
        raw.get("vision_config").ok_or("vision_config missing")?,
    )?;
    let mm = arch
        .multimodal()
        .ok_or("source declares no multimodal protocol")?;
    let larql_models::TokenBudget::Fixed(n) = mm.image_token_budget() else {
        return Err("V3 image input requires a fixed sequential image token budget".into());
    };
    let tower = larql_models::encoders::vision_tower::load_vision_tower_from_safetensors(
        dir,
        vision.clone(),
    )?;
    let projector = larql_models::connectors::projector::load_projector_from_safetensors(dir)?;
    if projector.text_hidden() != model.ops.hidden() {
        return Err("vision projector output width disagrees with the VINDEX3 component".into());
    }
    use larql_models::ModalEncoder as _;
    let encoder = larql_compute::encoders::vision_tower::VisionEncoder::new(&tower);
    // The protocol names the encoder the LM was trained against; the
    // host verifies the one it loaded is that encoder before wiring them.
    if mm.vision_encoder() != Some(encoder.family()) {
        return Err(format!(
            "the source declares vision encoder {:?}, but its weights load as `{}`",
            mm.vision_encoder(),
            encoder.family()
        )
        .into());
    }
    let connector =
        larql_compute::connectors::projector::VisionProjector::new(&projector, &vision, n)?;
    let plan = crate::commands::primary::run_cmd_image::prepare_multimodal_input(
        &*arch,
        &encoder,
        &connector,
        vision.image_size,
        &model.args.image,
        ids,
    )?;
    from_embedding_plan(plan)
}

pub(super) fn from_embedding_plan(
    plan: larql_compute::forward::EmbeddingPlan,
) -> Result<Vec<InputPosition>, BoxErr> {
    if !matches!(plan.positions, PositionScheme::Sequential) {
        return Err("V3 image input supports sequential positions only".into());
    }
    Ok(plan
        .chunks
        .into_iter()
        .flat_map(|chunk| match chunk {
            EmbeddingChunk::Tokens(ids) => ids
                .into_iter()
                .map(InputPosition::Token)
                .collect::<Vec<_>>(),
            EmbeddingChunk::Precomputed { rows, .. } => rows
                .rows()
                .into_iter()
                .map(|r| InputPosition::Embedding(r.to_vec()))
                .collect(),
        })
        .collect())
}
