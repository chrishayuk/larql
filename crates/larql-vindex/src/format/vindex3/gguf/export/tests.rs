//! The assembled export, end to end: a file that leaves, re-read by the
//! independent reader, and every refusal on the way there named.

use std::path::Path;

use larql_models::loading::gguf::{GgufFile, GgufValue};
use serde_json::json;

use super::*;
use crate::format::filenames::TOKENIZER_JSON;
use crate::format::vindex3::encode::checkpoint::snapshot_checkpoint_capabilities;
use crate::format::vindex3::encode::SYSTEM_GRAPH_JSON;
use crate::format::vindex3::fixtures::{
    encode_fixture_container, hybrid_lllf_f32_model, hybrid_lllf_f32_model_with_layers,
    norm_values, ShardBuilder, DENSE_HEAD_DIM, DENSE_HIDDEN, DENSE_VOCAB, HYBRID_CADENCE,
};
use crate::format::vindex3::gguf::emit::TYPE_F32;
use crate::format::vindex3::gguf::metadata::{ARCHITECTURE, ROPE_SECTION_SLOTS};
use crate::format::vindex3::gguf::vocab::VOCAB_PRE_BY_REGEX;

/// Two attending layers: the least depth from which the target's
/// `full_attention_interval` can be read rather than guessed.
const QWEN35_LAYERS: usize = 2 * HYBRID_CADENCE;
/// Half the head rotates — the fixture's `DENSE_HEAD_DIM` of 8 leaves
/// four rotary dims, two frequency pairs, split across the MRoPE axes.
const ROTARY_FRACTION: f64 = 0.5;
const MROPE_SECTION: [u32; 3] = [1, 1, 0];
/// The tokenizer the fixtures snapshot: two base tokens, one merge, one
/// special end-of-sequence token.
const BASE_TOKENS: [&str; 2] = ["a", "b"];
const EOS: &str = "<eos>";

/// A container the test encoded itself, kept alive with its checkpoint.
struct Encoded {
    _checkpoint: tempfile::TempDir,
    container: tempfile::TempDir,
}

impl Encoded {
    fn root(&self) -> &Path {
        self.container.path()
    }
}

/// Encode `write` into a container, then copy the checkpoint's
/// capability snapshot in exactly as `encode_checkpoint` does.
fn encode(write: impl FnOnce(&Path), name: &str) -> Encoded {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, checkpoint.path(), container.path(), name);
    snapshot_checkpoint_capabilities(checkpoint.path(), container.path()).unwrap();
    Encoded {
        _checkpoint: checkpoint,
        container,
    }
}

fn edit_config(dir: &Path, edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) {
    let path = dir.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    edit(config.as_object_mut().unwrap());
    std::fs::write(&path, config.to_string()).unwrap();
}

/// Declare MRoPE the way a qwen3_5 config does: a partial rotary
/// fraction and a `rope_parameters` block carrying the sections.
fn declare_mrope(dir: &Path) {
    edit_config(dir, |c| {
        c.insert("partial_rotary_factor".into(), json!(ROTARY_FRACTION));
        c.insert(
            "rope_parameters".into(),
            json!({
                "rope_theta": c["rope_theta"],
                "rope_type": "default",
                "partial_rotary_factor": ROTARY_FRACTION,
                "mrope_interleaved": true,
                "mrope_section": MROPE_SECTION,
            }),
        );
    });
}

/// A byte-level BPE tokenizer whose split regex is the one llama.cpp
/// calls `qwen35`.
fn write_tokenizer(dir: &Path) {
    let regex = VOCAB_PRE_BY_REGEX
        .iter()
        .find(|(id, _)| *id == ARCHITECTURE)
        .map(|(_, r)| *r)
        .unwrap();
    let vocab: serde_json::Map<String, serde_json::Value> = BASE_TOKENS
        .iter()
        .enumerate()
        .map(|(i, t)| (t.to_string(), json!(i)))
        .collect();
    let tokenizer = json!({
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": regex}},
        "model": {"type": "BPE", "vocab": vocab, "merges": [BASE_TOKENS.join(" ")]},
        "added_tokens": [{"content": EOS, "id": BASE_TOKENS.len(), "special": true}]
    });
    std::fs::write(dir.join(TOKENIZER_JSON), tokenizer.to_string()).unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        json!({ "eos_token": EOS }).to_string(),
    )
    .unwrap();
}

fn attends(layer: usize) -> bool {
    layer % HYBRID_CADENCE == HYBRID_CADENCE - 1
}

/// Per-head Q/K norms on the attending layers, in a second shard so the
/// base fixture stays exactly what every other test sees.
fn add_qk_norms(dir: &Path) {
    let mut shard = ShardBuilder::new();
    for layer in (0..QWEN35_LAYERS).filter(|l| attends(*l)) {
        for (i, name) in ["q_norm", "k_norm"].iter().enumerate() {
            shard.push(
                &format!("model.layers.{layer}.self_attn.{name}.weight"),
                &[DENSE_HEAD_DIM],
                &norm_values(DENSE_HEAD_DIM, (layer * 2 + i) as u64),
            );
        }
    }
    shard.write_as(dir, "model-qk-norm.safetensors");
}

/// The hybrid fixture made into everything qwen35 can carry: two
/// `LLLF` blocks, Q/K norms, MRoPE, an untied head declared as such,
/// and a tokenizer.
fn qwen35_checkpoint(dir: &Path) {
    hybrid_lllf_f32_model_with_layers(dir, QWEN35_LAYERS);
    add_qk_norms(dir);
    declare_mrope(dir);
    edit_config(dir, |c| {
        c.insert("tie_word_embeddings".into(), json!(false));
    });
    write_tokenizer(dir);
}

/// The container's own operation plan, closed.
fn op_plan(root: &Path) -> ComponentOpPlan {
    let inspection = inspect_container(root, false).unwrap();
    let component = inspection.graph.primary_text_component().unwrap();
    let outcome = plan_component_ops(&inspection, root, &component.id).unwrap();
    assert!(outcome.closed(), "{:?}", outcome.defects);
    outcome.plan.unwrap()
}

/// One tensor's stored bytes, read straight from the container segment
/// the plan's operand addresses — the authority the export copied from.
fn source_f32(root: &Path, object: &str, tensor: &str) -> Vec<f32> {
    let inspection = inspect_container(root, false).unwrap();
    let (_, entry) = inspection
        .index
        .representations
        .iter()
        .find(|(id, _)| id.split('@').next() == Some(object))
        .unwrap();
    let path = root.join(&entry.segment);
    let (header, data_start) = read_segment_header(&path).unwrap();
    let t = header.tensors.iter().find(|t| t.name == tensor).unwrap();
    assert_eq!(t.dtype, "F32", "the fixture is f32 throughout");
    let bytes = std::fs::read(&path).unwrap();
    let at = (data_start + t.offset) as usize;
    f32s(&bytes[at..at + t.len as usize])
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One tensor's payload out of the emitted file, through the reader.
fn emitted_f32(gguf: &GgufFile, file: &[u8], name: &str) -> Vec<f32> {
    let info = gguf.tensor_infos.iter().find(|t| t.name() == name).unwrap();
    assert_eq!(info.tensor_type(), TYPE_F32, "`{name}` is stored F32");
    let n: u64 = info.dims().iter().product();
    let at = (gguf.data_offset + info.offset()) as usize;
    f32s(&file[at..at + n as usize * 4])
}

fn export_err(root: &Path) -> String {
    let out = tempfile::tempdir().unwrap();
    let path = out.path().join("x.gguf");
    let err = export_qwen35(root, &path).expect_err("the export must refuse");
    assert!(
        !path.exists(),
        "a refused export leaves no file to mistake for a finished one"
    );
    err.to_string()
}

// ── the file that leaves ─────────────────────────────────────────────

/// **The whole pipeline, to a file, and back.** A container the test
/// encoded itself exports; the GGUF re-reads through the independent
/// reader with the target's architecture, the graph's depth and
/// cadence, the declared MRoPE sections padded to the target's slots,
/// and the tokenizer sized to the embedding. The counts the export
/// reports are the ones the file holds.
#[test]
fn a_qwen35_shaped_container_exports_and_reads_back_through_the_independent_reader() {
    let encoded = encode(qwen35_checkpoint, "export-happy");
    let out = tempfile::tempdir().unwrap();
    let path = out.path().join("model.gguf");
    let report = export_qwen35(encoded.root(), &path).expect("the export completes");

    assert_eq!(report.out, path);
    assert_eq!(report.bytes, std::fs::metadata(&path).unwrap().len());
    assert!(report.ledger.ready(), "{:?}", report.ledger.errors);
    assert_eq!(report.ledger.accounted, report.ledger.source_total);
    assert_eq!(report.ledger.geometry_reconciled, report.ledger.accounted);
    assert_eq!(
        report.selected_encoding, "canonical",
        "a container without a precision map carries its canonical bytes"
    );
    assert_eq!(report.ledger.generated_scale_tensors, 0, "nothing is NVFP4");
    assert_eq!(report.emit.tensors, report.ledger.accounted);
    assert_eq!(report.verify.tensors, report.emit.tensors);
    assert_eq!(report.verify.nvfp4_tensors, 0);
    assert_eq!(report.verify.metadata_keys, report.emit.metadata_keys);
    assert_eq!(report.vocab_tokens, BASE_TOKENS.len() + 1);
    assert_eq!(report.vocab_padded, DENSE_VOCAB - report.vocab_tokens);
    assert_eq!(report.vocab_merges, 1);

    let gguf = GgufFile::open(&path).unwrap();
    let meta = |k: &str| {
        gguf.metadata
            .get(k)
            .cloned()
            .unwrap_or_else(|| panic!("{k}"))
    };
    assert_eq!(
        meta("general.architecture"),
        GgufValue::String(ARCHITECTURE.into())
    );
    assert_eq!(
        meta("qwen35.block_count"),
        GgufValue::U32(QWEN35_LAYERS as u32)
    );
    assert_eq!(
        meta("qwen35.embedding_length"),
        GgufValue::U32(DENSE_HIDDEN as u32)
    );
    assert_eq!(
        meta("qwen35.full_attention_interval"),
        GgufValue::U32(HYBRID_CADENCE as u32),
        "the interval is read off the graph's per-layer operators"
    );
    let mut sections: Vec<GgufValue> = MROPE_SECTION.iter().map(|s| GgufValue::U32(*s)).collect();
    sections.resize(ROPE_SECTION_SLOTS, GgufValue::U32(0));
    assert_eq!(
        meta("qwen35.rope.dimension_sections"),
        GgufValue::Array(sections)
    );
    let GgufValue::Array(tokens) = meta("tokenizer.ggml.tokens") else {
        panic!("tokens are an array");
    };
    assert_eq!(tokens.len(), DENSE_VOCAB, "one token per embedding row");
    assert_eq!(tokens[0], GgufValue::String(BASE_TOKENS[0].into()));

    // Tensors: the embedding in GGUF's fastest-first dims, an untied
    // head (the fixture ships `lm_head`), recurrent operands only on
    // recurrent layers and attention operands only where it attends.
    let dims = |n: &str| {
        gguf.tensor_infos
            .iter()
            .find(|t| t.name() == n)
            .map(|t| t.dims().to_vec())
    };
    assert_eq!(
        dims("token_embd.weight"),
        Some(vec![DENSE_HIDDEN as u64, DENSE_VOCAB as u64])
    );
    assert!(
        dims("output.weight").is_some(),
        "the graph declares the head untied, so the head is required and leaves"
    );
    assert!(dims("output_norm.weight").is_some());
    let attending = HYBRID_CADENCE - 1;
    assert!(dims(&format!("blk.{attending}.attn_q.weight")).is_some());
    assert_eq!(
        dims(&format!("blk.{attending}.attn_q_norm.weight")),
        Some(vec![DENSE_HEAD_DIM as u64])
    );
    assert!(dims(&format!("blk.{attending}.ssm_a")).is_none());
    assert!(dims("blk.0.ssm_a").is_some());
    assert!(dims("blk.0.attn_q.weight").is_none());
    assert_eq!(gguf.tensor_infos.len(), report.verify.tensors);
}

/// **The bytes, not only the names.** The embedding has no transform,
/// so it leaves bit-identical to the container's segment; the log
/// decay leaves as the computed `-exp(A_log)` the target stores; and
/// a trunk norm leaves with the graph's declared weight offset folded.
#[test]
fn exported_payloads_are_the_containers_bytes_through_the_planned_arithmetic() {
    let encoded = encode(qwen35_checkpoint, "export-bytes");
    let out = tempfile::tempdir().unwrap();
    let path = out.path().join("model.gguf");
    export_qwen35(encoded.root(), &path).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let file = std::fs::read(&path).unwrap();
    let plan = op_plan(encoded.root());

    let embed = &plan.embedding.as_ref().unwrap().table;
    assert_eq!(
        emitted_f32(&gguf, &file, "token_embd.weight"),
        source_f32(encoded.root(), &embed.object, &embed.tensor),
        "an untransformed tensor passes through exactly"
    );

    let LayerAttention::GatedDelta(g) = &plan.layers[0].attention else {
        panic!("layer 0 of the LLLF cadence is recurrent");
    };
    let mut want: Vec<f32> = source_f32(encoded.root(), &g.a_log.object, &g.a_log.tensor)
        .iter()
        .map(|v| -v.exp())
        .collect();
    let mut got = emitted_f32(&gguf, &file, "blk.0.ssm_a");
    // The V-head reorder may move elements; the arithmetic may not
    // change any. Compare as multisets so this pins the value
    // transform without restating the layout.
    want.sort_by(f32::total_cmp);
    got.sort_by(f32::total_cmp);
    assert_eq!(got, want, "ssm_a holds -exp(A_log), bit-exact");

    let norm = plan.layers[0].pre_attention_norm.as_ref().unwrap();
    let source = source_f32(encoded.root(), &norm.weight.object, &norm.weight.tensor);
    let offset = inspect_container(encoded.root(), false)
        .unwrap()
        .graph
        .primary_text_component()
        .unwrap()
        .execution
        .as_ref()
        .unwrap()
        .norm
        .pre
        .weight_offset;
    let want: Vec<f32> = source.iter().map(|v| v + offset).collect();
    assert_eq!(emitted_f32(&gguf, &file, "blk.0.attn_norm.weight"), want);
}

// ── refusals on the export path ──────────────────────────────────────

/// The assembled pipeline on a real encoded container: the operation
/// plan's roles close, the walk passes coverage and geometry, and the
/// refusal that stops the fixture is the honest one — its graph
/// declares plain RoPE where qwen35's metadata needs MRoPE. Everything
/// before that gate worked, which is what this pins: roles arrive from
/// the plan, not from tensor names, on a container the test encoded
/// itself.
#[test]
fn export_reaches_the_metadata_gate_and_refuses_by_name_on_a_non_mrope_graph() {
    let encoded = encode(hybrid_lllf_f32_model, "export-gate");
    let msg = export_err(encoded.root());
    assert!(msg.contains("MRoPE"), "{msg}");
    assert!(
        msg.contains("Rope"),
        "the refusal names what the graph declares instead: {msg}"
    );
}

/// The hybrid fixture with its context length removed.
fn hybrid_without_context_length(dir: &Path) {
    hybrid_lllf_f32_model(dir);
    edit_config(dir, |c| {
        c.remove("max_position_embeddings");
    });
}

/// The preflight runs before anything is lowered or written: a graph
/// missing a fact the target needs is refused by the preflight, by
/// name, and leaves no file behind.
#[test]
fn export_runs_the_preflight_and_refuses_a_graph_missing_a_required_fact() {
    let encoded = encode(hybrid_without_context_length, "export-preflight");
    let msg = export_err(encoded.root());
    assert!(msg.contains("preflight refuses"), "{msg}");
    assert!(msg.contains("execution.context_length"), "{msg}");
}

/// One `LLLF` block declares where attention sits but not how often it
/// recurs. The metadata cannot state an interval it cannot read, and
/// the export refuses with the metadata's own error rather than
/// writing a guessed cadence.
#[test]
fn a_single_attending_layer_is_refused_at_the_metadata_not_given_a_guessed_interval() {
    let encoded = encode(
        |dir| {
            hybrid_lllf_f32_model(dir);
            declare_mrope(dir);
            write_tokenizer(dir);
        },
        "export-cadence",
    );
    let msg = export_err(encoded.root());
    assert!(msg.contains("export: metadata"), "{msg}");
    assert!(msg.contains("IrregularAttentionCadence"), "{msg}");
}

/// Edit the persisted graph of the container's `target` component in
/// place — the graph is the export's authority for every fact, so a
/// container whose graph says something else is how a defect in that
/// authority is reached.
fn mutate_graph(root: &Path, mutate: impl FnOnce(&mut serde_json::Value)) {
    let path = root.join(SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let target = graph["components"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["id"] == "target")
        .unwrap();
    mutate(target);
    std::fs::write(&path, graph.to_string()).unwrap();
}

/// A component with no execution surface has none of the facts the
/// target's metadata and geometry come from. Refused first, naming the
/// component.
#[test]
fn a_component_without_an_execution_surface_is_refused_by_name() {
    let encoded = encode(qwen35_checkpoint, "export-no-surface");
    mutate_graph(encoded.root(), |target| {
        target.as_object_mut().unwrap().remove("execution");
    });
    let msg = export_err(encoded.root());
    assert!(
        msg.contains("component `target` carries no execution surface"),
        "{msg}"
    );
}

/// **Roles come only from a plan that closes.** A graph whose declared
/// FFN width disagrees with the tensors it addresses yields a plan with
/// geometry defects; the export refuses with the plan's own defects
/// rather than taking roles from a plan that does not describe the
/// container.
#[test]
fn an_operation_plan_that_does_not_close_is_refused_with_its_defects() {
    let encoded = encode(qwen35_checkpoint, "export-open-plan");
    mutate_graph(encoded.root(), |target| {
        let ffn = &mut target["execution"]["ffn"]["intermediate_size"];
        *ffn = json!(2 * ffn.as_u64().unwrap());
    });
    let msg = export_err(encoded.root());
    assert!(msg.contains("the operation plan does not close"), "{msg}");
    assert!(msg.contains("GeometryMismatch"), "{msg}");
    assert!(
        msg.contains("mlp.gate_proj.weight"),
        "the defect names the tensor: {msg}"
    );
}

/// An interrupted segment write is refused by the export's inventory,
/// naming the truncation, before any plan is made or byte is written.
#[test]
fn a_truncated_segment_is_refused_by_the_export_before_anything_is_written() {
    let encoded = encode(qwen35_checkpoint, "export-truncated");
    let inspection = inspect_container(encoded.root(), false).unwrap();
    let segment = inspection
        .index
        .representations
        .values()
        .map(|e| encoded.root().join(&e.segment))
        .max_by_key(|p| std::fs::metadata(p).unwrap().len())
        .unwrap();
    let full = std::fs::metadata(&segment).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&segment)
        .unwrap()
        .set_len(full / 2)
        .unwrap();
    let msg = export_err(encoded.root());
    assert!(msg.contains("is truncated"), "{msg}");
}

/// A container whose checkpoint carried no tokenizer has no text
/// capability to export: refused by name, after every weight gate
/// passed, with no file written.
#[test]
fn a_container_without_a_tokenizer_is_refused_rather_than_exported_without_a_vocabulary() {
    let encoded = encode(
        |dir| {
            hybrid_lllf_f32_model_with_layers(dir, QWEN35_LAYERS);
            declare_mrope(dir);
        },
        "export-no-vocab",
    );
    let msg = export_err(encoded.root());
    assert!(msg.contains("no tokenizer.json"), "{msg}");
}

// ── roles from the operation plan ────────────────────────────────────

/// The happy path's role map is the plan's, exhaustively: every
/// operand the plan names is roled, once, with its layer.
#[test]
fn roles_cover_every_operand_the_plan_names_with_its_layer() {
    let encoded = encode(qwen35_checkpoint, "roles-exhaustive");
    let plan = op_plan(encoded.root());
    let roles = roles_from_plan(&plan).unwrap();

    let embed = &plan.embedding.as_ref().unwrap().table;
    assert_eq!(
        roles[&(embed.object.clone(), embed.tensor.clone())],
        ("embedding".to_string(), None)
    );
    let attending = HYBRID_CADENCE - 1;
    let LayerAttention::Softmax(a) = &plan.layers[attending].attention else {
        panic!("layer {attending} attends");
    };
    assert_eq!(
        roles[&(a.q.object.clone(), a.q.tensor.clone())],
        ("query".to_string(), Some(attending))
    );
    let LayerAttention::GatedDelta(g) = &plan.layers[0].attention else {
        panic!("layer 0 is recurrent");
    };
    assert_eq!(
        roles[&(g.a_log.object.clone(), g.a_log.tensor.clone())],
        (LOG_DECAY_ROLE.to_string(), Some(0))
    );
    let qk = a.qk_norm.as_ref().expect("the fixture ships Q/K norms");
    assert_eq!(
        roles[&(qk.k.object.clone(), qk.k.tensor.clone())],
        ("attention k norm".to_string(), Some(attending))
    );
    // Nine recurrent operands; four softmax projections and two Q/K
    // norms; two trunk norms and three FFN projections per layer; three
    // model-scope surfaces.
    let attending_layers = (0..QWEN35_LAYERS).filter(|l| attends(*l)).count();
    let recurrent = QWEN35_LAYERS - attending_layers;
    assert_eq!(
        roles.len(),
        3 + QWEN35_LAYERS * (2 + 3) + recurrent * 9 + attending_layers * (4 + 2)
    );
}

/// The four ways a layer's norms can fail to fit qwen35's two trunk
/// norms, each refused by layer and by what is wrong.
#[test]
fn norm_placements_qwen35_cannot_bind_are_refused_by_layer() {
    let encoded = encode(qwen35_checkpoint, "roles-norms");
    let plan = op_plan(encoded.root());
    // The plan spells the second trunk norm in one of two seats; the
    // other is empty, which is what every case below rearranges.
    let l0 = &plan.layers[0];
    let second = l0
        .post_attention_norm
        .clone()
        .or_else(|| l0.pre_ffn_norm.clone())
        .expect("a qwen35 layer carries a second trunk norm");
    let roles = roles_from_plan(&plan).unwrap();

    // Either spelling binds the same target role.
    let mut swapped = plan.clone();
    let l = &mut swapped.layers[0];
    std::mem::swap(&mut l.post_attention_norm, &mut l.pre_ffn_norm);
    assert_eq!(roles_from_plan(&swapped).unwrap(), roles);

    let mut both = plan.clone();
    both.layers[0].post_attention_norm = Some(second.clone());
    both.layers[0].pre_ffn_norm = Some(second.clone());
    let msg = roles_from_plan(&both).unwrap_err().to_string();
    assert!(
        msg.contains("layer 0") && msg.contains("not three"),
        "{msg}"
    );

    let mut neither = plan.clone();
    neither.layers[1].post_attention_norm = None;
    neither.layers[1].pre_ffn_norm = None;
    let msg = roles_from_plan(&neither).unwrap_err().to_string();
    assert!(
        msg.contains("layer 1") && msg.contains("no second trunk norm"),
        "{msg}"
    );

    let mut post_ffn = plan.clone();
    post_ffn.layers[2].post_ffn_norm = Some(second);
    let msg = roles_from_plan(&post_ffn).unwrap_err().to_string();
    assert!(
        msg.contains("layer 2") && msg.contains("post-FFN norm"),
        "{msg}"
    );

    // A post-norm stack has no pre-attention norm, and that is not a
    // refusal: there is nothing to write rather than a norm to invent.
    let mut post_norm = plan.clone();
    post_norm.layers[0].pre_attention_norm = None;
    let roles = roles_from_plan(&post_norm).unwrap();
    assert!(!roles
        .values()
        .any(|(r, l)| r == "input layer norm" && *l == Some(0)));
}

/// An FFN that is routed, or absent, has no qwen35 dense operands to
/// bind.
#[test]
fn ffns_other_than_one_dense_ffn_per_layer_are_refused() {
    use crate::format::vindex3::fixtures_qwen::miniature_qwen2_moe;

    let encoded = encode(qwen35_checkpoint, "roles-ffn");
    let plan = op_plan(encoded.root());

    let moe = encode(miniature_qwen2_moe, "roles-moe");
    let moe_plan = op_plan(moe.root());
    let routed = moe_plan
        .layers
        .iter()
        .find_map(|l| l.ffn.clone().filter(|f| f.dense().is_none()))
        .expect("the MoE fixture routes its FFN");
    let mut with_routed = plan.clone();
    with_routed.layers[0].ffn = Some(routed);
    let msg = roles_from_plan(&with_routed).unwrap_err().to_string();
    assert!(
        msg.contains("layer 0") && msg.contains("routed FFN"),
        "{msg}"
    );

    let mut without = plan;
    without.layers[1].ffn = None;
    let msg = roles_from_plan(&without).unwrap_err().to_string();
    assert!(msg.contains("layer 1") && msg.contains("no FFN"), "{msg}");
}

/// Operators qwen35 has no tensor names for are refused, naming the
/// operator. Each operator is transplanted from the plan of a container
/// that really carries it: KDA and MLA from a Kimi-shaped hybrid,
/// Mamba2 from a pure-SSM stack, conv-QKV from a Mamba2Attn hybrid.
#[test]
fn attention_operators_qwen35_cannot_name_are_refused_by_operator() {
    use crate::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
    use crate::format::vindex3::opplan::tests::{conv_qkv, mamba2};

    let encoded = encode(qwen35_checkpoint, "roles-ops");
    let plan = op_plan(encoded.root());
    let donors = [
        encode(hybrid_kda_mla_f32_model, "roles-kimi"),
        encode(|dir| mamba2::miniature_mamba2(dir, None), "roles-mamba2"),
        encode(conv_qkv::miniature_hybrid, "roles-conv-qkv"),
    ];

    let mut refused = Vec::new();
    for donor in &donors {
        for layer in &op_plan(donor.root()).layers {
            if matches!(
                layer.attention,
                LayerAttention::Softmax(_) | LayerAttention::GatedDelta(_)
            ) {
                continue;
            }
            let mut p = plan.clone();
            p.layers[0].attention = layer.attention.clone();
            let msg = roles_from_plan(&p).unwrap_err().to_string();
            assert!(msg.contains("layer 0"), "{msg}");
            refused.push(msg);
        }
    }
    let text = refused.join("\n");
    for operator in ["KDA", "MLA", "Mamba2", "conv-QKV attention"] {
        assert!(text.contains(&format!("binds {operator} ")), "{text}");
    }
}

/// qwen35 binds an attention output gate only as the second half of a
/// fused query projection. A gate on the query's own tensor is
/// accepted and adds no role; a gate with its own tensor is refused.
#[test]
fn only_a_gate_fused_into_the_query_projection_binds() {
    use crate::format::vindex3::fixtures::gated_q_f32_model;

    let gated = encode(gated_q_f32_model, "roles-gate");
    let gated_plan = op_plan(gated.root());
    let gate = gated_plan
        .layers
        .iter()
        .find_map(|l| match &l.attention {
            LayerAttention::Softmax(a) => a.output_gate.clone(),
            _ => None,
        })
        .expect("the fixture fuses a gate into q_proj");

    let encoded = encode(qwen35_checkpoint, "roles-gate-host");
    let plan = op_plan(encoded.root());
    let attending = HYBRID_CADENCE - 1;
    let with_gate = |projection: &crate::format::vindex3::opplan::OperandRef| {
        let mut p = plan.clone();
        let LayerAttention::Softmax(a) = &mut p.layers[attending].attention else {
            unreachable!()
        };
        let mut g = gate.clone();
        g.projection = projection.clone();
        a.output_gate = Some(g);
        p
    };
    let LayerAttention::Softmax(a) = &plan.layers[attending].attention else {
        panic!("layer {attending} attends");
    };

    let fused = roles_from_plan(&with_gate(&a.q)).unwrap();
    assert_eq!(fused, roles_from_plan(&plan).unwrap());

    let msg = roles_from_plan(&with_gate(&a.k)).unwrap_err().to_string();
    assert!(msg.contains("standalone attention output gate"), "{msg}");
    assert!(msg.contains(&a.k.tensor), "names the gate's tensor: {msg}");
}

/// **The precision programme is the container's, and the preflight
/// judges what it selects.** A represented container carries NVFP4
/// beside its canonical bytes and its precision map says NVFP4
/// executes, so the export selects the pack — and the fixture's value
/// heads are narrower than one NVFP4 group, which the V-head reorder
/// cannot split. The preflight refuses on that, by name, before
/// anything is lowered; on the canonical bytes the same weights pass.
#[test]
fn a_represented_container_is_judged_on_the_encoding_its_precision_map_selects() {
    use crate::format::vindex3::represent::{compile_representation, RepresentSpec};

    let encoded = encode(qwen35_checkpoint, "export-nvfp4");
    let represented = tempfile::tempdir().unwrap();
    compile_representation(encoded.root(), represented.path(), &RepresentSpec::nvfp4())
        .expect("the fixture compiles");
    let precision = inspect_container(represented.path(), false)
        .unwrap()
        .index
        .precision_map
        .expect("a represented container declares what executes");
    assert_eq!(precision.encoding.to_uppercase(), "NVFP4");

    let msg = export_err(represented.path());
    assert!(msg.contains("preflight refuses"), "{msg}");
    assert!(msg.contains("value_head_dim"), "{msg}");

    let out = tempfile::tempdir().unwrap();
    export_qwen35(encoded.root(), &out.path().join("canonical.gguf"))
        .expect("the canonical bytes of the same weights export");
}
