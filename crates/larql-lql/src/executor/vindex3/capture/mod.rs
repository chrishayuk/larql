//! `CAPTURE <site> [AT LAYER <n> | AT LAYERS (…)] FOR "<prompt>" [CONTINUE … | GENERATE <n>] [POSITIONS …] TO
//! "<file>.npy"`: the values the session's program computed at one site, written as a NumPy array with a sidecar
//! that names the model state that computed them.
//!
//! Nothing here computes a forward pass of its own. Every value is one the executor produced while running the
//! session's program, over the operands `INFER` resolves (the stored weights, or the patched bytes of the
//! stored-weight ops in effect, or a refusal where INFER refuses):
//!
//! - `residual`, `post_attention`, `ffn.input`, `final` and `logits` come from the one-shot traversal INFER runs
//!   ([`batch`]);
//! - `ffn.down_input` and `ffn.down_output` come from TRACE's decode steps, the only path that exposes the down
//!   projection ([`decode`]); the sidecar records whether those steps' FFN input equals INFER's pass bit for bit.

mod batch;
mod decode;
mod provenance;
mod sequence;

pub(crate) use provenance::AppliedPatchFile;

use larql_inference::vindex3::OperandOverrides;

use super::npy::{NpyDescr, NpyWriter};
use super::npy_output::{
    container_index, container_path, index_list, literal, output_paths, remove_partial,
    write_sidecar, LARQL_VERSION,
};
use super::{execution_overrides, v3_continuation};
use crate::ast::{CaptureContinuation, CaptureLayers, CapturePositions, CaptureSite};
use crate::error::LqlError;
use crate::executor::{Backend, Session};

const STATEMENT: &str = "CAPTURE";
/// The sidecar's schema id.
const SIDECAR_SCHEMA: &str = "larql-capture/v1";
/// The array type every capture writes: the f32 values the executor held.
const DESCR: NpyDescr = NpyDescr::F4;
/// The two execution paths a capture reads, as the sidecar names them.
const PATH_INFER_PASS: &str = "the one-shot traversal INFER runs, observed";
const PATH_DECODE_STEPS: &str = "TRACE's decode steps from position 0, observed";
/// How a prompt is encoded, as the sidecar says it.
const PROMPT_ENCODING: &str =
    "INFER's raw-prompt encoder: the tokenizer's own special tokens, plus the \
     container's declared BOS where the tokenizer did not add it; no chat template";
const CONTINUATION_ENCODING: &str = "no special tokens";

/// The rows of one block: one `[width]` vector per captured position.
type Rows = Vec<Vec<f32>>;

fn refused(e: impl std::fmt::Display) -> LqlError {
    super::npy_output::refused(STATEMENT, e)
}

/// The arguments of one `CAPTURE`, as the statement gives them.
pub(crate) struct CaptureRequest<'a> {
    pub site: CaptureSite,
    pub layers: Option<&'a CaptureLayers>,
    pub prompt: &'a str,
    pub continuation: Option<&'a CaptureContinuation>,
    pub positions: &'a CapturePositions,
    pub output: &'a str,
}

/// The statement as LQL text, in its canonical form: what the sidecar records.
fn statement_text(request: &CaptureRequest<'_>) -> String {
    let mut text = format!("{STATEMENT} {}", request.site.name());
    match request.layers {
        Some(CaptureLayers::One(layer)) => text.push_str(&format!(" AT LAYER {layer}")),
        Some(CaptureLayers::Many(layers)) => {
            text.push_str(&format!(" AT LAYERS ({})", index_list(layers)))
        }
        None => {}
    }
    text.push_str(&format!(" FOR {}", literal(request.prompt)));
    match request.continuation {
        Some(CaptureContinuation::Text(t)) => text.push_str(&format!(" CONTINUE {}", literal(t))),
        Some(CaptureContinuation::Tokens(ids)) => {
            text.push_str(&format!(" CONTINUE TOKENS ({})", index_list(ids)))
        }
        Some(CaptureContinuation::Generate(n)) => text.push_str(&format!(" GENERATE {n}")),
        None => {}
    }
    text.push_str(&format!(
        " POSITIONS {} TO {};",
        request.positions.spelled(),
        literal(request.output)
    ));
    text
}

/// Every block holds one row per position, and every row has the same width.
fn checked_shape(blocks: &[Rows], positions: usize, stacked: bool) -> Result<Vec<usize>, LqlError> {
    let width = blocks.first().and_then(|b| b.first()).map_or(0, Vec::len);
    let consistent = !blocks.is_empty()
        && width > 0
        && blocks
            .iter()
            .all(|b| b.len() == positions && b.iter().all(|row| row.len() == width));
    if !consistent {
        return Err(LqlError::Execution(format!(
            "{STATEMENT}: the executor did not report one value of a single width at every requested layer \
             and position"
        )));
    }
    Ok(if stacked {
        vec![blocks.len(), positions, width]
    } else {
        vec![positions, width]
    })
}

/// Write the rows, block after block, as one C-order array.
fn write_npy(path: &std::path::Path, shape: &[usize], blocks: &[Rows]) -> Result<String, LqlError> {
    let mut writer = NpyWriter::create_new(path, DESCR, shape)
        .map_err(|e| LqlError::exec(format!("create {}", path.display()), e))?;
    let body: std::io::Result<()> = blocks.iter().flatten().try_for_each(|row| {
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write(&bytes)
    });
    let finished = body.and_then(|()| writer.finish());
    finished.map_err(|e| {
        remove_partial(
            &[path],
            LqlError::exec(format!("write {}", path.display()), e),
        )
    })
}

impl Session {
    pub(crate) fn exec_capture(
        &self,
        request: &CaptureRequest<'_>,
    ) -> Result<Vec<String>, LqlError> {
        let (root, runtime, tokenizer, overlay, bos, patch_files) = match &self.backend {
            Backend::Vindex3 {
                path,
                runtime,
                tokenizer,
                overlay,
                bos_token,
                patch_files,
                ..
            } => (path, runtime, tokenizer, overlay, *bos_token, patch_files),
            Backend::None => return Err(LqlError::NoBackend),
            _ => return Err(refused(
                "CAPTURE runs a VINDEX3 container's program, and this session is not bound to one",
            )),
        };
        let tokenizer = tokenizer.as_ref().ok_or_else(|| {
            LqlError::Execution(
                "CAPTURE needs a tokenizer and this container carries no tokenizer.json — \
                 token-id capability only"
                    .into(),
            )
        })?;
        let site = request.site;
        let layers = sequence::layers(site, request.layers, runtime.plan().layers.len())?;
        let (npy_path, sidecar_path) = output_paths(STATEMENT, request.output)?;
        let (index, index_sha256) = container_index(root)?;
        let start = std::time::Instant::now();

        let sequence = sequence::build(
            runtime,
            overlay,
            tokenizer,
            bos,
            request.prompt,
            request.continuation,
        )?;
        let positions = sequence::positions(request.positions, &sequence)?;
        let overrides: Option<OperandOverrides> = execution_overrides(runtime, overlay)?;
        let continuation = v3_continuation(runtime)?;
        let (blocks, path, cross_check) = match site {
            CaptureSite::FfnDownInput | CaptureSite::FfnDownOutput => {
                let captured = decode::capture(
                    runtime,
                    overrides.as_ref(),
                    &continuation,
                    &sequence.ids,
                    site,
                    &layers,
                    &positions,
                )?;
                let check = serde_json::json!({
                    "compared": "ffn.input at the same layers and positions: these decode steps against INFER's one-shot pass",
                    "bit_identical": captured.ffn_input_matches_infer_pass,
                });
                (captured.blocks, PATH_DECODE_STEPS, check)
            }
            _ => {
                let blocks = batch::capture(
                    runtime,
                    overrides.as_ref(),
                    &continuation,
                    &sequence.ids,
                    site,
                    &layers,
                    &positions,
                )?;
                (blocks, PATH_INFER_PASS, serde_json::Value::Null)
            }
        };
        let stacked = matches!(request.layers, Some(CaptureLayers::Many(_)));
        let shape = checked_shape(&blocks, positions.len(), stacked)?;
        let npy_sha256 = write_npy(&npy_path, &shape, &blocks)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let decode = |id: u32| tokenizer.decode(&[id], false).unwrap_or_default();
        let position_records: Vec<serde_json::Value> = positions
            .iter()
            .map(|&p| {
                serde_json::json!({
                    "position": p,
                    "part": if p < sequence.prompt_len { "prompt" } else { "continuation" },
                    "token_id": sequence.ids[p],
                    "token": decode(sequence.ids[p]),
                    "predicts_token_id": sequence.predicts(p),
                })
            })
            .collect();
        let executes = provenance::executes(overlay, overrides.is_some());
        let sidecar = serde_json::json!({
            "schema": SIDECAR_SCHEMA,
            "statement": statement_text(request),
            "container": container_path(root),
            "index_sha256": index_sha256,
            "model": index.model,
            "site": site.name(),
            "layers": if site.per_layer() { serde_json::json!(layers) } else { serde_json::Value::Null },
            "positions": position_records,
            "sequence": {
                "token_ids": sequence.ids,
                "prompt_tokens": sequence.prompt_len,
                "prompt": request.prompt,
                "prompt_encoding": PROMPT_ENCODING,
                "bos_token_id": bos,
                "continuation": sequence.continuation,
                "continuation_encoding": CONTINUATION_ENCODING,
            },
            "executes": executes,
            "patches": provenance::patches(overlay, patch_files),
            "live_tensors": provenance::live_tensors(overlay),
            "execution": {
                "path": path,
                "binding": provenance::binding(runtime),
                "cross_check": cross_check,
            },
            "dtype": DESCR.label(),
            "shape": shape,
            "npy_sha256": npy_sha256,
            "larql_version": LARQL_VERSION,
        });
        write_sidecar(&sidecar_path, &sidecar, &[&npy_path])?;

        Ok(vec![
            format!(
                "Captured {} {:?} ({}) to {}",
                site.name(),
                shape,
                DESCR.label(),
                npy_path.display()
            ),
            format!(
                "  sequence: {} token(s), {} from the prompt; positions {}",
                sequence.ids.len(),
                sequence.prompt_len,
                index_list(&positions)
            ),
            format!("  executes: {executes} weights; {path}"),
            format!("  sha256:  {npy_sha256}"),
            format!("  sidecar: {}", sidecar_path.display()),
            format!("  {elapsed_ms:.0}ms"),
        ])
    }
}
