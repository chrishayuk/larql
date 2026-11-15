//! The operations: each one calls the typed Rust operation `larql_lql::operations` names (the same one the
//! LQL statement and the Python session call), after the policy's checks, and returns what the Python
//! session's method returns, as JSON.

use serde::Deserialize;
use serde_json::{json, Value};

use larql_lql::operations::{
    operand_bits_from_values, operand_origin, CaptureContinuation, CaptureLayers, CapturePositions,
    CaptureSite, CaptureSpec, NpyDescr, OperandValues,
};

use super::npy::{decode_operand, encode, EncodedArray, Operand};
use super::protocol::{args, Failure, Op};
use super::service::Service;

/// The origin an op staged from arrays records, as the Python session records it, so the op (and the
/// `.vlp` it is saved in) is byte-identical to the one the in-process session stages from the same arrays.
const ARRAY_ORIGIN: &str = "python arrays";
/// The `positions` words.
const POSITION_WORDS: &str = "LAST, PROMPT_END, REPLY or ALL";
/// What a capture's provenance says when the stored weights executed.
const EXECUTES_STORED: &str = "stored";
const DEFAULT_TOP: usize = 5;
/// The bytes of one f32 element.
const F32_BYTES: u64 = 4;
/// At most this many bytes of `.npy` header precede a capture's body (a 1.0 header for up to three dims).
const NPY_HEADER_BOUND: u64 = 256;

/// Refuse a capture whose array bytes, bounded from the plan before it runs, exceed `limit`. When the plan
/// gives the site no width, the capture is refused if the policy requires a bound (the default), and otherwise
/// left to the check made on the array after it runs.
pub(super) fn capture_bound(
    site: CaptureSite,
    width: Option<usize>,
    layers: u64,
    positions: u64,
    require: bool,
    limit: u64,
) -> Result<(), Failure> {
    let Some(width) = width else {
        return if require {
            Err(Failure::refused(format!(
                "the service cannot bound the size of a capture at the site {} before running it (the plan \
                 gives this site no width here), and the policy's require_capture_bound is true",
                site.name()
            )))
        } else {
            Ok(())
        };
    };
    let bound = layers
        .saturating_mul(positions)
        .saturating_mul(width as u64)
        .saturating_mul(F32_BYTES)
        .saturating_add(NPY_HEADER_BOUND);
    over(
        "the capture's array bytes, at most",
        bound,
        limit,
        "max_response_bytes",
    )
}

fn no_repeats(what: &str, list: &[usize]) -> Result<(), Failure> {
    let mut seen = std::collections::BTreeSet::new();
    match list.iter().find(|item| !seen.insert(**item)) {
        Some(repeat) => Err(Failure::invalid(format!("{what} {repeat} is listed twice"))),
        None => Ok(()),
    }
}

fn values(o: &Operand) -> OperandValues<'_> {
    match o {
        Operand::F64(v) => OperandValues::F64(v),
        Operand::F32(v) => OperandValues::F32(v),
    }
}

fn default_top() -> usize {
    DEFAULT_TOP
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorsArgs {
    #[serde(default)]
    filter: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorArgs {
    name: String,
    #[serde(default)]
    rows: Option<Vec<usize>>,
    #[serde(default)]
    cols: Option<Vec<usize>>,
    #[serde(default)]
    bits: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizeArgs {
    text: String,
    #[serde(default)]
    raw: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodeArgs {
    ids: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Positions {
    Word(String),
    List(Vec<usize>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureArgs {
    site: String,
    prompt: String,
    #[serde(default)]
    layer: Option<usize>,
    #[serde(default)]
    layers: Option<Vec<usize>>,
    #[serde(default)]
    continue_text: Option<String>,
    #[serde(default)]
    continue_tokens: Option<Vec<u32>>,
    #[serde(default)]
    generate: Option<u32>,
    #[serde(default)]
    positions: Option<Positions>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InferArgs {
    prompt: String,
    #[serde(default = "default_top")]
    top: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerateArgs {
    prompt: String,
    n: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PathArgs {
    path: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NpyArg {
    npy_base64: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LowRankArgs {
    tensor: String,
    a: NpyArg,
    b: NpyArg,
    columns: Vec<usize>,
    numerics: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateWeightsArgs {
    tensor: String,
    row: usize,
    cols: Vec<usize>,
    delta: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

fn positions_arg(positions: Option<Positions>) -> Result<CapturePositions, Failure> {
    match positions {
        None => Ok(CapturePositions::Last),
        Some(Positions::List(list)) => Ok(CapturePositions::List(list)),
        Some(Positions::Word(word)) => match word.to_ascii_uppercase().as_str() {
            "LAST" => Ok(CapturePositions::Last),
            "PROMPT_END" => Ok(CapturePositions::PromptEnd),
            "REPLY" => Ok(CapturePositions::Reply),
            "ALL" => Ok(CapturePositions::All),
            _ => Err(Failure::invalid(format!(
                "positions={word:?}: name {POSITION_WORDS}, or pass a list of positions"
            ))),
        },
    }
}

fn over(what: &str, value: u64, limit: u64, name: &str) -> Result<(), Failure> {
    if value > limit {
        return Err(Failure::refused(format!(
            "{what} is {value}; the policy's {name} is {limit}"
        )));
    }
    Ok(())
}

impl Service {
    pub(super) fn run(&mut self, op: Op, name: &str, a: &Value) -> Result<Value, Failure> {
        match op {
            Op::Tensors => self.tensors(args(name, a)?),
            Op::Tensor => self.tensor(args(name, a)?),
            Op::Tokenize => {
                let t: TokenizeArgs = args(name, a)?;
                let ids = self
                    .session
                    .tokenize(&t.text, !t.raw)
                    .map_err(Failure::failed)?;
                Ok(json!(ids))
            }
            Op::Decode => {
                let d: DecodeArgs = args(name, a)?;
                Ok(json!(self
                    .session
                    .decode(&d.ids)
                    .map_err(Failure::failed)?))
            }
            Op::Capture => self.capture(args(name, a)?),
            Op::Infer => self.infer(args(name, a)?),
            Op::Generate => self.generate(args(name, a)?),
            Op::BeginPatch => self.begin_patch(args(name, a)?),
            Op::AddLowRank => self.add_low_rank(args(name, a)?),
            Op::UpdateWeights => self.update_weights(args(name, a)?),
            Op::SavePatch => {
                let NoArgs {} = args(name, a)?;
                self.save_patch()
            }
            Op::ApplyPatch => self.apply_patch(args(name, a)?),
            Op::RemovePatch => self.remove_patch(args(name, a)?),
            Op::Patches => {
                let NoArgs {} = args(name, a)?;
                self.patches()
            }
            Op::DiscardWeightEdits => {
                let NoArgs {} = args(name, a)?;
                let d = self
                    .session
                    .discard_weight_edits()
                    .map_err(Failure::failed)?;
                Ok(json!({
                    "discarded": d.discarded,
                    "unrecorded": d.unrecorded,
                    "from_patches": d.from_patches,
                }))
            }
        }
    }

    fn tensors(&mut self, t: TensorsArgs) -> Result<Value, Failure> {
        let listings = self
            .session
            .tensors(t.filter.as_deref())
            .map_err(Failure::failed)?;
        Ok(Value::Array(
            listings
                .iter()
                .map(|l| {
                    json!({
                        "name": l.qualified_name(),
                        "representation": l.representation,
                        "object": l.object,
                        "tensor": l.tensor,
                        "canonical": l.canonical,
                        "dtype": l.dtype,
                        "shape": l.shape,
                        "bare": l.bare_name.label(),
                    })
                })
                .collect(),
        ))
    }

    /// Refuse a response whose arrays together exceed the limit, before they are sent.
    fn check_response_bytes(&self, bytes: u64) -> Result<(), Failure> {
        over(
            "the response's array bytes",
            bytes,
            self.limits.max_response_bytes,
            "max_response_bytes",
        )
    }

    fn readout(&mut self, encoded: EncodedArray, provenance: Value) -> Result<Value, Failure> {
        self.check_response_bytes(encoded.bytes)?;
        self.turn.arrays.push(json!({
            "sha256": encoded.sha256,
            "dtype": encoded.value["dtype"],
            "shape": encoded.value["shape"],
            "bytes": encoded.bytes,
        }));
        Ok(json!({"array": encoded.value, "provenance": provenance}))
    }

    /// The most bytes a read of `name` with this selection could return, from the tensor listing: so a read
    /// too large to send is refused before it is read into memory. `None` when the name is not listed (the
    /// read then refuses it itself).
    fn tensor_bytes_bound(&self, t: &TensorArgs) -> Result<Option<u64>, Failure> {
        let listings = self.session.tensors(None).map_err(Failure::failed)?;
        let bound = listings
            .iter()
            .filter(|l| l.qualified_name() == t.name || l.tensor == t.name)
            .map(|l| {
                let mut dims: Vec<u64> = l.shape.iter().map(|&d| d as u64).collect();
                if let (Some(rows), Some(first)) = (&t.rows, dims.first_mut()) {
                    *first = rows.len() as u64;
                }
                if let (Some(cols), Some(last)) = (&t.cols, dims.last_mut()) {
                    *last = cols.len() as u64;
                }
                // Values widen to f32; bits are at most four bytes an element.
                dims.iter().product::<u64>().saturating_mul(4)
            })
            .max();
        Ok(bound)
    }

    fn tensor(&mut self, t: TensorArgs) -> Result<Value, Failure> {
        if let Some(bound) = self.tensor_bytes_bound(&t)? {
            self.check_response_bytes(bound)?;
        }
        let read = self
            .session
            .read_weight(&t.name, t.rows.as_deref(), t.cols.as_deref(), t.bits)
            .map_err(Failure::failed)?;
        let encoded = encode(read.descr, &read.shape, &read.bytes)?;
        self.readout(encoded, read.provenance)
    }

    /// The prompt's length in tokens, as INFER and CAPTURE encode it.
    fn prompt_tokens(&self, prompt: &str) -> Result<u64, Failure> {
        Ok(self
            .session
            .tokenize(prompt, true)
            .map_err(Failure::failed)?
            .len() as u64)
    }

    fn check_sequence(&self, tokens: u64) -> Result<(), Failure> {
        over(
            "the sequence length in tokens",
            tokens,
            self.limits.max_sequence_tokens,
            "max_sequence_tokens",
        )
    }

    fn check_generate(&self, n: u64) -> Result<(), Failure> {
        over(
            "the tokens to generate",
            n,
            self.limits.max_generate_tokens,
            "max_generate_tokens",
        )
    }

    fn capture(&mut self, c: CaptureArgs) -> Result<Value, Failure> {
        let site = CaptureSite::from_name(&c.site).ok_or_else(|| {
            let names: Vec<&str> = CaptureSite::ALL.iter().map(|s| s.name()).collect();
            Failure::invalid(format!(
                "unknown site {:?}; sites are {}",
                c.site,
                names.join(", ")
            ))
        })?;
        let layers = match (c.layer, c.layers) {
            (Some(_), Some(_)) => return Err(Failure::invalid("pass layer= or layers=, not both")),
            (Some(one), None) => Some(CaptureLayers::One(one)),
            (None, Some(many)) => Some(CaptureLayers::Many(many)),
            (None, None) => None,
        };
        let continuation = match (c.continue_text, c.continue_tokens, c.generate) {
            (None, None, None) => None,
            (Some(t), None, None) => Some(CaptureContinuation::Text(t)),
            (None, Some(ids), None) => Some(CaptureContinuation::Tokens(ids)),
            (None, None, Some(n)) => Some(CaptureContinuation::Generate(n)),
            _ => {
                return Err(Failure::invalid(
                    "pass at most one of continue_text=, continue_tokens= and generate=",
                ))
            }
        };
        let positions = positions_arg(c.positions)?;
        self.check_capture(
            site,
            layers.as_ref(),
            &c.prompt,
            continuation.as_ref(),
            &positions,
        )?;

        // Charged now, before it runs: whatever happens from here on (a refusal LARQL makes after running, a
        // response too large to send), an edited model has executed.
        let edited = self.gate()?;
        let captured = self
            .session
            .capture(&CaptureSpec {
                site,
                layers: layers.as_ref(),
                prompt: &c.prompt,
                continuation: continuation.as_ref(),
                positions: &positions,
            })
            .map_err(Failure::failed)?;
        // The capture's own record of what executed is the second witness: if it says the stored weights
        // did not execute, the run is charged even if the session's report said otherwise.
        if !edited && captured.provenance["executes"] != EXECUTES_STORED {
            self.charge()?;
        }
        let body: Vec<u8> = captured
            .values()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let encoded = encode(NpyDescr::F4, &captured.shape, &body)?;
        self.readout(encoded, captured.provenance)
    }

    /// Every check a capture's arguments allow before it runs: the layers against the plan, the
    /// continuation, the positions against the sequence where its length is known, and the limits,
    /// including an upper bound on the response's array bytes.
    fn check_capture(
        &self,
        site: CaptureSite,
        layers: Option<&CaptureLayers>,
        prompt: &str,
        continuation: Option<&CaptureContinuation>,
        positions: &CapturePositions,
    ) -> Result<(), Failure> {
        let layer_list = layers.map(|l| l.layers()).unwrap_or_default();
        match (site.per_layer(), layers) {
            (true, None) => {
                return Err(Failure::invalid(format!(
                    "the site {} is inside a layer: pass layer= or layers=",
                    site.name()
                )))
            }
            (false, Some(_)) => {
                return Err(Failure::invalid(format!(
                    "the site {} is after the last layer and takes no layer",
                    site.name()
                )))
            }
            _ => {}
        }
        if site.per_layer() && layer_list.is_empty() {
            return Err(Failure::invalid("layers= is empty"));
        }
        no_repeats("layer", &layer_list)?;
        let geometry = self
            .session
            .capture_geometry(site, &layer_list)
            .map_err(Failure::failed)?;
        if let Some(bad) = layer_list.iter().find(|&&l| l >= geometry.plan_layers) {
            return Err(Failure::invalid(format!(
                "layer {bad} does not exist: the program has {} layers",
                geometry.plan_layers
            )));
        }
        over(
            "the layers to capture",
            layer_list.len().max(1) as u64,
            self.limits.max_capture_layers,
            "max_capture_layers",
        )?;

        let prompt_tokens = self.prompt_tokens(prompt)?;
        // The continuation's length: exact, or (for generate) at most.
        let (reply, exact) = match continuation {
            None => (0, true),
            Some(CaptureContinuation::Text(t)) => {
                let n = self
                    .session
                    .tokenize(t, false)
                    .map_err(Failure::failed)?
                    .len() as u64;
                if n == 0 {
                    return Err(Failure::invalid(format!(
                        "continue_text={t:?} encodes to no tokens"
                    )));
                }
                (n, true)
            }
            Some(CaptureContinuation::Tokens(ids)) => {
                if ids.is_empty() {
                    return Err(Failure::invalid("continue_tokens= is empty"));
                }
                if let (Some(vocab), Some(bad)) = (
                    geometry.vocab,
                    ids.iter()
                        .find(|&&id| id as usize >= geometry.vocab.unwrap_or(0)),
                ) {
                    return Err(Failure::invalid(format!(
                        "token id {bad} is outside the vocabulary of {vocab}"
                    )));
                }
                (ids.len() as u64, true)
            }
            Some(CaptureContinuation::Generate(n)) => {
                if *n == 0 {
                    return Err(Failure::invalid("generate= must be at least 1"));
                }
                self.check_generate(u64::from(*n))?;
                (u64::from(*n), false)
            }
        };
        let sequence = prompt_tokens + reply;
        self.check_sequence(sequence)?;
        let position_count = match positions {
            CapturePositions::Last | CapturePositions::PromptEnd => 1,
            CapturePositions::Reply => {
                if continuation.is_none() {
                    return Err(Failure::invalid(
                        "positions=\"REPLY\" needs a continuation (continue_text=, continue_tokens= or generate=)",
                    ));
                }
                reply
            }
            CapturePositions::All => sequence,
            CapturePositions::List(list) => {
                if list.is_empty() {
                    return Err(Failure::invalid("positions= is an empty list"));
                }
                no_repeats("position", list)?;
                if let Some(bad) = list.iter().find(|&&p| p as u64 >= sequence) {
                    return Err(Failure::invalid(if exact {
                        format!("position {bad} is outside the {sequence}-token sequence")
                    } else {
                        format!(
                            "position {bad} is beyond the prompt and the {reply} token(s) generate= may produce"
                        )
                    }));
                }
                list.len() as u64
            }
        };
        over(
            "the positions to capture",
            position_count,
            self.limits.max_capture_positions,
            "max_capture_positions",
        )?;
        #[cfg(test)]
        let geometry = larql_lql::operations::CaptureGeometry {
            width: geometry.width.filter(|_| !self.forget_capture_width),
            ..geometry
        };
        capture_bound(
            site,
            geometry.width,
            layer_list.len().max(1) as u64,
            position_count,
            self.require_capture_bound,
            self.limits.max_response_bytes,
        )?;
        Ok(())
    }

    fn infer(&mut self, i: InferArgs) -> Result<Value, Failure> {
        over(
            "top",
            i.top as u64,
            self.limits.max_infer_top,
            "max_infer_top",
        )?;
        let sequence = self.prompt_tokens(&i.prompt)?;
        self.check_sequence(sequence)?;
        self.gate()?;
        let predicted = self
            .session
            .predict(&i.prompt, i.top)
            .map_err(Failure::failed)?;
        Ok(json!({
            "prompt_ids": predicted.prompt_ids,
            "predictions": predicted.top.iter().map(|p| json!({
                "token_id": p.token_id,
                "token": p.token,
                // Widened exactly, as the Python session hands an f32 to Python.
                "probability": f64::from(p.probability),
            })).collect::<Vec<_>>(),
            "knn_override": predicted.knn_override.as_ref().map(|o| json!({
                "token": o.token,
                "cosine": f64::from(o.cosine),
                "layer": o.layer,
            })),
        }))
    }

    fn generate(&mut self, g: GenerateArgs) -> Result<Value, Failure> {
        self.check_generate(g.n as u64)?;
        let sequence = self.prompt_tokens(&g.prompt)? + g.n as u64;
        self.check_sequence(sequence)?;
        self.gate()?;
        let generated = self
            .session
            .generate(&g.prompt, g.n)
            .map_err(Failure::failed)?;
        Ok(json!({
            "prompt_ids": generated.prompt_ids,
            "token_ids": generated.token_ids,
            "text": generated.text,
        }))
    }

    /// The staging path for workdir name `name`, as LARQL is given it.
    fn staging_path(&self, name: &str) -> Result<String, Failure> {
        self.staged_path(name)
            .to_str()
            .map(str::to_string)
            .ok_or_else(|| Failure::refused("the staging path is not valid UTF-8"))
    }

    fn begin_patch(&mut self, p: PathArgs) -> Result<Value, Failure> {
        let name = self.workdir.name(&p.path)?;
        // Refuse a destination that could never be written now, rather than at SAVE PATCH, when the
        // recording could only be kept. (Checked again at the save: the workdir can change in between.)
        self.workdir.open_for_write(&name)?.discard();
        let staged = self.staging_path(&name)?;
        // LARQL creates the staged file at the save; nothing may be there before it.
        self.staged.clear(&name)?;
        self.session.begin_patch(&staged).map_err(Failure::failed)?;
        self.recording = Some(name);
        Ok(Value::Null)
    }

    fn add_low_rank(&mut self, l: LowRankArgs) -> Result<Value, Failure> {
        let a = decode_operand("a", &l.a.npy_base64)?;
        let b = decode_operand("b", &l.b.npy_base64)?;
        let rank = a.shape[1];
        if b.shape != [rank, l.columns.len()] {
            return Err(Failure::invalid(format!(
                "`b` is {}x{}; for `a` of rank {rank} over {} column(s) it must be {rank}x{}",
                b.shape[0],
                b.shape[1],
                l.columns.len(),
                l.columns.len()
            )));
        }
        let a_bits = operand_bits_from_values(&l.numerics, values(&a.values), "a")
            .map_err(Failure::failed)?;
        let b_bits = operand_bits_from_values(&l.numerics, values(&b.values), "b")
            .map_err(Failure::failed)?;
        let origin = operand_origin(ARRAY_ORIGIN, &a_bits, a.shape, &b_bits, b.shape);
        let staged = self
            .session
            .stage_low_rank(
                &l.tensor,
                &l.columns,
                &l.numerics,
                rank,
                &a_bits,
                &b_bits,
                Some(origin),
            )
            .map_err(Failure::failed)?;
        Ok(json!({
            "digest": staged.op.digest,
            "changed_elements": staged.changed_elements,
            "sham": staged.sham,
            "pending_structured_ops": staged.pending_structured_ops,
            "origin": staged.op.origin,
        }))
    }

    fn update_weights(&mut self, u: UpdateWeightsArgs) -> Result<Value, Failure> {
        let staged = self
            .session
            .stage_weight_delta(&u.tensor, u.row, &u.cols, u.delta)
            .map_err(Failure::failed)?;
        Ok(json!({
            "digest": staged.digest,
            "elements": staged.elements.iter().map(|e| json!({
                "col": e.col, "old_bits": e.old_bits, "new_bits": e.new_bits,
            })).collect::<Vec<_>>(),
            "pending_elements": staged.pending_elements,
        }))
    }

    fn save_patch(&mut self) -> Result<Value, Failure> {
        let Some(name) = self.recording.clone() else {
            // No recording this service began: LARQL's own refusal (none open, or the anonymous one, which
            // it never writes) is the answer.
            return match self.session.save_patch() {
                Err(e) => Err(Failure::failed(e)),
                Ok(_) => Err(Failure::failed(
                    "a recording the service did not begin was saved",
                )),
            };
        };
        // The destination is opened (and checked) before LARQL writes anything, so a refused destination
        // leaves the recording in the session.
        let mut target = self.workdir.open_for_write(&name)?;
        if let Err(f) = self.staged.clear(&name) {
            target.discard();
            return Err(f);
        }
        let saved = match self.session.save_patch() {
            Ok(saved) => saved,
            Err(e) => {
                target.discard();
                return Err(Failure::failed(e));
            }
        };
        self.recording = None;
        // Read back as confined as the workdir: a staged file that is not the regular, singly linked file
        // LARQL just wrote is refused.
        let bytes = self.staged.read(&name, self.limits.max_patch_bytes);
        let _ = self.staged.clear(&name);
        let bytes = bytes?;
        let delivered = {
            use std::io::Write;
            target
                .file
                .set_len(0)
                .and_then(|_| target.file.write_all(&bytes))
                .and_then(|_| target.file.sync_all())
        };
        delivered.map_err(|e| {
            Failure::failed(format!(
                "the patch was saved but could not be written to `{name}` in the workdir: {e}"
            ))
        })?;
        Ok(json!({
            "path": name,
            "operations": saved.operations,
            "inserts": saved.inserts,
            "updates": saved.updates,
            "deletes": saved.deletes,
            "weight_edits": saved.weight_edits,
        }))
    }

    fn apply_patch(&mut self, p: PathArgs) -> Result<Value, Failure> {
        let name = self.workdir.name(&p.path)?;
        let applied_now = self
            .session
            .patches()
            .map_err(Failure::failed)?
            .applied
            .len() as u64;
        over(
            "the patches applied",
            applied_now + 1,
            self.limits.max_applied_patches,
            "max_applied_patches",
        )?;
        let bytes = self.workdir.read(&name, self.limits.max_patch_bytes)?;
        let staged = self.staging_path(&name)?;
        self.staged.clear(&name)?;
        {
            use std::io::Write;
            let mut copy = self.staged.create_new(&name)?;
            copy.write_all(&bytes)
                .and_then(|_| copy.sync_all())
                .map_err(Failure::failed)?;
        }
        let applied = self.session.apply_patch(&staged).map_err(Failure::failed)?;
        Ok(json!({
            "path": applied.path,
            "sha256": applied.sha256,
            "operations": applied.operations,
            "inserts": applied.inserts,
            "updates": applied.updates,
            "deletes": applied.deletes,
            "weight_edits": applied.weight_edits,
            "structured_ops": applied.structured_ops,
        }))
    }

    fn remove_patch(&mut self, p: PathArgs) -> Result<Value, Failure> {
        // A patch applied from the workdir is named by its staged copy; one carrying a description of its own
        // is named by that description. Removal touches no file.
        let name = self.workdir.name(&p.path).ok();
        let staged = name.as_ref().and_then(|n| self.staging_path(n).ok());
        let removed = match staged.map(|s| self.session.remove_patch(&s)) {
            Some(Ok(removed)) => {
                if let Some(n) = &name {
                    // The staged copy is no longer applied.
                    let _ = self.staged.clear(n);
                }
                removed
            }
            _ => self
                .session
                .remove_patch(&p.path)
                .map_err(Failure::failed)?,
        };
        Ok(json!({
            "number": removed.number,
            "carried_weight_ops": removed.carried_weight_ops,
        }))
    }

    fn patches(&mut self) -> Result<Value, Failure> {
        let report = self.session.patches().map_err(Failure::failed)?;
        Ok(json!({
            "applied": report.applied.iter().map(|p| json!({
                "name": p.name,
                "path": p.path,
                "sha256": p.sha256,
                "operations": p.operations,
                "inserts": p.inserts,
                "updates": p.updates,
                "deletes": p.deletes,
            })).collect::<Vec<_>>(),
            "session_overrides": report.session_overrides,
            "weight_ops": report.weight_ops,
            "weight_elements": report.weight_elements,
            "recording": report.recording.as_ref().map(|(path, operations)| json!({
                "path": path, "operations": operations,
            })),
        }))
    }
}
