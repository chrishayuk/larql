//! GW-TEMPLATE-FACTORIAL stage 3: the generic cross-position subject-state
//! runner.
//!
//! It knows no template. A plan names token runs. Each target runs one token
//! run and reads out at its last token. Each arm names a donor run, a source
//! span in the donor and an equal-length target span in the target (mapped
//! position by position), a start layer `a`, and a mode:
//!
//! - `replace`: each target-span position takes the donor's FFN carrier at
//!   the mapped source position for layers a-1 .. L-1. The target's own
//!   key/value rows stand at layers 0 .. a-1, and from layer a on they are
//!   recomputed from the donor's carrier **at the target position**, under
//!   the target position's rotary encoding. So V rows equal the donor's
//!   (bit for bit, checked), and K rows equal the donor's only when the
//!   source and target positions coincide (checked). Otherwise they are
//!   recorded, and with `kv_dump` written out for an independent check.
//!   This is step C's W-subject-L{a} arm, with a position map.
//! - `norm_match`: as `replace`, but each injected vector is the donor
//!   vector rescaled to the L2 norm of the target's natural carrier at that
//!   layer and target position.
//!
//! Per target it checks:
//!
//! - **natural replay:** two natural runs are bit-identical; optionally the
//!   last-step logit hash equals a declared hash (the admitted baseline);
//! - **declared carriers:** every run's declared carrier hashes (from a
//!   seal) equal the run's natural carriers.
//!
//! Per arm it checks:
//!
//! - **landing:** one firing per address, and the carrier there is the
//!   injected one;
//! - **locality:** every step and every K/V row before the first target-span
//!   position equals the natural run;
//! - **provenance:** K/V at target-span positions are natural at layers
//!   0 .. a-1. From layer a: V equals the donor's, and K equals the donor's
//!   when the positions coincide;
//! - **identity:** an arm declared identity (the target as its own donor,
//!   same span) is bit-identical to the natural run at every step and every
//!   K/V row.
//!
//! Every plan fault refuses before the model loads, including:
//!
//! - spans not contiguous, of unequal length, outside the run, or reaching
//!   the readout;
//! - span token IDs that differ from the tokens at the declared positions,
//!   which catches wrong source or target positions;
//! - text that does not re-tokenise to the declared IDs;
//! - an identity arm that is not a self-transplant, or a self-transplant not
//!   declared identity;
//! - malformed declared carrier hashes.
//!
//! A carrier that differs from its declared hash refuses at run time. The
//! runner writes OUT.jsonl.partial while running and never overwrites.
//!
//! Usage:
//!   gwtemplate_xpos_runner PLAN.json OUT.jsonl
//!   gwtemplate_xpos_runner --check PLAN.json
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use larql_inference::vindex3::{CarrierWriteRecord, StepEvent, StepObserver, SublayerSite};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    intervene::{vector_sha256, Address, Intervention, InterventionPlan, VectorProvenance},
    intervene_heads::HeadInterventionPlan,
    kv::{KvState, RowKvState},
    operands::OperandStore,
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
};
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const PLAN_SCHEMA: &str = "larql.gwtemplate.xpos-plan.v1";
const RECORD_SCHEMA: &str = "larql.gwtemplate.xpos-record.v1";
const MANIFEST_SCHEMA: &str = "larql.gwtemplate.xpos-manifest.v1";
const TOKENIZER_FILE: &str = "tokenizer.json";
const MAX_TOP_K: u64 = 256;

fn refuse<T>(reason: impl Into<String>) -> Result<T> {
    Err(format!("REFUSED: {}", reason.into()).into())
}

// ── execution ─────────────────────────────────────────────────────────────

#[derive(Default)]
struct Carriers {
    ffn: BTreeMap<(usize, usize), (Vec<f32>, usize)>,
    scaled: usize,
}

impl StepObserver for Carriers {
    fn event(&mut self, _: StepEvent) {}
    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.site == SublayerSite::Ffn {
            let e = self.ffn.entry((record.layer, record.position)).or_default();
            e.0 = record.after.to_vec();
            e.1 += 1;
        }
        if record.layer_scale.is_some() {
            self.scaled += 1;
        }
    }
}

struct Run {
    carriers: Carriers,
    firings: usize,
    step_logit_sha: Vec<String>,
    last_logits: Vec<f32>,
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
}

fn row_sha(k: &[f32], v: &[f32]) -> String {
    let mut d = Sha256::new();
    for x in k.iter().chain(v) {
        d.update(x.to_bits().to_le_bytes());
    }
    format!("sha256:{:x}", d.finalize())
}

impl Run {
    fn kv_sha(&self, layer: usize, position: usize) -> String {
        row_sha(&self.keys[layer][position], &self.values[layer][position])
    }
    fn k_eq(&self, o: &Run, layer: usize, p: usize, q: usize) -> bool {
        self.keys[layer][p] == o.keys[layer][q]
    }
    fn v_eq(&self, o: &Run, layer: usize, p: usize, q: usize) -> bool {
        self.values[layer][p] == o.values[layer][q]
    }
}

struct Model {
    plan: ComponentOpPlan,
    ops: PreparedOperands,
    backend: ProductionBackend,
    layers: usize,
}

fn logits_sha(logits: &[f32]) -> String {
    let mut d = Sha256::new();
    for v in logits {
        d.update(v.to_bits().to_le_bytes());
    }
    format!("sha256:{:x}", d.finalize())
}

fn execute(model: &Model, tokens: &[u32], plan: &InterventionPlan) -> Result<Run> {
    let mut carriers = Carriers::default();
    let mut kv = RowKvState::default();
    let (mut firings, mut step_logit_sha, mut last_logits) = (0, Vec::new(), Vec::new());
    {
        let mut session = DecodeSession::over_prepared(&model.plan, &model.ops, &model.backend, &mut kv)?;
        let none = HeadInterventionPlan::none();
        for &t in tokens {
            let out = session.step_intervened(t, &mut carriers, plan, &none)?;
            firings += out.firings.len();
            let logits = out.logits.ok_or("a step returned no logits")?;
            step_logit_sha.push(logits_sha(&logits));
            last_logits = logits;
        }
    }
    if carriers.scaled != 0 {
        return refuse("a carrier write carried a layer scale");
    }
    let (mut keys, mut values) = (Vec::new(), Vec::new());
    for layer in 0..model.layers {
        kv.prepare_layer(layer);
        let (k, v) = kv.rows(layer).to_owned_rows();
        keys.push(k);
        values.push(v);
    }
    Ok(Run { carriers, firings, step_logit_sha, last_logits, keys, values })
}

fn argmax(logits: &[f32]) -> Result<usize> {
    if logits.is_empty() {
        return refuse("empty logits");
    }
    let mut best = 0;
    for (i, v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return refuse("non-finite logit");
        }
        if *v > logits[best] {
            best = i;
        }
    }
    Ok(best)
}

/// As the baseline executable: greedy token, log-normaliser, candidate
/// logits, top-k (descending, lower index first on a tie), logit hash.
fn readout(l: &[f32], candidates: &[u32], top_k: usize) -> Result<Value> {
    let best = argmax(l)?;
    let max = l[best] as f64;
    let lse = max + l.iter().map(|v| (*v as f64 - max).exp()).sum::<f64>().ln();
    let cand: Vec<f64> = candidates
        .iter()
        .map(|&c| l.get(c as usize).map(|v| *v as f64).ok_or("candidate id out of vocabulary"))
        .collect::<std::result::Result<_, _>>()?;
    let mut order: Vec<usize> = (0..l.len()).collect();
    order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).expect("finite").then(a.cmp(&b)));
    order.truncate(top_k);
    if order.first() != Some(&best) {
        return refuse("top-k does not begin with the greedy token");
    }
    Ok(json!({"next": best, "log_normalizer": lse, "candidate_logits": cand,
              "top_k_ids": order.iter().map(|&i| i as u32).collect::<Vec<_>>(),
              "top_k_logits": order.iter().map(|&i| l[i] as f64).collect::<Vec<_>>(),
              "next_logit_sha256": logits_sha(l)}))
}

fn hex_f32(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 8);
    for x in v {
        for b in x.to_le_bytes() {
            s.push_str(&format!("{b:02x}"));
        }
    }
    s
}

fn norm(v: &[f32]) -> f64 {
    v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
}

// ── plan ──────────────────────────────────────────────────────────────────

fn ids(v: &Value, what: &str) -> Result<Vec<u32>> {
    let a = match v.as_array() {
        Some(a) if !a.is_empty() => a,
        _ => return refuse(format!("{what}: empty or not a token list")),
    };
    a.iter()
        .map(|t| match t.as_u64() {
            Some(x) if x <= u32::MAX as u64 => Ok(x as u32),
            _ => refuse(format!("{what}: not a token id")),
        })
        .collect()
}

fn span(v: &Value, what: &str) -> Result<Vec<usize>> {
    let a = match v.as_array() {
        Some(a) if !a.is_empty() => a,
        _ => return refuse(format!("{what}: empty or not a position list")),
    };
    let p: Vec<usize> = a
        .iter()
        .map(|t| t.as_u64().map(|x| x as usize).ok_or("x"))
        .collect::<std::result::Result<_, _>>()
        .or_else(|_| refuse(format!("{what}: not a position")))?;
    if p.windows(2).any(|w| w[1] != w[0] + 1) {
        return refuse(format!("{what}: not contiguous ascending"));
    }
    Ok(p)
}

fn file_sha(path: &Path) -> Result<String> {
    Ok(format!("sha256:{:x}", Sha256::digest(std::fs::read(path)?)))
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Replace,
    NormMatch,
}

struct Arm {
    name: String,
    donor: String,
    source: Vec<usize>,
    target: Vec<usize>,
    from: usize,
    mode: Mode,
    identity: bool,
}

struct Target {
    id: String,
    run: String,
    set: String,
    expect_natural: Option<String>,
    arms: Vec<Arm>,
}

struct TokenRun {
    tokens: Vec<u32>,
    expected: BTreeMap<(usize, usize), String>,
}

struct Plan {
    json: Value,
    runs: BTreeMap<String, TokenRun>,
    targets: Vec<Target>,
    sets: BTreeMap<String, Vec<u32>>,
    top_k: usize,
    layers: usize,
    kv_dump: bool,
}

fn load_plan(path: &str) -> Result<Plan> {
    let plan: Value = match std::fs::read(path).map(|b| serde_json::from_slice::<Value>(&b)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return refuse(format!("plan is not JSON: {e}")),
        Err(e) => return refuse(format!("plan unreadable: {e}")),
    };
    if plan["schema"] != PLAN_SCHEMA {
        return refuse("plan schema");
    }
    let container = match plan["container"].as_str() {
        Some(c) => Path::new(c).to_path_buf(),
        None => return refuse("plan has no container"),
    };
    let layers = match plan["layers"].as_u64() {
        Some(l) if l >= 1 => l as usize,
        _ => return refuse("plan has no layer count"),
    };
    let top_k = match plan["top_k"].as_u64() {
        Some(k) if (1..=MAX_TOP_K).contains(&k) => k as usize,
        _ => return refuse(format!("top_k must be 1..={MAX_TOP_K}")),
    };
    let kv_dump = match &plan["kv_dump"] {
        Value::Bool(b) => *b,
        Value::Null => false,
        _ => return refuse("kv_dump must be a boolean"),
    };
    let tok_path = container.join(TOKENIZER_FILE);
    let want_tok = match plan["tokenizer_sha256"].as_str() {
        Some(s) => s.to_string(),
        None => return refuse("plan has no tokenizer_sha256"),
    };
    if !tok_path.exists() || file_sha(&tok_path)? != want_tok {
        return refuse("tokenizer hash differs from the plan's");
    }
    let tokenizer = Tokenizer::from_file(&tok_path).or_else(|e| refuse(format!("tokenizer: {e}")))?;
    let vocab = tokenizer.get_vocab_size(true) as u32;

    let mut sets = BTreeMap::new();
    match plan["candidate_sets"].as_object() {
        Some(m) if !m.is_empty() => {
            for (name, v) in m {
                let s = ids(v, &format!("candidate set {name}"))?;
                if s.iter().collect::<BTreeSet<_>>().len() != s.len() || s.iter().any(|&t| t >= vocab) {
                    return refuse(format!("candidate set {name}: repeated or outside the vocabulary"));
                }
                sets.insert(name.clone(), s);
            }
        }
        _ => return refuse("plan has no candidate_sets"),
    }

    let mut runs = BTreeMap::new();
    match plan["runs"].as_object() {
        Some(m) if !m.is_empty() => {
            for (id, r) in m {
                let tokens = ids(&r["tokens"], &format!("run {id} tokens"))?;
                if tokens.iter().any(|&t| t >= vocab) {
                    return refuse(format!("run {id}: token outside the vocabulary"));
                }
                if let Some(text) = r["text"].as_str() {
                    let enc = tokenizer
                        .encode_char_offsets(text, true)
                        .or_else(|e| refuse(format!("run {id}: tokenizer: {e}")))?;
                    if enc.get_ids() != tokens.as_slice() {
                        return refuse(format!("run {id}: tokens differ from the re-tokenised text"));
                    }
                } else if !r["text"].is_null() {
                    return refuse(format!("run {id}: text is not a string"));
                }
                let mut expected = BTreeMap::new();
                match &r["expected_carrier_sha256"] {
                    Value::Null => {}
                    Value::Object(e) => {
                        for (k, v) in e {
                            let parts: Vec<&str> = k.split('|').collect();
                            let lp = match (parts.as_slice(), v.as_str()) {
                                // vector_sha256's form: 64 lowercase hex digits, no prefix.
                                ([l, p], Some(s)) if s.len() == 64
                                    && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) => {
                                    match (l.parse::<usize>(), p.parse::<usize>()) {
                                        (Ok(l), Ok(p)) if l < layers && p < tokens.len() => (l, p),
                                        _ => return refuse(format!("run {id}: declared carrier key {k} out of range")),
                                    }
                                }
                                _ => return refuse(format!("run {id}: declared carrier {k} malformed")),
                            };
                            expected.insert(lp, v.as_str().unwrap().to_string());
                        }
                    }
                    _ => return refuse(format!("run {id}: expected_carrier_sha256 malformed")),
                }
                runs.insert(id.clone(), TokenRun { tokens, expected });
            }
        }
        _ => return refuse("plan has no runs"),
    }

    let list = match plan["targets"].as_array() {
        Some(t) if !t.is_empty() => t,
        _ => return refuse("plan has no targets"),
    };
    let mut targets = Vec::new();
    let mut seen = BTreeSet::new();
    for t in list {
        let id = match t["id"].as_str() {
            Some(s) => s.to_string(),
            None => return refuse("a target has no id"),
        };
        if !seen.insert(id.clone()) {
            return refuse(format!("{id}: target repeated"));
        }
        let run = match t["run"].as_str() {
            Some(r) if runs.contains_key(r) => r.to_string(),
            _ => return refuse(format!("{id}: unknown run")),
        };
        let set = match t["candidate_set"].as_str() {
            Some(s) if sets.contains_key(s) => s.to_string(),
            _ => return refuse(format!("{id}: unknown candidate_set")),
        };
        let expect_natural = match &t["expect_natural_logit_sha256"] {
            Value::Null => None,
            Value::String(s) if s.starts_with("sha256:") => Some(s.clone()),
            _ => return refuse(format!("{id}: expect_natural_logit_sha256 malformed")),
        };
        let ttoks = &runs[&run].tokens;
        let readout = ttoks.len() - 1;
        let arm_list = match t["arms"].as_array() {
            Some(a) if !a.is_empty() => a,
            _ => return refuse(format!("{id}: no arms")),
        };
        let mut arms = Vec::new();
        let mut names = BTreeSet::new();
        for a in arm_list {
            let name = match a["name"].as_str() {
                Some(n) => n.to_string(),
                None => return refuse(format!("{id}: an arm has no name")),
            };
            if !names.insert(name.clone()) {
                return refuse(format!("{id} {name}: arm repeated"));
            }
            let what = format!("{id} {name}");
            let donor = match a["donor_run"].as_str() {
                Some(d) if runs.contains_key(d) => d.to_string(),
                _ => return refuse(format!("{what}: unknown donor run")),
            };
            let dtoks = &runs[&donor].tokens;
            let source = span(&a["source_span"], &format!("{what} source_span"))?;
            let target = span(&a["target_span"], &format!("{what} target_span"))?;
            if source.len() != target.len() {
                return refuse(format!("{what}: source and target spans differ in length"));
            }
            if *source.last().unwrap() >= dtoks.len() {
                return refuse(format!("{what}: source span outside the donor run"));
            }
            if *target.last().unwrap() >= readout {
                return refuse(format!("{what}: target span reaches the readout position"));
            }
            let sids = ids(&a["source_span_ids"], &format!("{what} source_span_ids"))?;
            let tids = ids(&a["target_span_ids"], &format!("{what} target_span_ids"))?;
            if sids != source.iter().map(|&p| dtoks[p]).collect::<Vec<_>>() {
                return refuse(format!("{what}: source span token ids differ from the donor run at those positions"));
            }
            if tids != target.iter().map(|&p| ttoks[p]).collect::<Vec<_>>() {
                return refuse(format!("{what}: target span token ids differ from the target run at those positions"));
            }
            let from = match a["from_layer"].as_u64() {
                Some(f) if f >= 1 && (f as usize) <= layers - 1 => f as usize,
                _ => return refuse(format!("{what}: from_layer must be 1..={}", layers - 1)),
            };
            let mode = match a["mode"].as_str() {
                Some("replace") => Mode::Replace,
                Some("norm_match") => Mode::NormMatch,
                _ => return refuse(format!("{what}: mode must be replace or norm_match")),
            };
            let identity = a["identity"].as_bool().unwrap_or(false);
            let self_transplant = donor == run && source == target;
            if identity != self_transplant {
                return refuse(format!("{what}: identity must be declared exactly for a self-transplant"));
            }
            if identity && mode != Mode::Replace {
                return refuse(format!("{what}: an identity arm must use replace"));
            }
            arms.push(Arm { name, donor, source, target, from, mode, identity });
        }
        targets.push(Target { id, run, set, expect_natural, arms });
    }
    Ok(Plan { json: plan, runs, targets, sets, top_k, layers, kv_dump })
}

// ── one target ────────────────────────────────────────────────────────────

fn check_declared(id: &str, tr: &TokenRun, run: &Run) -> Result<()> {
    for ((l, p), want) in &tr.expected {
        match run.carriers.ffn.get(&(*l, *p)) {
            Some((v, 1)) if vector_sha256(v) == *want => {}
            _ => return refuse(format!("run {id}: carrier {l}|{p} differs from its declared hash")),
        }
    }
    Ok(())
}

fn run_target(model: &Model, plan: &Plan, t: &Target) -> Result<Value> {
    let tr = &plan.runs[&t.run];
    let tokens = &tr.tokens;
    let natural = execute(model, tokens, &InterventionPlan::none())?;
    let replay = execute(model, tokens, &InterventionPlan::none())?;
    if natural.step_logit_sha != replay.step_logit_sha || natural.keys != replay.keys
        || natural.values != replay.values || natural.last_logits != replay.last_logits {
        return refuse(format!("{}: natural replay is not bit-identical", t.id));
    }
    if natural.firings != 0 {
        return refuse(format!("{}: the natural run reported a firing", t.id));
    }
    if let Some(want) = &t.expect_natural {
        if natural.step_logit_sha.last() != Some(want) {
            return refuse(format!("{}: natural readout differs from the declared baseline hash", t.id));
        }
    }
    check_declared(&t.run, tr, &natural)?;
    let cands = &plan.sets[&t.set];
    let mut donors: BTreeMap<String, Run> = BTreeMap::new();
    let mut arm_records = serde_json::Map::new();
    for arm in &t.arms {
        let what = format!("{} {}", t.id, arm.name);
        if !donors.contains_key(&arm.donor) {
            let dr = &plan.runs[&arm.donor];
            let run = execute(model, &dr.tokens, &InterventionPlan::none())?;
            if run.firings != 0 {
                return refuse(format!("run {}: a donor's natural run reported a firing", arm.donor));
            }
            check_declared(&arm.donor, dr, &run)?;
            donors.insert(arm.donor.clone(), run);
        }
        let donor = &donors[&arm.donor];
        let mut iplan = InterventionPlan::none();
        let mut injected = Vec::new();
        let mut norms = serde_json::Map::new();
        for layer in arm.from - 1..model.layers {
            for (&sp, &tp) in arm.source.iter().zip(&arm.target) {
                let dv = match donor.carriers.ffn.get(&(layer, sp)) {
                    Some(c) => c.0.clone(),
                    None => return refuse(format!("{what}: donor carrier absent at {layer}|{sp}")),
                };
                let vector = match arm.mode {
                    Mode::Replace => dv,
                    Mode::NormMatch => {
                        let tv = &natural.carriers.ffn.get(&(layer, tp)).ok_or("target carrier absent")?.0;
                        let (nt, nd) = (norm(tv), norm(&dv));
                        if nd == 0.0 {
                            return refuse(format!("{what}: zero donor carrier at {layer}|{sp}"));
                        }
                        let s = nt / nd;
                        let out: Vec<f32> = dv.iter().map(|x| (*x as f64 * s) as f32).collect();
                        norms.insert(format!("{layer}|{tp}"),
                                     json!({"target": nt, "donor": nd, "injected": norm(&out)}));
                        out
                    }
                };
                let sha = vector_sha256(&vector);
                let provenance = VectorProvenance::Captured {
                    run_id: arm.donor.clone(), layer, site: SublayerSite::Ffn, position: sp, sha256: sha.clone(),
                };
                iplan = iplan.with(Intervention::replace(
                    Address::new(layer, SublayerSite::Ffn, [tp])?, vector, provenance)?)?;
                injected.push(((layer, tp), sha));
            }
        }
        let run = execute(model, tokens, &iplan)?;
        if run.firings != injected.len() {
            return refuse(format!("{what}: {} firings for {} addresses", run.firings, injected.len()));
        }
        for ((layer, position), sha) in &injected {
            match run.carriers.ffn.get(&(*layer, *position)) {
                Some((v, 1)) if vector_sha256(v) == *sha => {}
                _ => return refuse(format!("{what}: landing at {layer}|{position}")),
            }
        }
        let first = arm.target[0];
        if run.step_logit_sha[..first] != natural.step_logit_sha[..first] {
            return refuse(format!("{what}: a step before the target span changed"));
        }
        for layer in 0..model.layers {
            for position in 0..first {
                if run.kv_sha(layer, position) != natural.kv_sha(layer, position) {
                    return refuse(format!("{what}: earlier position {position} changed at layer {layer}"));
                }
            }
        }
        let mut kv_hashes = serde_json::Map::new();
        let mut k_equal_donor = serde_json::Map::new();
        let mut dump = serde_json::Map::new();
        for (&sp, &tp) in arm.source.iter().zip(&arm.target) {
            for layer in 0..model.layers {
                if layer < arm.from {
                    if !(run.k_eq(&natural, layer, tp, tp) && run.v_eq(&natural, layer, tp, tp)) {
                        return refuse(format!("{what}: provenance at {layer}|{tp}: not natural below the start layer"));
                    }
                } else if arm.mode == Mode::Replace {
                    if !run.v_eq(donor, layer, tp, sp) {
                        return refuse(format!("{what}: provenance at {layer}|{tp}: V is not the donor's"));
                    }
                    let keq = run.k_eq(donor, layer, tp, sp);
                    if sp == tp && !keq {
                        return refuse(format!("{what}: provenance at {layer}|{tp}: K is not the donor's at the same position"));
                    }
                    k_equal_donor.insert(format!("{layer}|{tp}"), json!(keq));
                    if plan.kv_dump {
                        dump.insert(format!("{layer}|{sp}|{tp}"), json!({
                            "target_k": hex_f32(&run.keys[layer][tp]),
                            "donor_k": hex_f32(&donor.keys[layer][sp]),
                        }));
                    }
                }
                kv_hashes.insert(format!("{layer}|{tp}"), json!(run.kv_sha(layer, tp)));
            }
        }
        let mut record = json!({
            "donor_run": arm.donor, "mode": if arm.mode == Mode::Replace { "replace" } else { "norm_match" },
            "from_layer": arm.from, "source_span": arm.source, "target_span": arm.target,
            "replacements": injected.len(), "firings": run.firings,
            "carriers": injected.iter().map(|((l, p), s)| (format!("{l}|{p}"), json!(s)))
                .collect::<serde_json::Map<_, _>>(),
            "kv_sha256": kv_hashes,
            "donor_kv_sha256": arm.source.iter().flat_map(|&sp| (0..model.layers)
                .map(move |l| (l, sp))).map(|(l, sp)| (format!("{l}|{sp}"), json!(donor.kv_sha(l, sp))))
                .collect::<serde_json::Map<_, _>>(),
            "k_equal_donor": k_equal_donor,
            "step_logit_sha256": run.step_logit_sha,
            "readout": readout(&run.last_logits, cands, plan.top_k)?,
            "donor_natural_last_logit_sha256": donor.step_logit_sha.last(),
        });
        if arm.mode == Mode::NormMatch {
            record["norms"] = Value::Object(norms);
        }
        if plan.kv_dump && arm.mode == Mode::Replace {
            record["kv_dump"] = Value::Object(dump);
        }
        if arm.identity {
            if run.step_logit_sha != natural.step_logit_sha || run.keys != natural.keys
                || run.values != natural.values || run.last_logits != natural.last_logits {
                return refuse(format!("{what}: identity is not bit-identical to the natural run"));
            }
            record["identity_bit_exact"] = json!(true);
        }
        arm_records.insert(arm.name.clone(), record);
    }
    Ok(json!({
        "schema": RECORD_SCHEMA, "id": t.id, "run": t.run, "n": tokens.len(),
        "natural": {"readout": readout(&natural.last_logits, cands, plan.top_k)?,
                    "step_logit_sha256": natural.step_logit_sha, "replay_bit_exact": true,
                    "declared_carriers_checked": tr.expected.len()},
        "arms": arm_records,
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 2 && args[0] == "--check" {
        let plan = load_plan(&args[1])?;
        let arms: usize = plan.targets.iter().map(|t| t.arms.len()).sum();
        println!("checked {} targets, {} arms, {} runs", plan.targets.len(), arms, plan.runs.len());
        return Ok(());
    }
    if args.len() != 2 {
        return Err("Usage: gwtemplate_xpos_runner PLAN.json OUT.jsonl | --check PLAN.json".into());
    }
    let plan = load_plan(&args[0])?;
    let out_path = &args[1];
    let partial = format!("{out_path}.partial");
    for p in [out_path.clone(), partial.clone(), format!("{out_path}.manifest.json")] {
        if Path::new(&p).exists() {
            return refuse("output exists; a run never overwrites");
        }
    }
    let container = Path::new(plan.json["container"].as_str().unwrap_or_default());
    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    let cplan = outcome.plan.ok_or("plan absent")?;
    if cplan.layers.iter().any(|l| l.layer_scale.is_some()) {
        return refuse("the plan declares a layer scale");
    }
    if cplan.layers.len() != plan.layers {
        return refuse("the plan's layer count differs from the model's");
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&cplan, &store, &backend, ExecutionSlice::Full)?;
    let layers = cplan.layers.len();
    let model = Model { plan: cplan, ops, backend, layers };
    let mut out = File::options().write(true).create_new(true).open(&partial)?;
    for t in &plan.targets {
        let record = run_target(&model, &plan, t)?;
        writeln!(out, "{record}")?;
        eprintln!("ran {}", t.id);
    }
    drop(out);
    std::fs::rename(&partial, out_path)?;
    let bytes = std::fs::read(out_path)?;
    let exe = std::fs::read(std::env::current_exe()?)?;
    let manifest = json!({
        "schema": MANIFEST_SCHEMA, "status": "complete",
        "executable_sha256": format!("sha256:{:x}", Sha256::digest(&exe)),
        "plan_sha256": format!("sha256:{:x}", Sha256::digest(std::fs::read(&args[0])?)),
        "output_sha256": format!("sha256:{:x}", Sha256::digest(&bytes)),
        "tokenizer_sha256": plan.json["tokenizer_sha256"],
        "targets": plan.targets.len(), "layers": layers, "top_k": plan.top_k, "kv_dump": plan.kv_dump,
    });
    std::fs::write(format!("{out_path}.manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}
