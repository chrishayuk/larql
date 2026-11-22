//! GW-TEMPLATE-FACTORIAL: template-agnostic natural baseline.
//!
//! Natural inference only. It has no intervention, no donor and no
//! comparison: every step passes `InterventionPlan::none()`. It knows no
//! template. Each cell of a plan gives its prompt text, its token IDs and its
//! semantic coordinates (subject span, relation span, readout position).
//!
//! Before the model is loaded, it independently re-tokenises every cell's
//! text with the container's tokenizer, after checking that tokenizer's hash
//! against the plan. It requires the plan's token IDs to equal the
//! re-tokenisation. It recomputes each span from the declared character
//! offsets and requires it to equal the declared token span. Any fault
//! refuses.
//!
//! For every cell it teacher-forces the prompt, then continues greedily for
//! `continuation_tokens` tokens. It records:
//!
//! - one logit hash per executed step;
//! - the continuation;
//! - the verified coordinates;
//! - at the readout position (the last prompt token), the logits that choose
//!   the first answer token: the greedy token, the full-vocabulary
//!   log-normaliser, the top-k token IDs and logits, the logits of the
//!   cell's candidate set, and the hash of those logits.
//!
//! It does not decide eligibility. That is the scoring code's job, applied to
//! this record.
//!
//! A run writes OUT.jsonl.partial while running, then OUT.jsonl and
//! OUT.jsonl.manifest.json only on completion. It never overwrites.
//! `--check PLAN` runs every plan and tokenisation check, loads no model and
//! writes nothing.
//!
//! Usage:
//!   gwtemplate_baseline PLAN.json OUT.jsonl
//!   gwtemplate_baseline --check PLAN.json
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use larql_inference::vindex3::{CarrierWriteRecord, StepEvent, StepObserver};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    intervene::InterventionPlan,
    intervene_heads::HeadInterventionPlan,
    kv::RowKvState,
    operands::OperandStore,
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
};
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const PLAN_SCHEMA: &str = "larql.gwtemplate.baseline-plan.v1";
const RECORD_SCHEMA: &str = "larql.gwtemplate.baseline-record.v1";
const MANIFEST_SCHEMA: &str = "larql.gwtemplate.baseline-manifest.v1";
const TOKENIZER_FILE: &str = "tokenizer.json";
const MAX_CONTINUATION: u64 = 32;
const MAX_TOP_K: u64 = 256;

fn refuse<T>(reason: impl Into<String>) -> Result<T> {
    Err(format!("REFUSED: {}", reason.into()).into())
}

/// Observes nothing; counts layer scales.
#[derive(Default)]
struct Watch {
    scaled: usize,
}

impl StepObserver for Watch {
    fn event(&mut self, _: StepEvent) {}
    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.layer_scale.is_some() {
            self.scaled += 1;
        }
    }
}

struct Model {
    plan: ComponentOpPlan,
    ops: PreparedOperands,
    backend: ProductionBackend,
}

/// sha256 over the f32 bits, little-endian (the runners' encoding).
fn logits_sha(logits: &[f32]) -> String {
    let mut d = Sha256::new();
    for v in logits {
        d.update(v.to_bits().to_le_bytes());
    }
    format!("sha256:{:x}", d.finalize())
}

/// The largest logit, the lowest index on a tie. A non-finite logit refuses.
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

/// The readout at the readout position. Candidate logits, the log-normaliser
/// (summed in vocabulary order), the greedy token and the logit hash are
/// computed exactly as the qualified runners compute them. Top-k is ordered
/// by logit, descending, the lower index first on a tie.
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
    let top_ids: Vec<u32> = order.iter().map(|&i| i as u32).collect();
    let top_logits: Vec<f64> = order.iter().map(|&i| l[i] as f64).collect();
    Ok(json!({"next": best, "log_normalizer": lse, "candidate_logits": cand,
              "top_k_ids": top_ids, "top_k_logits": top_logits,
              "next_logit_sha256": logits_sha(l)}))
}

struct Span {
    tokens: Vec<usize>,
}

struct Cell {
    id: String,
    prompt: Vec<u32>,
    candidate_set: String,
    subject: Span,
    relation: Span,
    readout: usize,
    text: String,
}

struct Plan {
    json: Value,
    cells: Vec<Cell>,
    sets: BTreeMap<String, Vec<u32>>,
    continuation: usize,
    top_k: usize,
}

fn ids(v: &Value, what: &str) -> Result<Vec<u32>> {
    let a = match v.as_array() {
        Some(a) => a,
        None => return refuse(format!("{what}: not a token list")),
    };
    a.iter()
        .map(|t| match t.as_u64() {
            Some(x) if x <= u32::MAX as u64 => Ok(x as u32),
            _ => refuse(format!("{what}: not a token id")),
        })
        .collect()
}

fn positions(v: &Value, what: &str) -> Result<Vec<usize>> {
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

fn char_span(v: &Value, what: &str) -> Result<(usize, usize)> {
    match (v.get(0).and_then(Value::as_u64), v.get(1).and_then(Value::as_u64), v.as_array().map(Vec::len)) {
        (Some(a), Some(b), Some(2)) if a < b => Ok((a as usize, b as usize)),
        _ => refuse(format!("{what}: character span is not [start, end) with start < end")),
    }
}

/// The token span that a character span [s0, s1) occupies, by the
/// tokenizer's character offsets. Each token may begin one character early
/// only if that character is a space (the tokenizer's leading-space marker),
/// and no token may extend past s1 or begin before s0 - 1. The span must
/// cover [s0, s1) exactly.
fn locate(offsets: &[(usize, usize)], chars: &[char], s0: usize, s1: usize, what: &str) -> Result<Vec<usize>> {
    let toks: Vec<usize> = offsets
        .iter()
        .enumerate()
        .filter(|(_, &(a, b))| b > a && a < s1 && b > s0)
        .map(|(i, _)| i)
        .collect();
    if toks.is_empty() {
        return refuse(format!("{what}: no token covers the span"));
    }
    for &i in &toks {
        let (a, b) = offsets[i];
        let early_ok = a >= s0 || (a + 1 == s0 && chars.get(a) == Some(&' '));
        if !early_ok || b > s1 {
            return refuse(format!("{what}: token boundaries do not align with the span"));
        }
    }
    let first = offsets[toks[0]].0;
    let last = offsets[*toks.last().unwrap()].1;
    if first > s0 || last != s1 {
        return refuse(format!("{what}: tokens do not cover the span"));
    }
    Ok(toks)
}

fn file_sha(path: &Path) -> Result<String> {
    Ok(format!("sha256:{:x}", Sha256::digest(std::fs::read(path)?)))
}

/// Every plan fault, including tokenisation and coordinates, before the
/// model is loaded.
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
    let continuation = match plan["continuation_tokens"].as_u64() {
        Some(c) if (1..=MAX_CONTINUATION).contains(&c) => c as usize,
        _ => return refuse(format!("continuation_tokens must be 1..={MAX_CONTINUATION}")),
    };
    let top_k = match plan["top_k"].as_u64() {
        Some(k) if (1..=MAX_TOP_K).contains(&k) => k as usize,
        _ => return refuse(format!("top_k must be 1..={MAX_TOP_K}")),
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
                if s.is_empty() || s.iter().collect::<BTreeSet<_>>().len() != s.len() {
                    return refuse(format!("candidate set {name}: empty or repeated"));
                }
                if s.iter().any(|&t| t >= vocab) {
                    return refuse(format!("candidate set {name}: id outside the vocabulary"));
                }
                sets.insert(name.clone(), s);
            }
        }
        _ => return refuse("plan has no candidate_sets"),
    }
    let list = match plan["cells"].as_array() {
        Some(r) if !r.is_empty() => r,
        _ => return refuse("plan has no cells"),
    };
    let mut cells = Vec::new();
    let mut seen = BTreeSet::new();
    for c in list {
        let id = match c["id"].as_str() {
            Some(s) => s.to_string(),
            None => return refuse("a cell has no id"),
        };
        if !seen.insert(id.clone()) {
            return refuse(format!("{id}: cell repeated"));
        }
        let text = match c["text"].as_str() {
            Some(t) if !t.is_empty() => t.to_string(),
            _ => return refuse(format!("{id}: no text")),
        };
        let prompt = ids(&c["token_ids"], &format!("{id} token_ids"))?;
        let set = match c["candidate_set"].as_str() {
            Some(s) if sets.contains_key(s) => s.to_string(),
            _ => return refuse(format!("{id}: unknown candidate_set")),
        };
        // Independent re-tokenisation.
        let enc = tokenizer
            .encode_char_offsets(text.as_str(), true)
            .or_else(|e| refuse(format!("{id}: tokenizer: {e}")))?;
        if enc.get_ids() != prompt.as_slice() {
            return refuse(format!("{id}: token_ids differ from the re-tokenised text"));
        }
        let chars: Vec<char> = text.chars().collect();
        let offsets = enc.get_offsets();
        let mut spans = Vec::new();
        for role in ["subject", "relation"] {
            let r = &c[role];
            let (s0, s1) = char_span(&r["char"], &format!("{id} {role}"))?;
            let want: String = match r["text"].as_str() {
                Some(t) => t.to_string(),
                None => return refuse(format!("{id} {role}: no text")),
            };
            if s1 > chars.len() || chars[s0..s1].iter().collect::<String>() != want {
                return refuse(format!("{id} {role}: character span does not hold its text"));
            }
            let declared = positions(&r["tokens"], &format!("{id} {role} tokens"))?;
            let located = locate(offsets, &chars, s0, s1, &format!("{id} {role}"))?;
            if declared != located {
                return refuse(format!("{id} {role}: declared token span differs from the located span"));
            }
            spans.push(Span { tokens: declared });
        }
        let relation = spans.pop().unwrap();
        let subject = spans.pop().unwrap();
        let n = prompt.len();
        let readout = match c["readout"].as_u64() {
            Some(r) if r as usize + 1 == n => r as usize,
            _ => return refuse(format!("{id}: readout is not the last prompt token")),
        };
        let (s_last, r_first) = (*subject.tokens.last().unwrap(), relation.tokens[0]);
        let (r_last, s_first) = (*relation.tokens.last().unwrap(), subject.tokens[0]);
        if !(s_last < r_first || r_last < s_first) {
            return refuse(format!("{id}: subject and relation spans overlap"));
        }
        if s_last >= readout || r_last >= readout {
            return refuse(format!("{id}: a span reaches the readout position"));
        }
        cells.push(Cell { id, prompt, candidate_set: set, subject, relation, readout, text });
    }
    Ok(Plan { json: plan, cells, sets, continuation, top_k })
}

fn run_cell(model: &Model, plan: &Plan, c: &Cell) -> Result<Value> {
    let mut watch = Watch::default();
    let mut kv = RowKvState::default();
    let none = HeadInterventionPlan::none();
    let nothing = InterventionPlan::none();
    let mut step_logit_sha = Vec::new();
    let mut continuation = Vec::with_capacity(plan.continuation);
    let mut firings = 0usize;
    let first;
    {
        let mut session = DecodeSession::over_prepared(&model.plan, &model.ops, &model.backend, &mut kv)?;
        let mut last = Vec::new();
        for &t in &c.prompt {
            let out = session.step_intervened(t, &mut watch, &nothing, &none)?;
            firings += out.firings.len();
            last = out.logits.ok_or("a step returned no logits")?;
            step_logit_sha.push(logits_sha(&last));
        }
        first = readout(&last, &plan.sets[&c.candidate_set], plan.top_k)?;
        for step in 0..plan.continuation {
            let token = argmax(&last)? as u32;
            continuation.push(token);
            if step + 1 < plan.continuation {
                let out = session.step_intervened(token, &mut watch, &nothing, &none)?;
                firings += out.firings.len();
                last = out.logits.ok_or("a step returned no logits")?;
                step_logit_sha.push(logits_sha(&last));
            }
        }
    }
    if firings != 0 {
        return refuse(format!("{}: a natural run reported a firing", c.id));
    }
    if watch.scaled != 0 {
        return refuse(format!("{}: a carrier write carried a layer scale", c.id));
    }
    if first["next"].as_u64() != Some(continuation[0] as u64) {
        return refuse(format!("{}: readout and continuation disagree", c.id));
    }
    Ok(json!({
        "schema": RECORD_SCHEMA,
        "id": c.id,
        "n": c.prompt.len(),
        "text": c.text,
        "prompt_tokens": c.prompt,
        "coordinates": {"subject": c.subject.tokens, "relation": c.relation.tokens, "readout": c.readout},
        "candidate_set": c.candidate_set,
        "continuation": continuation,
        "step_logit_sha256": step_logit_sha,
        "readout": first,
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 2 && args[0] == "--check" {
        let plan = load_plan(&args[1])?;
        println!("checked {} cells", plan.cells.len());
        return Ok(());
    }
    if args.len() != 2 {
        return Err("Usage: gwtemplate_baseline PLAN.json OUT.jsonl | --check PLAN.json".into());
    }
    let plan = load_plan(&args[0])?;
    let out_path = &args[1];
    let partial = format!("{out_path}.partial");
    if Path::new(out_path).exists() || Path::new(&format!("{out_path}.manifest.json")).exists() {
        return refuse("output exists; a run never overwrites");
    }
    if Path::new(&partial).exists() {
        return refuse("a partial output exists; a run never overwrites");
    }
    let container = Path::new(plan.json["container"].as_str().unwrap_or_default());
    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    let cplan = outcome.plan.ok_or("plan absent")?;
    if cplan.layers.iter().any(|l| l.layer_scale.is_some()) {
        return refuse("the plan declares a layer scale");
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&cplan, &store, &backend, ExecutionSlice::Full)?;
    let layers = cplan.layers.len();
    let model = Model { plan: cplan, ops, backend };
    let mut out = File::options().write(true).create_new(true).open(&partial)?;
    for c in &plan.cells {
        let record = run_cell(&model, &plan, c)?;
        writeln!(out, "{record}")?;
        eprintln!("ran {}", c.id);
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
        "cells": plan.cells.len(), "layers": layers,
        "continuation_tokens": plan.continuation, "top_k": plan.top_k,
    });
    std::fs::write(format!("{out_path}.manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}
