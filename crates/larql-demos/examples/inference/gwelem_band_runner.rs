//! GW-ELEMENT-1: the element-domain band runner. It is the qualified
//! GW-COMPRESS-1 band runner (`gwcompress_band_runner`, source e8f7de1c…,
//! binary 3ac83622…, left unchanged) plus ONE addition: a compiled-in
//! **sealed population** with an explicit, recorded release.
//!
//! Execution semantics, plan schema, record schema and every per-target and
//! per-arm check are the band runner's, byte for byte. See that file's header
//! for bands, locality, landing, the propagation fixpoint and identity.
//!
//! **Refusals (all before the model is loaded):**
//! - the plan's own holdout list, as before;
//! - the 28 spent Test 3 holdout countries, as before. This is **unconditional**:
//!   no release can lift it;
//! - **sealed population `gw-element-1/confirmation`** (the 35-element salted
//!   confirmation split). A run is refused if either:
//!   - its entity key (`country`) or any `|`-part of its id is a sealed key; or
//!   - its token sequence contains the subject token ids of any sealed element
//!     in any of four surface forms (lower-case or capitalised, mid-sentence or
//!     sentence-initial). This catches a mislabelled run whatever its key says.
//!
//! **Release.** A plan may declare `sealed_release`:
//! `{"population": "gw-element-1/confirmation", "authorisation_file": PATH,
//!   "authorisation_sha256": "sha256:…"}`.
//! The file must exist and hash to the declared value. It must be JSON with
//! - `schema` = `larql.gwelem.confirmation-authorisation.v1`;
//! - `population` = the sealed population name;
//! - a non-empty `owner_go`;
//! - `runs_digest` equal to this plan's runs digest: sha256 over the run ids
//!   in sorted order of `id|country|t1,t2,…\n`. So an authorisation releases
//!   exactly one set of runs and cannot be carried to another plan.
//!
//! **Single use.** A release is refused if `AUTHORISATION_FILE.used` exists.
//! An execution (not `--check`) creates that marker, exclusively, before the
//! model is loaded. So a released population is opened at most once, even if
//! the run then fails.
//!
//! The spent-holdout check also matches any `|`-part of a run id, not only the
//! `country` field. This is a deliberate tightening over the band runner.
//!
//! Any malformed or mismatching release refuses the whole plan. A valid
//! release lifts only this population's seal. The manifest records the release
//! (population and authorisation hash), or `null`.
//!
//! The release is a procedural guard against accidental execution, not a
//! cryptographic one. The confirmation orchestrator writes the authorisation
//! only from the owner's recorded GO.
//!
//! Usage:
//!   gwelem_band_runner --sealed                 (print the compiled seal as JSON)
//!   gwelem_band_runner --check PLAN.json
//!   gwelem_band_runner PLAN.json OUT.jsonl
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};

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

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const PLAN_SCHEMA: &str = "larql.gwcompress.band-plan.v1";
const RECORD_SCHEMA: &str = "larql.gwcompress.band-record.v1";
const MANIFEST_SCHEMA: &str = "larql.gwelem.band-manifest.v1";
/// The 28 Test 3 holdout countries, spent on 2026-11-20 (run …01238). No
/// plan can execute them, whatever its own holdout list says.
const SPENT_HOLDOUT: [&str; 28] = [
    "AF", "AS", "AW", "BF", "CR", "CZ", "ET", "GH", "IN", "KY", "LA", "MM", "MW", "NE",
    "NG", "PK", "RS", "SB", "SN", "SS", "SY", "TO", "TR", "UA", "VE", "VI", "YE", "ZA",
];
/// The sealed GW-ELEMENT-1 confirmation split (35 elements, entity key = atomic number).
const SEALED_POPULATION: &str = "gw-element-1/confirmation";
const AUTH_SCHEMA: &str = "larql.gwelem.confirmation-authorisation.v1";
const SEALED_KEYS: [&str; 35] = [
    "5", "7", "10", "21", "22", "23", "24", "26", "27", "28", "36", "47",
    "49", "53", "58", "61", "63", "64", "68", "71", "79", "80", "86", "87",
    "90", "92", "93", "95", "96", "97", "101", "102", "110", "113", "116",
];
/// (atomic number, surface form, subject token ids) for every sealed element: lower-case and
/// capitalised, mid-sentence (after "For ") and sentence-initial (after BOS).
const SEALED_SUBJECTS: [(u32, &str, &[u32]); 140] = [
    (5, "mid_cap", &[16585, 498]),
    (5, "mid_lower", &[83748]),
    (5, "start_cap", &[67157, 498]),
    (5, "start_lower", &[4431, 498]),
    (7, "mid_cap", &[95086]),
    (7, "mid_lower", &[19846]),
    (7, "start_cap", &[62486, 13017]),
    (7, "start_lower", &[186131]),
    (10, "mid_cap", &[71844]),
    (10, "mid_lower", &[23915]),
    (10, "start_cap", &[143443]),
    (10, "start_lower", &[155488]),
    (21, "mid_cap", &[99753, 1940]),
    (21, "mid_lower", &[82978, 1940]),
    (21, "start_cap", &[3900, 624, 1940]),
    (21, "start_lower", &[1166, 624, 1940]),
    (22, "mid_cap", &[124279]),
    (22, "mid_lower", &[50566]),
    (22, "start_cap", &[105968, 1940]),
    (22, "start_lower", &[106839, 1940]),
    (23, "mid_cap", &[11800, 13419]),
    (23, "mid_lower", &[130465]),
    (23, "start_cap", &[34142, 13419]),
    (23, "start_lower", &[6493, 13419]),
    (24, "mid_cap", &[138961]),
    (24, "mid_lower", &[65770]),
    (24, "start_cap", &[86202, 1940]),
    (24, "start_lower", &[136572]),
    (26, "mid_cap", &[21444]),
    (26, "mid_lower", &[8603]),
    (26, "start_cap", &[51024]),
    (26, "start_lower", &[2128]),
    (27, "mid_cap", &[148695]),
    (27, "mid_lower", &[56896]),
    (27, "start_cap", &[131660, 3404]),
    (27, "start_lower", &[100904, 3404]),
    (28, "mid_cap", &[79380]),
    (28, "mid_lower", &[36048]),
    (28, "start_cap", &[221202]),
    (28, "start_lower", &[179358]),
    (36, "mid_cap", &[159008, 498]),
    (36, "mid_lower", &[160522, 498]),
    (36, "start_cap", &[236855, 8867, 498]),
    (36, "start_lower", &[236767, 8867, 498]),
    (47, "mid_cap", &[18663]),
    (47, "mid_lower", &[10173]),
    (47, "start_cap", &[48659]),
    (47, "start_lower", &[56063]),
    (49, "mid_cap", &[1851, 1940]),
    (49, "mid_lower", &[192251]),
    (49, "start_cap", &[2704, 1940]),
    (49, "start_lower", &[724, 1940]),
    (53, "mid_cap", &[227028]),
    (53, "mid_lower", &[64736]),
    (53, "start_cap", &[236777, 590, 688]),
    (53, "start_lower", &[2397, 688]),
    (58, "mid_cap", &[24996, 1940]),
    (58, "mid_lower", &[7649, 1940]),
    (58, "start_cap", &[77185, 1940]),
    (58, "start_lower", &[2858, 1940]),
    (61, "mid_cap", &[10635, 1064, 1940]),
    (61, "mid_lower", &[2767, 1064, 1940]),
    (61, "start_cap", &[21301, 1064, 1940]),
    (61, "start_lower", &[15206, 1064, 1940]),
    (63, "mid_cap", &[18153, 1940]),
    (63, "mid_lower", &[35431, 1940]),
    (63, "start_cap", &[32889, 1682, 1940]),
    (63, "start_lower", &[17485, 1682, 1940]),
    (64, "mid_cap", &[61165, 216643]),
    (64, "mid_lower", &[29178, 216643]),
    (64, "start_cap", &[224079, 216643]),
    (64, "start_lower", &[93340, 216643]),
    (68, "mid_cap", &[203742, 1940]),
    (68, "mid_lower", &[82870, 1940]),
    (68, "start_cap", &[236788, 10370, 1940]),
    (68, "start_lower", &[90513, 1940]),
    (71, "mid_cap", &[639, 50880, 1940]),
    (71, "mid_lower", &[42864, 536, 1940]),
    (71, "start_cap", &[236798, 50880, 1940]),
    (71, "start_lower", &[236752, 50880, 1940]),
    (79, "mid_cap", &[7707]),
    (79, "mid_lower", &[5122]),
    (79, "start_cap", &[21713]),
    (79, "start_lower", &[22278]),
    (80, "mid_cap", &[50144]),
    (80, "mid_lower", &[37291]),
    (80, "start_cap", &[178430]),
    (80, "start_lower", &[1042, 23743]),
    (86, "mid_cap", &[229353]),
    (86, "mid_lower", &[114834]),
    (86, "start_cap", &[17222, 498]),
    (86, "start_lower", &[5514, 498]),
    (87, "mid_cap", &[8087, 1940]),
    (87, "mid_lower", &[28014, 1940]),
    (87, "start_cap", &[53279, 1940]),
    (87, "start_lower", &[147050, 1940]),
    (90, "mid_cap", &[29255, 1940]),
    (90, "mid_lower", &[150260]),
    (90, "start_cap", &[99444, 1940]),
    (90, "start_lower", &[15877, 1940]),
    (92, "mid_cap", &[149565]),
    (92, "mid_lower", &[37949]),
    (92, "start_cap", &[236836, 5259, 1940]),
    (92, "start_lower", &[13538, 1940]),
    (93, "mid_cap", &[98467, 602, 1940]),
    (93, "mid_lower", &[770, 693, 602, 1940]),
    (93, "start_cap", &[6855, 693, 602, 1940]),
    (93, "start_lower", &[866, 693, 602, 1940]),
    (95, "mid_cap", &[2362, 525, 1940]),
    (95, "mid_lower", &[228863, 1940]),
    (95, "start_cap", &[11555, 525, 1940]),
    (95, "start_lower", &[9197, 525, 1940]),
    (96, "mid_cap", &[11094, 1940]),
    (96, "mid_lower", &[1313, 1940]),
    (96, "start_cap", &[15413, 1940]),
    (96, "start_lower", &[1643, 1940]),
    (97, "mid_cap", &[5298, 12632, 1940]),
    (97, "mid_lower", &[169537, 1940]),
    (97, "start_cap", &[186817, 535, 1940]),
    (97, "start_lower", &[793, 12632, 1940]),
    (101, "mid_cap", &[183285, 3096, 1940]),
    (101, "mid_lower", &[520, 5664, 3096, 1940]),
    (101, "start_cap", &[236792, 5664, 3096, 1940]),
    (101, "start_lower", &[236757, 5664, 3096, 1940]),
    (102, "mid_cap", &[51194, 1940]),
    (102, "mid_lower", &[951, 8145, 1940]),
    (102, "start_cap", &[58681, 535, 1940]),
    (102, "start_lower", &[51962, 535, 1940]),
    (110, "mid_cap", &[157274, 1940]),
    (110, "mid_lower", &[513, 1730, 38262, 1940]),
    (110, "start_cap", &[236796, 1730, 38262, 1940]),
    (110, "start_lower", &[236753, 1730, 38262, 1940]),
    (113, "mid_cap", &[127260, 30775]),
    (113, "mid_lower", &[50265, 30775]),
    (113, "start_cap", &[236797, 2839, 30775]),
    (113, "start_lower", &[21586, 30775]),
    (116, "mid_cap", &[20614, 28176, 1940]),
    (116, "mid_lower", &[15961, 28176, 1940]),
    (116, "start_cap", &[81972, 28176, 1940]),
    (116, "start_lower", &[25314, 28176, 1940]),
];
const TOKENIZER_FILE: &str = "tokenizer.json";
const MAX_TOP_K: usize = 256;

fn refuse<T>(reason: impl Into<String>) -> Result<T> {
    Err(format!("REFUSED: {}", reason.into()).into())
}

fn sha_bytes(b: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(b))
}

fn row_sha(k: &[f32], v: &[f32]) -> String {
    let mut d = Sha256::new();
    for x in k.iter().chain(v) {
        d.update(x.to_bits().to_le_bytes());
    }
    format!("sha256:{:x}", d.finalize())
}

fn logits_sha(l: &[f32]) -> String {
    let mut d = Sha256::new();
    for v in l {
        d.update(v.to_bits().to_le_bytes());
    }
    format!("sha256:{:x}", d.finalize())
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

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

// ── execution (as the qualified runner) ──────────────────────────────────

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
    fired: Vec<(usize, SublayerSite, usize)>,
    step_logit_sha: Vec<String>,
    last_logits: Vec<f32>,
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
}

impl Run {
    fn kv_sha(&self, l: usize, p: usize) -> String {
        row_sha(&self.keys[l][p], &self.values[l][p])
    }
}

struct Model {
    plan: ComponentOpPlan,
    ops: PreparedOperands,
    backend: ProductionBackend,
    layers: usize,
}

fn execute(model: &Model, tokens: &[u32], plan: &InterventionPlan) -> Result<Run> {
    let mut carriers = Carriers::default();
    let mut kv = RowKvState::default();
    let (mut firings, mut step_logit_sha, mut last_logits) = (0, Vec::new(), Vec::new());
    let mut fired = Vec::new();
    {
        let mut session = DecodeSession::over_prepared(&model.plan, &model.ops, &model.backend, &mut kv)?;
        let none = HeadInterventionPlan::none();
        for &t in tokens {
            let out = session.step_intervened(t, &mut carriers, plan, &none)?;
            firings += out.firings.len();
            fired.extend(out.firings.iter().map(|f| (f.layer, f.site, f.position)));
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
    Ok(Run { carriers, firings, fired, step_logit_sha, last_logits, keys, values })
}

fn argmax(l: &[f32]) -> Result<usize> {
    let mut best = 0;
    for (i, v) in l.iter().enumerate() {
        if !v.is_finite() {
            return refuse("non-finite logit");
        }
        if *v > l[best] {
            best = i;
        }
    }
    Ok(best)
}

fn readout(l: &[f32], candidates: &[u32], top_k: usize) -> Result<Value> {
    let best = argmax(l)?;
    let max = l[best] as f64;
    let lse = max + l.iter().map(|v| (*v as f64 - max).exp()).sum::<f64>().ln();
    let cand: Vec<f64> = candidates.iter().map(|&c| l[c as usize] as f64).collect();
    let mut order: Vec<usize> = (0..l.len()).collect();
    order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).expect("finite").then(a.cmp(&b)));
    order.truncate(top_k);
    Ok(json!({"next": best, "log_normalizer": lse, "candidate_logits": cand,
              "top_k_ids": order.iter().map(|&i| i as u32).collect::<Vec<_>>(),
              "top_k_logits": order.iter().map(|&i| l[i] as f64).collect::<Vec<_>>(),
              "next_logit_sha256": logits_sha(l)}))
}

// ── plan ──────────────────────────────────────────────────────────────────

struct VEntry {
    offset: usize,
    sha: String,
    recipe: String,
}

struct Arm {
    name: String,
    from: usize,
    /// Last replaced carrier layer, inclusive.
    to: usize,
    span: Vec<usize>,
    vectors: BTreeMap<(usize, usize), String>,
    identity: bool,
}

struct Target {
    id: String,
    run: String,
    set: String,
    arms: Vec<Arm>,
}

struct TRun {
    tokens: Vec<u32>,
    window: Vec<usize>,
    steps: Vec<String>,
}

struct Plan {
    container: PathBuf,
    layers: usize,
    top_k: usize,
    kv_dump: bool,
    propagation_check: bool,
    hidden: usize,
    store: PathBuf,
    store_sha: String,
    index: BTreeMap<String, VEntry>,
    runs: BTreeMap<String, TRun>,
    sets: BTreeMap<String, Vec<u32>>,
    targets: Vec<Target>,
    release: Option<Value>,
}

fn u64s(v: &Value, what: &str) -> Result<Vec<u64>> {
    match v.as_array() {
        Some(a) if !a.is_empty() => a.iter().map(|x| x.as_u64().ok_or_else(|| format!("REFUSED: {what}").into())).collect(),
        _ => refuse(format!("{what}: empty or not a list")),
    }
}

/// Why a run is sealed, or None. Checks the key, every id part, and the tokens.
fn sealed_hit(id: &str, country: &str, tokens: &[u32]) -> Option<String> {
    if SEALED_KEYS.contains(&country) || id.split('|').any(|p| SEALED_KEYS.contains(&p)) {
        return Some("sealed entity key".into());
    }
    for (z, ph, subj) in SEALED_SUBJECTS.iter() {
        if !subj.is_empty() && tokens.windows(subj.len()).any(|w| w == *subj) {
            return Some(format!("tokens contain the subject of sealed element {z} ({ph})"));
        }
    }
    None
}

/// A declared release, validated in full; absent = None. Anything malformed refuses.
/// sha256 over sorted run ids of "id|country|t1,t2,...\n" (BTreeMap iteration is sorted).
fn runs_digest(runs: &serde_json::Map<String, Value>) -> Result<String> {
    let sorted: BTreeMap<&String, &Value> = runs.iter().collect();
    let mut d = Sha256::new();
    for (id, r) in sorted {
        let country = r["country"].as_str().ok_or_else(|| format!("REFUSED: run {id}: no country"))?;
        let toks: Vec<String> = u64s(&r["tokens"], "tokens")?.iter().map(|t| t.to_string()).collect();
        d.update(format!("{id}|{country}|{}\n", toks.join(",")).as_bytes());
    }
    Ok(format!("sha256:{:x}", d.finalize()))
}

fn used_marker(file: &str) -> PathBuf {
    PathBuf::from(format!("{file}.used"))
}

fn load_release(r: &Value, runs: &Value) -> Result<Option<Value>> {
    if r.is_null() {
        return Ok(None);
    }
    if r["population"].as_str() != Some(SEALED_POPULATION) {
        return refuse("sealed_release names another population");
    }
    let file = r["authorisation_file"].as_str().ok_or("REFUSED: sealed_release has no authorisation_file")?;
    let want = r["authorisation_sha256"].as_str().ok_or("REFUSED: sealed_release has no authorisation_sha256")?;
    let bytes = std::fs::read(file).or_else(|e| refuse(format!("sealed_release authorisation unreadable: {e}")))?;
    let got = sha_bytes(&bytes);
    if got != want {
        return refuse("sealed_release authorisation differs from its declared hash");
    }
    let a: Value = serde_json::from_slice(&bytes).or_else(|e| refuse(format!("authorisation is not JSON: {e}")))?;
    if a["schema"] != AUTH_SCHEMA || a["population"].as_str() != Some(SEALED_POPULATION)
        || a["owner_go"].as_str().map_or(true, |g| g.trim().is_empty())
    {
        return refuse("authorisation schema, population or owner_go");
    }
    let digest = runs_digest(runs.as_object().ok_or("REFUSED: no runs")?)?;
    if a["runs_digest"].as_str() != Some(digest.as_str()) {
        return refuse("authorisation does not release this plan's runs (runs_digest)");
    }
    if used_marker(file).exists() {
        return refuse("authorisation already used: a released population is opened once");
    }
    Ok(Some(json!({"population": SEALED_POPULATION, "authorisation_sha256": got,
                   "authorisation_file": file, "runs_digest": digest})))
}

fn load_plan(path: &Path) -> Result<Plan> {
    let v: Value = serde_json::from_slice(&std::fs::read(path)?).or_else(|e| refuse(format!("plan is not JSON: {e}")))?;
    if v["schema"] != PLAN_SCHEMA {
        return refuse("plan schema");
    }
    let container = PathBuf::from(v["container"].as_str().ok_or("REFUSED: no container")?);
    let layers = v["layers"].as_u64().filter(|&l| l >= 1).ok_or("REFUSED: no layer count")? as usize;
    let top_k = v["top_k"].as_u64().filter(|&k| k >= 1 && (k as usize) <= MAX_TOP_K).ok_or("REFUSED: top_k")? as usize;
    let kv_dump = match &v["kv_dump"] {
        Value::Bool(b) => *b,
        Value::Null => false,
        _ => return refuse("kv_dump must be a boolean"),
    };
    let propagation_check = match &v["propagation_check"] {
        Value::Bool(b) => *b,
        _ => return refuse("propagation_check must be declared as a boolean"),
    };
    let tok = container.join(TOKENIZER_FILE);
    if !tok.exists() || Some(sha_bytes(&std::fs::read(&tok)?).as_str()) != v["tokenizer_sha256"].as_str() {
        return refuse("tokenizer hash differs from the plan's");
    }
    let holdout: BTreeSet<String> = match v["holdout"].as_array() {
        Some(a) if !a.is_empty() => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => return refuse("plan has no holdout list"),
    };
    let release = load_release(&v["sealed_release"], &v["runs"])?;
    let mut sets = BTreeMap::new();
    for (k, s) in v["candidate_sets"].as_object().ok_or("REFUSED: no candidate_sets")? {
        sets.insert(k.clone(), u64s(s, "candidate set")?.into_iter().map(|x| x as u32).collect());
    }
    let st = &v["vector_store"];
    let store = PathBuf::from(st["file"].as_str().ok_or("REFUSED: no vector store file")?);
    let store_sha = st["sha256"].as_str().ok_or("REFUSED: no vector store sha")?.to_string();
    let hidden = st["hidden"].as_u64().ok_or("REFUSED: no hidden size")? as usize;
    let mut index = BTreeMap::new();
    for (vid, e) in st["index"].as_object().ok_or("REFUSED: no vector index")? {
        let sha = e["sha256"].as_str().filter(|s| is_hex64(s)).ok_or_else(|| format!("REFUSED: vector {vid} sha"))?;
        let recipe = e["recipe_sha256"].as_str().filter(|s| s.starts_with("sha256:")).ok_or_else(|| format!("REFUSED: vector {vid} recipe"))?;
        let offset = e["offset"].as_u64().ok_or_else(|| format!("REFUSED: vector {vid} offset"))? as usize;
        index.insert(vid.clone(), VEntry { offset, sha: sha.into(), recipe: recipe.into() });
    }
    let mut runs = BTreeMap::new();
    for (id, r) in v["runs"].as_object().filter(|m| !m.is_empty()).ok_or("REFUSED: no runs")? {
        let country = r["country"].as_str().ok_or_else(|| format!("REFUSED: run {id}: no country"))?;
        if holdout.contains(country) {
            return refuse(format!("run {id}: a holdout country"));
        }
        if SPENT_HOLDOUT.contains(&country) || id.split('|').any(|p| SPENT_HOLDOUT.contains(&p)) {
            return refuse(format!("run {id}: a spent Test 3 holdout country"));
        }
        let tokens: Vec<u32> = u64s(&r["tokens"], "tokens")?.into_iter().map(|x| x as u32).collect();
        if release.is_none() {
            if let Some(why) = sealed_hit(id, country, &tokens) {
                return refuse(format!("run {id}: sealed population {SEALED_POPULATION}: {why}"));
            }
        }
        let window: Vec<usize> = u64s(&r["window"], "window")?.into_iter().map(|x| x as usize).collect();
        if u64s(&r["window_ids"], "window_ids")? != window.iter().map(|&p| tokens[p] as u64).collect::<Vec<_>>() {
            return refuse(format!("run {id}: window token ids differ from the run"));
        }
        if window.windows(2).any(|w| w[1] != w[0] + 1) || *window.last().unwrap() + 1 >= tokens.len() {
            return refuse(format!("run {id}: window not contiguous or reaches the readout"));
        }
        let steps: Vec<String> = match r["expected_step_logit_sha256"].as_array() {
            Some(a) if a.len() == tokens.len() => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
            _ => return refuse(format!("run {id}: step hashes")),
        };
        runs.insert(id.clone(), TRun { tokens, window, steps });
    }
    let mut targets = Vec::new();
    let mut seen = BTreeSet::new();
    for t in v["targets"].as_array().filter(|a| !a.is_empty()).ok_or("REFUSED: no targets")? {
        let id = t["id"].as_str().ok_or("REFUSED: target id")?.to_string();
        if !seen.insert(id.clone()) {
            return refuse(format!("{id}: repeated"));
        }
        let run = t["run"].as_str().filter(|r| runs.contains_key(*r)).ok_or_else(|| format!("REFUSED: {id}: unknown run"))?.to_string();
        let set = t["candidate_set"].as_str().filter(|s| sets.contains_key(*s)).ok_or_else(|| format!("REFUSED: {id}: candidate set"))?.to_string();
        let tr = &runs[&run];
        let mut arms = Vec::new();
        let mut names = BTreeSet::new();
        for a in t["arms"].as_array().filter(|a| !a.is_empty()).ok_or_else(|| format!("REFUSED: {id}: no arms"))? {
            let name = a["name"].as_str().ok_or("REFUSED: arm name")?.to_string();
            if !names.insert(name.clone()) {
                return refuse(format!("{id} {name}: repeated"));
            }
            let what = format!("{id} {name}");
            let from = a["from_layer"].as_u64().filter(|&f| f >= 1 && (f as usize) <= layers).ok_or_else(|| format!("REFUSED: {what}: from_layer"))? as usize;
            let span: Vec<usize> = u64s(&a["target_span"], "target_span")?.into_iter().map(|x| x as usize).collect();
            let to = a["to_layer"].as_u64().ok_or_else(|| format!("REFUSED: {what}: to_layer"))? as usize;
            if to + 1 < from || to >= layers {
                return refuse(format!("{what}: band [{}, {to}] is not inside 0..{}", from - 1, layers - 1));
            }
            if !span.iter().all(|p| tr.window.contains(p)) || span.windows(2).any(|w| w[1] != w[0] + 1) {
                return refuse(format!("{what}: target span outside the window"));
            }
            if span.iter().any(|&p| p + 1 >= tr.tokens.len()) {
                return refuse(format!("{what}: the span reaches the readout"));
            }
            let mut vectors = BTreeMap::new();
            for (k, vid) in a["vectors"].as_object().ok_or_else(|| format!("REFUSED: {what}: no vectors"))? {
                let p: Vec<&str> = k.split('|').collect();
                let lp = match p.as_slice() {
                    [l, q] => (l.parse::<usize>()?, q.parse::<usize>()?),
                    _ => return refuse(format!("{what}: vector key {k}")),
                };
                let vid = vid.as_str().filter(|x| index.contains_key(*x)).ok_or_else(|| format!("REFUSED: {what}: unknown vector {k}"))?;
                vectors.insert(lp, vid.to_string());
            }
            let want: BTreeSet<(usize, usize)> = (from - 1..=to).flat_map(|l| span.iter().map(move |&p| (l, p))).collect();
            if vectors.keys().cloned().collect::<BTreeSet<_>>() != want {
                return refuse(format!("{what}: vectors do not cover exactly layers {}..={to} × the span", from - 1));
            }
            arms.push(Arm { name, from, to, span, vectors, identity: a["identity"].as_bool().unwrap_or(false) });
        }
        targets.push(Target { id, run, set, arms });
    }
    Ok(Plan { container, layers, top_k, kv_dump, propagation_check, hidden, store, store_sha, index, runs, sets, targets, release })
}

fn load_store(plan: &Plan) -> Result<Vec<f32>> {
    let bytes = std::fs::read(&plan.store)?;
    if sha_bytes(&bytes) != plan.store_sha {
        return refuse("vector store differs from its declared sha");
    }
    let data: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    for (vid, e) in &plan.index {
        let v = data.get(e.offset..e.offset + plan.hidden).ok_or_else(|| format!("REFUSED: vector {vid} out of range"))?;
        if vector_sha256(v) != e.sha {
            return refuse(format!("vector {vid} differs from its index hash"));
        }
    }
    Ok(data)
}

// ── run ───────────────────────────────────────────────────────────────────

/// vector_sha256 of the carrier at every (layer, window position).
fn span_carriers(run: &Run, window: &[usize], layers: usize) -> serde_json::Map<String, Value> {
    let mut m = serde_json::Map::new();
    for l in 0..layers {
        for &p in window {
            if let Some((v, _)) = run.carriers.ffn.get(&(l, p)) {
                m.insert(format!("{l}|{p}"), json!(vector_sha256(v)));
            }
        }
    }
    m
}

fn run_target(model: &Model, plan: &Plan, store: &[f32], t: &Target) -> Result<Value> {
    let tr = &plan.runs[&t.run];
    let natural = execute(model, &tr.tokens, &InterventionPlan::none())?;
    let replay = execute(model, &tr.tokens, &InterventionPlan::none())?;
    if natural.step_logit_sha != replay.step_logit_sha || natural.keys != replay.keys || natural.values != replay.values {
        return refuse(format!("{}: natural replay is not bit-identical", t.id));
    }
    if natural.step_logit_sha != tr.steps {
        return refuse(format!("{}: natural steps differ from the declared hashes", t.id));
    }
    let cands = &plan.sets[&t.set];
    let mut arms = serde_json::Map::new();
    for a in &t.arms {
        let what = format!("{} {}", t.id, a.name);
        let mut ip = InterventionPlan::none();
        let mut injected = Vec::new();
        let mut recipes = BTreeMap::new();
        for ((l, p), vid) in &a.vectors {
            let e = &plan.index[vid];
            let v = store[e.offset..e.offset + plan.hidden].to_vec();
            ip = ip.with(Intervention::replace(Address::new(*l, SublayerSite::Ffn, [*p])?, v,
                                               VectorProvenance::Literal { sha256: e.sha.clone() })?)?;
            injected.push(((*l, *p), e.sha.clone()));
            *recipes.entry(e.recipe.clone()).or_insert(0usize) += 1;
        }
        let run = execute(model, &tr.tokens, &ip)?;
        if run.firings != injected.len() {
            return refuse(format!("{what}: {} firings for {} addresses", run.firings, injected.len()));
        }
        let fired: BTreeSet<(usize, usize)> = run.fired.iter().map(|&(l, _, p)| (l, p)).collect();
        if fired.len() != run.fired.len()
            || run.fired.iter().any(|&(l, s, p)| s != SublayerSite::Ffn || l + 1 < a.from || l > a.to || !a.span.contains(&p))
            || fired != a.vectors.keys().cloned().collect::<BTreeSet<_>>()
        {
            return refuse(format!("{what}: a firing outside the declared band × span, or repeated"));
        }
        for ((l, p), sha) in &injected {
            match run.carriers.ffn.get(&(*l, *p)) {
                Some((v, 1)) if vector_sha256(v) == *sha => {}
                _ => return refuse(format!("{what}: landing at {l}|{p}")),
            }
        }
        let first = a.span[0];
        if run.step_logit_sha[..first] != natural.step_logit_sha[..first] {
            return refuse(format!("{what}: a step before the span changed"));
        }
        for l in 0..model.layers {
            for p in 0..first {
                if run.kv_sha(l, p) != natural.kv_sha(l, p) {
                    return refuse(format!("{what}: earlier position {p} changed at layer {l}"));
                }
            }
            if l < a.from {
                for &p in &a.span {
                    if run.kv_sha(l, p) != natural.kv_sha(l, p) {
                        return refuse(format!("{what}: K/V at {l}|{p} not natural below the start layer"));
                    }
                }
            }
            if l + 1 < a.from {
                for &p in &a.span {
                    if run.carriers.ffn.get(&(l, p)).map(|c| &c.0) != natural.carriers.ffn.get(&(l, p)).map(|c| &c.0) {
                        return refuse(format!("{what}: carrier at {l}|{p} not natural below the band"));
                    }
                }
            }
        }
        let mut changed_above = Vec::new();
        for l in a.to + 1..model.layers {
            for &p in &a.span {
                match (run.carriers.ffn.get(&(l, p)), natural.carriers.ffn.get(&(l, p))) {
                    (Some((v, 1)), Some((n, _))) => {
                        if v != n {
                            changed_above.push(format!("{l}|{p}"));
                        }
                    }
                    _ => return refuse(format!("{what}: carrier at {l}|{p} above the band was not written exactly once")),
                }
            }
        }
        let mut fixpoint = Value::Null;
        if plan.propagation_check && a.to + 1 < model.layers {
            let mut ip2 = ip.clone();
            let mut n2 = injected.len();
            for l in a.to + 1..model.layers {
                for &p in &a.span {
                    let v = run.carriers.ffn[&(l, p)].0.clone();
                    let sha = vector_sha256(&v);
                    ip2 = ip2.with(Intervention::replace(Address::new(l, SublayerSite::Ffn, [p])?, v,
                                                         VectorProvenance::Literal { sha256: sha })?)?;
                    n2 += 1;
                }
            }
            let re = execute(model, &tr.tokens, &ip2)?;
            let same_carriers = run.carriers.ffn.iter().all(|(k, (v, _))| re.carriers.ffn.get(k).map(|c| &c.0) == Some(v));
            if re.firings != n2 || re.step_logit_sha != run.step_logit_sha || re.keys != run.keys || re.values != run.values || !same_carriers {
                return refuse(format!("{what}: above-band states are not a fixed point of the model's own computation"));
            }
            fixpoint = json!({"bit_exact": true, "replayed_addresses": n2 - injected.len()});
        }
        let identical = run.step_logit_sha == natural.step_logit_sha && run.keys == natural.keys && run.values == natural.values;
        if a.identity && !identical {
            return refuse(format!("{what}: identity is not bit-identical to the natural run"));
        }
        let mut rec = json!({
            "from_layer": a.from, "to_layer": a.to, "band": [a.from - 1, a.to],
            "target_span": a.span, "replacements": injected.len(), "firings": run.firings,
            "span_carrier_sha256": span_carriers(&run, &tr.window, model.layers),
            "above_band_changed": changed_above, "propagation_fixpoint": fixpoint,
            "vectors": injected.iter().map(|((l, p), s)| (format!("{l}|{p}"), json!(s))).collect::<serde_json::Map<_, _>>(),
            "recipes": recipes, "identity_bit_exact": a.identity && identical,
            "step_logit_sha256": run.step_logit_sha, "readout": readout(&run.last_logits, cands, plan.top_k)?,
        });
        if plan.kv_dump {
            let mut d = serde_json::Map::new();
            for l in a.from..=a.to.min(model.layers - 1) {
                for &p in &a.span {
                    d.insert(format!("{l}|{p}"), json!({"k": hex_f32(&run.keys[l][p]), "v": hex_f32(&run.values[l][p])}));
                }
            }
            rec["kv_dump"] = Value::Object(d);
        }
        arms.insert(a.name.clone(), rec);
    }
    Ok(json!({"schema": RECORD_SCHEMA, "id": t.id, "run": t.run,
              "natural": {"readout": readout(&natural.last_logits, cands, plan.top_k)?, "replay_bit_exact": true,
                          "span_carrier_sha256": span_carriers(&natural, &tr.window, model.layers)},
              "arms": arms}))
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() == 1 && a[0] == "--sealed" {
        let subj: Vec<Value> = SEALED_SUBJECTS.iter().map(|(z, ph, t)| json!({"z": z, "form": ph, "tokens": t})).collect();
        println!("{}", json!({"population": SEALED_POPULATION, "keys": SEALED_KEYS.to_vec(), "subjects": subj,
                              "spent_holdout": SPENT_HOLDOUT.to_vec()}));
        return Ok(());
    }
    if a.len() == 2 && a[0] == "--check" {
        let plan = load_plan(Path::new(&a[1]))?;
        load_store(&plan)?;
        let arms: usize = plan.targets.iter().map(|t| t.arms.len()).sum();
        println!("checked {} targets, {} arms, {} vectors; sealed release: {}", plan.targets.len(), arms, plan.index.len(),
                 plan.release.as_ref().map_or("none".to_string(), |r| r.to_string()));
        return Ok(());
    }
    if a.len() != 2 {
        return Err("Usage: gwelem_band_runner --sealed | --check PLAN.json | PLAN.json OUT.jsonl".into());
    }
    let plan = load_plan(Path::new(&a[0]))?;
    let out = Path::new(&a[1]);
    let partial = PathBuf::from(format!("{}.partial", a[1]));
    let man = PathBuf::from(format!("{}.manifest.json", a[1]));
    if out.exists() || partial.exists() || man.exists() {
        return refuse("output exists; a run never overwrites");
    }
    let store = load_store(&plan)?;
    if let Some(r) = &plan.release {
        let f = r["authorisation_file"].as_str().ok_or("REFUSED: release file")?;
        std::fs::File::options().write(true).create_new(true).open(used_marker(f))
            .or_else(|e| refuse(format!("authorisation already used or marker not creatable: {e}")))?
            .write_all(sha_bytes(&std::fs::read(&a[0])?).as_bytes())?;
        eprintln!("sealed release in use: {r}");
    }
    let inspection = inspect_container(&plan.container, true)?;
    let cplan = plan_component_ops(&inspection, &plan.container, "target")?.plan.ok_or("plan absent")?;
    if cplan.layers.iter().any(|l| l.layer_scale.is_some()) || cplan.layers.len() != plan.layers {
        return refuse("the model plan differs from the declared layers or declares a layer scale");
    }
    let ostore = OperandStore::open(&plan.container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&cplan, &ostore, &backend, ExecutionSlice::Full)?;
    let layers = cplan.layers.len();
    let model = Model { plan: cplan, ops, backend, layers };
    let mut f = std::fs::File::options().write(true).create_new(true).open(&partial)?;
    for t in &plan.targets {
        writeln!(f, "{}", run_target(&model, &plan, &store, t)?)?;
        eprintln!("ran {}", t.id);
    }
    drop(f);
    std::fs::rename(&partial, out)?;
    let exe = std::fs::read(std::env::current_exe()?)?;
    let m = json!({"schema": MANIFEST_SCHEMA, "status": "complete", "executable_sha256": sha_bytes(&exe),
                   "plan_sha256": sha_bytes(&std::fs::read(&a[0])?), "output_sha256": sha_bytes(&std::fs::read(out)?),
                   "vector_store_sha256": plan.store_sha, "targets": plan.targets.len(),
                   "propagation_check": plan.propagation_check, "sealed_release": plan.release});
    std::fs::write(&man, serde_json::to_vec_pretty(&m)?)?;
    Ok(())
}
