//! GW-TEMPLATE-FACTORIAL test 3: the derived-vector runner (ENGINEERING; not
//! qualified, and no candidate execution is authorised).
//!
//! It is a sibling of the qualified `gwtemplate_xpos_runner`, which it does not
//! change. Instead of taking a donor run's live carriers, each arm names a
//! vector for every (layer, target position) it replaces. The vectors come
//! from a hashed **vector store**: a raw f32 little-endian file plus an index.
//! Every vector carries its `vector_sha256` and the sha256 of the **recipe**
//! that produced it, e.g. a captured carrier, a leave-one-target-out mean or a
//! translation. The engine sees `Literal` provenance; the runner records the
//! recipe hash beside it in the output. The candidate builder
//! (`scripts/gwtemplate_test3_candidates.py`) writes the store and enforces
//! the information rules.
//!
//! **Per target:**
//! - natural replay is bit-identical and equals the declared step hashes;
//! - a target run whose country is on the holdout list refuses;
//! - the target span lies inside the declared window, ends before the
//!   readout, and its token ids match.
//!
//! **Per arm (mode `replace`, layers from-1 .. L-1, as test 2):**
//! - the declared vector keys cover exactly those layers × the target span;
//! - every loaded vector's hash equals the index;
//! - landing: one firing per address, carrying the injected vector;
//! - locality: every step and K/V row before the span equals the natural run;
//! - K/V below the start layer at span positions are natural;
//! - an `identity` arm (declared) must be bit-identical to the natural run.
//!
//! The readout record has the same form as the qualified runner's, so the
//! test 2 analysis can be adapted. Output is `OUT.jsonl.partial` until
//! complete, and never overwrites.
//!
//! Usage:
//!   gwtemplate_test3_derived_runner --check PLAN.json
//!   gwtemplate_test3_derived_runner PLAN.json OUT.jsonl
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

const PLAN_SCHEMA: &str = "larql.gwtemplate.test3-derived-plan.v1";
const RECORD_SCHEMA: &str = "larql.gwtemplate.test3-derived-record.v1";
const MANIFEST_SCHEMA: &str = "larql.gwtemplate.test3-derived-manifest.v1";
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
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
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
    {
        let mut session =
            DecodeSession::over_prepared(&model.plan, &model.ops, &model.backend, &mut kv)?;
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
    Ok(Run {
        carriers,
        firings,
        step_logit_sha,
        last_logits,
        keys,
        values,
    })
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
    Ok(
        json!({"next": best, "log_normalizer": lse, "candidate_logits": cand,
              "top_k_ids": order.iter().map(|&i| i as u32).collect::<Vec<_>>(),
              "top_k_logits": order.iter().map(|&i| l[i] as f64).collect::<Vec<_>>(),
              "next_logit_sha256": logits_sha(l)}),
    )
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
    hidden: usize,
    store: PathBuf,
    store_sha: String,
    index: BTreeMap<String, VEntry>,
    runs: BTreeMap<String, TRun>,
    sets: BTreeMap<String, Vec<u32>>,
    targets: Vec<Target>,
}

fn u64s(v: &Value, what: &str) -> Result<Vec<u64>> {
    match v.as_array() {
        Some(a) if !a.is_empty() => a
            .iter()
            .map(|x| x.as_u64().ok_or_else(|| format!("REFUSED: {what}").into()))
            .collect(),
        _ => refuse(format!("{what}: empty or not a list")),
    }
}

fn load_plan(path: &Path) -> Result<Plan> {
    let v: Value = serde_json::from_slice(&std::fs::read(path)?)
        .or_else(|e| refuse(format!("plan is not JSON: {e}")))?;
    if v["schema"] != PLAN_SCHEMA {
        return refuse("plan schema");
    }
    let container = PathBuf::from(v["container"].as_str().ok_or("REFUSED: no container")?);
    let layers = v["layers"]
        .as_u64()
        .filter(|&l| l >= 1)
        .ok_or("REFUSED: no layer count")? as usize;
    let top_k = v["top_k"]
        .as_u64()
        .filter(|&k| k >= 1 && (k as usize) <= MAX_TOP_K)
        .ok_or("REFUSED: top_k")? as usize;
    let kv_dump = match &v["kv_dump"] {
        Value::Bool(b) => *b,
        Value::Null => false,
        _ => return refuse("kv_dump must be a boolean"),
    };
    let tok = container.join(TOKENIZER_FILE);
    if !tok.exists()
        || Some(sha_bytes(&std::fs::read(&tok)?).as_str()) != v["tokenizer_sha256"].as_str()
    {
        return refuse("tokenizer hash differs from the plan's");
    }
    let holdout: BTreeSet<String> = match v["holdout"].as_array() {
        Some(a) if !a.is_empty() => a
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect(),
        _ => return refuse("plan has no holdout list"),
    };
    let mut sets = BTreeMap::new();
    for (k, s) in v["candidate_sets"]
        .as_object()
        .ok_or("REFUSED: no candidate_sets")?
    {
        sets.insert(
            k.clone(),
            u64s(s, "candidate set")?
                .into_iter()
                .map(|x| x as u32)
                .collect(),
        );
    }
    let st = &v["vector_store"];
    let store = PathBuf::from(st["file"].as_str().ok_or("REFUSED: no vector store file")?);
    let store_sha = st["sha256"]
        .as_str()
        .ok_or("REFUSED: no vector store sha")?
        .to_string();
    let hidden = st["hidden"].as_u64().ok_or("REFUSED: no hidden size")? as usize;
    let mut index = BTreeMap::new();
    for (vid, e) in st["index"].as_object().ok_or("REFUSED: no vector index")? {
        let sha = e["sha256"]
            .as_str()
            .filter(|s| is_hex64(s))
            .ok_or_else(|| format!("REFUSED: vector {vid} sha"))?;
        let recipe = e["recipe_sha256"]
            .as_str()
            .filter(|s| s.starts_with("sha256:"))
            .ok_or_else(|| format!("REFUSED: vector {vid} recipe"))?;
        let offset = e["offset"]
            .as_u64()
            .ok_or_else(|| format!("REFUSED: vector {vid} offset"))? as usize;
        index.insert(
            vid.clone(),
            VEntry {
                offset,
                sha: sha.into(),
                recipe: recipe.into(),
            },
        );
    }
    let mut runs = BTreeMap::new();
    for (id, r) in v["runs"]
        .as_object()
        .filter(|m| !m.is_empty())
        .ok_or("REFUSED: no runs")?
    {
        let country = r["country"]
            .as_str()
            .ok_or_else(|| format!("REFUSED: run {id}: no country"))?;
        if holdout.contains(country) {
            return refuse(format!("run {id}: a holdout country"));
        }
        let tokens: Vec<u32> = u64s(&r["tokens"], "tokens")?
            .into_iter()
            .map(|x| x as u32)
            .collect();
        let window: Vec<usize> = u64s(&r["window"], "window")?
            .into_iter()
            .map(|x| x as usize)
            .collect();
        if u64s(&r["window_ids"], "window_ids")?
            != window.iter().map(|&p| tokens[p] as u64).collect::<Vec<_>>()
        {
            return refuse(format!("run {id}: window token ids differ from the run"));
        }
        if window.windows(2).any(|w| w[1] != w[0] + 1)
            || *window.last().unwrap() + 1 >= tokens.len()
        {
            return refuse(format!(
                "run {id}: window not contiguous or reaches the readout"
            ));
        }
        let steps: Vec<String> = match r["expected_step_logit_sha256"].as_array() {
            Some(a) if a.len() == tokens.len() => a
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect(),
            _ => return refuse(format!("run {id}: step hashes")),
        };
        runs.insert(
            id.clone(),
            TRun {
                tokens,
                window,
                steps,
            },
        );
    }
    let mut targets = Vec::new();
    let mut seen = BTreeSet::new();
    for t in v["targets"]
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or("REFUSED: no targets")?
    {
        let id = t["id"].as_str().ok_or("REFUSED: target id")?.to_string();
        if !seen.insert(id.clone()) {
            return refuse(format!("{id}: repeated"));
        }
        let run = t["run"]
            .as_str()
            .filter(|r| runs.contains_key(*r))
            .ok_or_else(|| format!("REFUSED: {id}: unknown run"))?
            .to_string();
        let set = t["candidate_set"]
            .as_str()
            .filter(|s| sets.contains_key(*s))
            .ok_or_else(|| format!("REFUSED: {id}: candidate set"))?
            .to_string();
        let tr = &runs[&run];
        let mut arms = Vec::new();
        let mut names = BTreeSet::new();
        for a in t["arms"]
            .as_array()
            .filter(|a| !a.is_empty())
            .ok_or_else(|| format!("REFUSED: {id}: no arms"))?
        {
            let name = a["name"].as_str().ok_or("REFUSED: arm name")?.to_string();
            if !names.insert(name.clone()) {
                return refuse(format!("{id} {name}: repeated"));
            }
            let what = format!("{id} {name}");
            let from = a["from_layer"]
                .as_u64()
                .filter(|&f| f >= 1 && (f as usize) < layers)
                .ok_or_else(|| format!("REFUSED: {what}: from_layer"))?
                as usize;
            let span: Vec<usize> = u64s(&a["target_span"], "target_span")?
                .into_iter()
                .map(|x| x as usize)
                .collect();
            if !span.iter().all(|p| tr.window.contains(p))
                || span.windows(2).any(|w| w[1] != w[0] + 1)
            {
                return refuse(format!("{what}: target span outside the window"));
            }
            let mut vectors = BTreeMap::new();
            for (k, vid) in a["vectors"]
                .as_object()
                .ok_or_else(|| format!("REFUSED: {what}: no vectors"))?
            {
                let p: Vec<&str> = k.split('|').collect();
                let lp = match p.as_slice() {
                    [l, q] => (l.parse::<usize>()?, q.parse::<usize>()?),
                    _ => return refuse(format!("{what}: vector key {k}")),
                };
                let vid = vid
                    .as_str()
                    .filter(|x| index.contains_key(*x))
                    .ok_or_else(|| format!("REFUSED: {what}: unknown vector {k}"))?;
                vectors.insert(lp, vid.to_string());
            }
            let want: BTreeSet<(usize, usize)> = (from - 1..layers)
                .flat_map(|l| span.iter().map(move |&p| (l, p)))
                .collect();
            if vectors.keys().cloned().collect::<BTreeSet<_>>() != want {
                return refuse(format!(
                    "{what}: vectors do not cover exactly layers {}..{} × the span",
                    from - 1,
                    layers - 1
                ));
            }
            arms.push(Arm {
                name,
                from,
                span,
                vectors,
                identity: a["identity"].as_bool().unwrap_or(false),
            });
        }
        targets.push(Target { id, run, set, arms });
    }
    Ok(Plan {
        container,
        layers,
        top_k,
        kv_dump,
        hidden,
        store,
        store_sha,
        index,
        runs,
        sets,
        targets,
    })
}

fn load_store(plan: &Plan) -> Result<Vec<f32>> {
    let bytes = std::fs::read(&plan.store)?;
    if sha_bytes(&bytes) != plan.store_sha {
        return refuse("vector store differs from its declared sha");
    }
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for (vid, e) in &plan.index {
        let v = data
            .get(e.offset..e.offset + plan.hidden)
            .ok_or_else(|| format!("REFUSED: vector {vid} out of range"))?;
        if vector_sha256(v) != e.sha {
            return refuse(format!("vector {vid} differs from its index hash"));
        }
    }
    Ok(data)
}

// ── run ───────────────────────────────────────────────────────────────────

fn run_target(model: &Model, plan: &Plan, store: &[f32], t: &Target) -> Result<Value> {
    let tr = &plan.runs[&t.run];
    let natural = execute(model, &tr.tokens, &InterventionPlan::none())?;
    let replay = execute(model, &tr.tokens, &InterventionPlan::none())?;
    if natural.step_logit_sha != replay.step_logit_sha
        || natural.keys != replay.keys
        || natural.values != replay.values
    {
        return refuse(format!("{}: natural replay is not bit-identical", t.id));
    }
    if natural.step_logit_sha != tr.steps {
        return refuse(format!(
            "{}: natural steps differ from the declared hashes",
            t.id
        ));
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
            ip = ip.with(Intervention::replace(
                Address::new(*l, SublayerSite::Ffn, [*p])?,
                v,
                VectorProvenance::Literal {
                    sha256: e.sha.clone(),
                },
            )?)?;
            injected.push(((*l, *p), e.sha.clone()));
            *recipes.entry(e.recipe.clone()).or_insert(0usize) += 1;
        }
        let run = execute(model, &tr.tokens, &ip)?;
        if run.firings != injected.len() {
            return refuse(format!(
                "{what}: {} firings for {} addresses",
                run.firings,
                injected.len()
            ));
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
                        return refuse(format!(
                            "{what}: K/V at {l}|{p} not natural below the start layer"
                        ));
                    }
                }
            }
        }
        let identical = run.step_logit_sha == natural.step_logit_sha
            && run.keys == natural.keys
            && run.values == natural.values;
        if a.identity && !identical {
            return refuse(format!(
                "{what}: identity is not bit-identical to the natural run"
            ));
        }
        let mut rec = json!({
            "from_layer": a.from, "target_span": a.span, "replacements": injected.len(), "firings": run.firings,
            "vectors": injected.iter().map(|((l, p), s)| (format!("{l}|{p}"), json!(s))).collect::<serde_json::Map<_, _>>(),
            "recipes": recipes, "identity_bit_exact": a.identity && identical,
            "step_logit_sha256": run.step_logit_sha, "readout": readout(&run.last_logits, cands, plan.top_k)?,
        });
        if plan.kv_dump {
            let mut d = serde_json::Map::new();
            for l in a.from..model.layers {
                for &p in &a.span {
                    d.insert(
                        format!("{l}|{p}"),
                        json!({"k": hex_f32(&run.keys[l][p]), "v": hex_f32(&run.values[l][p])}),
                    );
                }
            }
            rec["kv_dump"] = Value::Object(d);
        }
        arms.insert(a.name.clone(), rec);
    }
    Ok(json!({"schema": RECORD_SCHEMA, "id": t.id, "run": t.run,
              "natural": {"readout": readout(&natural.last_logits, cands, plan.top_k)?, "replay_bit_exact": true},
              "arms": arms}))
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() == 2 && a[0] == "--check" {
        let plan = load_plan(Path::new(&a[1]))?;
        load_store(&plan)?;
        let arms: usize = plan.targets.iter().map(|t| t.arms.len()).sum();
        println!(
            "checked {} targets, {} arms, {} vectors",
            plan.targets.len(),
            arms,
            plan.index.len()
        );
        return Ok(());
    }
    if a.len() != 2 {
        return Err(
            "Usage: gwtemplate_test3_derived_runner --check PLAN.json | PLAN.json OUT.jsonl".into(),
        );
    }
    let plan = load_plan(Path::new(&a[0]))?;
    let out = Path::new(&a[1]);
    let partial = PathBuf::from(format!("{}.partial", a[1]));
    let man = PathBuf::from(format!("{}.manifest.json", a[1]));
    if out.exists() || partial.exists() || man.exists() {
        return refuse("output exists; a run never overwrites");
    }
    let store = load_store(&plan)?;
    let inspection = inspect_container(&plan.container, true)?;
    let cplan = plan_component_ops(&inspection, &plan.container, "target")?
        .plan
        .ok_or("plan absent")?;
    if cplan.layers.iter().any(|l| l.layer_scale.is_some()) || cplan.layers.len() != plan.layers {
        return refuse("the model plan differs from the declared layers or declares a layer scale");
    }
    let ostore = OperandStore::open(&plan.container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&cplan, &ostore, &backend, ExecutionSlice::Full)?;
    let layers = cplan.layers.len();
    let model = Model {
        plan: cplan,
        ops,
        backend,
        layers,
    };
    let mut f = std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(&partial)?;
    for t in &plan.targets {
        writeln!(f, "{}", run_target(&model, &plan, &store, t)?)?;
        eprintln!("ran {}", t.id);
    }
    drop(f);
    std::fs::rename(&partial, out)?;
    let exe = std::fs::read(std::env::current_exe()?)?;
    let m = json!({"schema": MANIFEST_SCHEMA, "status": "complete", "executable_sha256": sha_bytes(&exe),
                   "plan_sha256": sha_bytes(&std::fs::read(&a[0])?), "output_sha256": sha_bytes(&std::fs::read(out)?),
                   "vector_store_sha256": plan.store_sha, "targets": plan.targets.len()});
    std::fs::write(&man, serde_json::to_vec_pretty(&m)?)?;
    Ok(())
}
