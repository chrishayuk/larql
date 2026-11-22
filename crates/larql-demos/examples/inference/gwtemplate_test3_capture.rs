//! GW-TEMPLATE-FACTORIAL test 3, stage 1b: development-only activation
//! capture and its reconstruction checks.
//!
//! **Capture** (`capture PLAN OUTDIR`): for each plan run (a natural, unintervened
//! token run) it runs the model twice. The two runs must be bit-identical, and
//! must equal:
//! - the declared step-logit hashes;
//! - the declared K/V row hashes at the window positions;
//! - the declared FFN-carrier hashes at the window positions, all from the
//!   test 2 seal.
//!
//! It then writes the post-FFN residual carrier for every layer at every
//! window slot and at the readout position. Each run gets one raw f32
//! little-endian file, laid out [layer][slot][hidden]; the slots are the
//! window positions in order, then the readout. A manifest records every
//! vector's `vector_sha256`. A run whose country is on the plan's holdout
//! list refuses. OUTDIR must not exist.
//!
//! **Verify** (`verify PLAN CAPDIR VPLAN OUT.json`): every vector is loaded
//! from the capture files, never from a live donor run, and each file and
//! vector hash is checked against the manifest. Then:
//! - **reinject:** each listed run gets its own captured window carriers
//!   injected at every layer. Every step, every K/V row and the last logits
//!   must be bit-identical to a fresh natural run.
//! - **arms:** each declared arm injects the donor's captured carriers at the
//!   source slots into the target's target positions, for layers
//!   from-1 .. L-1. The last-logit hash must equal the declared hash (a
//!   test 2 record). With `kv_dump`, the target K rows at the target
//!   positions and the donor's natural K rows at the source positions are
//!   written, for an independent RoPE check.
//! - Source spans must lie in the donor's captured window and target spans
//!   in the target's window, or the run refuses.
//!
//! `check PLAN` runs the static plan checks only.
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

const PLAN_SCHEMA: &str = "larql.gwtemplate.test3-capture-plan.v1";
const VERIFY_SCHEMA: &str = "larql.gwtemplate.test3-capture-verify.v1";
const MANIFEST_SCHEMA: &str = "larql.gwtemplate.test3-capture-manifest.v1";
const RESULT_SCHEMA: &str = "larql.gwtemplate.test3-capture-verify-result.v1";
const TOKENIZER_FILE: &str = "tokenizer.json";

fn refuse<T>(reason: impl Into<String>) -> Result<T> {
    Err(format!("REFUSED: {}", reason.into()).into())
}

fn sha_bytes(b: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(b))
}

fn file_sha(p: &Path) -> Result<String> {
    Ok(sha_bytes(&std::fs::read(p)?))
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

impl Run {
    fn kv_sha(&self, layer: usize, pos: usize) -> String {
        row_sha(&self.keys[layer][pos], &self.values[layer][pos])
    }
    fn carrier(&self, layer: usize, pos: usize) -> Result<&Vec<f32>> {
        match self.carriers.ffn.get(&(layer, pos)) {
            Some((v, 1)) => Ok(v),
            _ => refuse(format!(
                "carrier {layer}|{pos} absent or written more than once"
            )),
        }
    }
}

struct Model {
    plan: ComponentOpPlan,
    ops: PreparedOperands,
    backend: ProductionBackend,
    layers: usize,
}

fn load_model(container: &Path, layers: usize) -> Result<Model> {
    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    let cplan = outcome.plan.ok_or("plan absent")?;
    if cplan.layers.iter().any(|l| l.layer_scale.is_some()) {
        return refuse("the plan declares a layer scale");
    }
    if cplan.layers.len() != layers {
        return refuse("the plan's layer count differs from the model's");
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&cplan, &store, &backend, ExecutionSlice::Full)?;
    Ok(Model {
        plan: cplan,
        ops,
        backend,
        layers,
    })
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

fn same(a: &Run, b: &Run) -> bool {
    a.step_logit_sha == b.step_logit_sha
        && a.keys == b.keys
        && a.values == b.values
        && a.last_logits == b.last_logits
}

// ── plan ──────────────────────────────────────────────────────────────────

struct CRun {
    tokens: Vec<u32>,
    window: Vec<usize>,
    readout: usize,
    carriers: BTreeMap<(usize, usize), String>,
    kv: BTreeMap<(usize, usize), String>,
    steps: Vec<String>,
}

struct Plan {
    container: PathBuf,
    layers: usize,
    runs: BTreeMap<String, CRun>,
}

fn ids(v: &Value, what: &str) -> Result<Vec<u64>> {
    match v.as_array() {
        Some(a) if !a.is_empty() => a
            .iter()
            .map(|x| x.as_u64().ok_or_else(|| format!("REFUSED: {what}").into()))
            .collect(),
        _ => refuse(format!("{what}: empty or not a list")),
    }
}

fn keyed(
    v: &Value,
    what: &str,
    prefixed: bool,
    layers: usize,
    n: usize,
) -> Result<BTreeMap<(usize, usize), String>> {
    let m = v
        .as_object()
        .ok_or_else(|| format!("REFUSED: {what} missing"))?;
    let mut out = BTreeMap::new();
    for (k, s) in m {
        let s = s
            .as_str()
            .ok_or_else(|| format!("REFUSED: {what} {k} not a string"))?;
        let body = if prefixed {
            s.strip_prefix("sha256:")
        } else {
            Some(s)
        };
        match body {
            Some(b)
                if b.len() == 64
                    && b.bytes()
                        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) => {}
            _ => return refuse(format!("{what} {k} malformed")),
        }
        let p: Vec<&str> = k.split('|').collect();
        let lp = match p.as_slice() {
            [l, q] => match (l.parse::<usize>(), q.parse::<usize>()) {
                (Ok(l), Ok(q)) if l < layers && q < n => (l, q),
                _ => return refuse(format!("{what} {k} out of range")),
            },
            _ => return refuse(format!("{what} {k} malformed")),
        };
        out.insert(lp, s.to_string());
    }
    Ok(out)
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
    let tok = container.join(TOKENIZER_FILE);
    if !tok.exists() || Some(file_sha(&tok)?.as_str()) != v["tokenizer_sha256"].as_str() {
        return refuse("tokenizer hash differs from the plan's");
    }
    let holdout: BTreeSet<String> = match v["holdout"].as_array() {
        Some(a) if !a.is_empty() => a
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect(),
        _ => return refuse("plan has no holdout list"),
    };
    let mut runs = BTreeMap::new();
    for (id, r) in v["runs"]
        .as_object()
        .filter(|m| !m.is_empty())
        .ok_or("REFUSED: no runs")?
    {
        let country = r["country"]
            .as_str()
            .ok_or_else(|| format!("REFUSED: run {id}: no country"))?;
        if holdout.contains(country) || id.split('|').any(|p| holdout.contains(p)) {
            return refuse(format!("run {id}: a holdout country"));
        }
        let tokens: Vec<u32> = ids(&r["tokens"], &format!("run {id} tokens"))?
            .into_iter()
            .map(|x| x as u32)
            .collect();
        let window: Vec<usize> = ids(&r["window"], &format!("run {id} window"))?
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let wids = ids(&r["window_ids"], &format!("run {id} window_ids"))?;
        let readout = tokens.len() - 1;
        if window.windows(2).any(|w| w[1] != w[0] + 1) || *window.last().unwrap() >= readout {
            return refuse(format!(
                "run {id}: window not contiguous or reaches the readout"
            ));
        }
        if wids != window.iter().map(|&p| tokens[p] as u64).collect::<Vec<_>>() {
            return refuse(format!(
                "run {id}: window token ids differ from the run at those positions"
            ));
        }
        let carriers = keyed(
            &r["expected_carrier_sha256"],
            &format!("run {id} carriers"),
            false,
            layers,
            tokens.len(),
        )?;
        let kv = keyed(
            &r["expected_kv_sha256"],
            &format!("run {id} kv"),
            true,
            layers,
            tokens.len(),
        )?;
        let want: BTreeSet<(usize, usize)> = (0..layers)
            .flat_map(|l| window.iter().map(move |&p| (l, p)))
            .collect();
        if carriers.keys().cloned().collect::<BTreeSet<_>>() != want
            || kv.keys().cloned().collect::<BTreeSet<_>>() != want
        {
            return refuse(format!(
                "run {id}: declared hashes do not cover exactly the window at every layer"
            ));
        }
        let steps: Vec<String> = match r["expected_step_logit_sha256"].as_array() {
            Some(a) if a.len() == tokens.len() => a
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect(),
            _ => {
                return refuse(format!(
                    "run {id}: expected step hashes do not match the run length"
                ))
            }
        };
        runs.insert(
            id.clone(),
            CRun {
                tokens,
                window,
                readout,
                carriers,
                kv,
                steps,
            },
        );
    }
    Ok(Plan {
        container,
        layers,
        runs,
    })
}

fn file_name(i: usize) -> String {
    format!("run-{i:04}.f32")
}

// ── capture ───────────────────────────────────────────────────────────────

fn capture(plan_path: &Path, out: &Path) -> Result<()> {
    let plan = load_plan(plan_path)?;
    if out.exists() {
        return refuse("output directory exists; a capture never overwrites");
    }
    let model = load_model(&plan.container, plan.layers)?;
    let partial = out.with_extension("partial");
    if partial.exists() {
        return refuse("a partial capture exists");
    }
    std::fs::create_dir_all(partial.join("vectors"))?;
    let mut entries = serde_json::Map::new();
    for (i, (id, r)) in plan.runs.iter().enumerate() {
        let a = execute(&model, &r.tokens, &InterventionPlan::none())?;
        let b = execute(&model, &r.tokens, &InterventionPlan::none())?;
        if !same(&a, &b) || a.firings != 0 {
            return refuse(format!("run {id}: natural replay is not bit-identical"));
        }
        if a.step_logit_sha != r.steps {
            return refuse(format!(
                "run {id}: step logits differ from the declared hashes"
            ));
        }
        for ((l, p), want) in &r.carriers {
            if vector_sha256(a.carrier(*l, *p)?) != *want {
                return refuse(format!("run {id}: carrier {l}|{p} differs from the seal"));
            }
        }
        for ((l, p), want) in &r.kv {
            if a.kv_sha(*l, *p) != *want {
                return refuse(format!("run {id}: K/V {l}|{p} differs from the seal"));
            }
        }
        let slots: Vec<usize> = r.window.iter().cloned().chain([r.readout]).collect();
        let mut bytes = Vec::new();
        let mut hashes = serde_json::Map::new();
        let mut hidden = 0;
        for l in 0..plan.layers {
            for &p in &slots {
                let v = a.carrier(l, p)?;
                hidden = v.len();
                hashes.insert(format!("{l}|{p}"), json!(vector_sha256(v)));
                for x in v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        let name = file_name(i);
        std::fs::write(partial.join("vectors").join(&name), &bytes)?;
        entries.insert(id.clone(), json!({
            "file": format!("vectors/{name}"), "file_sha256": sha_bytes(&bytes), "positions": slots,
            "window": r.window, "readout": r.readout, "hidden": hidden, "layers": plan.layers,
            "vector_sha256": hashes, "step_logit_sha256": a.step_logit_sha,
            "kv_sha256": slots.iter().flat_map(|&p| (0..plan.layers).map(move |l| (l, p)))
                .map(|(l, p)| (format!("{l}|{p}"), json!(a.kv_sha(l, p)))).collect::<serde_json::Map<_, _>>(),
        }));
        eprintln!("captured {id}");
    }
    let exe = std::fs::read(std::env::current_exe()?)?;
    let manifest = json!({
        "schema": MANIFEST_SCHEMA, "status": "complete", "layout": "f32 little-endian [layer][slot][hidden]; slots = window positions then readout",
        "executable_sha256": sha_bytes(&exe), "plan_sha256": file_sha(plan_path)?, "runs": entries,
    });
    let mut f = std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(partial.join("manifest.json"))?;
    f.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    drop(f);
    std::fs::rename(&partial, out)?;
    println!(
        "captured {} runs; manifest {}",
        plan.runs.len(),
        file_sha(&out.join("manifest.json"))?
    );
    Ok(())
}

// ── verify ────────────────────────────────────────────────────────────────

struct Captured {
    positions: Vec<usize>,
    window: Vec<usize>,
    hidden: usize,
    data: Vec<f32>,
    hashes: BTreeMap<(usize, usize), String>,
}

impl Captured {
    fn vector(&self, layer: usize, pos: usize) -> Result<Vec<f32>> {
        let slot = self
            .positions
            .iter()
            .position(|&p| p == pos)
            .ok_or_else(|| format!("REFUSED: position {pos} not captured"))?;
        let n = self.positions.len();
        let start = (layer * n + slot) * self.hidden;
        let v = self.data[start..start + self.hidden].to_vec();
        if self.hashes.get(&(layer, pos)) != Some(&vector_sha256(&v)) {
            return refuse(format!(
                "captured vector {layer}|{pos} differs from the manifest"
            ));
        }
        Ok(v)
    }
}

fn load_capture(dir: &Path, layers: usize) -> Result<(String, BTreeMap<String, Captured>)> {
    let mpath = dir.join("manifest.json");
    let msha = file_sha(&mpath)?;
    let m: Value = serde_json::from_slice(&std::fs::read(&mpath)?)?;
    if m["schema"] != MANIFEST_SCHEMA || m["status"] != "complete" {
        return refuse("capture manifest schema or status");
    }
    let mut out = BTreeMap::new();
    for (id, e) in m["runs"]
        .as_object()
        .ok_or("REFUSED: manifest has no runs")?
    {
        let bytes = std::fs::read(dir.join(e["file"].as_str().ok_or("REFUSED: no file")?))?;
        if Some(sha_bytes(&bytes).as_str()) != e["file_sha256"].as_str() {
            return refuse(format!("run {id}: capture file differs from the manifest"));
        }
        let positions: Vec<usize> = ids(&e["positions"], "positions")?
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let window: Vec<usize> = ids(&e["window"], "window")?
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let hidden = e["hidden"].as_u64().ok_or("REFUSED: no hidden size")? as usize;
        if bytes.len() != layers * positions.len() * hidden * 4 {
            return refuse(format!("run {id}: capture file has the wrong size"));
        }
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let hashes = keyed(
            &e["vector_sha256"],
            "manifest vectors",
            false,
            layers,
            usize::MAX,
        )?;
        out.insert(
            id.clone(),
            Captured {
                positions,
                window,
                hidden,
                data,
                hashes,
            },
        );
    }
    Ok((msha, out))
}

fn verify(plan_path: &Path, cap: &Path, vplan_path: &Path, out_path: &Path) -> Result<()> {
    if out_path.exists() {
        return refuse("verify output exists");
    }
    let plan = load_plan(plan_path)?;
    let (msha, captured) = load_capture(cap, plan.layers)?;
    let vp: Value = serde_json::from_slice(&std::fs::read(vplan_path)?)?;
    if vp["schema"] != VERIFY_SCHEMA {
        return refuse("verify plan schema");
    }
    if vp["capture_manifest_sha256"].as_str() != Some(msha.as_str()) {
        return refuse("verify plan names a different capture manifest");
    }
    let reinject: Vec<String> = vp["reinject"]
        .as_array()
        .ok_or("REFUSED: no reinject list")?
        .iter()
        .filter_map(|x| x.as_str().map(String::from))
        .collect();
    let arms = vp["arms"].as_array().ok_or("REFUSED: no arms")?;
    // static arm checks before the model loads
    let mut parsed = Vec::new();
    for a in arms {
        let id = a["id"].as_str().ok_or("REFUSED: arm id")?.to_string();
        let (t, d) = (
            a["target_run"].as_str().unwrap_or_default(),
            a["donor_run"].as_str().unwrap_or_default(),
        );
        let (tr, dc) = match (plan.runs.get(t), captured.get(d)) {
            (Some(tr), Some(dc)) => (tr, dc),
            _ => return refuse(format!("{id}: unknown target or donor run")),
        };
        let src: Vec<usize> = ids(&a["source_span"], "source_span")?
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let tgt: Vec<usize> = ids(&a["target_span"], "target_span")?
            .into_iter()
            .map(|x| x as usize)
            .collect();
        if src.len() != tgt.len()
            || !src.iter().all(|p| dc.window.contains(p))
            || !tgt.iter().all(|p| tr.window.contains(p))
        {
            return refuse(format!("{id}: slot map outside the captured windows"));
        }
        let from = a["from_layer"]
            .as_u64()
            .filter(|&f| f >= 1 && (f as usize) < plan.layers)
            .ok_or_else(|| format!("REFUSED: {id}: from_layer"))? as usize;
        let want = a["expect_last_logit_sha256"]
            .as_str()
            .ok_or_else(|| format!("REFUSED: {id}: no expected hash"))?
            .to_string();
        parsed.push((
            id,
            t.to_string(),
            d.to_string(),
            src,
            tgt,
            from,
            want,
            a["kv_dump"].as_bool().unwrap_or(false),
        ));
    }
    for id in &reinject {
        if !plan.runs.contains_key(id) || !captured.contains_key(id) {
            return refuse(format!("reinject {id}: unknown run"));
        }
    }
    let model = load_model(&plan.container, plan.layers)?;
    let mut rein = serde_json::Map::new();
    for id in &reinject {
        let (r, c) = (&plan.runs[id], &captured[id]);
        let natural = execute(&model, &r.tokens, &InterventionPlan::none())?;
        let mut ip = InterventionPlan::none();
        let mut n = 0;
        for l in 0..plan.layers {
            for &p in &r.window {
                let v = c.vector(l, p)?;
                let prov = VectorProvenance::Captured {
                    run_id: id.clone(),
                    layer: l,
                    site: SublayerSite::Ffn,
                    position: p,
                    sha256: vector_sha256(&v),
                };
                ip = ip.with(Intervention::replace(
                    Address::new(l, SublayerSite::Ffn, [p])?,
                    v,
                    prov,
                )?)?;
                n += 1;
            }
        }
        let run = execute(&model, &r.tokens, &ip)?;
        if run.firings != n || !same(&run, &natural) {
            return refuse(format!(
                "reinject {id}: not bit-identical to the natural run"
            ));
        }
        rein.insert(
            id.clone(),
            json!({"replacements": n, "bit_identical": true}),
        );
        eprintln!("reinjected {id}");
    }
    let mut arm_out = serde_json::Map::new();
    for (id, t, d, src, tgt, from, want, dump) in parsed {
        let (tr, dc) = (&plan.runs[&t], &captured[&d]);
        let mut ip = InterventionPlan::none();
        let mut n = 0;
        for l in from - 1..plan.layers {
            for (&sp, &tp) in src.iter().zip(&tgt) {
                let v = dc.vector(l, sp)?;
                let prov = VectorProvenance::Captured {
                    run_id: d.clone(),
                    layer: l,
                    site: SublayerSite::Ffn,
                    position: sp,
                    sha256: vector_sha256(&v),
                };
                ip = ip.with(Intervention::replace(
                    Address::new(l, SublayerSite::Ffn, [tp])?,
                    v,
                    prov,
                )?)?;
                n += 1;
            }
        }
        let run = execute(&model, &tr.tokens, &ip)?;
        if run.firings != n {
            return refuse(format!(
                "{id}: {} firings for {n} replacements",
                run.firings
            ));
        }
        let got = logits_sha(&run.last_logits);
        let mut rec = json!({"replacements": n, "last_logit_sha256": got, "expected": want, "reproduced": got == want});
        if dump {
            let donor = execute(&model, &plan.runs[&d].tokens, &InterventionPlan::none())?;
            let mut kd = serde_json::Map::new();
            for l in from..plan.layers {
                for (&sp, &tp) in src.iter().zip(&tgt) {
                    kd.insert(format!("{l}|{sp}|{tp}"), json!({"target_k": hex_f32(&run.keys[l][tp]), "donor_k": hex_f32(&donor.keys[l][sp])}));
                }
            }
            rec["kv_dump"] = Value::Object(kd);
        }
        arm_out.insert(id.clone(), rec);
        eprintln!("arm {id}");
    }
    let exe = std::fs::read(std::env::current_exe()?)?;
    let res = json!({"schema": RESULT_SCHEMA, "executable_sha256": sha_bytes(&exe), "capture_manifest_sha256": msha,
                     "verify_plan_sha256": file_sha(vplan_path)?, "reinject": rein, "arms": arm_out});
    let mut f = std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(out_path)?;
    f.write_all(&serde_json::to_vec(&res)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["check", p] => {
            let plan = load_plan(Path::new(p))?;
            println!("checked {} runs", plan.runs.len());
            Ok(())
        }
        ["capture", p, o] => capture(Path::new(p), Path::new(o)),
        ["verify", p, c, v, o] => verify(Path::new(p), Path::new(c), Path::new(v), Path::new(o)),
        _ => Err("Usage: gwtemplate_test3_capture check PLAN | capture PLAN OUTDIR | verify PLAN CAPDIR VPLAN OUT.json".into()),
    }
}
