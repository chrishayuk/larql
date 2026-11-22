//! GW-BREADCRUMB-2 seal evidence: pre-treatment carrier and key/value hashes.
//!
//! Natural inference only, independent of the intervention runner: it reads
//! no runner output and has no intervention path (every step passes
//! `InterventionPlan::none()`). For every run of a seal plan, a token
//! sequence and a position range, it records:
//!
//! - the FFN carrier hash (`vector_sha256`, as the runner) at every layer and
//!   every position in the range;
//! - the key/value row hash (sha256 over the key bits then the value bits,
//!   little-endian, as the runner) at every layer and position in the range;
//! - the logit hash of every step.
//!
//! The seal binds these before any transplant; admission checks the runner's
//! records against them.
//!
//! It refuses on any plan fault before the model is loaded, writes
//! OUT.jsonl.partial while running and OUT.jsonl plus its manifest only on
//! completion, and never overwrites.
//!
//! Usage: gwbreadcrumb2_seal SEAL_PLAN.json OUT.jsonl
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use larql_inference::vindex3::{CarrierWriteRecord, StepEvent, StepObserver, SublayerSite};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    intervene::{vector_sha256, InterventionPlan},
    intervene_heads::HeadInterventionPlan,
    kv::{KvState, RowKvState},
    operands::OperandStore,
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
};
use larql_vindex::format::vindex3::opplan::plan_component_ops;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const PLAN_SCHEMA: &str = "larql.gwbreadcrumb2.seal-plan.v1";

fn refuse<T>(reason: impl Into<String>) -> Result<T> {
    Err(format!("REFUSED: {}", reason.into()).into())
}

#[derive(Default)]
struct Ffn {
    carriers: BTreeMap<(usize, usize), (Vec<f32>, usize)>,
    scaled: usize,
}

impl StepObserver for Ffn {
    fn event(&mut self, _: StepEvent) {}
    fn carrier_write(&mut self, r: CarrierWriteRecord<'_>) {
        if r.site == SublayerSite::Ffn {
            let e = self.carriers.entry((r.layer, r.position)).or_default();
            e.0 = r.after.to_vec();
            e.1 += 1;
        }
        if r.layer_scale.is_some() {
            self.scaled += 1;
        }
    }
}

fn f32_sha(parts: &[&[f32]]) -> String {
    let mut d = Sha256::new();
    for p in parts {
        for v in *p {
            d.update(v.to_bits().to_le_bytes());
        }
    }
    format!("sha256:{:x}", d.finalize())
}

struct SealRun {
    id: String,
    tokens: Vec<u32>,
    lo: usize,
    hi: usize,
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

fn load_plan(path: &str) -> Result<(Value, Vec<SealRun>)> {
    let plan: Value = match serde_json::from_slice(&std::fs::read(path)?) {
        Ok(v) => v,
        Err(e) => return refuse(format!("plan is not JSON: {e}")),
    };
    if plan["schema"] != PLAN_SCHEMA {
        return refuse("plan schema");
    }
    if plan["container"].as_str().is_none() {
        return refuse("plan has no container");
    }
    let list = match plan["runs"].as_array() {
        Some(r) if !r.is_empty() => r,
        _ => return refuse("plan has no runs"),
    };
    let (mut runs, mut seen) = (Vec::new(), BTreeSet::new());
    for r in list {
        let id = match r["id"].as_str() {
            Some(c) => c.to_string(),
            None => return refuse("a run has no id"),
        };
        if !seen.insert(id.clone()) {
            return refuse(format!("{id}: run repeated"));
        }
        let tokens = ids(&r["tokens"], &format!("{id} tokens"))?;
        let (lo, hi) = match (r["positions"][0].as_u64(), r["positions"][1].as_u64()) {
            (Some(a), Some(b)) => (a as usize, b as usize),
            _ => return refuse(format!("{id}: positions must be [lo, hi]")),
        };
        if tokens.is_empty() || lo > hi || hi >= tokens.len() {
            return refuse(format!("{id}: positions outside the tokens"));
        }
        runs.push(SealRun { id, tokens, lo, hi });
    }
    Ok((plan, runs))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("Usage: gwbreadcrumb2_seal SEAL_PLAN.json OUT.jsonl".into());
    }
    let (plan_json, runs) = load_plan(&args[0])?;
    let out_path = &args[1];
    let partial = format!("{out_path}.partial");
    if Path::new(out_path).exists()
        || Path::new(&format!("{out_path}.manifest.json")).exists()
        || Path::new(&partial).exists()
    {
        return refuse("output exists; a run never overwrites");
    }
    let container = Path::new(plan_json["container"].as_str().unwrap_or_default());
    let inspection = inspect_container(container, true)?;
    let plan = plan_component_ops(&inspection, container, "target")?
        .plan
        .ok_or("plan absent")?;
    if plan.layers.iter().any(|l| l.layer_scale.is_some()) {
        return refuse("the plan declares a layer scale");
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    let layers = plan.layers.len();
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(&partial)?;
    for r in &runs {
        let mut ffn = Ffn::default();
        let mut kv = RowKvState::default();
        let mut steps = Vec::new();
        let mut firings = 0;
        {
            let mut s = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
            let none = HeadInterventionPlan::none();
            let nothing = InterventionPlan::none();
            for &t in &r.tokens {
                let o = s.step_intervened(t, &mut ffn, &nothing, &none)?;
                firings += o.firings.len();
                steps.push(f32_sha(&[&o.logits.ok_or("a step returned no logits")?]));
            }
        }
        if firings != 0 || ffn.scaled != 0 {
            return refuse(format!(
                "{}: a natural run fired or carried a layer scale",
                r.id
            ));
        }
        let (mut carriers, mut kvs) = (serde_json::Map::new(), serde_json::Map::new());
        for layer in 0..layers {
            kv.prepare_layer(layer);
            let (k, v) = kv.rows(layer).to_owned_rows();
            for p in r.lo..=r.hi {
                let c = match ffn.carriers.get(&(layer, p)) {
                    Some((c, 1)) => c,
                    _ => {
                        return refuse(format!(
                            "{}: carrier at {layer}|{p} absent or written twice",
                            r.id
                        ))
                    }
                };
                carriers.insert(format!("{layer}|{p}"), json!(vector_sha256(c)));
                kvs.insert(format!("{layer}|{p}"), json!(f32_sha(&[&k[p], &v[p]])));
            }
        }
        writeln!(
            out,
            "{}",
            json!({"id": r.id, "tokens": r.tokens, "positions": [r.lo, r.hi],
                                   "carriers": carriers, "kv_sha256": kvs, "step_logit_sha256": steps})
        )?;
        eprintln!("sealed {}", r.id);
    }
    drop(out);
    std::fs::rename(&partial, out_path)?;
    let bytes = std::fs::read(out_path)?;
    let exe = std::fs::read(std::env::current_exe()?)?;
    let manifest = json!({
        "schema": "larql.gwbreadcrumb2.seal-evidence-manifest.v1", "status": "complete",
        "executable_sha256": format!("sha256:{:x}", Sha256::digest(&exe)),
        "plan_sha256": format!("sha256:{:x}", Sha256::digest(std::fs::read(&args[0])?)),
        "output_sha256": format!("sha256:{:x}", Sha256::digest(&bytes)),
        "runs": runs.len(), "layers": layers,
    });
    std::fs::write(
        format!("{out_path}.manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}
