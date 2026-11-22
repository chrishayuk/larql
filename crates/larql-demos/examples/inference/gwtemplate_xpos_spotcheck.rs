//! GW-TEMPLATE-FACTORIAL stage 3 spot check: an application-level
//! reproduction of the cross-position runner's records for named targets.
//!
//! It is written separately from the runner and shares none of its code. It
//! has its own plan reading (only what it needs, trusting the runner's
//! validation, which it does not repeat), its own capture and its own
//! intervention assembly. It shares only the VINDEX3 library, so it is a
//! reproduction, not an independent implementation.
//!
//! For every named target it recomputes the natural run, every donor run and
//! every arm. It then compares with the runner's record:
//!
//! - the injected carrier at every address;
//! - the K/V hash at every target-span position and layer;
//! - every step's logit hash;
//! - the greedy token and the candidate logits.
//!
//! It decides nothing; it reports differences.
//!
//! Usage: gwtemplate_xpos_spotcheck PLAN.json RUNNER_OUT.jsonl REPORT.json ID [ID ...]
use std::collections::BTreeMap;
use std::error::Error;

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
use larql_vindex::format::vindex3::opplan::plan_component_ops;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

struct Seen(BTreeMap<(usize, usize), Vec<f32>>);

impl StepObserver for Seen {
    fn event(&mut self, _: StepEvent) {}
    fn carrier_write(&mut self, r: CarrierWriteRecord<'_>) {
        if r.site == SublayerSite::Ffn {
            self.0.insert((r.layer, r.position), r.after.to_vec());
        }
    }
}

struct Pass {
    seen: Seen,
    steps: Vec<String>,
    last: Vec<f32>,
    kv: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
}

fn sha_f32(parts: &[&[f32]]) -> String {
    let mut d = Sha256::new();
    for p in parts {
        for x in *p {
            d.update(x.to_bits().to_le_bytes());
        }
    }
    format!("sha256:{:x}", d.finalize())
}

fn toks(v: &Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect()).unwrap_or_default()
}

fn positions(v: &Value) -> Vec<usize> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as usize)).collect()).unwrap_or_default()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        return Err("Usage: gwtemplate_xpos_spotcheck PLAN.json RUNNER_OUT.jsonl REPORT.json ID [ID ...]".into());
    }
    let plan: Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let records: BTreeMap<String, Value> = std::fs::read_to_string(&args[1])?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Value>(l).map(|v| (v["id"].as_str().unwrap_or("").to_string(), v)))
        .collect::<std::result::Result<_, _>>()?;
    let container = std::path::Path::new(plan["container"].as_str().ok_or("container")?);
    let inspection = inspect_container(container, true)?;
    let cplan = plan_component_ops(&inspection, container, "target")?.plan.ok_or("plan absent")?;
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&cplan, &store, &backend, ExecutionSlice::Full)?;
    let nl = cplan.layers.len();

    let pass = |tokens: &[u32], iv: &InterventionPlan| -> Result<Pass> {
        let mut seen = Seen(BTreeMap::new());
        let mut kvs = RowKvState::default();
        let mut steps = Vec::new();
        let mut last = Vec::new();
        {
            let mut s = DecodeSession::over_prepared(&cplan, &ops, &backend, &mut kvs)?;
            let heads = HeadInterventionPlan::none();
            for &t in tokens {
                let o = s.step_intervened(t, &mut seen, iv, &heads)?;
                last = o.logits.ok_or("no logits")?;
                steps.push(sha_f32(&[&last]));
            }
        }
        let mut kv = Vec::new();
        for l in 0..nl {
            kvs.prepare_layer(l);
            kv.push(kvs.rows(l).to_owned_rows());
        }
        Ok(Pass { seen, steps, last, kv })
    };

    let mut report = serde_json::Map::new();
    let mut total_diffs = 0usize;
    for id in &args[3..] {
        let t = plan["targets"].as_array().ok_or("targets")?.iter()
            .find(|t| t["id"] == id.as_str()).ok_or(format!("{id}: not in the plan"))?;
        let rec = records.get(id).ok_or(format!("{id}: not in the runner output"))?;
        let ttoks = toks(&plan["runs"][t["run"].as_str().unwrap_or("")]["tokens"]);
        let cands = toks(&plan["candidate_sets"][t["candidate_set"].as_str().unwrap_or("")]);
        let natural = pass(&ttoks, &InterventionPlan::none())?;
        let mut diffs: Vec<String> = Vec::new();
        if Value::from(natural.steps.clone()) != rec["natural"]["step_logit_sha256"] {
            diffs.push("natural steps".into());
        }
        let mut donor_cache: BTreeMap<String, Pass> = BTreeMap::new();
        let mut checked = 0usize;
        for a in t["arms"].as_array().ok_or("arms")? {
            let name = a["name"].as_str().unwrap_or("");
            let got = &rec["arms"][name];
            let drun = a["donor_run"].as_str().unwrap_or("").to_string();
            if !donor_cache.contains_key(&drun) {
                let d = pass(&toks(&plan["runs"][&drun]["tokens"]), &InterventionPlan::none())?;
                donor_cache.insert(drun.clone(), d);
            }
            let donor = &donor_cache[&drun];
            let (src, tgt) = (positions(&a["source_span"]), positions(&a["target_span"]));
            let from = a["from_layer"].as_u64().unwrap_or(0) as usize;
            let scale = a["mode"] == "norm_match";
            let mut iv = InterventionPlan::none();
            let mut want = BTreeMap::new();
            for l in from - 1..nl {
                for (&sp, &tp) in src.iter().zip(&tgt) {
                    let mut v = donor.seen.0.get(&(l, sp)).ok_or("donor carrier")?.clone();
                    if scale {
                        let n = |x: &[f32]| x.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
                        let s = n(natural.seen.0.get(&(l, tp)).ok_or("target carrier")?) / n(&v);
                        v = v.iter().map(|y| (*y as f64 * s) as f32).collect();
                    }
                    let h = vector_sha256(&v);
                    want.insert(format!("{l}|{tp}"), h.clone());
                    let prov = VectorProvenance::Captured { run_id: drun.clone(), layer: l, site: SublayerSite::Ffn,
                                                            position: sp, sha256: h };
                    iv = iv.with(Intervention::replace(Address::new(l, SublayerSite::Ffn, [tp])?, v, prov)?)?;
                }
            }
            let run = pass(&ttoks, &iv)?;
            for (k, h) in &want {
                if got["carriers"][k] != h.as_str() {
                    diffs.push(format!("{name}: carrier {k}"));
                }
            }
            for &tp in &tgt {
                for l in 0..nl {
                    let h = sha_f32(&[&run.kv[l].0[tp], &run.kv[l].1[tp]]);
                    if got["kv_sha256"][format!("{l}|{tp}")] != h.as_str() {
                        diffs.push(format!("{name}: K/V {l}|{tp}"));
                    }
                }
            }
            if Value::from(run.steps.clone()) != got["step_logit_sha256"] {
                diffs.push(format!("{name}: steps"));
            }
            let mut best = 0;
            for (i, v) in run.last.iter().enumerate() {
                if *v > run.last[best] {
                    best = i;
                }
            }
            if got["readout"]["next"].as_u64() != Some(best as u64) {
                diffs.push(format!("{name}: next"));
            }
            let cl: Vec<f64> = cands.iter().map(|&c| run.last[c as usize] as f64).collect();
            if Value::from(cl) != got["readout"]["candidate_logits"] {
                diffs.push(format!("{name}: candidate logits"));
            }
            checked += 1;
        }
        total_diffs += diffs.len();
        report.insert(id.clone(), json!({"arms_checked": checked, "differences": diffs}));
        eprintln!("spot-checked {id}");
    }
    let out = json!({"schema": "larql.gwtemplate.xpos-spotcheck.v1", "targets": report,
                     "total_differences": total_diffs,
                     "result": if total_diffs == 0 { "REPRODUCED" } else { "DIFFERENCES" }});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&out)?)?;
    println!("{}", out["result"]);
    Ok(())
}
