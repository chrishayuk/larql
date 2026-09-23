//! Manifest-bound GW-0 execution using the canonical CPU observation seam.
use super::*;
use larql_vindex::format::vindex3::opplan::exec::observe::NoopObserver;
use std::collections::BTreeSet;

#[derive(Clone)]
struct Write {
    layer: usize,
    site: SublayerSite,
    before: Vec<f32>,
    after: Vec<f32>,
    delta: Vec<f32>,
    layer_scale: Option<f32>,
}

struct FinalRecorder {
    target_position: usize,
    current: Option<Vec<f32>>,
    pending_scale: Option<f32>,
    writes: Vec<Write>,
}

impl FinalRecorder {
    fn new(target_position: usize) -> Self {
        Self {
            target_position,
            current: None,
            pending_scale: None,
            writes: Vec::new(),
        }
    }
}

impl StepObserver for FinalRecorder {
    fn event(&mut self, event: StepEvent) {
        if let StepEvent::FfnDone { .. } = event {
            if let (Some(scale), Some(current)) = (self.pending_scale.take(), self.current.as_mut())
            {
                current.iter_mut().for_each(|value| *value *= scale);
            }
        }
    }

    fn entering_carrier(&mut self, position: usize, values: &[f32]) {
        if position == self.target_position {
            self.current = Some(values.to_vec());
            self.pending_scale = None;
        }
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.position != self.target_position {
            return;
        }
        let before = self
            .current
            .take()
            .expect("entering/previous carrier precedes write");
        self.writes.push(Write {
            layer: record.layer,
            site: record.site,
            before,
            after: record.after.to_vec(),
            delta: record.delta.to_vec(),
            layer_scale: record.layer_scale,
        });
        self.current = Some(record.after.to_vec());
        self.pending_scale = record.layer_scale;
    }
}

fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        let existing = std::fs::read(path)?;
        if existing == bytes {
            return Ok(());
        }
        return Err(format!("content-addressed artifact collision: {}", path.display()).into());
    }
    atomic_json(path, bytes)
}

fn artifact(root: &Path, kind: &str, values: &[f32]) -> Result<Value> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let digest = sha(&bytes);
    let name = format!("sha256-{digest}.{kind}.f32");
    atomic_bytes(&root.join(&name), &bytes)?;
    Ok(
        json!({"kind": kind, "path": name, "sha256": format!("sha256:{digest}"),
        "bytes": bytes.len(), "dtype": "f32-le"}),
    )
}

fn logits_bytes(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn readout(logits: &[f32], target: usize) -> Result<Value> {
    if target >= logits.len() || logits.iter().any(|value| !value.is_finite()) {
        return Err("invalid target vocabulary readout".into());
    }
    let target_logit = f64::from(logits[target]);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_z = logits
        .iter()
        .map(|value| (f64::from(*value) - max).exp())
        .sum::<f64>()
        .ln()
        + max;
    let rank = logits
        .iter()
        .enumerate()
        .filter(|(index, value)| {
            value.total_cmp(&logits[target]).is_gt()
                || (value.to_bits() == logits[target].to_bits() && *index < target)
        })
        .count()
        + 1;
    let best = logits
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
        .unwrap()
        .0;
    Ok(
        json!({"token_id": target, "rank": rank, "logit": target_logit,
        "log_probability": target_logit-log_z, "top_token_id": best}),
    )
}

fn site_name(site: SublayerSite) -> &'static str {
    if site == SublayerSite::Attention {
        "attention"
    } else {
        "ffn"
    }
}

fn norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn sha_file(path: &Path) -> Result<String> {
    Ok(format!("sha256:{}", sha(&std::fs::read(path)?)))
}

/// `--gw0-batch CONTAINER MANIFEST.json INPUT.jsonl OUTPUT_DIR [START [LIMIT]]`
pub(super) fn run(args: &[String]) -> Result<()> {
    if !(4..=6).contains(&args.len()) {
        return Err("Usage: observatory_record --gw0-batch CONTAINER MANIFEST.json INPUT.jsonl OUTPUT_DIR [START [LIMIT]]".into());
    }
    let root = Path::new(&args[0]);
    let manifest_path = Path::new(&args[1]);
    let input_path = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    let start = args
        .get(4)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(0usize);
    let limit = args
        .get(5)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(usize::MAX);
    std::fs::create_dir_all(output.join("artifacts"))?;
    std::fs::create_dir_all(output.join("records"))?;
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest_path)?)?;
    let manifest_hash = manifest["manifest_sha256"]
        .as_str()
        .ok_or("manifest hash absent")?;
    if manifest["status"] != "frozen_input_unexecuted"
        || sha_file(input_path)? != manifest["rows"]["sha256"]
    {
        return Err("manifest status or input hash mismatch".into());
    }
    let rows: Vec<Value> = std::fs::read_to_string(input_path)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;

    eprintln!("Verifying container once and preparing canonical CPU operands...");
    let inspection = inspect_container(root, true)?;
    if !inspection.is_coherent() {
        return Err(format!("container defects: {:?}", inspection.defects).into());
    }
    let outcome = plan_component_ops(&inspection, root, "target")?;
    if !outcome.closed() {
        return Err(format!("plan refused: {:?}", outcome.defects).into());
    }
    let plan = outcome.plan.ok_or("no closed target plan")?;
    if !plan.residual_topology.is_single_stream() {
        return Err("GW-0 batch requires Single topology".into());
    }
    let plan_hash = sha(&serde_json::to_vec(&plan)?);
    let container_hash = sha(&serde_json::to_vec(
        &json!({"index": inspection.index, "graph": inspection.graph}),
    )?);
    let store = OperandStore::open(root, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    let execution = ExecutionProvenance::of(&ops);
    let execution_fingerprint = execution.fingerprint();
    let tokenizer_bytes = std::fs::read(root.join("tokenizer.json"))?;
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)?;
    let executable_hash = sha(&std::fs::read(std::env::current_exe()?)?);
    let expected_writes = plan
        .layers
        .iter()
        .map(|layer| 1 + usize::from(layer.ffn.is_some()))
        .sum::<usize>();

    for (row_index, row) in rows.iter().enumerate().skip(start).take(limit) {
        let edge_id = row["edge_id"].as_str().ok_or("edge_id missing")?;
        let record_path = output.join("records").join(format!("{edge_id}.json"));
        if record_path.exists() {
            eprintln!(
                "[{}/{}] {} already complete",
                row_index + 1,
                rows.len(),
                edge_id
            );
            continue;
        }
        let prompt = row["prompt"]["text"].as_str().ok_or("prompt missing")?;
        let encoding = tokenizer.encode(prompt, true)?;
        let tokens = encoding.get_ids();
        let declared: Vec<u32> = row["prompt"]["token_ids"]
            .as_array()
            .ok_or("prompt token ids missing")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .map(|v| v as u32)
                    .ok_or("bad prompt token id")
            })
            .collect::<std::result::Result<_, _>>()?;
        if tokens != declared
            || row["capture_request"]["position"].as_u64() != Some((tokens.len() - 1) as u64)
        {
            return Err(format!("{edge_id}: frozen tokenization/capture position mismatch").into());
        }
        let target = row["semantic_edge"]["target_token_ids"][0]
            .as_u64()
            .ok_or("target id missing")? as usize;
        let run_id = format!(
            "gw0-{}-{}",
            edge_id,
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        );
        let started = Instant::now();
        let mut recorder = FinalRecorder::new(tokens.len() - 1);
        let mut observed = Vec::new();
        {
            let mut kv = RowKvState::default();
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
            for token in tokens {
                observed.push(
                    session
                        .step_observed(*token, &mut recorder)?
                        .logits
                        .ok_or("missing logits")?,
                );
            }
        }
        let mut control = Vec::new();
        {
            let mut kv = RowKvState::default();
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
            for token in tokens {
                control.push(
                    session
                        .step_observed(*token, &mut NoopObserver)?
                        .logits
                        .ok_or("missing control logits")?,
                );
            }
        }
        if !bit_parity(&observed, &control) || recorder.writes.len() != expected_writes {
            return Err(format!("{edge_id}: parity or final-position coverage failure").into());
        }

        let mut before_payload = Vec::new();
        let mut after_payload = Vec::new();
        let mut delta_payload = Vec::new();
        let mut sites = Vec::new();
        for write in &recorder.writes {
            before_payload.extend_from_slice(&write.before);
            after_payload.extend_from_slice(&write.after);
            delta_payload.extend_from_slice(&write.delta);
            let before_logits = ops.readout_carrier(&backend, &write.before)?;
            let mut after_state = write.after.clone();
            if let Some(scale) = write.layer_scale {
                backend.scale_row(&mut after_state, scale);
            }
            let after_logits = ops.readout_carrier(&backend, &after_state)?;
            let before = readout(&before_logits, target)?;
            let after = readout(&after_logits, target)?;
            sites.push(json!({"layer": write.layer, "site": site_name(write.site),
                "before": before, "after": after,
                "delta_target_log_probability": after["log_probability"].as_f64().unwrap()-before["log_probability"].as_f64().unwrap(),
                "write_norm": norm(&write.delta), "carrier_norm": norm(&write.after), "layer_scale": write.layer_scale}));
        }
        let artifacts = vec![
            artifact(&output.join("artifacts"), "carrier-before", &before_payload)?,
            artifact(&output.join("artifacts"), "carrier-after", &after_payload)?,
            artifact(&output.join("artifacts"), "delta", &delta_payload)?,
        ];

        let mut strongest: Vec<usize> = (0..sites.len())
            .filter(|index| {
                sites[*index]["delta_target_log_probability"]
                    .as_f64()
                    .unwrap()
                    > 0.0
            })
            .collect();
        strongest.sort_by(|a, b| {
            sites[*b]["delta_target_log_probability"]
                .as_f64()
                .unwrap()
                .total_cmp(&sites[*a]["delta_target_log_probability"].as_f64().unwrap())
                .then_with(|| a.cmp(b))
        });
        strongest.truncate(3);
        let stable_emergence = (0..sites.len()).find(|index| {
            sites[*index..]
                .iter()
                .all(|site| site["after"]["rank"].as_u64().unwrap() <= 10)
        });
        let fallback = (0..sites.len()).max_by_key(|index| {
            sites[*index]["before"]["rank"]
                .as_u64()
                .unwrap()
                .saturating_sub(sites[*index]["after"]["rank"].as_u64().unwrap())
        });
        let emergence = stable_emergence.or(fallback);
        let mut candidates: BTreeSet<usize> = strongest.into_iter().collect();
        if let Some(index) = emergence {
            candidates.insert(index);
        }
        let candidate_sites: Vec<_> = candidates.into_iter().map(|index| json!({"layer": sites[index]["layer"], "site": sites[index]["site"], "operator_ids": []})).collect();
        let final_logits = observed.last().ok_or("no logits")?;
        let final_target = readout(final_logits, target)?;
        let produced = final_logits
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .unwrap()
            .0;
        let observed_hash = sha(&logits_bytes(&observed));
        let record = json!({"schema":"larql.gw0.execution-record.v1", "edge_id":edge_id, "run_id":run_id,
            "manifest_sha256":manifest_hash, "prompt_row_index":row_index, "capture_position":tokens.len()-1,
            "identity":{"container":format!("sha256:{container_hash}"),"container_index":manifest["model"]["container_identity"],
                "plan":format!("sha256:{plan_hash}"),"tokenizer":format!("sha256:{}",sha(&tokenizer_bytes)),
                "execution_fingerprint":execution_fingerprint,"runner":format!("sha256:{executable_hash}")},
            "prompt":{"text":prompt,"token_ids":tokens}, "target":{"token_id":target,"final":final_target},
            "produced":{"token_id":produced,"text":tokenizer.decode(&[produced as u32],false)?},
            "parity":{"result":"pass","criterion":"all finite logits bit-identical at every prompt position","observed_sha256":format!("sha256:{observed_hash}"),"unobserved_sha256":format!("sha256:{}",sha(&logits_bytes(&control)))},
            "coverage":{"result":"complete","writes":recorder.writes.len(),"expected_writes":expected_writes},
            "sites":sites,"transition_candidate":{"rule":"stable target top-10 emergence (fallback: largest rank improvement) union top-3 positive target log-probability writes","sites":candidate_sites,"feature_addresses":[]},
            "surface_status":{"relation_readable_band":"unavailable_no_frozen_decoder","candidate_read_operators":"refused_post_attention_norm"},
            "causal_status":"untested","causal_evidence_ids":[],"artifacts":artifacts,"elapsed_ns":started.elapsed().as_nanos().to_string()});
        let bytes = serde_json::to_vec_pretty(&record)?;
        atomic_json(&record_path, &bytes)?;
        eprintln!(
            "[{}/{}] {} PASS {:.2}s",
            row_index + 1,
            rows.len(),
            edge_id,
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
