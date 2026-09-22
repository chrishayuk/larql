//! Cost-instrument parity over every captured V2 subject token.
use super::*;
use larql_vindex::format::vindex3::opplan::exec::payload_prefix::PayloadPrefix;

/// GW-V2's frozen source: query head 1, which reads KV head 0 of the
/// Gemma 3 4B source layer's unbiased 8Q/4KV/256 V projection. Declared
/// here, by the experiment, so the library adapter stays geometry-free;
/// a plan of any other shape refuses before a prefix is prepared.
const SOURCE_HEAD: usize = 1;
const FROZEN_Q_HEADS: usize = 8;
const FROZEN_KV_HEADS: usize = 4;

fn frozen_source_geometry(plan: &ComponentOpPlan, depth: usize) -> Result<()> {
    let op = plan
        .layers
        .get(depth)
        .and_then(|layer| layer.attention.softmax())
        .ok_or("GW-V2 source layer is not softmax")?;
    if op.num_q_heads != FROZEN_Q_HEADS || op.num_kv_heads != FROZEN_KV_HEADS || op.head_dim != D {
        return Err("GW-V2 requires the frozen 8Q/4KV/256 source geometry".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn check(
    e: &Value,
    output: &Path,
    capture_path: &Path,
    plan: &ComponentOpPlan,
    store: &OperandStore,
    full: &PreparedOperands,
    backend: &ProductionBackend,
) -> Result<()> {
    let capture = read_json(capture_path)?;
    if capture["lineage"] != e["lineage"] {
        return Err("prefix capture lineage mismatch".into());
    }
    let v = checked(
        capture_path.parent().unwrap(),
        &capture,
        "source-depth-v.f32",
        usize_at(&e["subject_rows"])? * 8 * D,
    )?;
    let mut reports = Vec::new();
    for (depth_index, depth) in DEPTHS[..7].iter().copied().enumerate() {
        frozen_source_geometry(plan, depth)?;
        let prefix = PayloadPrefix::prepare(plan, store, backend, depth)?;
        let mut positions_checked = 0;
        for item in e["rows"].as_array().unwrap() {
            let tokens: Vec<u32> = ids(&item["row"]["prompt"]["token_ids"])?
                .into_iter()
                .map(|x| x as u32)
                .collect();
            let positions = ids(&item["roles"]["subject_entity"])?;
            let offset = usize_at(&item["subject_offset"])?;
            let constructed =
                prefix.values(&tokens, &positions, SOURCE_HEAD, plan, full, backend)?;
            for (ordinal, vector) in constructed.iter().enumerate() {
                let start = ((offset + ordinal) * 8 + depth_index) * D;
                if !same(vector, &v[start..start + D]) {
                    return Err(format!("prefix V parity failed at depth={depth}, subject_offset={offset}, ordinal={ordinal}").into());
                }
                positions_checked += 1;
            }
        }
        reports.push(json!({"depth":depth,"subject_positions_checked":positions_checked,"bit_mismatches":0,"prefix_prepared_resident_bytes":prefix.resident_bytes()}));
        eprintln!("GW-V2 prefix L{depth}: {positions_checked} subject positions bit-exact");
    }
    let report = json!({"schema":"larql.gwv2.prefix-parity.v1","lineage":e["lineage"],"capture_file_sha256":file_sha(capture_path)?,"depths":reports,"executable_sha256":file_sha(&std::env::current_exe()?)?});
    atomic_json(
        &output.join("prefix-parity.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
