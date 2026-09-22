//! Head analysis of borrowed values captured during canonical CPU attention.
//! Stored W_O/head rows are used ONLY after execution; this is a linear
//! direction probe, not a normalized logit lens or a causal intervention.
use super::*;
use larql_vindex::format::vindex3::opplan::exec::observe::AttentionHeadRecord;

#[derive(serde::Serialize)]
pub(super) struct CapturedHead {
    pub layer: usize,
    pub position: usize,
    pub head: usize,
    pub kv_head: usize,
    pub source_start: usize,
    pub weights: Vec<f32>,
    pub values: Vec<f32>,
}
impl CapturedHead {
    pub fn capture(layer: usize, r: AttentionHeadRecord<'_>) -> Self {
        Self {
            layer,
            position: r.position,
            head: r.head,
            kv_head: r.kv_head,
            source_start: r.source_start,
            weights: r.weights.to_vec(),
            values: r.values.to_vec(),
        }
    }
}

pub(super) fn preflight(plan: &ComponentOpPlan, positions: usize, hidden: usize) -> Result<usize> {
    let mut heads = 0usize;
    let mut bytes = 0usize;
    for layer in &plan.layers {
        let a = layer
            .attention
            .softmax()
            .ok_or("head capture requires softmax layers")?;
        if a.output_gate.is_some() || a.o_bias.is_some() || layer.post_attention_norm.is_some() {
            return Err("head content adapter requires ungated, bias-free attention without post-attention norm; capture semantics must be extended explicitly".into());
        }
        if a.o.shape != [hidden, a.num_q_heads * a.head_dim] {
            return Err("unsupported W_O geometry".into());
        }
        if !matches!(a.o.dtype.as_str(), "BF16" | "F16" | "F32") {
            return Err("head analysis requires stored floating W_O".into());
        }
        heads += positions * a.num_q_heads;
        bytes += positions * a.num_q_heads * (a.head_dim + positions) * 4;
        if bytes > 64 * 1024 * 1024
            || heads > 16384
            || hidden * a.num_q_heads * a.head_dim * 4 > 128 * 1024 * 1024
        {
            return Err("head capture or per-layer analysis exceeds its memory bound".into());
        }
    }
    Ok(heads)
}

pub(super) fn analyse(
    captured: &[CapturedHead],
    carriers: &[CapturedCarrier],
    plan: &ComponentOpPlan,
    store: &OperandStore,
    targets: &[(u32, String)],
    hidden: usize,
    basis: &str,
) -> Result<Value> {
    let token_rows = head_rows(
        plan,
        store,
        &targets.iter().map(|t| t.0).collect::<Vec<_>>(),
        hidden,
    )?;
    let mut rows = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut sums = std::collections::BTreeMap::<(usize, usize), Vec<f64>>::new();
    for layer in &plan.layers {
        let a = layer.attention.softmax().ok_or("missing softmax op")?;
        // One layer at a time, never the whole model. This is analysis, not a
        // replacement executor: no attention scores or model traversal are rerun.
        let wo = store.load(&a.o)?;
        let width = a.num_q_heads * a.head_dim;
        if wo.len() != hidden * width || wo.iter().any(|v| !v.is_finite()) {
            return Err("invalid stored W_O".into());
        }
        for c in captured.iter().filter(|h| h.layer == layer.layer) {
            if !seen.insert((c.position, c.layer, c.head))
                || c.head >= a.num_q_heads
                || c.kv_head != c.head / (a.num_q_heads / a.num_kv_heads)
                || c.values.len() != a.head_dim
                || c.source_start > c.position
                || c.weights.len() != c.position + 1 - c.source_start
                || c.values.iter().chain(&c.weights).any(|v| !v.is_finite())
                || c.weights.iter().any(|v| *v < 0.0 || *v > 1.0)
                || c.weights.iter().map(|v| f64::from(*v)).sum::<f64>() > 1.00001
            {
                return Err("invalid or duplicate head observation".into());
            }
            let scale = f64::from(layer.residual_scale.unwrap_or(1.0));
            let contribution: Vec<f64> = (0..hidden)
                .map(|i| {
                    wo[i * width + c.head * a.head_dim..i * width + (c.head + 1) * a.head_dim]
                        .iter()
                        .zip(&c.values)
                        .map(|(&w, &v)| f64::from(w) * f64::from(v))
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let sum = sums
                .entry((c.position, c.layer))
                .or_insert_with(|| vec![0.; hidden]);
            for (sum, value) in sum.iter_mut().zip(&contribution) {
                *sum += value;
            }
            let coefficients: Vec<f64> = token_rows
                .iter()
                .map(|r| {
                    r.iter()
                        .zip(&contribution)
                        .map(|(&w, &v)| f64::from(w) * v)
                        .sum()
                })
                .collect();
            let norm = contribution.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !norm.is_finite() || coefficients.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite head projection".into());
            }
            let projections: Vec<_> = targets.iter().zip(&coefficients).map(|((id,label),value)|
                json!({"token_id": id, "token": label, "coefficient": value})).collect();
            let sources: Vec<_> = c
                .weights
                .iter()
                .enumerate()
                .map(|(i, &w)| json!({"position": c.source_start+i, "weight": f64::from(w)}))
                .collect();
            for ((_, target), coefficient) in targets.iter().zip(&coefficients).take(1) {
                rows.push(json!({"site": format!("{}:{}:attention_write:main:residual", c.position,c.layer),
                    "head": c.head, "target": target, "dla": coefficient, "sources": sources,
                    "content": {"method": "Stored W_O × captured head × residual scale; selected token directions, no final norm.",
                        "basis": "stored-W_O-and-head-v1 / identity in heads.basis", "vector_norm": norm, "projections": projections}}));
            }
        }
        eprintln!(
            "Head content: layer {} / {}",
            layer.layer + 1,
            plan.layers.len()
        );
    }
    let mut max_relative_l2 = 0.0_f64;
    for ((position, layer), sum) in &sums {
        let actual = carriers
            .iter()
            .find(|c| {
                c.position == *position && c.layer == *layer && c.site == SublayerSite::Attention
            })
            .ok_or("missing actual attention carrier write")?;
        if actual.delta.len() != hidden {
            return Err("missing captured write values".into());
        }
        let error = sum
            .iter()
            .zip(&actual.delta)
            .map(|(a, b)| (a - f64::from(*b)).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = actual
            .delta
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();
        let relative = if norm == 0.0 { error } else { error / norm };
        max_relative_l2 = max_relative_l2.max(relative);
    }
    if !max_relative_l2.is_finite() || max_relative_l2 > 1e-4 {
        return Err(format!("stored head projections do not reconstruct actual writes within 1e-4 relative L2: {max_relative_l2}").into());
    }
    eprintln!(
        "Head reconstruction: {} writes, max relative L2 {max_relative_l2:e}",
        sums.len()
    );
    Ok(
        json!({"method": "Measured canonical per-head source weights and weighted V; derived raw direction attribution via stored W_O and selected unembedding rows. Signed score is not probability or a normalized logit change. No final norm, output multiplier or softcap; physical projection rounding may differ. Selected token directions only, not vocabulary top-k or additive normalized-logit attribution. Separate intrusive head capture, not Standard.",
        "basis": basis, "rows": rows, "reconstruction": {"writes": sums.len(), "max_relative_l2": max_relative_l2, "threshold": 1e-4, "method": "Sum stored-W_O per-head projections versus captured applied attention delta; f64 analysis vs production rounding", "result": "pass"}}),
    )
}
