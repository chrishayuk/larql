use larql_compute::ComputeBackend;
use larql_inference::engines::test_utils::{make_test_vindex, make_test_weights};
use larql_inference::layer_graph::{CudaResidentAttentionMmapExpertsGraph, LayerGraph};
use larql_inference::vindex::WalkFfnConfig;
use ndarray::Array2;
use std::time::Instant;

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(cuda) = larql_compute::CudaBackend::new() else {
        println!("status=skipped reason=cuda-unavailable");
        return Ok(());
    };

    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let input = Array2::from_shape_vec(
        (1, weights.hidden_size),
        (0..weights.hidden_size)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect(),
    )?;

    let before_rss_kb = current_rss_kb();
    let residency_start = Instant::now();
    let residents: Vec<_> = (0..weights.num_layers)
        .map(|layer| {
            larql_inference::attention::CudaAttentionResidency::from_layer(&weights, &cuda, layer)
                .ok_or_else(|| format!("failed to make resident attention for layer {layer}"))
        })
        .collect::<Result<_, _>>()?;
    let residency_ms = residency_start.elapsed().as_secs_f64() * 1000.0;
    let after_residency_rss_kb = current_rss_kb();

    let graph = CudaResidentAttentionMmapExpertsGraph {
        index: &index,
        cuda: &cuda,
        resident_attention: &residents,
        expert_backend: None,
        expert_config: WalkFfnConfig::sparse(weights.num_layers, 1),
        layer_range: 0..weights.num_layers,
    };

    let start = Instant::now();
    let out = graph
        .forward_layer(&weights, &input, 0)
        .expect("layer 0 should run");
    let forward_ms = start.elapsed().as_secs_f64() * 1000.0;
    let after_forward_rss_kb = current_rss_kb();

    let max_abs = out.residual.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));

    println!(
        "status=ok graph={} attention_backend={} expert_backend=none expert_config=sparse layers={} hidden={} residency_ms={:.3} forward_ms={:.3} rss_before_kb={} rss_after_residency_kb={} rss_after_forward_kb={} residual_shape={:?} residual_max_abs={:.6}",
        graph.name(),
        cuda.name(),
        weights.num_layers,
        weights.hidden_size,
        residency_ms,
        forward_ms,
        before_rss_kb,
        after_residency_rss_kb,
        after_forward_rss_kb,
        out.residual.shape(),
        max_abs,
    );
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn main() {
    println!("status=skipped reason=requires-linux-cuda-feature");
}

fn current_rss_kb() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    status
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:").and_then(|rest| {
                rest.split_whitespace()
                    .next()
                    .and_then(|kb| kb.parse::<u64>().ok())
            })
        })
        .unwrap_or(0)
}
