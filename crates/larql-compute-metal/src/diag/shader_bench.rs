//! Metal shader bench and pipeline inventory.
//!
//! This harness is intentionally separate from Criterion benches:
//! it measures GPU command-buffer behavior directly, reports the active
//! shader inventory, and keeps isolated timings visibly separate from
//! production-shaped batched timings.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use crate::buffers::read_buffer_f32;
use crate::kernels::KernelHandle;
use crate::ops::q4_common::quantize_to_q8;
use crate::MetalBackend;
use larql_compute::cpu::ops::q4_common::{
    quantize_q4_0, quantize_q4_k, quantize_q4_kf, quantize_q6_k,
};
use larql_compute::cpu::ops::q8_matvec::quantize_weights_q8;

const GEMMA3_4B_KV_ROWS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Smoke,
    Gemma3,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub profile: Profile,
    pub warmup: usize,
    pub iters: usize,
    pub n_layers: usize,
    pub json: Option<PathBuf>,
    pub compare: Option<PathBuf>,
    pub threshold_pct: f64,
    pub inventory_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            profile: Profile::Smoke,
            warmup: 2,
            iters: 8,
            n_layers: 4,
            json: None,
            compare: None,
            threshold_pct: 5.0,
            inventory_only: false,
        }
    }
}

impl Config {
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        let mut cfg = Self::default();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--profile" => {
                    i += 1;
                    let Some(value) = args.get(i) else {
                        return Err("--profile requires smoke or gemma3".into());
                    };
                    match value.as_str() {
                        "smoke" => {
                            cfg.profile = Profile::Smoke;
                            cfg.warmup = 2;
                            cfg.iters = 8;
                            cfg.n_layers = 4;
                        }
                        "gemma3" => {
                            cfg.profile = Profile::Gemma3;
                            cfg.warmup = 5;
                            cfg.iters = 30;
                            cfg.n_layers = 34;
                        }
                        _ => return Err(format!("unknown profile `{value}`")),
                    }
                }
                "--warmup" => {
                    i += 1;
                    cfg.warmup = parse_usize(args.get(i), "--warmup")?;
                }
                "--iters" => {
                    i += 1;
                    cfg.iters = parse_usize(args.get(i), "--iters")?;
                }
                "--layers" => {
                    i += 1;
                    cfg.n_layers = parse_usize(args.get(i), "--layers")?;
                }
                "--json" => {
                    i += 1;
                    let Some(path) = args.get(i) else {
                        return Err("--json requires a path".into());
                    };
                    cfg.json = Some(PathBuf::from(path));
                }
                "--compare" => {
                    i += 1;
                    let Some(path) = args.get(i) else {
                        return Err("--compare requires a path".into());
                    };
                    cfg.compare = Some(PathBuf::from(path));
                }
                "--threshold" => {
                    i += 1;
                    cfg.threshold_pct = parse_f64(args.get(i), "--threshold")?;
                }
                "--inventory-only" => cfg.inventory_only = true,
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown argument `{other}`")),
            }
            i += 1;
        }
        if cfg.warmup == 0 || cfg.iters == 0 || cfg.n_layers == 0 {
            return Err("--warmup, --iters, and --layers must be non-zero".into());
        }
        if !cfg.threshold_pct.is_finite() || cfg.threshold_pct < 0.0 {
            return Err("--threshold must be a non-negative percentage".into());
        }
        Ok(cfg)
    }
}

pub fn usage() -> String {
    "Usage: cargo run --release --features metal -p larql-compute --example diag_shader_bench -- [--profile smoke|gemma3] [--warmup N] [--iters N] [--layers N] [--inventory-only] [--json PATH] [--compare PATH] [--threshold PCT]".into()
}

fn parse_usize(value: Option<&String>, flag: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse::<usize>()
        .map_err(|_| format!("{flag} requires a positive integer"))
}

fn parse_f64(value: Option<&String>, flag: &str) -> Result<f64, String> {
    value
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse::<f64>()
        .map_err(|_| format!("{flag} requires a number"))
}

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    hidden: usize,
    inter: usize,
    q_rows: usize,
    kv_rows: usize,
    lm_rows: usize,
}

impl Shape {
    fn for_profile(profile: Profile) -> Self {
        match profile {
            Profile::Smoke => Self {
                label: "smoke",
                hidden: 512,
                inter: 2048,
                q_rows: 1024,
                kv_rows: 512,
                lm_rows: 8192,
            },
            Profile::Gemma3 => Self {
                label: "gemma3-4b",
                hidden: 2560,
                inter: 10240,
                q_rows: 8192,
                kv_rows: GEMMA3_4B_KV_ROWS,
                // Full Gemma 3 vocab would allocate ~2.7GB for f32
                // lm_head input alone. Keep shader bench usable by
                // capping the synthetic f32/f16 gemv case while other
                // kernels use production layer shapes.
                lm_rows: 32768,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: &'static str,
    pub family: &'static str,
    pub status: &'static str,
    pub shape: String,
    pub rows_per_tg: Option<u64>,
    pub threads_per_tg: Option<u64>,
    pub bytes_per_call: u64,
    pub isolated_ms: Option<f64>,
    pub isolated_sd_ms: Option<f64>,
    pub batched_ms: Option<f64>,
    pub batched_gbs: Option<f64>,
    pub output_nonzero: Option<usize>,
    pub sanity: &'static str,
    pub note: &'static str,
}

struct InventoryItem {
    name: &'static str,
    family: &'static str,
    status: &'static str,
    note: &'static str,
}

pub fn run(cfg: &Config) -> Result<Vec<BenchResult>, String> {
    let shape = Shape::for_profile(cfg.profile);

    println!("Metal shader bench");
    println!(
        "profile={} hidden={} inter={} q_rows={} kv_rows={} lm_rows={} layers={} warmup={} iters={}",
        shape.label,
        shape.hidden,
        shape.inter,
        shape.q_rows,
        shape.kv_rows,
        shape.lm_rows,
        cfg.n_layers,
        cfg.warmup,
        cfg.iters
    );
    println!();

    print_inventory();

    let mut results = inventory_results(cfg.inventory_only);
    if cfg.inventory_only {
        print_inventory_rows(&results);
        if let Some(path) = &cfg.json {
            std::fs::write(path, to_json(&results)).map_err(|e| format!("write json: {e}"))?;
            println!();
            println!("wrote {}", path.display());
        }
        return Ok(results);
    }

    let metal = MetalBackend::new().ok_or("Metal backend unavailable")?;

    results.extend(run_benches(&metal, cfg, shape));
    print_results(&results);

    if let Some(path) = &cfg.compare {
        let baseline = load_baseline(path)?;
        print_compare(&results, &baseline, path, cfg.threshold_pct);
    }

    if let Some(path) = &cfg.json {
        std::fs::write(path, to_json(&results)).map_err(|e| format!("write json: {e}"))?;
        println!();
        println!("wrote {}", path.display());
    }

    Ok(results)
}

fn run_benches(metal: &MetalBackend, cfg: &Config, shape: Shape) -> Vec<BenchResult> {
    let mut out = Vec::new();

    out.push(bench_q4_0_matvec(metal, cfg, shape));
    out.push(bench_q8_matvec(metal, cfg, shape));

    let q4k_w = quantize_q4_k(&synth_f32(shape.hidden * shape.hidden, 0.11));
    let q6k_w = quantize_q6_k(&synth_f32(shape.hidden * shape.inter, 0.12));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q4k_matvec_active",
        "q4k-matvec",
        &metal.quant.q4k_matvec_pipeline,
        &q4k_w,
        shape.hidden,
        shape.hidden,
        "active production Q4_K matvec handle after env selection",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q4k_matvec_4sg",
        "q4k-matvec",
        &metal.quant.q4k_matvec_4sg_pipeline,
        &q4k_w,
        shape.hidden,
        shape.hidden,
        "explicit 4-simdgroup Q4_K variant",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q4k_matvec_8sg",
        "q4k-matvec",
        &metal.quant.q4k_matvec_8sg_pipeline,
        &q4k_w,
        shape.hidden,
        shape.hidden,
        "explicit 8-simdgroup Q4_K variant",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q4k_matvec_stride32",
        "q4k-matvec",
        &metal.quant.q4k_matvec_stride32_pipeline,
        &q4k_w,
        shape.hidden,
        shape.hidden,
        "LM-head correctness variant at hidden-square shape",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q6k_matvec_active",
        "q6k-matvec",
        &metal.quant.q6k_matvec_pipeline,
        &q6k_w,
        shape.hidden,
        shape.inter,
        "active production Q6_K matvec handle after env selection",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q6k_matvec_4sg",
        "q6k-matvec",
        &metal.quant.q6k_matvec_4sg_pipeline,
        &q6k_w,
        shape.hidden,
        shape.inter,
        "explicit 4-simdgroup Q6_K variant",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q6k_matvec_8sg",
        "q6k-matvec",
        &metal.quant.q6k_matvec_8sg_pipeline,
        &q6k_w,
        shape.hidden,
        shape.inter,
        "explicit 8-simdgroup Q6_K variant",
    ));

    out.extend(bench_gate_up_family(metal, cfg, shape));
    out.extend(bench_geglu_down_family(metal, cfg, shape));
    out.extend(bench_qkv_family(metal, cfg, shape));
    out.push(bench_f32_gemv(metal, cfg, shape));

    out
}

fn bench_q4_0_matvec(metal: &MetalBackend, cfg: &Config, shape: Shape) -> BenchResult {
    let n = shape.hidden;
    let k = shape.hidden;
    let w = quantize_q4_0(&synth_f32(n * k, 0.21));
    let x = synth_f32(k, 0.31);
    let (q8_x, q8_scales) = quantize_to_q8(&x);
    let bufs = metal.bufs();
    let wb = bufs.get_bytes(&w);
    let xb = bufs.transient_from_i8(&q8_x);
    let sb = bufs.transient_from_f32(&q8_scales);
    let ob = bufs.output((n * 4) as u64);
    let kh = &metal.q4.matvec;
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        "q4_matvec_v4",
        "q4-0-matvec",
        kh,
        format!("N={n} K={k}"),
        w.len() as u64 + q8_x.len() as u64 + (q8_scales.len() * 4) as u64,
        &ob,
        n,
        "checked",
        "Q4_0 x Q8 input matvec",
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&sb), 0);
            enc.set_buffer(3, Some(&ob), 0);
            enc.set_bytes(4, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

fn bench_q8_matvec(metal: &MetalBackend, cfg: &Config, shape: Shape) -> BenchResult {
    let n = shape.hidden;
    let k = shape.hidden;
    let (w_q8, w_scales) = quantize_weights_q8(&synth_f32(n * k, 0.22), n, k);
    let x = synth_f32(k, 0.32);
    let (x_q8, x_scales) = quantize_to_q8(&x);
    let bufs = metal.bufs();
    let wb = bufs.transient_from_i8(&w_q8);
    let wsb = bufs.transient_from_f32(&w_scales);
    let xb = bufs.transient_from_i8(&x_q8);
    let xsb = bufs.transient_from_f32(&x_scales);
    let ob = bufs.output((n * 4) as u64);
    let kh = &metal.quant.q8_matvec_pipeline;
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        "q8_matvec",
        "q8-matvec",
        kh,
        format!("N={n} K={k}"),
        w_q8.len() as u64 + (w_scales.len() * 4) as u64,
        &ob,
        n,
        "checked",
        "Q8_0 x Q8 input matvec",
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&wsb), 0);
            enc.set_buffer(3, Some(&xsb), 0);
            enc.set_buffer(4, Some(&ob), 0);
            enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn bench_qk_matvec(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
    name: &'static str,
    family: &'static str,
    kh: &KernelHandle,
    w: &[u8],
    n: usize,
    k: usize,
    note: &'static str,
) -> BenchResult {
    let x = synth_f32(k, 0.41);
    let bufs = metal.bufs();
    let wb = bufs.get_bytes(w);
    let xb = bufs.transient_from_f32(&x);
    let ob = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        name,
        family,
        kh,
        format!("{} N={n} K={k}", shape.label),
        w.len() as u64,
        &ob,
        n,
        "checked",
        note,
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

fn bench_gate_up_family(metal: &MetalBackend, cfg: &Config, shape: Shape) -> Vec<BenchResult> {
    let n = shape.inter;
    let k = shape.hidden;
    let gate_q4k = quantize_q4_k(&synth_f32(n * k, 0.51));
    let up_q4k = quantize_q4_k(&synth_f32(n * k, 0.52));
    let gate_q4kf = quantize_q4_kf(&synth_f32(n * k, 0.53));
    let up_q4kf = quantize_q4_kf(&synth_f32(n * k, 0.54));
    let mut out = Vec::new();
    for (name, kh, gate, up, sanity, note) in [
        (
            "q4k_ffn_gate_up",
            &metal.ffn.q4k_ffn_gate_up_pipeline,
            gate_q4k.as_slice(),
            up_q4k.as_slice(),
            "checked",
            "baseline Q4_K gate+up",
        ),
        (
            "q4k_ffn_gate_up_8sg",
            &metal.ffn.q4k_ffn_gate_up_8sg_pipeline,
            gate_q4k.as_slice(),
            up_q4k.as_slice(),
            "checked",
            "8-simdgroup Q4_K gate+up candidate/default path",
        ),
        (
            "q4k_ffn_gate_up_f16acc",
            &metal.ffn.q4k_ffn_gate_up_f16acc_pipeline,
            gate_q4k.as_slice(),
            up_q4k.as_slice(),
            "checked",
            "f16 accumulator candidate",
        ),
        (
            "q4k_ffn_gate_up_coop",
            &metal.ffn.q4k_ffn_gate_up_coop_pipeline,
            gate_q4k.as_slice(),
            up_q4k.as_slice(),
            "checked",
            "cooperative scale-load candidate",
        ),
        (
            "q4kf_ffn_gate_up",
            &metal.ffn.q4kf_ffn_gate_up_pipeline,
            gate_q4kf.as_slice(),
            up_q4kf.as_slice(),
            "layout-sensitive",
            "Q4_KF/GGUF-layout gate+up; synthetic Q4_KF may not exercise every row",
        ),
    ] {
        out.push(bench_gate_up(
            metal, cfg, shape, name, kh, gate, up, n, k, sanity, note,
        ));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn bench_gate_up(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
    name: &'static str,
    kh: &KernelHandle,
    gate: &[u8],
    up: &[u8],
    n: usize,
    k: usize,
    sanity: &'static str,
    note: &'static str,
) -> BenchResult {
    let x = synth_f32(k, 0.61);
    let bufs = metal.bufs();
    let gb = bufs.get_bytes(gate);
    let ub = bufs.get_bytes(up);
    let xb = bufs.transient_from_f32(&x);
    let go = bufs.output((n * 4) as u64);
    let uo = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg) * 2;

    measure_tiled(
        metal,
        cfg,
        name,
        "ffn-gate-up",
        kh,
        format!("{} N={n} K={k}", shape.label),
        (gate.len() + up.len()) as u64,
        &go,
        n,
        sanity,
        note,
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&gb), 0);
            enc.set_buffer(1, Some(&ub), 0);
            enc.set_buffer(2, Some(&xb), 0);
            enc.set_buffer(3, Some(&go), 0);
            enc.set_buffer(4, Some(&uo), 0);
            enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

fn bench_geglu_down_family(metal: &MetalBackend, cfg: &Config, shape: Shape) -> Vec<BenchResult> {
    let n = shape.hidden;
    let k = shape.inter;
    let q4k_down = quantize_q4_k(&synth_f32(n * k, 0.71));
    let q6k_down = quantize_q6_k(&synth_f32(n * k, 0.72));
    let gate = synth_f32(k, 0.73);
    let up = synth_f32(k, 0.74);
    vec![
        bench_geglu_down(
            metal,
            cfg,
            shape,
            "q4k_geglu_silu_down",
            "ffn-down",
            &metal.ffn.q4k_geglu_silu_down_pipeline,
            &q4k_down,
            &gate,
            &up,
            "checked",
            "Q4_K fused SiLU GEGLU down",
        ),
        bench_geglu_down(
            metal,
            cfg,
            shape,
            "q4k_geglu_gelu_tanh_down",
            "ffn-down",
            &metal.ffn.q4k_geglu_gelu_tanh_down_pipeline,
            &q4k_down,
            &gate,
            &up,
            "checked",
            "Q4_K fused GELU-tanh GEGLU down",
        ),
        bench_geglu_down(
            metal,
            cfg,
            shape,
            "q6k_geglu_silu_down",
            "ffn-down",
            &metal.ffn.q6k_geglu_silu_down_pipeline,
            &q6k_down,
            &gate,
            &up,
            "checked",
            "Q6_K fused SiLU GEGLU down",
        ),
        bench_geglu_down(
            metal,
            cfg,
            shape,
            "q6k_geglu_gelu_tanh_down",
            "ffn-down",
            &metal.ffn.q6k_geglu_gelu_tanh_down_pipeline,
            &q6k_down,
            &gate,
            &up,
            "checked",
            "Q6_K fused GELU-tanh GEGLU down",
        ),
        bench_geglu_down(
            metal,
            cfg,
            shape,
            "q6k_geglu_gelu_tanh_down_cached",
            "ffn-down",
            &metal.ffn.q6k_geglu_gelu_tanh_down_cached_pipeline,
            &q6k_down,
            &gate,
            &up,
            "checked",
            "Q6_K cached-activation GELU-tanh GEGLU down",
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn bench_geglu_down(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
    name: &'static str,
    family: &'static str,
    kh: &KernelHandle,
    weights: &[u8],
    gate: &[f32],
    up: &[f32],
    sanity: &'static str,
    note: &'static str,
) -> BenchResult {
    let n = shape.hidden;
    let k = shape.inter;
    let bufs = metal.bufs();
    let wb = bufs.get_bytes(weights);
    let gb = bufs.transient_from_f32(gate);
    let ub = bufs.transient_from_f32(up);
    let ob = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        name,
        family,
        kh,
        format!("{} N={n} K={k}", shape.label),
        weights.len() as u64 + (gate.len() * 8) as u64,
        &ob,
        n,
        sanity,
        note,
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&gb), 0);
            enc.set_buffer(2, Some(&ub), 0);
            enc.set_buffer(3, Some(&ob), 0);
            enc.set_bytes(4, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

fn bench_qkv_family(metal: &MetalBackend, cfg: &Config, shape: Shape) -> Vec<BenchResult> {
    let q4_q = quantize_q4_k(&synth_f32(shape.q_rows * shape.hidden, 0.81));
    let q4_k = quantize_q4_k(&synth_f32(shape.kv_rows * shape.hidden, 0.82));
    let q4_v = quantize_q4_k(&synth_f32(shape.kv_rows * shape.hidden, 0.83));
    let q6_v = quantize_q6_k(&synth_f32(shape.kv_rows * shape.hidden, 0.84));
    let q4kf_q = quantize_q4_kf(&synth_f32(shape.q_rows * shape.hidden, 0.85));
    let q4kf_k = quantize_q4_kf(&synth_f32(shape.kv_rows * shape.hidden, 0.86));
    let q4kf_v = quantize_q4_kf(&synth_f32(shape.kv_rows * shape.hidden, 0.87));
    vec![
        bench_q4k_qkv(
            metal,
            cfg,
            shape,
            "q4k_qkv_proj",
            &metal.attention.q4k_qkv_proj_pipeline,
            &q4_q,
            &q4_k,
            &q4_v,
            "checked",
            "Q4_K fused QKV projection",
        ),
        bench_q4k_qkv(
            metal,
            cfg,
            shape,
            "q4kf_qkv_proj",
            &metal.attention.q4kf_qkv_proj_pipeline,
            &q4kf_q,
            &q4kf_k,
            &q4kf_v,
            "layout-sensitive",
            "Q4_KF/GGUF fused QKV projection; synthetic Q4_KF may not exercise every row",
        ),
        bench_q4k_q6k_qkv(
            metal,
            cfg,
            shape,
            "q4k_q6k_qkv_proj",
            &metal.attention.q4k_q6k_qkv_proj_pipeline,
            &q4_q,
            &q4_k,
            &q6_v,
            false,
            "checked",
            "mixed Q4_K Q/K + Q6_K V fused QKV projection",
        ),
        bench_q4k_q6k_qkv(
            metal,
            cfg,
            shape,
            "q4k_q6k_qkv_proj_normed",
            &metal.attention.q4k_q6k_qkv_proj_normed_pipeline,
            &q4_q,
            &q4_k,
            &q6_v,
            true,
            "checked",
            "mixed Q4_K/Q6_K fused QKV projection with RMS norm",
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn bench_q4k_qkv(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
    name: &'static str,
    kh: &KernelHandle,
    wq: &[u8],
    wk: &[u8],
    wv: &[u8],
    sanity: &'static str,
    note: &'static str,
) -> BenchResult {
    let x = synth_f32(shape.hidden, 0.91);
    let bufs = metal.bufs();
    let wqb = bufs.get_bytes(wq);
    let wkb = bufs.get_bytes(wk);
    let wvb = bufs.get_bytes(wv);
    let xb = bufs.transient_from_f32(&x);
    let qb = bufs.output((shape.q_rows * 4) as u64);
    let kb = bufs.output((shape.kv_rows * 4) as u64);
    let vb = bufs.output((shape.kv_rows * 4) as u64);
    let q_rows = shape.q_rows as u32;
    let k_rows = shape.kv_rows as u32;
    let v_rows = shape.kv_rows as u32;
    let hidden = shape.hidden as u32;
    let tgs = ((shape.q_rows + 2 * shape.kv_rows) as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        name,
        "qkv",
        kh,
        format!(
            "{} Q={} K/V={} hidden={}",
            shape.label, shape.q_rows, shape.kv_rows, shape.hidden
        ),
        (wq.len() + wk.len() + wv.len()) as u64,
        &qb,
        shape.q_rows,
        sanity,
        note,
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wqb), 0);
            enc.set_buffer(1, Some(&wkb), 0);
            enc.set_buffer(2, Some(&wvb), 0);
            enc.set_buffer(3, Some(&xb), 0);
            enc.set_buffer(4, Some(&qb), 0);
            enc.set_buffer(5, Some(&kb), 0);
            enc.set_buffer(6, Some(&vb), 0);
            enc.set_bytes(7, 4, &q_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &k_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &v_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &hidden as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn bench_q4k_q6k_qkv(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
    name: &'static str,
    kh: &KernelHandle,
    wq: &[u8],
    wk: &[u8],
    wv: &[u8],
    normed: bool,
    sanity: &'static str,
    note: &'static str,
) -> BenchResult {
    let x = synth_f32(shape.hidden, 0.92);
    let norm_w = vec![1.0f32; shape.hidden];
    let bufs = metal.bufs();
    let wqb = bufs.get_bytes(wq);
    let wkb = bufs.get_bytes(wk);
    let wvb = bufs.get_bytes(wv);
    let xb = bufs.transient_from_f32(&x);
    let nb = bufs.transient_from_f32(&norm_w);
    let qb = bufs.output((shape.q_rows * 4) as u64);
    let kb = bufs.output((shape.kv_rows * 4) as u64);
    let vb = bufs.output((shape.kv_rows * 4) as u64);
    let q_rows = shape.q_rows as u32;
    let k_rows = shape.kv_rows as u32;
    let v_rows = shape.kv_rows as u32;
    let hidden = shape.hidden as u32;
    let eps = larql_compute::RMSNORM_EPSILON_DEFAULT;
    let offset = 0.0f32;
    let tgs = ((shape.q_rows + 2 * shape.kv_rows) as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        name,
        "qkv",
        kh,
        format!(
            "{} Q={} K/V={} hidden={}",
            shape.label, shape.q_rows, shape.kv_rows, shape.hidden
        ),
        (wq.len() + wk.len() + wv.len()) as u64,
        &qb,
        shape.q_rows,
        sanity,
        note,
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wqb), 0);
            enc.set_buffer(1, Some(&wkb), 0);
            enc.set_buffer(2, Some(&wvb), 0);
            enc.set_buffer(3, Some(&xb), 0);
            if normed {
                enc.set_buffer(4, Some(&nb), 0);
                enc.set_buffer(5, Some(&qb), 0);
                enc.set_buffer(6, Some(&kb), 0);
                enc.set_buffer(7, Some(&vb), 0);
                enc.set_bytes(8, 4, &q_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(9, 4, &k_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(10, 4, &v_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(11, 4, &hidden as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(12, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(13, 4, &offset as *const f32 as *const std::ffi::c_void);
            } else {
                enc.set_buffer(4, Some(&qb), 0);
                enc.set_buffer(5, Some(&kb), 0);
                enc.set_buffer(6, Some(&vb), 0);
                enc.set_bytes(7, 4, &q_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &k_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(9, 4, &v_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(10, 4, &hidden as *const u32 as *const std::ffi::c_void);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

fn bench_f32_gemv(metal: &MetalBackend, cfg: &Config, shape: Shape) -> BenchResult {
    let n = shape.lm_rows;
    let k = shape.hidden;
    let weights = synth_f32(n * k, 1.01);
    let x = synth_f32(k, 1.02);
    let bufs = metal.bufs();
    let wb = bufs.get_f32(&weights);
    let xb = bufs.transient_from_f32(&x);
    let ob = bufs.output((n * 4) as u64);
    let kh = &metal.f32_gemv_pipeline;
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        "f32_gemv",
        "lm-head",
        kh,
        format!("{} N={n} K={k}", shape.label),
        (weights.len() * 4) as u64,
        &ob,
        n,
        "checked",
        "f32 row-per-simdgroup GEMV; Gemma3 profile caps N to avoid multi-GB synthetic allocation",
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn measure_tiled(
    metal: &MetalBackend,
    cfg: &Config,
    name: &'static str,
    family: &'static str,
    kh: &KernelHandle,
    shape: String,
    bytes_per_call: u64,
    output: &Buffer,
    output_len: usize,
    sanity: &'static str,
    note: &'static str,
    encode: impl Fn(&ComputeCommandEncoderRef),
) -> BenchResult {
    let (isolated_ms, isolated_sd_ms) = measure_isolated(metal, cfg.warmup, cfg.iters, &encode);
    let batched_ms = measure_batched(metal, cfg.warmup, cfg.iters, cfg.n_layers, &encode);
    let output = read_buffer_f32(output, output_len);
    let output_nonzero = output.iter().filter(|v| v.abs() > 1e-10).count();
    BenchResult {
        name,
        family,
        status: "bench",
        shape,
        rows_per_tg: Some(kh.rows_per_tg),
        threads_per_tg: Some(kh.threads_per_tg),
        bytes_per_call,
        isolated_ms: Some(isolated_ms),
        isolated_sd_ms: Some(isolated_sd_ms),
        batched_ms: Some(batched_ms),
        batched_gbs: Some(gbs(bytes_per_call, batched_ms)),
        output_nonzero: Some(output_nonzero),
        sanity,
        note,
    }
}

fn measure_isolated(
    metal: &MetalBackend,
    warmup: usize,
    iters: usize,
    encode: &impl Fn(&ComputeCommandEncoderRef),
) -> (f64, f64) {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode(enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        if i >= warmup {
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
    }
    (mean(&times), stddev(&times))
}

fn measure_batched(
    metal: &MetalBackend,
    warmup: usize,
    iters: usize,
    n_layers: usize,
    encode: &impl Fn(&ComputeCommandEncoderRef),
) -> f64 {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..n_layers {
            encode(enc);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        if i >= warmup {
            times.push(t.elapsed().as_secs_f64() * 1000.0 / n_layers as f64);
        }
    }
    mean(&times)
}

fn gbs(bytes: u64, ms: f64) -> f64 {
    bytes as f64 / 1e6 / ms
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn stddev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn synth_f32(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            ((seed + f * 0.013).sin() * 0.35) + ((seed * 0.3 + f * 0.007).cos() * 0.15)
        })
        .collect()
}

fn inventory() -> &'static [InventoryItem] {
    &[
        InventoryItem {
            name: "sgemm",
            family: "dense",
            status: "inventory",
            note: "flat matmul; covered by Criterion matmul bench",
        },
        InventoryItem {
            name: "sgemm_transb",
            family: "dense",
            status: "inventory",
            note: "flat transposed matmul; covered by Criterion matmul bench",
        },
        InventoryItem {
            name: "q4_matvec_v4",
            family: "q4-0-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q8_matvec",
            family: "q8-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_matvec",
            family: "q4k-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_matvec_8sg",
            family: "q4k-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_matvec_stride32",
            family: "q4k-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q6k_matvec",
            family: "q6k-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q6k_matvec_8sg",
            family: "q6k-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_ffn_gate_up",
            family: "ffn-gate-up",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_ffn_gate_up_8sg",
            family: "ffn-gate-up",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_ffn_gate_up_f16acc",
            family: "ffn-gate-up",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_ffn_gate_up_coop",
            family: "ffn-gate-up",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4kf_ffn_gate_up",
            family: "ffn-gate-up",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_geglu_silu_down",
            family: "ffn-down",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_geglu_gelu_tanh_down",
            family: "ffn-down",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q6k_geglu_silu_down",
            family: "ffn-down",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q6k_geglu_gelu_tanh_down",
            family: "ffn-down",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q6k_geglu_gelu_tanh_down_cached",
            family: "ffn-down",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_qkv_proj",
            family: "qkv",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4kf_qkv_proj",
            family: "qkv",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_q6k_qkv_proj",
            family: "qkv",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_q6k_qkv_proj_normed",
            family: "qkv",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "f32_gemv",
            family: "lm-head",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "f16_gemv",
            family: "lm-head",
            status: "inventory",
            note: "requires synthetic half buffer; not timed in first pass",
        },
        InventoryItem {
            name: "rms_norm",
            family: "norm",
            status: "inventory",
            note: "flat reduction kernel; stage diagnostics cover decode use",
        },
        InventoryItem {
            name: "residual_add",
            family: "residual",
            status: "inventory",
            note: "flat elementwise kernel",
        },
        InventoryItem {
            name: "rms_norm_q8",
            family: "norm+quant",
            status: "inventory",
            note: "flat fused kernel; shape-sensitive q8 staging",
        },
        InventoryItem {
            name: "residual_norm",
            family: "norm",
            status: "inventory",
            note: "flat fused kernel",
        },
        InventoryItem {
            name: "residual_norm_q8",
            family: "norm+quant",
            status: "inventory",
            note: "flat fused kernel",
        },
        InventoryItem {
            name: "residual_norm_store",
            family: "norm",
            status: "inventory",
            note: "flat fused kernel",
        },
        InventoryItem {
            name: "qk_norm",
            family: "norm",
            status: "inventory",
            note: "head-shaped reduction kernel",
        },
        InventoryItem {
            name: "qk_norm_qk",
            family: "norm",
            status: "inventory",
            note: "Q/K paired norm kernel",
        },
        InventoryItem {
            name: "qk_norm_rope_fused",
            family: "attention",
            status: "inventory",
            note: "complex head-shaped fused kernel",
        },
        InventoryItem {
            name: "rope_at_pos",
            family: "rope",
            status: "inventory",
            note: "flat rope kernel",
        },
        InventoryItem {
            name: "rope_at_pos_batched",
            family: "rope",
            status: "inventory",
            note: "flat rope kernel",
        },
        InventoryItem {
            name: "rope_at_pos_batched_qk",
            family: "rope",
            status: "inventory",
            note: "flat Q/K rope kernel",
        },
        InventoryItem {
            name: "kv_attention",
            family: "attention",
            status: "inventory",
            note: "cache-shaped attention kernel",
        },
        InventoryItem {
            name: "kv_cache_append",
            family: "attention",
            status: "inventory",
            note: "cache-write kernel",
        },
        InventoryItem {
            name: "kv_append_attend_fused",
            family: "attention",
            status: "inventory",
            note: "cache-shaped fused attention kernel",
        },
        InventoryItem {
            name: "attn_fused",
            family: "attention",
            status: "inventory",
            note: "experimental fused attention kernel",
        },
        InventoryItem {
            name: "fused_attention",
            family: "attention",
            status: "inventory",
            note: "prefill/attention-shaped kernel",
        },
        InventoryItem {
            name: "post_attn_residual_norm_store",
            family: "norm",
            status: "inventory",
            note: "complex fused decode-stage kernel",
        },
        InventoryItem {
            name: "post_ffn_norm_residual_add",
            family: "norm",
            status: "inventory",
            note: "complex fused decode-stage kernel",
        },
        InventoryItem {
            name: "silu",
            family: "activation",
            status: "inventory",
            note: "flat activation kernel",
        },
        InventoryItem {
            name: "gelu_tanh",
            family: "activation",
            status: "inventory",
            note: "flat activation kernel",
        },
        InventoryItem {
            name: "geglu_silu",
            family: "activation",
            status: "inventory",
            note: "flat activation kernel",
        },
        InventoryItem {
            name: "geglu_gelu_tanh",
            family: "activation",
            status: "inventory",
            note: "flat activation kernel",
        },
        InventoryItem {
            name: "quantize_q8",
            family: "quant",
            status: "inventory",
            note: "flat quantization kernel",
        },
        InventoryItem {
            name: "layer_norm",
            family: "norm",
            status: "inventory",
            note: "LayerNorm reduction kernel",
        },
        InventoryItem {
            name: "layer_norm_no_bias",
            family: "norm",
            status: "inventory",
            note: "LayerNorm reduction kernel",
        },
        InventoryItem {
            name: "v_norm",
            family: "norm",
            status: "inventory",
            note: "V-norm reduction kernel",
        },
        InventoryItem {
            name: "v_norm_batched",
            family: "norm",
            status: "inventory",
            note: "batched V-norm reduction kernel",
        },
        InventoryItem {
            name: "scale_vector",
            family: "residual",
            status: "inventory",
            note: "flat scalar multiply kernel",
        },
        InventoryItem {
            name: "q4_vecmat",
            family: "q4",
            status: "inventory",
            note: "scatter/vector-matrix helper",
        },
        InventoryItem {
            name: "q4_f32_matvec",
            family: "q4",
            status: "inventory",
            note: "transposed f32-input helper",
        },
        InventoryItem {
            name: "q4_sparse_matvec",
            family: "q4",
            status: "inventory",
            note: "experimental sparse helper",
        },
        InventoryItem {
            name: "q4k_matmul",
            family: "q4k-matmul",
            status: "inventory",
            note: "covered by targeted matmul tests; not in decode hot path",
        },
        InventoryItem {
            name: "q8_qkv_proj",
            family: "qkv",
            status: "inventory",
            note: "Q8 fused QKV projection",
        },
        InventoryItem {
            name: "q8_proj_rope",
            family: "qkv",
            status: "inventory",
            note: "Q8 projection+rope helper",
        },
        InventoryItem {
            name: "f32_argmax_partial",
            family: "lm-head",
            status: "inventory",
            note: "partial reduction helper after f32_gemv",
        },
        InventoryItem {
            name: "f32_topk_partial",
            family: "lm-head",
            status: "inventory",
            note: "partial top-k helper after f32_gemv",
        },
        InventoryItem {
            name: "causal_attention",
            family: "attention",
            status: "inventory",
            note: "causal attention kernel",
        },
        InventoryItem {
            name: "turboquant_encode",
            family: "turboquant",
            status: "inventory",
            note: "KV compression utility",
        },
        InventoryItem {
            name: "turboquant_decode",
            family: "turboquant",
            status: "inventory",
            note: "KV decompression utility",
        },
        InventoryItem {
            name: "graph_walk_knn",
            family: "graph-walk",
            status: "inventory",
            note: "KNN graph walk utility",
        },
    ]
}

fn print_inventory() {
    let total = inventory().len();
    let benched = inventory().iter().filter(|i| i.status == "bench").count();
    println!("inventory: {total} shader functions ({benched} timed by this harness)");
    println!();
}

fn inventory_results(include_benched: bool) -> Vec<BenchResult> {
    inventory()
        .iter()
        .filter(|i| include_benched || i.status != "bench")
        .map(|i| BenchResult {
            name: i.name,
            family: i.family,
            status: i.status,
            shape: String::new(),
            rows_per_tg: None,
            threads_per_tg: None,
            bytes_per_call: 0,
            isolated_ms: None,
            isolated_sd_ms: None,
            batched_ms: None,
            batched_gbs: None,
            output_nonzero: None,
            sanity: inventory_sanity(i),
            note: i.note,
        })
        .collect()
}

fn inventory_sanity(i: &InventoryItem) -> &'static str {
    match i.name {
        "q4kf_ffn_gate_up" | "q4kf_qkv_proj" => "layout-sensitive",
        _ if i.status == "bench" => "timed-mode",
        _ => "not-timed",
    }
}

fn print_inventory_rows(results: &[BenchResult]) {
    println!(
        "{:<34} {:<14} {:<10} {:<16} Note",
        "Kernel", "Family", "Status", "Sanity"
    );
    println!("{}", "-".repeat(96));
    for r in results {
        println!(
            "{:<34} {:<14} {:<10} {:<16} {}",
            r.name, r.family, r.status, r.sanity, r.note
        );
    }
}

fn print_results(results: &[BenchResult]) {
    println!(
        "{:<34} {:<14} {:>5} {:>5} {:>9} {:>9} {:>9} {:>9} {:>8} {:<16}",
        "Kernel",
        "Family",
        "rows",
        "thr",
        "iso_ms",
        "iso_sd",
        "bat_ms",
        "GB/s",
        "nonzero",
        "Sanity"
    );
    println!("{}", "-".repeat(130));
    for r in results.iter().filter(|r| r.status == "bench") {
        println!(
            "{:<34} {:<14} {:>5} {:>5} {:>9.4} {:>9.4} {:>9.4} {:>9.1} {:>8} {:<16}",
            r.name,
            r.family,
            r.rows_per_tg.unwrap_or_default(),
            r.threads_per_tg.unwrap_or_default(),
            r.isolated_ms.unwrap_or_default(),
            r.isolated_sd_ms.unwrap_or_default(),
            r.batched_ms.unwrap_or_default(),
            r.batched_gbs.unwrap_or_default(),
            r.output_nonzero.unwrap_or_default(),
            r.sanity,
        );
    }
    println!();
    println!("Use batched ms/GB/s for promotion decisions; isolated numbers include per-call command-buffer overhead.");
}

#[derive(Debug, Clone)]
struct BaselineResult {
    family: String,
    batched_ms: Option<f64>,
}

fn load_baseline(path: &PathBuf) -> Result<HashMap<String, BaselineResult>, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read compare json: {e}"))?;
    let mut out = HashMap::new();
    let mut rest = src.as_str();
    while let Some(start) = rest.find('{') {
        rest = &rest[start + 1..];
        let Some(end) = rest.find('}') else {
            break;
        };
        let obj = &rest[..end];
        rest = &rest[end + 1..];
        let Some(name) = json_field_string(obj, "name") else {
            continue;
        };
        let family = json_field_string(obj, "family").unwrap_or_default();
        let batched_ms = json_field_number(obj, "batched_ms");
        out.insert(name, BaselineResult { family, batched_ms });
    }
    if out.is_empty() {
        return Err(format!(
            "compare json `{}` did not contain shader bench results",
            path.display()
        ));
    }
    Ok(out)
}

fn print_compare(
    current: &[BenchResult],
    baseline: &HashMap<String, BaselineResult>,
    path: &Path,
    threshold_pct: f64,
) {
    println!();
    println!(
        "Comparison vs {} (batched_ms, threshold={threshold_pct:.1}%):",
        path.display()
    );
    println!(
        "{:<34} {:<14} {:>10} {:>10} {:>9} {:<10}",
        "Kernel", "Family", "base_ms", "cur_ms", "delta", "Verdict"
    );
    println!("{}", "-".repeat(94));

    let mut improved = 0usize;
    let mut flat = 0usize;
    let mut regressed = 0usize;
    let mut missing = 0usize;

    for r in current.iter().filter(|r| r.status == "bench") {
        let Some(cur_ms) = r.batched_ms else {
            continue;
        };
        let Some(base) = baseline.get(r.name) else {
            missing += 1;
            continue;
        };
        let Some(base_ms) = base.batched_ms else {
            missing += 1;
            continue;
        };
        if base_ms <= 0.0 {
            missing += 1;
            continue;
        }
        let delta = (cur_ms - base_ms) / base_ms * 100.0;
        let verdict = if delta > threshold_pct {
            regressed += 1;
            "regressed"
        } else if delta < -threshold_pct {
            improved += 1;
            "improved"
        } else {
            flat += 1;
            "flat"
        };
        let family = if base.family.is_empty() {
            r.family
        } else {
            base.family.as_str()
        };
        println!(
            "{:<34} {:<14} {:>10.4} {:>10.4} {:>8.1}% {:<10}",
            r.name, family, base_ms, cur_ms, delta, verdict
        );
    }

    println!("summary: improved={improved} flat={flat} regressed={regressed} missing={missing}");
}

fn json_field_string(obj: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\":\"");
    let start = obj.find(&pattern)? + pattern.len();
    let mut out = String::new();
    let mut escaped = false;
    for ch in obj[start..].chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn json_field_number(obj: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{key}\":");
    let start = obj.find(&pattern)? + pattern.len();
    let tail = obj[start..].trim_start();
    if tail.starts_with("null") {
        return None;
    }
    let len = tail
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E'))
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()?;
    tail[..len].parse::<f64>().ok()
}

fn to_json(results: &[BenchResult]) -> String {
    let mut s = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            s.push_str(",\n");
        }
        s.push_str("  {");
        write!(s, "\"name\":\"{}\"", json_escape(r.name)).unwrap();
        write!(s, ",\"family\":\"{}\"", json_escape(r.family)).unwrap();
        write!(s, ",\"status\":\"{}\"", json_escape(r.status)).unwrap();
        write!(s, ",\"shape\":\"{}\"", json_escape(&r.shape)).unwrap();
        write!(s, ",\"rows_per_tg\":{}", opt_u64(r.rows_per_tg)).unwrap();
        write!(s, ",\"threads_per_tg\":{}", opt_u64(r.threads_per_tg)).unwrap();
        write!(s, ",\"bytes_per_call\":{}", r.bytes_per_call).unwrap();
        write!(s, ",\"isolated_ms\":{}", opt_f64(r.isolated_ms)).unwrap();
        write!(s, ",\"isolated_sd_ms\":{}", opt_f64(r.isolated_sd_ms)).unwrap();
        write!(s, ",\"batched_ms\":{}", opt_f64(r.batched_ms)).unwrap();
        write!(s, ",\"batched_gbs\":{}", opt_f64(r.batched_gbs)).unwrap();
        write!(s, ",\"output_nonzero\":{}", opt_usize(r.output_nonzero)).unwrap();
        write!(s, ",\"sanity\":\"{}\"", json_escape(r.sanity)).unwrap();
        write!(s, ",\"note\":\"{}\"", json_escape(r.note)).unwrap();
        s.push('}');
    }
    s.push_str("\n]\n");
    s
}

fn opt_u64(v: Option<u64>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "null".into())
}

fn opt_usize(v: Option<usize>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "null".into())
}

fn opt_f64(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "null".into())
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn compare_json_parser_reads_batched_ms() {
        let path = std::env::temp_dir().join(format!(
            "larql-shader-bench-compare-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"[
  {"name":"q4k_matvec","family":"q4k-matvec","batched_ms":0.025000,"batched_gbs":147.7},
  {"name":"f16_gemv","family":"lm-head","batched_ms":null}
]"#,
        )
        .unwrap();

        let parsed = load_baseline(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let q4k = parsed.get("q4k_matvec").unwrap();
        assert_eq!(q4k.family, "q4k-matvec");
        assert_eq!(q4k.batched_ms, Some(0.025));
        assert_eq!(parsed.get("f16_gemv").unwrap().batched_ms, None);
    }

    #[test]
    fn config_default_uses_smoke_profile_with_baseline_iters() {
        let c = Config::default();
        assert_eq!(c.profile, Profile::Smoke);
        assert_eq!(c.warmup, 2);
        assert_eq!(c.iters, 8);
        assert_eq!(c.n_layers, 4);
        assert!(!c.inventory_only);
        assert!(c.json.is_none());
        assert!(c.compare.is_none());
    }

    #[test]
    fn config_from_args_parses_smoke_profile_flag() {
        let cfg = Config::from_args(&args(&["--profile", "smoke"])).unwrap();
        assert_eq!(cfg.profile, Profile::Smoke);
        assert_eq!(cfg.warmup, 2);
        assert_eq!(cfg.iters, 8);
        assert_eq!(cfg.n_layers, 4);
    }

    #[test]
    fn config_from_args_parses_gemma3_profile_flag() {
        let cfg = Config::from_args(&args(&["--profile", "gemma3"])).unwrap();
        assert_eq!(cfg.profile, Profile::Gemma3);
        assert_eq!(cfg.warmup, 5);
        assert_eq!(cfg.iters, 30);
        assert_eq!(cfg.n_layers, 34);
    }

    #[test]
    fn config_from_args_parses_numeric_flags() {
        let cfg = Config::from_args(&args(&[
            "--warmup",
            "7",
            "--iters",
            "11",
            "--layers",
            "3",
            "--threshold",
            "12.5",
        ]))
        .unwrap();
        assert_eq!(cfg.warmup, 7);
        assert_eq!(cfg.iters, 11);
        assert_eq!(cfg.n_layers, 3);
        assert!((cfg.threshold_pct - 12.5).abs() < f64::EPSILON);
    }

    #[test]
    fn config_from_args_parses_path_flags_and_inventory_only() {
        let cfg = Config::from_args(&args(&[
            "--json",
            "/tmp/x.json",
            "--compare",
            "/tmp/baseline.json",
            "--inventory-only",
        ]))
        .unwrap();
        assert_eq!(
            cfg.json.as_deref().unwrap().to_string_lossy(),
            "/tmp/x.json"
        );
        assert_eq!(
            cfg.compare.as_deref().unwrap().to_string_lossy(),
            "/tmp/baseline.json"
        );
        assert!(cfg.inventory_only);
    }

    #[test]
    fn config_from_args_unknown_arg_errors() {
        assert!(Config::from_args(&args(&["--bogus"])).is_err());
    }

    #[test]
    fn config_from_args_help_returns_usage_as_err() {
        let err = Config::from_args(&args(&["--help"])).unwrap_err();
        assert!(err.contains("diag_shader_bench"));
    }

    #[test]
    fn config_from_args_rejects_unknown_profile_value() {
        assert!(Config::from_args(&args(&["--profile", "oversized"])).is_err());
    }

    #[test]
    fn config_from_args_rejects_missing_flag_values() {
        assert!(Config::from_args(&args(&["--profile"])).is_err());
        assert!(Config::from_args(&args(&["--warmup"])).is_err());
        assert!(Config::from_args(&args(&["--iters"])).is_err());
        assert!(Config::from_args(&args(&["--layers"])).is_err());
        assert!(Config::from_args(&args(&["--json"])).is_err());
        assert!(Config::from_args(&args(&["--compare"])).is_err());
        assert!(Config::from_args(&args(&["--threshold"])).is_err());
    }

    #[test]
    fn config_from_args_rejects_zero_warmup_iters_or_layers() {
        assert!(Config::from_args(&args(&["--warmup", "0"])).is_err());
        assert!(Config::from_args(&args(&["--iters", "0"])).is_err());
        assert!(Config::from_args(&args(&["--layers", "0"])).is_err());
    }

    #[test]
    fn config_from_args_rejects_negative_threshold() {
        assert!(Config::from_args(&args(&["--threshold", "-1.0"])).is_err());
        assert!(Config::from_args(&args(&["--threshold", "nan"])).is_err());
    }

    #[test]
    fn config_from_args_rejects_non_numeric_warmup_iters_layers() {
        assert!(Config::from_args(&args(&["--warmup", "abc"])).is_err());
        assert!(Config::from_args(&args(&["--iters", "x"])).is_err());
        assert!(Config::from_args(&args(&["--layers", "y"])).is_err());
        assert!(Config::from_args(&args(&["--threshold", "z"])).is_err());
    }

    #[test]
    fn usage_string_mentions_the_example_entry_point() {
        let u = usage();
        assert!(u.contains("diag_shader_bench"));
    }

    #[test]
    fn shape_for_smoke_and_gemma3_have_distinct_dimensions() {
        let smoke = Shape::for_profile(Profile::Smoke);
        let gemma3 = Shape::for_profile(Profile::Gemma3);
        assert_eq!(smoke.label, "smoke");
        assert_eq!(gemma3.label, "gemma3-4b");
        assert!(gemma3.hidden > smoke.hidden);
        assert!(gemma3.inter > smoke.inter);
    }

    #[test]
    fn inventory_results_include_or_exclude_benched_entries() {
        let with_bench = inventory_results(true);
        let no_bench = inventory_results(false);
        assert!(!with_bench.is_empty());
        assert!(no_bench.len() < with_bench.len());
        assert!(no_bench.iter().all(|r| r.status != "bench"));
    }

    /// Smoke-run the full bench loop at the small `Smoke` profile
    /// with warmup=1 iters=1 layers=1. Drives every `bench_*` helper,
    /// `print_results`, and the `--compare` baseline pipeline so the
    /// bench-dispatch code path isn't dead from the unit-test
    /// perspective. Skips silently if no Metal device is available.
    #[test]
    fn run_smoke_profile_drives_bench_dispatchers_end_to_end() {
        if MetalBackend::new().is_none() {
            return;
        }
        let json_path = std::env::temp_dir().join(format!(
            "larql-shader-bench-smoke-json-{}.json",
            std::process::id()
        ));
        let compare_path = std::env::temp_dir().join(format!(
            "larql-shader-bench-smoke-cmp-{}.json",
            std::process::id()
        ));
        let cfg = Config {
            profile: Profile::Smoke,
            warmup: 1,
            iters: 1,
            n_layers: 1,
            json: Some(json_path.clone()),
            compare: None,
            threshold_pct: 5.0,
            inventory_only: false,
        };
        let results = run(&cfg).expect("smoke run should succeed on Metal-capable host");
        assert!(results.iter().any(|r| r.status == "bench"));
        // Use the JSON we just wrote as the compare baseline so the
        // `print_compare` branch runs end-to-end on real data.
        std::fs::copy(&json_path, &compare_path).unwrap();
        let cfg_cmp = Config {
            profile: Profile::Smoke,
            warmup: 1,
            iters: 1,
            n_layers: 1,
            json: None,
            compare: Some(compare_path.clone()),
            threshold_pct: 5.0,
            inventory_only: false,
        };
        let _ = run(&cfg_cmp).expect("compare run should succeed");
        std::fs::remove_file(&json_path).ok();
        std::fs::remove_file(&compare_path).ok();
    }

    #[test]
    fn run_with_inventory_only_writes_json_and_returns_results() {
        let path = std::env::temp_dir().join(format!(
            "larql-shader-bench-inv-{}.json",
            std::process::id()
        ));
        let cfg = Config {
            inventory_only: true,
            json: Some(path.clone()),
            ..Config::default()
        };
        let results = run(&cfg).expect("inventory-only run should succeed");
        assert!(!results.is_empty());
        let json = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(json.starts_with('['));
        assert!(json.contains("\"name\":"));
    }

    #[test]
    fn to_json_handles_optional_fields_and_escapes_quotes() {
        let r = BenchResult {
            name: "k\"k",
            family: "fam",
            status: "bench",
            shape: "x".into(),
            rows_per_tg: Some(4),
            threads_per_tg: Some(64),
            bytes_per_call: 1024,
            isolated_ms: Some(1.25),
            isolated_sd_ms: None,
            batched_ms: Some(0.5),
            batched_gbs: None,
            output_nonzero: Some(7),
            sanity: "ok",
            note: "",
        };
        let s = to_json(&[r]);
        assert!(s.contains(r#""name":"k\"k""#));
        assert!(s.contains("\"isolated_sd_ms\":null"));
        assert!(s.contains("\"batched_gbs\":null"));
        assert!(s.contains("\"isolated_ms\":1.250000"));
        assert!(s.contains("\"bytes_per_call\":1024"));
    }

    #[test]
    fn opt_helpers_round_trip_some_and_null_for_none() {
        assert_eq!(opt_u64(Some(42)), "42");
        assert_eq!(opt_u64(None), "null");
        assert_eq!(opt_usize(Some(7)), "7");
        assert_eq!(opt_usize(None), "null");
        assert_eq!(opt_f64(Some(1.5)), "1.500000");
        assert_eq!(opt_f64(None), "null");
    }

    #[test]
    fn json_escape_handles_quotes_and_backslashes() {
        assert_eq!(json_escape(r#"a"b\c"#), r#"a\"b\\c"#);
    }

    #[test]
    fn json_field_helpers_handle_missing_and_null() {
        assert_eq!(json_field_string("{\"x\":\"y\"}", "x"), Some("y".into()));
        assert_eq!(json_field_string("{}", "missing"), None);
        assert_eq!(json_field_number("{\"v\":1.25}", "v"), Some(1.25));
        assert_eq!(json_field_number("{\"v\":null}", "v"), None);
        assert_eq!(json_field_number("{}", "v"), None);
    }

    #[test]
    fn load_baseline_errors_on_empty_array() {
        let path = std::env::temp_dir().join(format!(
            "larql-shader-bench-empty-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, "[]").unwrap();
        let err = load_baseline(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("did not contain"));
    }

    #[test]
    fn load_baseline_errors_when_path_missing() {
        let p = std::env::temp_dir().join(format!(
            "larql-shader-bench-missing-{}.json",
            std::process::id()
        ));
        assert!(load_baseline(&p).is_err());
    }

    #[test]
    fn print_compare_summarises_regression_vs_baseline() {
        // Smoke test: builds a fake baseline + current and walks the
        // verdict branches (improved / flat / regressed / missing).
        let path = std::env::temp_dir().join(format!(
            "larql-shader-bench-cmp-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"[
  {"name":"a","family":"fam","batched_ms":1.000000},
  {"name":"b","family":"fam","batched_ms":1.000000},
  {"name":"c","family":"fam","batched_ms":1.000000},
  {"name":"e","family":"fam","batched_ms":null},
  {"name":"f","family":"fam","batched_ms":0.000000}
]"#,
        )
        .unwrap();
        let baseline = load_baseline(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let mk = |name: &'static str, ms: Option<f64>| BenchResult {
            name,
            family: "fam",
            status: "bench",
            shape: String::new(),
            rows_per_tg: None,
            threads_per_tg: None,
            bytes_per_call: 0,
            isolated_ms: None,
            isolated_sd_ms: None,
            batched_ms: ms,
            batched_gbs: None,
            output_nonzero: None,
            sanity: "ok",
            note: "",
        };
        let current = vec![
            mk("a", Some(0.5)), // improved
            mk("b", Some(1.0)), // flat
            mk("c", Some(1.5)), // regressed
            mk("d", Some(1.0)), // missing from baseline
            mk("e", Some(1.0)), // baseline.batched_ms = None → missing
            mk("f", Some(1.0)), // baseline.batched_ms = 0.0 → missing
            mk("g", None),      // skipped, no cur_ms
        ];
        print_compare(
            &current,
            &baseline,
            std::path::Path::new("baseline.json"),
            5.0,
        );
    }
}
