//! Rung 6a: one complete MLA attention operation on device, with the
//! compressed-latent cache resident across positions.
//!
//! The real trajectory measured the GPU idling ~24 ms a token across
//! seven CPU MLA layers — more than the 19.2 ms of GPU work it was
//! waiting between, and it is why a 2.3x compute win came out flat.
//! MLA is therefore not a compute port: it is the operator that lets a
//! token stay GPU-owned across a KDA→MLA→KDA boundary.
//!
//! **The cache is what makes this different from KDA.** KDA's state is a
//! fixed `[H][D][D]` matrix; MLA's grows by one raw compressed latent a
//! position, and every step re-derives every prior position's
//! `k_nope`/`v` from it. So the gate checks the cache as carefully as
//! the output: it must grow `1, 2, 3, …` exactly, hold what the CPU's
//! holds, and never come back to the host in the production path.
//!
//! Every boundary `exec::mla::MlaTrace` names is compared at every
//! position, against the proven CPU operator on the checkpoint's own
//! layer-3 weights.
//!
//! ```text
//! LARQL_KIMI_MLA_LAYER_FIXTURE=/tmp/kimi_mla_layer_fixture \
//!   cargo test -p larql-vindex --features gpu --release --lib mla_metal -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_compute_metal::trait_impl::kimi_layer::ExpertEncoding;
use larql_compute_metal::trait_impl::mla::{MlaDeviceState, MlaDeviceWeights, MlaShape};
use larql_compute_metal::MetalBackend;
use larql_models::config::MlaGeometry;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::mla::{mla_forward, MlaState, MlaWeights, Mutation};

const FIXTURE_ENV: &str = "LARQL_KIMI_MLA_LAYER_FIXTURE";
/// The same ceiling every real-weight Kimi gate uses.
const TOLERANCE: f32 = 3e-4;
const WARMUP: usize = 8;
const ITERS: usize = 10;

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os(FIXTURE_ENV).map(PathBuf::from)
}

fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(format!("{name}.f32")))
        .unwrap_or_else(|e| panic!("{name}.f32: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// f32 values as the bf16 codes the checkpoint actually stores.
///
/// The fixture writes MLA's matrices f32 because the host operator
/// consumes f32, but they are a LOSSLESS upcast of the checkpoint's own
/// bf16 — so truncating back recovers exactly those bits, and the device
/// path computes on the same values the host does. The same argument
/// P4c-4 made for KDA's q/k/v/o.
fn to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// The f32 values those codes denote — what the HOST arm must be given,
/// so the two arms differ in where they compute and not in what.
fn widened(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|f| f32::from_bits(f.to_bits() & 0xFFFF_0000))
        .collect()
}

struct Fixture {
    hidden: usize,
    geometry: MlaGeometry,
    eps: f64,
    inputs: Vec<Vec<f32>>,
    /// Host-side, already widened to the bf16 grid.
    q_proj: Vec<f32>,
    kv_a_proj: Vec<f32>,
    kv_b_proj: Vec<f32>,
    o_proj: Vec<f32>,
    kv_a_norm: Vec<f32>,
    /// The same matrices as device bf16 codes.
    q_bytes: Vec<u8>,
    kv_a_bytes: Vec<u8>,
    kv_b_bytes: Vec<u8>,
    o_bytes: Vec<u8>,
}

impl Fixture {
    fn host(&self) -> MlaWeights<'_> {
        MlaWeights {
            q_proj: WeightRows::F32(&self.q_proj),
            kv_a_proj: WeightRows::F32(&self.kv_a_proj),
            kv_a_norm: &self.kv_a_norm,
            kv_b_proj: WeightRows::F32(&self.kv_b_proj),
            o_proj: WeightRows::F32(&self.o_proj),
            kv_a_norm_eps: self.eps,
        }
    }

    fn device(&self) -> MlaDeviceWeights<'_> {
        MlaDeviceWeights {
            q_proj: &self.q_bytes,
            kv_a_proj: &self.kv_a_bytes,
            kv_a_norm: &self.kv_a_norm,
            kv_b_proj: &self.kv_b_bytes,
            o_proj: &self.o_bytes,
            kv_a_norm_eps: self.eps as f32,
            projection_encoding: ExpertEncoding::Bf16,
        }
    }

    fn shape(&self) -> MlaShape {
        MlaShape {
            hidden: self.hidden,
            num_heads: self.geometry.num_heads,
            kv_lora_rank: self.geometry.kv_lora_rank,
            qk_nope_head_dim: self.geometry.qk_nope_head_dim,
            qk_rope_head_dim: self.geometry.qk_rope_head_dim,
            v_head_dim: self.geometry.v_head_dim,
        }
    }
}

fn load(dir: &Path) -> Fixture {
    let m: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("m"))
        .expect("manifest parses");
    let g = |k: &str| m[k].as_u64().unwrap() as usize;
    let positions = g("positions");
    let mla = |n: &str| read_f32(dir, &format!("mla_{n}"));
    let (q, ka, kb, o) = (
        mla("q_proj"),
        mla("kv_a_proj"),
        mla("kv_b_proj"),
        mla("o_proj"),
    );
    Fixture {
        hidden: g("hidden"),
        geometry: MlaGeometry {
            num_heads: g("num_heads"),
            kv_lora_rank: g("kv_lora_rank"),
            qk_nope_head_dim: g("qk_nope_head_dim"),
            qk_rope_head_dim: g("qk_rope_head_dim"),
            v_head_dim: g("v_head_dim"),
        },
        eps: m["kv_a_norm_eps"].as_f64().unwrap(),
        inputs: (0..positions)
            .map(|p| {
                // The MLA operator reads the layer's NORMED hidden state,
                // which the fixture already exports per position.
                read_f32(dir, &format!("out_input_normed_{p}"))
            })
            .collect(),
        q_bytes: to_bf16_bytes(&q),
        kv_a_bytes: to_bf16_bytes(&ka),
        kv_b_bytes: to_bf16_bytes(&kb),
        o_bytes: to_bf16_bytes(&o),
        q_proj: widened(&q),
        kv_a_proj: widened(&ka),
        kv_b_proj: widened(&kb),
        o_proj: widened(&o),
        kv_a_norm: read_f32(dir, "mla_kv_a_norm"),
    }
}

fn setup() -> Option<(MetalBackend, Fixture)> {
    let dir = match fixture_dir() {
        Some(d) => d,
        None => {
            eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
            return None;
        }
    };
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => {
            #[cfg(target_os = "macos")]
            panic!(
                "MetalBackend::new() returned None on macOS — the shader library almost \
                 certainly failed to compile. Run `cargo test -p larql-compute-metal --lib`."
            );
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("skipped: no Metal device on this host");
                return None;
            }
        }
    };
    Some((metal, load(&dir)))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// **R6a's gate.** Every boundary, every position, plus the cache.
#[test]
fn device_mla_matches_the_cpu_operator_across_positions() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let mut host_state = MlaState::default();
    let device_state = MlaDeviceState::with_capacity(&metal, fx.shape(), 64);

    for (p, x) in fx.inputs.iter().enumerate() {
        let want = mla_forward(
            x,
            fx.hidden,
            fx.host(),
            fx.geometry,
            &mut host_state,
            Mutation::None,
        );
        let got = metal
            .mla_attention_step_traced(fx.device(), fx.shape(), &device_state, x)
            .expect("device mla step");

        for (name, a, b) in [
            ("q_proj", &got.q_proj, &want.q_proj),
            ("compressed_kv", &got.compressed_kv, &want.compressed_kv),
            ("kv_a_normed", &got.kv_a_normed, &want.kv_a_normed),
            ("kv_b", &got.kv_b, &want.kv_b),
            ("attn_weights", &got.attn_weights, &want.attn_weights),
            ("attn_value", &got.attn_value, &want.attn_value),
            ("output", &got.output, &want.output),
        ] {
            let d = max_abs(a, b);
            eprintln!("[r6a] pos {p} {name:>14}: max|Δ| {d:e}");
            assert!(d < TOLERANCE, "pos {p} {name}: max|Δ| {d:e}");
        }

        // The cache: it must GROW by exactly one and hold what the CPU's
        // holds. A device path that recomputed from scratch, or appended
        // twice, diverges here and nowhere else.
        assert_eq!(
            device_state.len(),
            p + 1,
            "the cache must grow by exactly one position a step"
        );
        let cached = device_state.read_back();
        assert_eq!(cached.len(), host_state.len());
        for (i, (a, b)) in cached.iter().zip(host_state.rows()).enumerate() {
            let d = max_abs(a, b);
            assert!(d < TOLERANCE, "pos {p}, cache entry {i}: max|Δ| {d:e}");
        }
    }
    eprintln!(
        "[r6a] {} positions, cache resident and growing 1..{}, every boundary within {TOLERANCE:e}",
        fx.inputs.len(),
        fx.inputs.len()
    );
}

/// **Control.** A reset cache must not reproduce an advanced one.
///
/// Without it the gate above would pass for a device path that ignored
/// its cache and attended only to the current position — which at
/// position 0 is indistinguishable from correct.
#[test]
fn a_reset_cache_changes_the_answer() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    assert!(fx.inputs.len() >= 2, "this control needs two positions");
    let state = MlaDeviceState::with_capacity(&metal, fx.shape(), 64);
    let (first, _) = metal
        .mla_attention_step(fx.device(), fx.shape(), &state, &fx.inputs[0])
        .expect("pos 0");
    let (second, _) = metal
        .mla_attention_step(fx.device(), fx.shape(), &state, &fx.inputs[1])
        .expect("pos 1, attending to both");
    state.reset();
    let (alone, _) = metal
        .mla_attention_step(fx.device(), fx.shape(), &state, &fx.inputs[1])
        .expect("pos 1 alone");

    let d = max_abs(&second, &alone);
    assert!(
        d > TOLERANCE,
        "attending to two positions moved the output by only {d:e} — the cache is \
         not being read"
    );
    assert_eq!(state.len(), 1, "a reset cache restarts at one position");
    eprintln!("[r6a] control: the cached position moves the answer by {d:e}");
    let _ = first;
}

/// A sequence longer than the cache was built for is refused, not
/// silently truncated — a Metal buffer cannot grow.
#[test]
fn overrunning_the_cache_is_refused() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let state = MlaDeviceState::with_capacity(&metal, fx.shape(), 1);
    assert!(metal
        .mla_attention_step(fx.device(), fx.shape(), &state, &fx.inputs[0])
        .is_ok());
    assert!(
        metal
            .mla_attention_step(fx.device(), fx.shape(), &state, &fx.inputs[0])
            .is_err(),
        "a cache at capacity must refuse the next position"
    );
}

/// What one MLA step costs on each side, at the fixture's position
/// count. Reported, not asserted.
#[test]
fn report_mla_step_cost() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let x = &fx.inputs[fx.inputs.len() - 1];
    let host = || {
        let mut st = MlaState::default();
        for prior in &fx.inputs {
            let _ = mla_forward(
                prior,
                fx.hidden,
                fx.host(),
                fx.geometry,
                &mut st,
                Mutation::None,
            );
        }
        let t = Instant::now();
        std::hint::black_box(mla_forward(
            x,
            fx.hidden,
            fx.host(),
            fx.geometry,
            &mut st,
            Mutation::None,
        ));
        t.elapsed().as_secs_f64() * 1000.0
    };
    let state = MlaDeviceState::with_capacity(&metal, fx.shape(), 64);
    let device = || {
        state.reset();
        for prior in &fx.inputs {
            let _ = metal.mla_attention_step(fx.device(), fx.shape(), &state, prior);
        }
        let t = Instant::now();
        let (out, gpu) = metal
            .mla_attention_step(fx.device(), fx.shape(), &state, x)
            .expect("device step");
        std::hint::black_box(out);
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };

    for _ in 0..WARMUP {
        host();
        device();
    }
    let (mut h, mut dw, mut dg) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ITERS {
        h.push(host());
        let (w, g) = device();
        dw.push(w);
        dg.push(g);
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (hm, dm, gm) = (median(&mut h), median(&mut dw), median(&mut dg));
    eprintln!(
        "[r6a] one MLA step at {} cached positions: host {hm:.3} ms, device {dm:.3} ms \
         (gpu-busy {gm:.3}, host {:.3} over 1 crossing)  [{:.2}x]",
        fx.inputs.len() + 1,
        dm - gm,
        hm / dm,
    );
    assert!(hm.is_finite() && dm.is_finite());
}
