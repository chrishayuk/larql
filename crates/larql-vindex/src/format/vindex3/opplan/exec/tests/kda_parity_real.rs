//! Full-width KDA parity against real Kimi Linear layer-0 weights.
//!
//! The committed fixture is 2 heads × 4 and proves the *arithmetic*. It
//! cannot prove indexing, stride, state sizing, convolution layout or
//! flatten order: a transposed head axis or a wrong `h*D + d` is invisible
//! at `D = 4` and fatal at `D = 128`. This runs the same executor, with no
//! changes, at Kimi Linear's real 32 × 128.
//!
//! Env-gated because the fixture is ~196 MiB of f32 — too large to commit,
//! and regenerable in seconds:
//!
//! ```text
//! python scripts/kda_fixture_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --layer 0 --out /tmp/kdafix
//! LARQL_KDA_FIXTURE=/tmp/kdafix cargo test -p larql-vindex --lib kda_parity_real
//! ```
//!
//! `N = 64` and `N = 65` straddle the point where the reference switches
//! from `fused_recurrent_kda` to `chunk_kda`. LARQL does not implement the
//! chunked algorithm and does not need to — but it must stay mathematically
//! equivalent across that boundary, and only a fixture that crosses it can
//! say so.

use std::path::PathBuf;

use larql_models::config::KdaGeometry;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::kda::{layer_forward, KdaState, KdaWeights, Mutation};

/// Directory written by `scripts/kda_fixture_export.py`.
const FIXTURE_ENV: &str = "LARQL_KDA_FIXTURE";

/// Positions the ladder runs at; the last two straddle the reference's
/// implementation seam.
const POSITIONS: [usize; 4] = [8, 32, 64, 65];

/// Looser than the 2e-5 the tiny fixture holds, and deliberately so: at
/// 2304 hidden and 128 head dim each value is a sum over hundreds of
/// terms, so two orderings of the same arithmetic separate further. Still
/// four orders below every control's effect.
const TOLERANCE: f32 = 3e-4;

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os(FIXTURE_ENV).map(PathBuf::from)
}

fn read_f32(dir: &std::path::Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// q/k/v/o_proj only (P4c-4 — `KdaWeights`'s four widest operands are BF16
/// code units). See `stack_real.rs`'s own `read_bf16` for why truncation
/// recovers the checkpoint's own bits exactly.
fn read_bf16(dir: &std::path::Path, name: &str) -> Vec<u16> {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn full_width_boundaries_and_state_match_the_oracle() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");
    let geometry = KdaGeometry {
        num_heads: manifest["num_heads"].as_u64().unwrap() as usize,
        head_dim: manifest["head_dim"].as_u64().unwrap() as usize,
        conv_kernel: manifest["conv_kernel"].as_u64().unwrap() as usize,
    };
    let hidden = manifest["hidden"].as_u64().unwrap() as usize;
    let eps = manifest["rms_eps"].as_f64().unwrap() as f32;
    assert_eq!(
        (geometry.num_heads, geometry.head_dim),
        (32, 128),
        "this gate exists to run REAL geometry"
    );

    let load = |n: &str| read_f32(&dir, &format!("w_{n}.f32"));
    let load_bf16 = |n: &str| read_bf16(&dir, &format!("w_{n}.bf16"));
    let (qp, kp, vp) = (
        load_bf16("q_proj"),
        load_bf16("k_proj"),
        load_bf16("v_proj"),
    );
    let (qc, kc, vc) = (load("q_conv1d"), load("k_conv1d"), load("v_conv1d"));
    let (fa, fb) = (load("f_a_proj"), load("f_b_proj"));
    let (ga, gb) = (load("g_a_proj"), load("g_b_proj"));
    let (bp, al, dt) = (load("b_proj"), load("a_log"), load("dt_bias"));
    let (on, op) = (load("o_norm"), load_bf16("o_proj"));
    let weights = KdaWeights {
        q_proj: &qp,
        k_proj: &kp,
        v_proj: &vp,
        q_conv1d: &qc,
        k_conv1d: &kc,
        v_conv1d: &vc,
        f_a_proj: &fa,
        f_b_proj: &fb,
        g_a_proj: &ga,
        g_b_proj: &gb,
        b_proj: &bp,
        a_log: &al,
        dt_bias: &dt,
        o_norm: &on,
        o_proj: &op,
        norm_eps: eps,
    };

    const BOUNDARIES: [&str; 15] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "q_conv",
        "k_conv",
        "v_conv",
        "q_norm",
        "k_norm",
        "f_lowrank",
        "g_decay",
        "beta",
        "recurrent_out",
        "o_gate",
        "o_norm",
        "output",
    ];
    for n in POSITIONS {
        let x = read_f32(&dir, &format!("n{n}_input.f32"));
        let mut state = KdaState::zeros(geometry);
        let planes = layer_forward(&x, hidden, weights, geometry, &mut state, Mutation::None);
        let got = |name: &str| -> &Vec<f32> {
            match name {
                "q_proj" => &planes.q_proj,
                "k_proj" => &planes.k_proj,
                "v_proj" => &planes.v_proj,
                "q_conv" => &planes.q_conv,
                "k_conv" => &planes.k_conv,
                "v_conv" => &planes.v_conv,
                "q_norm" => &planes.q_norm,
                "k_norm" => &planes.k_norm,
                "f_lowrank" => &planes.f_lowrank,
                "g_decay" => &planes.g_decay,
                "beta" => &planes.beta,
                "recurrent_out" => &planes.recurrent_out,
                "o_gate" => &planes.o_gate,
                "o_norm" => &planes.o_norm,
                "output" => &planes.output,
                other => panic!("unknown boundary `{other}`"),
            }
        };
        for name in BOUNDARIES {
            let want = read_f32(&dir, &format!("n{n}_{name}.f32"));
            let d = max_abs_diff(got(name), &want);
            assert!(d < TOLERANCE, "N={n} boundary `{name}`: max|Δ| {d:e}");
        }
        // The state is the gate. An implementation can match every token
        // and still carry the wrong thing into the next call.
        let d = max_abs_diff(
            &state.recurrent,
            &read_f32(&dir, &format!("n{n}_state.f32")),
        );
        assert!(d < TOLERANCE, "N={n} recurrent state: max|Δ| {d:e}");
        // And the convolution windows, which are the other half of what a
        // continuation resumes from.
        for (i, window) in state.conv.iter().enumerate() {
            let want = read_f32(&dir, &format!("n{n}_conv{i}.f32"));
            let d = max_abs_diff(window, &want);
            assert!(d < TOLERANCE, "N={n} conv window {i}: max|Δ| {d:e}");
        }
        eprintln!("  N={n:>3}: 15 boundaries + recurrent + 3 conv windows match");
    }
}
