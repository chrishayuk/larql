//! Rung 1 of the Kimi Metal ladder: one BF16 expert FFN, CPU against
//! Metal, on the real checkpoint's own weights.
//!
//! The Kimi CPU arc closed at ~11 tok/s with every operator proven
//! byte-identical to `modeling_kimi.py`. Porting it to Metal starts
//! here because the survey that preceded this rung found no BF16 GEMV
//! kernel anywhere in `larql-compute-metal` — every quant-matvec path
//! refused `QuantFormat::BF16` by name. `f16_gemv` is not a substitute:
//! IEEE-754 binary16 and bf16 share a width and nothing else.
//!
//! **What this gate licenses, and what it does not.** Three claims, in
//! the order the evidence has to arrive:
//!
//! 1. *Controls.* The same real bytes read through the f16 kernel must
//!    NOT reproduce the oracle — otherwise a tolerance that passes here
//!    would pass for the wrong decode too. And the gate must fail when
//!    the arms are handed different experts' weights.
//! 2. *Parity against a reference outside this system.* Each expert's
//!    Metal output is scored against `expert_output_i.f32`, which
//!    `scripts/kimi_moe_export.py` captured from the checkpoint's own
//!    `KimiBlockSparseMLP.forward`. That is what makes this more than
//!    two copies of larql agreeing with each other — the hazard
//!    [[parity gate blind to shared input]] names.
//! 3. *CPU-vs-Metal agreement*, which is the substitution claim: the
//!    two arms share the fixture bytes, the activation
//!    (`kimi_moe_block::silu`) and the composition order, and differ in
//!    the matvec alone.
//!
//! It does NOT license anything about routing, the shared-expert
//! combine, dispatch shape at MoE scale, or residency — one expert, one
//! GEMV kernel, nothing else. The timing it prints is one expert's three
//! projections in three separate command buffers, which is the
//! *unfused* shape; Granite's lowered pilot is the standing warning that
//! per-op Metal barely beats CPU and only the fused shape reaches the
//! roofline.
//!
//! Rung 1 (`single`) put one expert FFN on the GPU and found the arm
//! submission-bound. Rung 2 (`grouped`) asks the question that left
//! open: whether what remains is occupancy.
//!
//! ```text
//! python scripts/kimi_moe_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --layer 1 --out /tmp/kimi_moe_fixture
//! LARQL_KIMI_MOE_FIXTURE=/tmp/kimi_moe_fixture \
//!     cargo test -p larql-vindex --features gpu --release --lib kimi_moe_metal -- --nocapture
//! ```

mod grouped;
mod single;

use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_compute::backend::MatMul;
use larql_compute_metal::MetalBackend;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::kimi_moe_block::{expert_ffn, silu, ExpertWeights};

const FIXTURE_ENV: &str = "LARQL_KIMI_MOE_FIXTURE";

/// Scored relative to the oracle vector's own magnitude, so one ceiling
/// covers both projection shapes.
///
/// Measured, not assumed — see the run this rung reports. The two arms
/// widen the identical codes; all that separates them is that the GPU
/// sums 32 lane partials in a tree with `fma` while `FusedBf16` sums
/// serially. At K=2304 that is a handful of f32 ULP. The f16-decode
/// control below lands at rel ~1e0, four orders past this.
const REL_TOLERANCE: f32 = 1e-4;

/// Bytes per bf16 code, in the fixture file and in the device buffer.
const BF16_BYTES: usize = 2;

/// Warmup dispatches discarded before timing, then the number of timed
/// repeats reported as a minimum. Min-of-N rather than a mean because
/// the interference this machine shows is one-sided.
const BENCH_WARMUP: usize = 2;
const BENCH_ITERS: usize = 15;

/// A gemv small enough that its bytes cannot explain its cost — the
/// per-submission floor, measured through the same entry point as the
/// real ones so nothing but the shape differs.
/// Working set the cold-bandwidth control builds, in bytes.
///
/// One expert bank for one stage is ~40 MiB, which an Apple-silicon
/// system level cache can hold outright — so a repeat-the-same-bank loop
/// measures the cache. 512 MiB is past any plausible SLC and past the
/// point where the previous cycle's first bank could survive.
const COLD_WORKING_SET_BYTES: usize = 512 * 1024 * 1024;

/// Shape and repeat count of the pre-measurement GPU ramp.
///
/// Sized by measurement, not by feel: at 40 dispatches the heaviest test
/// here still reported a ramp factor of 1.24-1.31, meaning its first
/// samples were taken on a GPU a quarter slower than its last. Each
/// dispatch is ~8 MiB, so this streams ~1.6 GB before anything is timed
/// — comparable to what the block test itself moves during a run.
const RAMP_ROWS: usize = 2048;
const RAMP_COLS: usize = 2048;
const RAMP_DISPATCHES: usize = 200;

const TINY_ROWS: usize = 8;
const TINY_COLS: usize = 32;

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

/// The checkpoint's own bytes, unwidened. Both arms are fed from this
/// one read: Metal binds the bytes, the CPU reads the `u16` codes those
/// same bytes spell, so "same input" is structural rather than asserted.
fn read_bf16_bytes(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{name}.bf16"))).unwrap_or_else(|e| panic!("{name}.bf16: {e}"))
}

fn codes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(BF16_BYTES)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(
        got.len(),
        want.len(),
        "length {} vs {}",
        got.len(),
        want.len()
    );
    let max_abs = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "degenerate oracle: every element is zero");
    max_abs / scale
}

/// One expert's three projections on the GPU, with the CPU's own
/// activation and composition between them — `w2(silu(w1 x) * w3 x)`,
/// the same shape `kimi_moe_block::expert_ffn` runs.
///
/// `bf16_gemv_force` rather than the threshold-gated entry: the caller
/// has already decided this work belongs on the GPU, and a silent `None`
/// would turn a parity gate into a skip.
fn metal_expert_ffn(
    metal: &MetalBackend,
    x: &[f32],
    w: MetalExpert<'_>,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let gate = metal
        .bf16_gemv_force(w.gate, x, inter, hidden)
        .expect("gate projection dispatches");
    let up = metal
        .bf16_gemv_force(w.up, x, inter, hidden)
        .expect("up projection dispatches");
    let h: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
    metal
        .bf16_gemv_force(w.down, &h, hidden, inter)
        .expect("down projection dispatches")
}

/// The device-side mirror of [`ExpertWeights`] — the same three
/// projections in the checkpoint's own gate/up/down naming, as raw
/// bytes instead of widened codes.
#[derive(Clone, Copy)]
struct MetalExpert<'a> {
    gate: &'a [u8],
    up: &'a [u8],
    down: &'a [u8],
}

/// Everything one expert needs, read once.
struct Expert {
    id: String,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    gate_codes: Vec<u16>,
    up_codes: Vec<u16>,
    down_codes: Vec<u16>,
    /// `modeling_kimi.py`'s own output for this expert on this input.
    oracle: Vec<f32>,
}

impl Expert {
    fn load(dir: &Path, name: &str, oracle: Vec<f32>) -> Self {
        let gate = read_bf16_bytes(dir, &format!("{name}_w1"));
        let up = read_bf16_bytes(dir, &format!("{name}_w3"));
        let down = read_bf16_bytes(dir, &format!("{name}_w2"));
        Self {
            id: name.to_string(),
            gate_codes: codes(&gate),
            up_codes: codes(&up),
            down_codes: codes(&down),
            gate,
            up,
            down,
            oracle,
        }
    }

    fn cpu(&self) -> ExpertWeights<'_> {
        ExpertWeights {
            gate: &self.gate_codes,
            up: &self.up_codes,
            down: &self.down_codes,
        }
    }

    fn metal(&self) -> MetalExpert<'_> {
        MetalExpert {
            gate: &self.gate,
            up: &self.up,
            down: &self.down,
        }
    }
}

struct Fixture {
    hidden: usize,
    inter: usize,
    x: Vec<f32>,
    experts: Vec<Expert>,
}

fn load(dir: &Path) -> Fixture {
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");
    let hidden = manifest["hidden"].as_u64().unwrap() as usize;
    let inter = manifest["moe_intermediate_size"].as_u64().unwrap() as usize;
    let ids: Vec<usize> = manifest["selected_ids_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(
        (
            manifest["experts"].as_u64().unwrap(),
            manifest["top_k"].as_u64().unwrap()
        ),
        (256, 8),
        "this gate exists to run REAL geometry"
    );

    // The routed experts in the oracle's own order (`expert_output_i`
    // is `selected_ids_order[i]`), then the shared branch, which runs
    // the identical FFN shape and is scored the same way.
    let mut experts: Vec<Expert> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            Expert::load(
                dir,
                &format!("expert{id}"),
                read_f32(dir, &format!("expert_output_{i}")),
            )
        })
        .collect();
    experts.push(Expert::load(dir, "shared", read_f32(dir, "shared_output")));

    Fixture {
        hidden,
        inter,
        x: read_f32(dir, "input"),
        experts,
    }
}

/// What one dispatch cost, in the two units that answer different
/// questions.
///
/// Wall is what a caller pays. The GPU window (`GPUEndTime -
/// GPUStartTime`) is what the kernel costs — and rung 1 measured ~0.2 ms
/// of fixed submission at these shapes, comparable to a whole dispatch,
/// so a bandwidth figure taken from wall time prices the stack rather
/// than the kernel.
///
/// Both a min and a median of the GPU window, because at these
/// durations (0.1-0.3 ms) the min alone has been seen to return a window
/// short enough to imply a bandwidth above this machine's DRAM peak —
/// physically impossible, so a timer artifact. When min and median
/// disagree materially, the min is the one to distrust.
#[derive(Clone, Copy)]
struct Timing {
    wall_min_ms: f64,
    gpu_min_ms: f64,
    gpu_median_ms: f64,
}

impl Timing {
    /// How far the GPU timer's median sits above its own minimum.
    ///
    /// Reported rather than hidden because it is the artifact detector:
    /// the same dispatch measured fifteen times should not vary much, so
    /// a spread well above 1 means some windows came back implausibly
    /// short and the minimum is not a floor of anything real. A run that
    /// implied 448 GB/s — past this chip's DRAM peak — announced itself
    /// this way first.
    fn gpu_spread(self) -> f64 {
        self.gpu_median_ms / self.gpu_min_ms
    }
}

/// Run `run` (which returns its own GPU-busy ms) and summarise.
fn measure(warmup: usize, iters: usize, mut run: impl FnMut() -> f64) -> Timing {
    for _ in 0..warmup {
        run();
    }
    let mut wall = f64::INFINITY;
    let mut gpus = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let g = run();
        wall = wall.min(t.elapsed().as_secs_f64() * 1000.0);
        gpus.push(g);
    }
    gpus.sort_by(f64::total_cmp);
    Timing {
        wall_min_ms: wall,
        gpu_min_ms: gpus[0],
        gpu_median_ms: gpus[gpus.len() / 2],
    }
}

/// Two arms measured against each other, **interleaved**.
///
/// Blocked arms (`AAAA…BBBB…`) attribute drift to whichever ran second,
/// and this machine drifts hard: the same probe re-run three times in a
/// row climbed 175 → 253 → 294 GB/s as the GPU ramped its clock, so a
/// second arm measured after a first is measured on a different machine.
/// Alternating them puts both arms' samples across the whole window.
///
/// Returns each arm's median GPU-busy ms, plus a **ramp factor**: the
/// first third of arm A's samples over its last third. Far from 1 means
/// the machine moved during the measurement and the comparison is worth
/// less than it looks — reported rather than hidden, because a ratio
/// taken from a moving baseline is exactly the failure this helper
/// exists to make visible.
struct Interleaved {
    a_median_ms: f64,
    b_median_ms: f64,
    ramp: f64,
}

fn interleave(
    warmup: usize,
    iters: usize,
    mut a: impl FnMut() -> f64,
    mut b: impl FnMut() -> f64,
) -> Interleaved {
    for _ in 0..warmup {
        a();
        b();
    }
    let (mut a_s, mut b_s) = (Vec::with_capacity(iters), Vec::with_capacity(iters));
    for _ in 0..iters {
        a_s.push(a());
        b_s.push(b());
    }
    let third = (a_s.len() / 3).max(1);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let ramp = mean(&a_s[..third]) / mean(&a_s[a_s.len() - third..]);
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    Interleaved {
        a_median_ms: median(&mut a_s),
        b_median_ms: median(&mut b_s),
        ramp,
    }
}

/// The fixture and a device, or `None` with a reason printed.
///
/// Every test in this tree needs both and neither is guaranteed: the
/// fixture is ~245 MiB of real weights that lives in `/tmp`, and the
/// device is absent on any non-Metal host. Skipping loudly beats a test
/// that quietly asserts nothing.
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
            // On macOS this is NOT an absent device — it is a shader
            // library that failed to compile, and `MetalBackend::new`
            // reports both the same way. Skipping there turns a broken
            // build into a green run: a `log1p` that MSL does not have
            // once made every Metal gate in this tree "pass" by
            // skipping. Fail loudly where a device is supposed to exist.
            #[cfg(target_os = "macos")]
            panic!(
                "MetalBackend::new() returned None on macOS — the shader library \
                 almost certainly failed to compile. Run `cargo test -p \
                 larql-compute-metal --lib` to see the compiler's own message."
            );
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("skipped: no Metal device on this host");
                return None;
            }
        }
    };
    ramp_up(&metal);
    Some((metal, load(&dir)))
}

/// Bring the GPU to a steady clock before anything is measured.
///
/// Not politeness — a correctness fix for every number below it. The
/// same probe re-run in three fresh processes climbed 175 → 253 → 294
/// GB/s purely as the GPU ramped, which means the first arm measured in
/// a process is measured on a slower machine than the second. That
/// artifact once read as "the narrow projection shape sustains half the
/// bandwidth of the wide one"; interleaving the arms showed the two
/// within ~13% of each other and the difference was position in the run.
///
/// Ramping first removes the confound at the source, so a test does not
/// have to be interleaved to be trustworthy — though the ones making
/// comparisons still are, and still report their ramp factor.
fn ramp_up(metal: &MetalBackend) {
    let w = vec![0x3Cu8; RAMP_ROWS * RAMP_COLS * 2];
    let x = vec![1.0f32; RAMP_COLS];
    for _ in 0..RAMP_DISPATCHES {
        std::hint::black_box(metal.bf16_gemv_force(&w, &x, RAMP_ROWS, RAMP_COLS));
    }
}

/// Min-of-N wall time in milliseconds, after warmup.
fn min_ms(warmup: usize, iters: usize, mut run: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        run();
    }
    (0..iters)
        .map(|_| {
            let t = Instant::now();
            run();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .fold(f64::INFINITY, f64::min)
}
