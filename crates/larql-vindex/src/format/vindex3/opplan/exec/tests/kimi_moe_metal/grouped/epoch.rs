//! Rung 5a: how much performance is waiting for a longer GPU execution
//! epoch.
//!
//! Rung 3 collapsed one block's three submissions into one and took 0.38
//! ms of wall off it. What remained was ~40% of block wall in host time —
//! one submission floor plus readback — paid once per block. Rung 4 then
//! established there is no kernel-level mechanism left worth chasing at
//! these shapes: the grouped kernel runs at ~302 GB/s, row tiling is
//! flat, and gate/up fusion is an 8-9% regression. So the only lever of
//! that size remaining is how many blocks share one command buffer.
//!
//! **This is a ceiling, not a production shape, and the difference
//! matters.** Encoding N blocks ahead requires knowing all N inputs and
//! all N routing decisions before the command buffer is built. A real
//! Kimi decoder cannot: the next layer's hidden state comes out of
//! KDA/MLA/residual/norm work that still runs on the host, and its
//! routing decision depends on that state. Every layer therefore ends in
//! a CPU round-trip today.
//!
//! What this measures is what those round-trips cost — the prize for
//! migrating enough of the layer that the encoder can keep ownership
//! across the dependency chain. It says how much is there; it does not
//! collect it.
//!
//! **Cold by construction.** Each block reads its own bank set, so the
//! sweep streams `26 x 121.5 MiB ≈ 3.2 GB` — which is also roughly what
//! one real token of Kimi's active expert weights costs across 27
//! layers. Repeating one layer's banks would have measured the cache.
//!
//! ```text
//! LARQL_KIMI_MOE_FIXTURE=/tmp/kimi_moe_fixture \
//!   cargo test -p larql-vindex --features gpu --release --lib epoch -- --nocapture
//! ```

use larql_compute_metal::trait_impl::bf16_moe_block::{
    BlockLowering, ExpertBankRef, MoeBlockCall, MoeFfnBanks,
};

use super::*;

/// Blocks a full sweep covers — Kimi Linear's layer count, so the
/// largest arm is one token's worth of MoE work.
const BLOCKS: usize = 26;

/// Blocks per command buffer, the swept variable. Every arm does the
/// same `BLOCKS` blocks over the same bytes; only the grouping differs.
const BLOCKS_PER_CB: [usize; 5] = [1, 2, 4, 8, 26];

/// Repeats of the whole sweep. Fewer than the lighter tests use: one
/// pass already streams ~3.2 GB, so the arms are long enough to average
/// themselves and short enough that the machine stays put.
const EPOCH_WARMUP: usize = 2;
const EPOCH_ITERS: usize = 5;

/// One block's own weights. Distinct per block so the sweep measures the
/// memory system rather than the cache, and so a scratch race between
/// blocks in one command buffer could not pass unnoticed.
struct BlockWeights {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    offsets: Vec<ExpertOffset>,
}

impl BlockWeights {
    fn refs<'a>(&'a self, fx: &Fixture) -> MoeFfnBanks<'a> {
        let r = |w: &'a [u8]| ExpertBankRef {
            weights: w,
            offsets: &self.offsets,
        };
        MoeFfnBanks {
            gate: r(&self.gate),
            up: r(&self.up),
            down: r(&self.down),
            hidden: fx.hidden,
            inter: fx.inter,
        }
    }

    fn bytes(&self) -> usize {
        self.gate.len() + self.up.len() + self.down.len()
    }
}

#[test]
fn report_execution_epoch_ceiling() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let templates = (
        Bank::build(&fx, Stage::Gate),
        Bank::build(&fx, Stage::Up),
        Bank::build(&fx, Stage::Down),
    );
    // Every block's weights held alive at once — ~3.2 GB. Cloning and
    // dropping same-sized banks would let the device buffer cache, which
    // keys on `(ptr, len)`, hand a later block an earlier one's buffer.
    let blocks: Vec<BlockWeights> = (0..BLOCKS)
        .map(|_| BlockWeights {
            gate: templates.0.bytes.clone(),
            up: templates.1.bytes.clone(),
            down: templates.2.bytes.clone(),
            offsets: templates.0.offsets.clone(),
        })
        .collect();
    let per_block = blocks[0].bytes() as f64;
    let total = per_block * BLOCKS as f64;

    let calls: Vec<MoeBlockCall<'_>> = blocks
        .iter()
        .map(|b| MoeBlockCall {
            banks: b.refs(&fx),
            x: &fx.x,
        })
        .collect();

    // One full sweep of BLOCKS blocks, grouped `per_cb` at a time.
    let sweep = |per_cb: usize| -> (f64, f64) {
        let t = Instant::now();
        let mut gpu = 0.0;
        for chunk in calls.chunks(per_cb) {
            let (outs, g) = metal
                .bf16_moe_ffn_blocks(chunk, BlockLowering::Separate)
                .expect("batched blocks");
            std::hint::black_box(outs);
            gpu += g;
        }
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };

    // Correctness before timing: the largest arm must agree with the
    // smallest, block for block. Batching changes when work is
    // submitted, never what it computes.
    let one_at_a_time = sweep(1);
    let (single, _) = metal
        .bf16_moe_ffn_blocks(&calls[..1], BlockLowering::Separate)
        .expect("single");
    let (whole, _) = metal
        .bf16_moe_ffn_blocks(&calls, BlockLowering::Separate)
        .expect("whole token");
    assert_eq!(whole.len(), BLOCKS);
    for (i, out) in whole.iter().enumerate() {
        assert_eq!(
            out, &single[0],
            "block {i} of a {BLOCKS}-block command buffer differs from the same \
             block alone — batching changed a value"
        );
    }
    let _ = one_at_a_time;

    for _ in 0..EPOCH_WARMUP {
        for per_cb in BLOCKS_PER_CB {
            sweep(per_cb);
        }
    }
    let mut wall: Vec<Vec<f64>> = BLOCKS_PER_CB.iter().map(|_| Vec::new()).collect();
    let mut gpu: Vec<Vec<f64>> = BLOCKS_PER_CB.iter().map(|_| Vec::new()).collect();
    // Interleaved: this machine's clock drifts, and a blocked sweep
    // would credit the last arm with it.
    for _ in 0..EPOCH_ITERS {
        for (i, per_cb) in BLOCKS_PER_CB.into_iter().enumerate() {
            let (w, g) = sweep(per_cb);
            wall[i].push(w);
            gpu[i].push(g);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let third = (wall[0].len() / 3).max(1);
    let ramp = mean(&wall[0][..third]) / mean(&wall[0][wall[0].len() - third..]);
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };

    eprintln!(
        "[epoch] {BLOCKS} blocks x {:.1} MiB = {:.2} GiB per sweep, {} slots, \
         hidden={} inter={}",
        per_block / (1024.0 * 1024.0),
        total / (1024.0 * 1024.0 * 1024.0),
        fx.experts.len(),
        fx.hidden,
        fx.inter,
    );
    let mut baseline: Option<f64> = None;
    for (i, per_cb) in BLOCKS_PER_CB.into_iter().enumerate() {
        let (w, g) = (median(&mut wall[i]), median(&mut gpu[i]));
        let (wb, gb) = (w / BLOCKS as f64, g / BLOCKS as f64);
        eprintln!(
            "[epoch]   {per_cb:>2} block/CB ({:>2} submissions)  wall {wb:.3} ms/block  \
             gpu {gb:.3} ms/block  host {:.3} ms/block  effective {:>6.1} GB/s{}",
            BLOCKS.div_ceil(per_cb),
            wb - gb,
            per_block / 1e6 / wb,
            match baseline {
                None => String::new(),
                Some(b) => format!("  [{:.2}x vs 1/CB]", b / wb),
            }
        );
        if baseline.is_none() {
            baseline = Some(wb);
        }
    }
    // Host cost per block should be `a + b/N`: a fixed cost per command
    // buffer, plus whatever each block still pays on its own (its
    // readback, mostly). Fitting it names the two rather than leaving a
    // reader to eyeball the curve — and `b` is the number that says how
    // much a longer execution epoch is worth, while `a` is the floor no
    // amount of batching removes.
    let n: Vec<f64> = BLOCKS_PER_CB.iter().map(|&c| c as f64).collect();
    let host: Vec<f64> = (0..BLOCKS_PER_CB.len())
        .map(|i| mean(&wall[i]) / BLOCKS as f64 - mean(&gpu[i]) / BLOCKS as f64)
        .collect();
    let (a, b) = least_squares_inverse(&n, &host);
    eprintln!(
        "[epoch]   host/block fits {a:.4} + {b:.3}/N ms  =>  per-command-buffer cost \
         {b:.3} ms, irreducible per-block host {a:.4} ms",
    );
    let gpu_floor = mean(&gpu[0]) / BLOCKS as f64;
    eprintln!(
        "[epoch]   ceiling if the CPU never intervened: {:.3} ms/block ({:.1} GB/s), \
         {:.2}x over 1 block/CB",
        gpu_floor + a,
        per_block / 1e6 / (gpu_floor + a),
        (mean(&wall[0]) / BLOCKS as f64) / (gpu_floor + a),
    );
    eprintln!("[epoch]   ramp {ramp:.2}x — 1.00 means the machine held still");
    assert!(ramp.is_finite());
}

/// Least squares for `y = a + b/x`, which is linear in `1/x`.
fn least_squares_inverse(x: &[f64], y: &[f64]) -> (f64, f64) {
    let inv: Vec<f64> = x.iter().map(|v| 1.0 / v).collect();
    let n = x.len() as f64;
    let (sx, sy) = (inv.iter().sum::<f64>(), y.iter().sum::<f64>());
    let sxx: f64 = inv.iter().map(|v| v * v).sum();
    let sxy: f64 = inv.iter().zip(y).map(|(a, b)| a * b).sum();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    ((sy - b * sx) / n, b)
}
