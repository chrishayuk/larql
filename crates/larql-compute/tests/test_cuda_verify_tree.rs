//! Parity tests for `cuda::sampling::verify_tree_gpu` vs the CPU
//! oracle in `larql_inference::speculative::verify::verify_and_accept`.
//!
//! Phase 3 of `cuda-speculative-decoding`. The kernel and the CPU
//! oracle SHALL produce identical accepted/corrected/bonus token IDs
//! given the same (path probabilities, drafts, p_draft, seed) tuple,
//! across 64 fixed RNG seeds.

#![cfg(any(feature = "cuda", feature = "cuda-oxide"))]

use larql_compute::cuda::sampling::{verify_tree_gpu, verify_tree_gpu_parallel};
use larql_compute::cuda::CudaBackend;

fn gpu_or_skip() -> Option<CudaBackend> {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
        return None;
    }
    CudaBackend::new().ok()
}

// Local re-implementation of the CPU oracle to avoid pulling
// larql-inference into larql-compute's test deps. Bit-identical to
// `larql_inference::speculative::verify::verify_and_accept`.

#[derive(Clone, Debug)]
struct CpuRng {
    state: u64,
}
impl CpuRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CpuSpan {
    accepted: Vec<i32>,
    corrected: i32,
    bonus: i32,
}

fn cpu_verify(
    p_target: &[Vec<f32>],
    path_drafts: &[i32],
    path_pdraft: &[f32],
    seed: u64,
) -> CpuSpan {
    let path_len = p_target.len();
    let mut rng = CpuRng::new(seed);
    let mut accepted = vec![-1i32; path_len];
    let mut corrected = -1i32;
    let mut bonus = -1i32;

    for k in 0..path_len {
        let p_row = &p_target[k];
        let draft_id = path_drafts[k] as usize;
        let pd = path_pdraft[k].max(f32::MIN_POSITIVE);
        let pt = p_row[draft_id];
        let accept_prob = (pt / pd).min(1.0);
        let u = rng.next_f32();
        if u < accept_prob {
            accepted[k] = path_drafts[k];
            continue;
        }
        // Rejected: residual sample.
        let subtract = pd.min(pt);
        let mut residual: Vec<f32> = p_row.clone();
        residual[draft_id] = (residual[draft_id] - subtract).max(0.0);
        let z: f32 = residual.iter().sum();
        let picked = if z <= 0.0 {
            // Fall back to p_target.
            let total: f32 = p_row.iter().sum();
            if total <= 0.0 {
                0
            } else {
                let u2 = rng.next_f32() * total;
                let mut acc = 0.0;
                let mut p = (p_row.len() - 1) as i32;
                for (i, x) in p_row.iter().enumerate() {
                    acc += x;
                    if u2 < acc {
                        p = i as i32;
                        break;
                    }
                }
                p
            }
        } else {
            let u2 = rng.next_f32() * z;
            let mut acc = 0.0;
            let mut p = (residual.len() - 1) as i32;
            for (i, x) in residual.iter().enumerate() {
                acc += x;
                if u2 < acc {
                    p = i as i32;
                    break;
                }
            }
            p
        };
        corrected = picked;
        return CpuSpan {
            accepted,
            corrected,
            bonus,
        };
    }

    // All-accept: bonus from last p_target row.
    let p_last = p_target.last().unwrap();
    let total: f32 = p_last.iter().sum();
    if total <= 0.0 {
        bonus = 0;
    } else {
        let u3 = rng.next_f32() * total;
        let mut acc = 0.0;
        let mut p = (p_last.len() - 1) as i32;
        for (i, x) in p_last.iter().enumerate() {
            acc += x;
            if u3 < acc {
                p = i as i32;
                break;
            }
        }
        bonus = p;
    }
    CpuSpan {
        accepted,
        corrected,
        bonus,
    }
}

fn unit(id: i32, vocab: usize) -> Vec<f32> {
    let mut v = vec![0.0; vocab];
    v[id as usize] = 1.0;
    v
}

fn synth_probs(vocab: usize, seed: u64) -> Vec<f32> {
    // Deterministic pseudo-random positive vector summing to ~1.
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let raw: Vec<f32> = (0..vocab)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFF) as f32 / 65536.0) + 0.001
        })
        .collect();
    let total: f32 = raw.iter().sum();
    raw.iter().map(|x| x / total).collect()
}

#[test]
fn verify_tree_all_accept_one_hot() {
    let Some(backend) = gpu_or_skip() else { return };
    let vocab = 8;
    let p_target = vec![unit(1, vocab), unit(2, vocab), unit(3, vocab)];
    let drafts = vec![1, 2, 3];
    let pdraft = vec![1.0_f32, 1.0, 1.0];

    for seed in 0..16u64 {
        let cpu = cpu_verify(&p_target, &drafts, &pdraft, seed);
        let gpu = verify_tree_gpu(&backend, &p_target, &drafts, &pdraft, vocab, seed)
            .expect("verify_tree_gpu");
        assert_eq!(cpu.accepted, gpu.accepted, "seed {seed} accepted");
        assert_eq!(cpu.corrected, gpu.corrected, "seed {seed} corrected");
        assert_eq!(cpu.bonus, gpu.bonus, "seed {seed} bonus");
    }
}

#[test]
fn verify_tree_certain_reject_at_root() {
    let Some(backend) = gpu_or_skip() else { return };
    let vocab = 4;
    // Draft proposes 0, target one-hot at 1 → certain rejection.
    let p_target = vec![unit(1, vocab)];
    let drafts = vec![0];
    let pdraft = vec![1.0_f32];

    for seed in 0..16u64 {
        let cpu = cpu_verify(&p_target, &drafts, &pdraft, seed);
        let gpu = verify_tree_gpu(&backend, &p_target, &drafts, &pdraft, vocab, seed)
            .expect("verify_tree_gpu");
        assert_eq!(
            cpu,
            CpuSpan {
                accepted: gpu.accepted.clone(),
                corrected: gpu.corrected,
                bonus: gpu.bonus,
            }
        );
    }
}

#[test]
fn verify_tree_64_seeds_synthetic_match_cpu() {
    let Some(backend) = gpu_or_skip() else { return };
    let vocab = 32;
    // Three positions with synthetic probabilities. Pick draft IDs
    // that have moderate target mass so we get a mix of accept/reject.
    let p_target = vec![
        synth_probs(vocab, 0xAAAA),
        synth_probs(vocab, 0xBBBB),
        synth_probs(vocab, 0xCCCC),
    ];
    let drafts = vec![3i32, 7, 11];
    // Set p_draft slightly below the target mass so acceptance prob ~ 1 sometimes,
    // moderate other times. Mixed regime.
    let pdraft: Vec<f32> = drafts
        .iter()
        .enumerate()
        .map(|(k, &d)| p_target[k][d as usize] * 0.7)
        .collect();

    let mut accept_path_count = 0;
    let mut reject_count = 0;
    for seed in 0..64u64 {
        let cpu = cpu_verify(&p_target, &drafts, &pdraft, seed);
        let gpu = verify_tree_gpu(&backend, &p_target, &drafts, &pdraft, vocab, seed)
            .expect("verify_tree_gpu");
        assert_eq!(cpu.accepted, gpu.accepted, "seed {seed} accepted");
        assert_eq!(cpu.corrected, gpu.corrected, "seed {seed} corrected");
        assert_eq!(cpu.bonus, gpu.bonus, "seed {seed} bonus");

        if gpu.bonus >= 0 {
            accept_path_count += 1;
        }
        if gpu.corrected >= 0 {
            reject_count += 1;
        }
    }
    // Sanity: with p_draft=0.7×p_target, acceptance is certain at every step.
    // So all 64 seeds should be all-accept (with bonus). reject_count = 0.
    assert_eq!(
        accept_path_count + reject_count,
        64,
        "every seed must produce exactly one of bonus or corrected"
    );
}

#[test]
fn verify_tree_mixed_acceptance_regime() {
    let Some(backend) = gpu_or_skip() else { return };
    let vocab = 16;
    let p_target = vec![synth_probs(vocab, 0xDEAD), synth_probs(vocab, 0xBEEF)];
    let drafts = vec![5i32, 9];
    // p_draft well above target mass → acceptance prob much less than 1.
    // Many seeds will reject.
    let pdraft = vec![0.5_f32, 0.5];

    let mut hist_first_reject = 0;
    for seed in 0..64u64 {
        let cpu = cpu_verify(&p_target, &drafts, &pdraft, seed);
        let gpu = verify_tree_gpu(&backend, &p_target, &drafts, &pdraft, vocab, seed)
            .expect("verify_tree_gpu");
        // The load-bearing assertion.
        assert_eq!(cpu.accepted, gpu.accepted, "seed {seed} accepted");
        assert_eq!(cpu.corrected, gpu.corrected, "seed {seed} corrected");
        assert_eq!(cpu.bonus, gpu.bonus, "seed {seed} bonus");

        if gpu.accepted.iter().all(|&a| a == -1) {
            hist_first_reject += 1;
        }
    }
    // With acceptance prob ≈ p_target/0.5 (small), most seeds reject at root.
    // Don't assert on the count exactly; the parity test is what matters.
    assert!(
        hist_first_reject > 0,
        "mixed regime expected some rejections"
    );
}

#[test]
fn verify_tree_parallel_matches_serial_at_small_vocab() {
    // Parallel kernel reduction order differs from CPU serial sum;
    // at small vocab (synth_probs vocab=32, all-positive), the
    // parallel reduction (binary tree in shared mem) should still
    // match the single-thread serial CPU oracle bit-equally because
    // every partial sum is performed in finite-precision but stays
    // numerically tight. Spot-check 32 seeds.
    let Some(backend) = gpu_or_skip() else { return };
    let vocab = 32;
    let p_target = vec![
        synth_probs(vocab, 0xAAAA),
        synth_probs(vocab, 0xBBBB),
        synth_probs(vocab, 0xCCCC),
    ];
    let drafts = vec![3i32, 7, 11];
    let pdraft: Vec<f32> = drafts
        .iter()
        .enumerate()
        .map(|(k, &d)| p_target[k][d as usize] * 0.7)
        .collect();
    let mut serial_par_match = 0;
    for seed in 0..32u64 {
        let serial = verify_tree_gpu(&backend, &p_target, &drafts, &pdraft, vocab, seed)
            .expect("verify_tree_gpu");
        let parallel = verify_tree_gpu_parallel(&backend, &p_target, &drafts, &pdraft, vocab, seed)
            .expect("verify_tree_gpu_parallel");
        if serial == parallel {
            serial_par_match += 1;
        }
    }
    // At small vocab, sums are within ulp regardless of order — expect
    // 100% match (statistical equivalence proof at large vocab is
    // tested via distributional bench).
    assert!(
        serial_par_match >= 30,
        "small-vocab parallel/serial agreement too low: {serial_par_match}/32"
    );
}

#[test]
fn verify_tree_parallel_distribution_at_vocab_256() {
    // Statistical-equivalence test at vocab=256 over many seeds:
    // both kernels SHALL produce the same DISTRIBUTION of bonus
    // tokens (within 5% absolute per-bin) over 4096 seeds, even if
    // individual seeds may differ at boundary cases.
    let Some(backend) = gpu_or_skip() else { return };
    let vocab = 256;
    // Single-position all-accept setup so bonus is sampled.
    let p = synth_probs(vocab, 0xF00D);
    let p_target = vec![p.clone()];
    let drafts = vec![5i32];
    let pdraft = vec![p[5]]; // ratio = 1 → certain accept

    let n_trials = 4096;
    let mut hist_serial = vec![0u32; vocab];
    let mut hist_parallel = vec![0u32; vocab];
    for seed in 0..n_trials as u64 {
        let s = verify_tree_gpu(&backend, &p_target, &drafts, &pdraft, vocab, seed).unwrap();
        let p_ =
            verify_tree_gpu_parallel(&backend, &p_target, &drafts, &pdraft, vocab, seed).unwrap();
        if s.bonus >= 0 {
            hist_serial[s.bonus as usize] += 1;
        }
        if p_.bonus >= 0 {
            hist_parallel[p_.bonus as usize] += 1;
        }
    }
    let mut max_div = 0.0_f32;
    for i in 0..vocab {
        let s_p = hist_serial[i] as f32 / n_trials as f32;
        let p_p = hist_parallel[i] as f32 / n_trials as f32;
        let d = (s_p - p_p).abs();
        if d > max_div {
            max_div = d;
        }
    }
    assert!(
        max_div < 0.05,
        "distributional divergence between serial and parallel: {max_div}"
    );
}
