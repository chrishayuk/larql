//! The oracle is the decoder, and the comparison is exact.
//!
//! A tolerance would hide the thing this kernel exists to guarantee. Its
//! whole claim is that executing the stored blocks is the SAME
//! arithmetic as decoding them and multiplying — so any difference at
//! all is a different program, not rounding, and the assertions say
//! `assert_eq!` rather than "within epsilon".

use super::{q8_0_gemv, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
use larql_models::quant::ggml::dequantize_q8_0;

/// Deterministic weights spanning several magnitudes, so blocks get
/// different scales and a single shared scale cannot pass by luck.
fn weights(n: usize, k: usize) -> Vec<f32> {
    (0..n * k)
        .map(|i| {
            let t = (i % 37) as f32 / 37.0 - 0.5;
            t * (1.0 + (i / k) as f32 * 0.25)
        })
        .collect()
}

/// Encode to Q8_0 the way ggml defines it: per 32 elements, `d =
/// amax/127`, codes are `round(w/d)`.
fn encode_q8_0(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.len() / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES);
    for block in w.chunks(Q8_0_BLOCK_ELEMS) {
        let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let dh = half_bits(d);
        // Round-trip the scale through f16 so the codes are chosen
        // against the scale that will actually be stored.
        let ds = larql_models::quant::half::f16_to_f32(dh);
        out.extend_from_slice(&dh.to_le_bytes());
        for &v in block {
            let q = if ds == 0.0 { 0.0 } else { (v / ds).round() };
            out.push((q.clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    out
}

/// f32 -> f16 bits, round-to-nearest-even, for the normal range this
/// test exercises.
fn half_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let mut m = (mant >> 13) as u16;
    // Round to nearest even on the dropped bits.
    let dropped = mant & 0x1fff;
    if dropped > 0x1000 || (dropped == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            exp += 1;
            if exp >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((exp as u16) << 10) | m
}

fn decoded_gemv(w: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|r| {
            let mut acc = 0.0f32;
            for c in 0..k {
                acc += w[r * k + c] * x[c];
            }
            acc
        })
        .collect()
}

fn activations(k: usize) -> Vec<f32> {
    (0..k).map(|i| ((i % 11) as f32 - 5.0) / 7.0).collect()
}

/// The oracle: decoding the pack and multiplying must give exactly what
/// the kernel gives. Not "within a tolerance" — the kernel is the same
/// arithmetic in the same association, so any difference is a different
/// program.
#[test]
fn the_kernel_equals_dequantise_then_multiply_bit_for_bit() {
    // 70 rows span three parallel work units, so the schedule is
    // exercised as well as the arithmetic.
    for (n, k) in [(1, 32), (3, 64), (8, 128), (5, 256), (70, 64)] {
        let w = weights(n, k);
        let packed = encode_q8_0(&w);
        let decoded = dequantize_q8_0(&packed, n * k).expect("decode");
        let x = activations(k);
        let got = q8_0_gemv(&packed, &x, n, k).expect("kernel");
        let want = decoded_gemv(&decoded, &x, n, k);
        assert_eq!(got, want, "[{n},{k}] kernel diverged from the decoder");
    }
}

/// The kernel must be reading the stored codes, not reconstructing the
/// original floats: quantisation is lossy, so an exact match against the
/// SOURCE weights would mean this is not testing a quantised path.
#[test]
fn the_result_is_lossy_against_the_source_weights() {
    let (n, k) = (4, 128);
    let w = weights(n, k);
    let packed = encode_q8_0(&w);
    let x = activations(k);
    let got = q8_0_gemv(&packed, &x, n, k).expect("kernel");
    let exact = decoded_gemv(&w, &x, n, k);
    assert_ne!(
        got, exact,
        "kernel matched the unquantised weights exactly — it is not reading codes"
    );
    // …but it must still be close, or it is reading them wrongly.
    for (g, e) in got.iter().zip(&exact) {
        assert!(
            (g - e).abs() < 0.05 * e.abs().max(1.0),
            "kernel {g} is not a Q8_0 approximation of {e}"
        );
    }
}

/// Geometry that does not describe the pack returns `None` rather than a
/// plausible vector read across a row boundary.
#[test]
fn a_geometry_that_does_not_describe_the_pack_is_refused() {
    let (n, k) = (2, 64);
    let packed = encode_q8_0(&weights(n, k));
    let x = activations(k);
    assert!(q8_0_gemv(&packed, &x, n, k).is_some(), "the control case");

    // k off the block grid.
    assert!(q8_0_gemv(&packed, &activations(48), n, 48).is_none());
    // x the wrong length for k.
    assert!(q8_0_gemv(&packed, &activations(32), n, k).is_none());
    // a stream short by one block.
    let short = &packed[..packed.len() - Q8_0_BLOCK_BYTES];
    assert!(q8_0_gemv(short, &x, n, k).is_none());
    // more rows than the stream holds.
    assert!(q8_0_gemv(&packed, &x, n + 1, k).is_none());
    // zero k.
    assert!(q8_0_gemv(&packed, &[], n, 0).is_none());
}

/// A block whose weights are all zero has a zero scale, which must not
/// become a NaN through a division that never happens.
#[test]
fn an_all_zero_block_contributes_nothing() {
    let (n, k) = (1, 64);
    let mut w = weights(n, k);
    for v in w.iter_mut().take(Q8_0_BLOCK_ELEMS) {
        *v = 0.0;
    }
    let packed = encode_q8_0(&w);
    let x = activations(k);
    let got = q8_0_gemv(&packed, &x, n, k).expect("kernel");
    assert!(got[0].is_finite(), "a zero-scale block produced {}", got[0]);
    let decoded = dequantize_q8_0(&packed, n * k).expect("decode");
    assert_eq!(got, decoded_gemv(&decoded, &x, n, k));
}

/// One block per scale, with codes covering the full range. `half_bits` above
/// flushes subnormals, so these are written directly.
fn blocks_with_scales(scales: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(scales.len() * Q8_0_BLOCK_BYTES);
    for (b, d) in scales.iter().enumerate() {
        out.extend_from_slice(&d.to_le_bytes());
        for j in 0..Q8_0_BLOCK_ELEMS {
            let code = -127 + ((b * 13 + j * 8) % 255) as i32;
            out.push((code as i8) as u8);
        }
    }
    out
}

/// f16 subnormals, from the smallest up to and across the normal boundary, in
/// both signs. An encoder that rounds to nearest stores these for small blocks.
const SUBNORMAL_SCALES: [u16; 8] = [
    0x0001, 0x0002, 0x00e3, 0x03fc, 0x03ff, 0x0400, 0x8001, 0x83ff,
];

/// Below the normal range an f16 is `m * 2^-24`, and the smallest normal, `0x0400`,
/// continues the same formula with `m = 1024`. Each is normal in f32, so the decode
/// has an exact value that needs no f16 code to state.
#[test]
fn a_subnormal_scale_decodes_to_its_exact_value() {
    let packed = blocks_with_scales(&SUBNORMAL_SCALES);
    let decoded =
        dequantize_q8_0(&packed, SUBNORMAL_SCALES.len() * Q8_0_BLOCK_ELEMS).expect("decode");
    for (b, d) in SUBNORMAL_SCALES.iter().enumerate() {
        let magnitude = f64::from(d & 0x7fff) * 2f64.powi(-24);
        let scale = if d & 0x8000 != 0 {
            -magnitude
        } else {
            magnitude
        };
        for j in 0..Q8_0_BLOCK_ELEMS {
            let code = packed[b * Q8_0_BLOCK_BYTES + 2 + j] as i8;
            let want = (f64::from(code) * scale) as f32;
            assert_eq!(
                decoded[b * Q8_0_BLOCK_ELEMS + j].to_bits(),
                want.to_bits(),
                "scale {d:#06x} element {j}"
            );
        }
    }
}

/// The kernel on subnormal scales: exactly the decoder's arithmetic, and not the
/// zeros a flushed scale would give.
#[test]
fn the_kernel_executes_subnormal_scales_exactly() {
    let k = SUBNORMAL_SCALES.len() * Q8_0_BLOCK_ELEMS;
    let packed = blocks_with_scales(&SUBNORMAL_SCALES);
    let decoded = dequantize_q8_0(&packed, k).expect("decode");
    let x = activations(k);
    let got = q8_0_gemv(&packed, &x, 1, k).expect("kernel");
    assert_eq!(got, decoded_gemv(&decoded, &x, 1, k));
    assert_ne!(got[0], 0.0, "a flushed scale would give exactly this");
    let flushed = blocks_with_scales(&[0x0000; 8]);
    assert_eq!(
        q8_0_gemv(&flushed, &x, 1, k).expect("kernel")[0],
        0.0,
        "the control"
    );
}

/// The encoder and the kernel together, on the input this path exists for: a
/// block too small for a normal f16 scale.
#[test]
fn a_block_with_a_subnormal_scale_survives_encoding_and_execution() {
    let k = Q8_0_BLOCK_ELEMS;
    let w: Vec<f32> = (0..k)
        .map(|i| if i == 0 { 0.0077 } else { 0.001 })
        .collect();
    let packed = larql_models::quant::ggml::quantize_q8_0(&w);
    assert_eq!(
        u16::from_le_bytes([packed[0], packed[1]]) & 0x7c00,
        0,
        "a subnormal scale"
    );
    let x = vec![1.0f32; k];
    let got = q8_0_gemv(&packed, &x, 1, k).expect("kernel")[0];
    let exact: f32 = w.iter().sum();
    // Rounding moves each element by at most half a step, so the sum by at most
    // 32 half-steps. A flushed scale would give 0 and miss this by far more.
    let half_step = 0.0077 / 127.0 / 2.0;
    assert!(
        (got - exact).abs() <= 32.0 * half_step,
        "{got} against {exact}"
    );
}

// ---- A panic inside a worker must reach the caller; it must never wedge it or become a result ----
//
// The failpoint is keyed on the job's row count, so these tests cannot disturb the kernel tests running beside them.
mod worker_panics {
    use super::super::{
        q8_0_gemv, FAIL_IN_HELPER, FAIL_N, FAIL_TWO_PANICS, GEMV_GATE, HELPER_PANICKED,
        IN_EXCLUSIVE_SCENARIO,
    };
    use std::sync::atomic::Ordering;
    use std::sync::{mpsc, Mutex};
    use std::time::Duration;

    /// Rows for the injected job: past one chunk of a 32-column matrix, off the 16-row grouping, and used by no other test.
    const ROWS: usize = 4113;
    const K: usize = 32;
    /// The failpoint is process-wide state: one injected job at a time.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    /// Marks the current thread as part of the exclusive scenario for as long as it lives.
    struct InScenario;

    impl InScenario {
        fn enter() -> Self {
            IN_EXCLUSIVE_SCENARIO.with(|c| c.set(true));
            InScenario
        }
    }

    impl Drop for InScenario {
        fn drop(&mut self) {
            IN_EXCLUSIVE_SCENARIO.with(|c| c.set(false));
        }
    }

    /// `q8_0_gemv` as the scenario's own call: it bypasses the gate its owner holds.
    fn scenario_gemv(blocks: &[u8], x: &[f32]) -> Option<Vec<f32>> {
        let _in = InScenario::enter();
        q8_0_gemv(blocks, x, ROWS, K)
    }

    /// Exclusive use of the kernel: waits for other tests' gemv calls to finish and holds
    /// off new ones, so the scenario's calls never meet the fallback path.
    fn exclusive() -> std::sync::RwLockWriteGuard<'static, ()> {
        GEMV_GATE.write().unwrap_or_else(|e| e.into_inner())
    }

    fn run(in_helper: bool, swallow_panics: bool) {
        let _guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let _exclusive = exclusive();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut builder = rayon::ThreadPoolBuilder::new().num_threads(4);
            if swallow_panics {
                builder = builder.panic_handler(|_| {});
            }
            let pool = builder.build().unwrap();
            let blocks = vec![0u8; ROWS * (K / 32) * 34];
            let x = vec![1.0f32; K];
            for round in 0..25 {
                FAIL_N.store(ROWS, Ordering::SeqCst);
                FAIL_IN_HELPER.store(in_helper, Ordering::SeqCst);
                HELPER_PANICKED.store(false, Ordering::SeqCst);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    pool.install(|| scenario_gemv(&blocks, &x))
                }));
                FAIL_N.store(0, Ordering::SeqCst);
                let payload = result.expect_err(&format!(
                    "round {round}: a panicking worker must not yield a result"
                ));
                let text = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                assert!(
                    text.starts_with("injected"),
                    "round {round}: the original panic reaches the caller, got {text:?}"
                );
                let ok = pool
                    .install(|| scenario_gemv(&blocks, &x))
                    .expect("geometry is valid");
                assert!(
                    ok.iter().all(|v| *v == 0.0),
                    "round {round}: the next call is correct"
                );
            }
            tx.send(()).unwrap();
        });
        rx.recv_timeout(Duration::from_secs(60))
            .expect("a worker panic left the caller waiting, or the thread died");
    }

    #[test]
    fn a_helper_panic_reaches_the_caller_on_a_stock_pool() {
        run(true, false);
    }

    #[test]
    fn a_helper_panic_reaches_the_caller_when_the_pool_swallows_panics() {
        run(true, true);
    }

    #[test]
    fn a_caller_panic_reaches_the_caller_and_the_kernel_recovers() {
        run(false, false);
    }

    /// Two workers panic together and the second payload's destructor panics. Dropping that payload must not unwind out of
    /// the worker: it would lose the worker's finished-chunk count and leave the caller waiting forever.
    #[test]
    fn a_payload_whose_destructor_panics_does_not_strand_the_caller() {
        let _guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let _exclusive = exclusive();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .panic_handler(|_| {})
                .build()
                .unwrap();
            let blocks = vec![0u8; ROWS * (K / 32) * 34];
            let x = vec![1.0f32; K];
            FAIL_N.store(ROWS, Ordering::SeqCst);
            FAIL_IN_HELPER.store(false, Ordering::SeqCst);
            FAIL_TWO_PANICS.store(true, Ordering::SeqCst);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.install(|| scenario_gemv(&blocks, &x))
            }));
            FAIL_N.store(0, Ordering::SeqCst);
            FAIL_TWO_PANICS.store(false, Ordering::SeqCst);
            let payload = result.expect_err("a panicking worker must not yield a result");
            let text = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .unwrap_or_default();
            assert_eq!(
                text, "injected caller panic",
                "the first panic is the one re-raised"
            );
            assert!(
                pool.install(|| scenario_gemv(&blocks, &x)).is_some(),
                "the kernel still works"
            );
            tx.send(()).unwrap();
        });
        rx.recv_timeout(Duration::from_secs(20))
            .expect("the caller was left waiting for a chunk that was never counted");
    }

    /// The scenarios own the kernel: other threads calling it at the same time (as the rest of
    /// the test binary does) must neither disturb them nor be starved.
    #[test]
    fn the_scenarios_hold_while_other_threads_call_the_kernel() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let stop = Arc::new(AtomicBool::new(false));
        let noise: Vec<_> = (0..3)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let blocks = vec![0u8; 5000 * 34];
                    let x = vec![1.0f32; 32];
                    while !stop.load(Ordering::Relaxed) {
                        let _ = q8_0_gemv(&blocks, &x, 5000, 32);
                    }
                })
            })
            .collect();
        run(false, false);
        run(true, false);
        stop.store(true, Ordering::Relaxed);
        for n in noise {
            n.join().unwrap();
        }
    }
}
