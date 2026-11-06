//! `FfnGramAccumulator`: the f64 streaming `KᵀK / N` behind
//! `estimate_ffn_covariance`, exercised on synthetic activations — no model.

use larql_inference::forward::ffn_covariance::FfnGramAccumulator;
use ndarray::Array2;

/// Activation magnitude reported on Gemma 4 26B-A4B, where the f32
/// accumulator lost enough precision to break the MEMIT Cholesky.
const MAGNITUDE: f32 = 1.0e3;
/// Enough token positions that each Gram entry grows to ~1e6 · N.
const ROWS: usize = 1 << 16;
const FFN_DIM: usize = 4;

/// Deterministic `(rows, FFN_DIM)` activations: a large shared offset plus a
/// small per-column variation, so the off-diagonal entries are dominated by
/// the offset and the low-order terms are what an f32 sum drops.
fn activations(rows: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, FFN_DIM), |(r, c)| {
        MAGNITUDE + ((r * 7 + c * 13) % 97) as f32 * 0.01
    })
}

/// Exact `KᵀK / N` in f64, the reference both accumulators are judged by.
fn reference(k: &Array2<f32>) -> Array2<f64> {
    let k64 = k.mapv(|v| v as f64);
    k64.t().dot(&k64) / k.nrows() as f64
}

/// The pre-fix accumulator: `KᵀK` summed in f32.
fn naive_f32(k: &Array2<f32>) -> Array2<f64> {
    let mut ktk = Array2::<f32>::zeros((FFN_DIM, FFN_DIM));
    for row in k.rows() {
        for i in 0..FFN_DIM {
            for j in 0..FFN_DIM {
                ktk[[i, j]] += row[i] * row[j];
            }
        }
    }
    ktk.mapv(|v| v as f64 / k.nrows() as f64)
}

fn max_relative_error(got: &Array2<f64>, want: &Array2<f64>) -> f64 {
    got.iter()
        .zip(want.iter())
        .map(|(g, w)| ((g - w) / w).abs())
        .fold(0.0, f64::max)
}

/// f32 output precision: the result is narrowed once, so a correct
/// accumulator is within a few f32 ulps of the f64 reference.
const F32_NARROWING_TOLERANCE: f64 = 1.0e-6;

#[test]
fn accumulator_matches_the_f64_reference_at_large_magnitude() {
    let k = activations(ROWS);
    let mut acc = FfnGramAccumulator::new(FFN_DIM);
    acc.add(k.view());
    let (cov, n) = acc.finish().expect("rows were accumulated");
    assert_eq!(n, ROWS);
    let err = max_relative_error(&cov.mapv(|v| v as f64), &reference(&k));
    assert!(
        err < F32_NARROWING_TOLERANCE,
        "f64 accumulator relative error {err:e}"
    );
}

#[test]
fn f32_accumulation_is_the_failure_this_replaces() {
    // Control: on the same input, summing in f32 misses the tolerance the
    // f64 accumulator meets. If this stops failing the tolerance, the
    // fixture no longer exercises the precision loss and the test above
    // proves nothing.
    let k = activations(ROWS);
    let err = max_relative_error(&naive_f32(&k), &reference(&k));
    assert!(
        err > F32_NARROWING_TOLERANCE,
        "f32 accumulation relative error {err:e} — fixture too easy"
    );
}

#[test]
fn batches_accumulate_like_one_matrix() {
    let k = activations(1000);
    let mut whole = FfnGramAccumulator::new(FFN_DIM);
    whole.add(k.view());
    let mut split = FfnGramAccumulator::new(FFN_DIM);
    split.add(k.slice(ndarray::s![..300, ..]));
    split.add(k.slice(ndarray::s![300.., ..]));
    assert_eq!(split.samples(), 1000);
    assert_eq!(whole.finish(), split.finish());
}

#[test]
fn zero_activations_contribute_samples_but_no_mass() {
    let mut acc = FfnGramAccumulator::new(2);
    acc.add(
        Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 3.0, 0.0])
            .unwrap()
            .view(),
    );
    let (cov, n) = acc.finish().unwrap();
    assert_eq!(n, 2);
    assert_eq!(
        cov,
        Array2::from_shape_vec((2, 2), vec![4.5, 0.0, 0.0, 0.0]).unwrap()
    );
}

#[test]
fn empty_accumulator_finishes_to_none() {
    let acc = FfnGramAccumulator::new(FFN_DIM);
    assert_eq!(acc.ffn_dim(), FFN_DIM);
    assert!(acc.finish().is_none());
}

#[test]
#[should_panic(expected = "does not match accumulator ffn_dim")]
fn width_mismatch_is_refused() {
    let mut acc = FfnGramAccumulator::new(FFN_DIM);
    acc.add(Array2::<f32>::zeros((1, FFN_DIM + 1)).view());
}
