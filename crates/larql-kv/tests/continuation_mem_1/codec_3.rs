//! CONTINUATION-CODEC-3 reconnaissance: the real recent-K provider,
//! `codec-recent/v1`, against the simulated map it replaces.
//!
//! On fixtures, before any model arm:
//! - SIM-W: for each window w, decode logits bit-identical to MappedCodec's
//!   `all:v:all|all:k:older<w>`, with the same retention after every
//!   append and the same reads (the recorder's trace);
//! - window 0 is codec/v1, bit for bit and trace for trace;
//! - residency is exact against what the window DECLARES, every live
//!   append-born pointer is a declared row or list, and two seeded
//!   violations are caught: a hidden f32 copy, and a window wider than
//!   declared.
//!
//! The real arms on gemma3-4b-it run from `codec_3_run.rs`.

use larql_kv::{CodecKvState, CodecRecentKvState};
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    LatentKvRows, LayerContinuationGeometry, RecurrentState,
};
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, ContinuationProvider, LayerKvGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::kv_view::KvView;

use super::codec_map_sim::{CompressMap, MappedCodec, Recorder, Retained};
use super::measured::{CodeList, MixedLayout};
use super::*;

const BITS: u8 = 4;
/// Windows below, at and above the fixture's sliding history (G_WINDOW),
/// and past every position of the journey.
const WINDOWS: [usize; 5] = [1, 2, G_WINDOW, 5, 64];

/// The simulator's map that a window of `w` replaces.
pub(super) fn protection_spec(window: usize) -> String {
    format!("all:v:all|all:k:older{window}")
}

impl Retained for CodecRecentKvState {
    fn retained(&self, layer: usize) -> (usize, usize) {
        let base = self.rows_base(layer);
        (base, base + self.encoded_values(layer).len())
    }
}

/// A journey that crosses the fixture's sliding window in prefill and in
/// decode, so release and ageing interleave in both.
fn journey() -> Journey {
    Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5, 9, 13],
        decode: vec![1, 2, 3, 4, 6, 7, 8],
    }
}

fn subject() -> (Subject, ReferenceBackend) {
    (
        subjects::fixture(miniature_glimmer, "codec-3-recon"),
        ReferenceBackend::new(),
    )
}

/// Decode logits, the recorder's trace, and `read` of the measured
/// provider after the journey.
fn traced<P, R>(
    inner: P,
    read: impl Fn(&Measured<Recorder<P>>) -> R,
) -> (Vec<Vec<f32>>, (u64, u64), R)
where
    P: Inspect + Retained,
{
    let (subject, backend) = subject();
    let ops = subject.prepare(&backend);
    let mut kv = Measured::new(Recorder::new(inner));
    let out = subjects::run(&subject, &ops, &backend, &mut kv, &journey());
    let rows = out
        .logits
        .into_iter()
        .filter(|(phase, _)| *phase == subjects::DECODE)
        .map(|(_, r)| r)
        .collect();
    (rows, kv.inner.trace(), read(&kv))
}

fn bits(rows: &[Vec<f32>]) -> Vec<u32> {
    rows.iter().flatten().map(|x| x.to_bits()).collect()
}

#[test]
fn sim_w_the_real_provider_is_the_simulated_protection_map_bit_for_bit() {
    let _serial = serial();
    let (codec, codec_trace, _) = traced(CodecKvState::new(BITS), |_| ());
    for w in WINDOWS {
        let spec = protection_spec(w);
        let map = CompressMap::parse(&format!("p{w}={spec}")).unwrap();
        let (sim, sim_trace, _) = traced(MappedCodec::new(BITS, map), |_| ());
        let (real, real_trace, _) = traced(CodecRecentKvState::new(BITS, w), |_| ());
        assert!(!real.is_empty());
        assert_eq!(
            bits(&real),
            bits(&sim),
            "w {w}: logits bit-identical to {spec}"
        );
        assert_eq!(real_trace, sim_trace, "w {w}: same retention and reads");
        assert_eq!(real_trace, codec_trace, "w {w}: retention is codec/v1's");
        assert_ne!(
            bits(&real),
            bits(&codec),
            "w {w}: the window moves the logits"
        );
    }
}

#[test]
fn a_window_of_zero_is_codec_v1_bit_for_bit() {
    let _serial = serial();
    let (codec, codec_trace, _) = traced(CodecKvState::new(BITS), |_| ());
    let (real, real_trace, _) = traced(CodecRecentKvState::new(BITS, 0), |_| ());
    assert_eq!(bits(&real), bits(&codec));
    assert_eq!(real_trace, codec_trace);
}

#[test]
fn residency_is_exactly_what_the_window_declares() {
    let _serial = serial();
    for bits_ in [4u8, 3] {
        for w in WINDOWS {
            let (_, _, (residency, appends)) = traced(CodecRecentKvState::new(bits_, w), |kv| {
                let (calls, _) = kv.take_records();
                let appends: Vec<_> = calls.iter().filter_map(|c| c.append).collect();
                (kv.mixed_residency(w).expect("a mixed layout"), appends)
            });
            assert!(residency.holds(), "{bits_}-bit w {w}: {residency:?}");
            assert!(residency.k_exact_bytes > 0 && residency.v_code_bytes > 0);
            assert!(!appends.is_empty());
            for t in &appends {
                assert_eq!(t.incoming_freed, 2, "every incoming row is freed: {t:?}");
            }
        }
    }
}

/// The RED controls: a hidden f32 copy is a stray by pointer, and a
/// provider holding a wider window than declared misses the declaration.
#[test]
fn residency_catches_a_hidden_copy_and_an_undeclared_window() {
    let _serial = serial();
    let (_, _, hidden) = traced(HidesACopy::new(CodecRecentKvState::new(BITS, 2)), |kv| {
        kv.mixed_residency(2).unwrap()
    });
    assert!(hidden.strays > 0 && !hidden.holds(), "{hidden:?}");
    let (_, _, wider) = traced(CodecRecentKvState::new(BITS, 4), |kv| {
        kv.mixed_residency(2).unwrap()
    });
    assert!(
        wider.window_mismatches > 0 && wider.append_born_live != wider.expected,
        "{wider:?}"
    );
}

/// A seeded violation: codec-recent/v1 plus an f32 copy of every key it
/// is handed, kept where no trait method shows it.
struct HidesACopy {
    inner: CodecRecentKvState,
    hidden: Vec<Vec<f32>>,
}

impl HidesACopy {
    fn new(inner: CodecRecentKvState) -> Self {
        Self {
            inner,
            hidden: Vec::new(),
        }
    }
}

impl Retained for HidesACopy {
    fn retained(&self, layer: usize) -> (usize, usize) {
        self.inner.retained(layer)
    }
}

impl ContinuationProvider for HidesACopy {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.inner.prepare(layers)
    }
    fn prepare_continuation(
        &mut self,
        layers: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        self.inner.prepare_continuation(layers)
    }
    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.hidden.push(key.clone());
        self.inner.append(layer, key, value)
    }
    fn rows(&self, layer: usize) -> KvView<'_> {
        self.inner.rows(layer)
    }
    fn prepare_layer(&mut self, layer: usize) {
        self.inner.prepare_layer(layer)
    }
    fn position(&self) -> usize {
        self.inner.position()
    }
    fn set_position(&mut self, position: usize) {
        self.inner.set_position(position)
    }
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.inner.recurrent_state(layer)
    }
    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.inner.latent_state(layer)
    }
}

impl Inspect for HidesACopy {
    fn matrix_ptrs(&self, _: usize) -> Option<(usize, usize)> {
        None
    }
    fn matrix_rows(&self, _: usize) -> Option<usize> {
        None
    }
    fn code_list(&self, _: usize) -> Option<CodeList> {
        None
    }
    fn code_rows(&self, _: usize, _: std::ops::Range<usize>) -> Vec<usize> {
        Vec::new()
    }
    fn scratch_ptrs(&self) -> Option<[usize; 2]> {
        self.inner.scratch_ptrs()
    }
    fn mixed_layout(&self, layer: usize) -> Option<MixedLayout> {
        self.inner.mixed_layout(layer)
    }
}
