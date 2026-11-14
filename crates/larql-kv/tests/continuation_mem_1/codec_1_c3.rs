//! CONTINUATION-CODEC-1 C3: the instrument, its stop guards, and the
//! frozen arms.
//!
//! The instrument (codec storage by pointer, the exact residency
//! equation, the scratch bound) and the three guards (the frozen token
//! bank, the NULL arm, the yardstick's bound pack) are exercised here on
//! fixtures. The real arms — R, NULL, C, H, H3 on gemma3-4b-it — run
//! progressively from `codec_1_c3_run.rs`.
//! Frozen: docs/represent/forecasts/continuation-codec-1.json.

use std::collections::BTreeMap;

use super::*;

use larql_kv::CodecKvState;
use larql_vindex::format::vindex3::opplan::exec::operands::SelectedRepresentation;
use larql_vindex::format::vindex3::represent::measure::plan::metrics::{
    position_metrics, PositionMetrics, Summary, MARGIN_BANDS,
};
use sha2::{Digest, Sha256};

use super::measured::{CodecResidency, Inspect};

/// The frozen token bank, its length and digest (sha256 over u32 LE ids).
const BANK_PATH: &str = "../../docs/represent/forecasts/continuation-codec-1-token-bank.json";
const BANK_LEN: usize = 8193;
const BANK_IDS_SHA256: &str = "c5631deeb358b1eda28d33975d1bcf0536eb2a10de2c9fffab6f50f07b82b3ae";

/// The frozen ladder: every rung ends in RESUME resumed and DECODE scored
/// teacher-forced positions.
pub(super) const RUNGS: [usize; 3] = [1_024, 4_096, 8_192];
pub(super) const RESUME: usize = 16;
pub(super) const DECODE: usize = 512;

/// The rule's confident band (R's margin ≥ 0.5) and its floor; the
/// yardstick's informativeness floor.
const CONFIDENT_BAND: (f64, f64) = MARGIN_BANDS[2];
const CONFIDENT_TOP1_MIN: f64 = 0.995;
pub(super) const YARDSTICK_TOP1_MIN: f64 = 0.90;

/// The yardstick's pack, as the freeze names it.
pub(super) const YARDSTICK_ENCODING: &str = "Q4_K";
const YARDSTICK_REVISION: u32 = 1;
const DECODER_STACK: &str = "target.decoder_stack";

/// The bank, refused unless it is the frozen one.
pub(super) fn bank() -> Result<Vec<u32>, String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(BANK_PATH);
    let raw = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let doc: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let ids: Vec<u32> = doc["ids"]
        .as_array()
        .ok_or("the bank has no ids")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("an id is not a u32")
        })
        .collect::<Result<_, _>>()?;
    let mut hash = Sha256::new();
    for id in &ids {
        hash.update(id.to_le_bytes());
    }
    let digest = format!("{:x}", hash.finalize());
    if ids.len() != BANK_LEN || digest != BANK_IDS_SHA256 {
        return Err(format!(
            "token bank is not the frozen one: {} ids, sha256 {digest}",
            ids.len()
        ));
    }
    Ok(ids)
}

/// sha256 of a file's bytes, hex.
pub(super) fn file_sha256(path: &std::path::Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    format!("{:x}", Sha256::digest(bytes))
}

/// The four authorities a C3 record carries: the code, the binary that
/// ran it, the containers with what each actually bound, and the bank.
pub(super) fn authorities(exact: (&str, &Subject), yard: (&str, &Subject)) -> Value {
    let binary = std::env::current_exe().expect("the running test binary");
    let container = |(dir, subject): (&str, &Subject)| {
        json!({
            "dir_name": std::path::Path::new(dir).file_name().map(|n| n.to_string_lossy().into_owned()),
            "index_json_sha256": file_sha256(&std::path::Path::new(dir).join("index.json")),
            "bound": format!("{:?}", subject.store.selection()),
        })
    };
    json!({
        "implementation_git_sha": git_sha(),
        "binary_sha256": file_sha256(&binary),
        "containers": {"exact": container(exact), "yardstick": container(yard)},
        "token_bank_ids_sha256": BANK_IDS_SHA256,
    })
}

/// The NULL guard: window/v1 against R must be KL exactly 0 with the same
/// argmax at every scored position, or nothing after it is informative.
pub(super) fn null_guard(reference: &[Vec<f32>], null: &[Vec<f32>]) -> Result<(), String> {
    if reference.len() != null.len() {
        return Err(format!(
            "NULL scored {} positions, R {}",
            null.len(),
            reference.len()
        ));
    }
    for (i, (r, n)) in reference.iter().zip(null).enumerate() {
        let s = position_metrics(r, n, None).map_err(|e| format!("{e:?}"))?;
        if s.kl != 0.0 || !s.top1_agree {
            return Err(format!(
                "UNINFORMATIVE: NULL differs from R at scored position {i} (KL {:e}, top-1 {})",
                s.kl, s.top1_agree
            ));
        }
    }
    Ok(())
}

/// The yardstick guard: arm C must have bound the decoder stack to the
/// frozen pack, stored, with its codec identity — a plain open binds the
/// canonical stack and measures the reference against itself (P0a).
pub(super) fn yardstick_guard(
    selection: &BTreeMap<String, SelectedRepresentation>,
) -> Result<(), String> {
    let bound = selection.get(DECODER_STACK);
    let ok = bound.is_some_and(|s| {
        s.encoding == YARDSTICK_ENCODING
            && s.stored
            && s.codec
                .as_ref()
                .is_some_and(|c| c.family == YARDSTICK_ENCODING && c.revision == YARDSTICK_REVISION)
    });
    if ok {
        Ok(())
    } else {
        Err(format!(
            "UNINFORMATIVE: arm C bound `{DECODER_STACK}` as {bound:?}, not the stored \
             {YARDSTICK_ENCODING} rev {YARDSTICK_REVISION} pack"
        ))
    }
}

/// Every decode row of one arm's journey, and codec residency if any.
pub(super) fn decode_rows<P: Inspect, B: PlanBackend>(
    subject: &Subject,
    ops: &larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands,
    backend: &B,
    inner: P,
    journey: &Journey,
) -> (Vec<Vec<f32>>, Option<CodecResidency>) {
    let mut kv = Measured::new(inner);
    let out = subjects::run(subject, ops, backend, &mut kv, journey);
    let rows = out
        .logits
        .into_iter()
        .filter(|(phase, _)| *phase == subjects::DECODE)
        .map(|(_, row)| row)
        .collect();
    (rows, kv.codec_residency())
}

/// Score a candidate's decode rows against R's; position `start + i`
/// predicts `next[i]`.
pub(super) fn score(
    reference: &[Vec<f32>],
    candidate: &[Vec<f32>],
    next: &[u32],
    rung: usize,
    start: usize,
) -> Vec<PositionMetrics> {
    reference
        .iter()
        .zip(candidate)
        .zip(next)
        .enumerate()
        .map(|(i, ((r, c), &token))| {
            let s = position_metrics(r, c, Some(token)).expect("finite rows of one width");
            PositionMetrics {
                sample: rung,
                position: start + i,
                category: format!("n{rung}"),
                kl: s.kl,
                top1_agree: s.top1_agree,
                top5_overlap: s.top5_overlap,
                delta_nll: s.delta_nll,
                reference_margin: s.reference_margin,
                reference_entropy: s.reference_entropy,
                max_abs_delta: s.max_abs_delta,
                mean_abs_delta: s.mean_abs_delta,
            }
        })
        .collect()
}

/// The frozen rule, pooled: H is acceptable only if its mean and p99 KL
/// do not exceed C's and its confident-band top-1 is at least the floor.
pub(super) fn verdict(h: &Summary, c: &Summary) -> Value {
    let band = |s: &Summary| {
        s.by_margin_band
            .iter()
            .find(|(b, _)| *b == CONFIDENT_BAND)
            .map(|(_, a)| a.top1_agreement)
    };
    let mean = h.all.kl_mean <= c.all.kl_mean;
    let tail = h.all.kl_p99 <= c.all.kl_p99;
    let confident = band(h).is_some_and(|t| t >= CONFIDENT_TOP1_MIN);
    json!({
        "mean_kl": {"h": h.all.kl_mean, "c": c.all.kl_mean, "holds": mean},
        "p99_kl": {"h": h.all.kl_p99, "c": c.all.kl_p99, "holds": tail},
        "confident_band_top1": {"h": band(h), "min": CONFIDENT_TOP1_MIN, "holds": confident},
        "acceptable": mean && tail && confident,
    })
}

// ---- fixture controls: the instrument and the guards ----------------------

#[test]
fn the_bank_is_the_frozen_authority() {
    let ids = bank().unwrap();
    assert_eq!(ids.len(), BANK_LEN);
    assert_eq!(ids[0], 2, "position 0 is <bos>");
}

#[test]
fn the_null_guard_passes_identity_and_stops_on_one_ulp() {
    let rows = vec![vec![0.5, -1.0, 2.0, 0.25]; 3];
    null_guard(&rows, &rows).unwrap();
    let mut moved = rows.clone();
    moved[2][1] = f32::from_bits((-1.0f32).to_bits() + 1);
    let stop = null_guard(&rows, &moved).unwrap_err();
    assert!(
        stop.contains("UNINFORMATIVE") && stop.contains("position 2"),
        "{stop}"
    );
}

#[test]
fn the_yardstick_guard_refuses_a_store_that_did_not_bind_the_pack() {
    let subject = subjects::fixture(miniature_glimmer, "c3-unbound");
    let stop = yardstick_guard(subject.store.selection()).unwrap_err();
    assert!(
        stop.contains("UNINFORMATIVE") && stop.contains(DECODER_STACK),
        "{stop}"
    );
}

#[test]
fn codec_residency_is_exact_and_every_append_event_is_accounted() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "c3-residency");
    let backend = ReferenceBackend::new();
    let ops = subject.prepare(&backend);
    let journey = Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5, 9, 13],
        decode: vec![1, 2, 3, 4],
    };
    for bits in [4, 3] {
        let mut kv = Measured::new(CodecKvState::new(bits));
        let out = subjects::run(&subject, &ops, &backend, &mut kv, &journey);
        let residency = kv.codec_residency().expect("codec/v1 has codes");
        assert!(residency.holds(), "{bits}-bit residency: {residency:?}");
        assert!(residency.expected > 0);
        let (calls, _) = kv.take_records();
        let appends: Vec<_> = calls
            .iter()
            .filter_map(|c| Some((c.layer?, c.append?)))
            .collect();
        assert!(!appends.is_empty());
        for (layer, t) in &appends {
            let row_bytes = kv.inner.encoded_row_bytes(*layer) as u64;
            assert_eq!((t.encoded_allocs, t.encoded_bytes), (1, row_bytes), "{t:?}");
            assert_eq!((t.incoming_freed, t.unclassified), (2, 0), "{t:?}");
        }
        let dropped: u64 = appends.iter().map(|(_, t)| t.encoded_dropped).sum();
        assert!(dropped > 0, "the journey must cross the sliding window");
        let codes = out
            .inventories
            .last()
            .unwrap()
            .1
            .iter()
            .filter(|b| b.kind == "kv_code")
            .count();
        let held: usize = (0..subject.geometry.len())
            .filter_map(|l| kv.inner.code_list(l))
            .map(|c| c.end - c.base)
            .sum();
        assert_eq!(codes, held, "the inventory sees every held encoded row");
    }
}

/// The RED control: a provider that keeps an f32 copy of each row beside
/// its codes is caught by pointer, whatever it reports.
#[test]
fn residency_catches_a_hidden_copy() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "c3-cheat");
    let backend = ReferenceBackend::new();
    let ops = subject.prepare(&backend);
    let journey = Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5],
        decode: vec![1, 2],
    };
    let mut kv = Measured::new(cheat::HidesACopy::new(CodecKvState::new(4)));
    subjects::run(&subject, &ops, &backend, &mut kv, &journey);
    let residency = kv.codec_residency().unwrap();
    assert!(residency.strays > 0 && !residency.holds(), "{residency:?}");
}

mod cheat {
    //! A seeded violation for the residency control: codec/v1 plus an f32
    //! copy of every key it is handed, kept where no trait method shows it.
    use super::*;
    use larql_vindex::format::vindex3::opplan::exec::continuation::{
        LatentKvRows, LayerContinuationGeometry, RecurrentState,
    };
    use larql_vindex::format::vindex3::opplan::exec::kv::{
        ContinuationError, ContinuationProvider, LayerKvGeometry,
    };
    use larql_vindex::format::vindex3::opplan::exec::kv_view::KvView;

    pub struct HidesACopy {
        inner: CodecKvState,
        hidden: Vec<Vec<f32>>,
    }

    impl HidesACopy {
        pub fn new(inner: CodecKvState) -> Self {
            Self {
                inner,
                hidden: Vec::new(),
            }
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
        fn invalidate(&mut self, reason: &str) {
            self.inner.invalidate(reason)
        }
        fn invalidated(&self) -> Option<&str> {
            self.inner.invalidated()
        }
        fn recurrent_state(
            &mut self,
            layer: usize,
        ) -> Result<&mut RecurrentState, ContinuationError> {
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
        fn code_list(&self, layer: usize) -> Option<measured::CodeList> {
            self.inner.code_list(layer)
        }
        fn code_rows(&self, layer: usize, range: std::ops::Range<usize>) -> Vec<usize> {
            self.inner.code_rows(layer, range)
        }
        fn scratch_ptrs(&self) -> Option<[usize; 2]> {
            self.inner.scratch_ptrs()
        }
    }
}
