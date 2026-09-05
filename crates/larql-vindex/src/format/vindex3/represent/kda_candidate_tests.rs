//! Synthetic proof for the KDA candidate family — same discipline as
//! `compiler_tests`: a hand-built fake source through the REAL compile
//! path, at a 32-aligned geometry small enough to reason about.

use std::collections::BTreeMap;

use super::super::arena::{SourceOperands, StoredOperand};
use super::super::compiler::{SourceDependency, SourceIdentity};
use super::super::map::{Exception, PrecisionMap};
use super::*;

/// Q8_0 needs every reduction axis %32: qkv k = HIDDEN, o k = WIDTH.
const WIDTH: usize = 32;
const HIDDEN: usize = 64;
const OBJECT: &str = "target.kda_bank";

struct Fake {
    bytes: BTreeMap<String, Vec<u8>>,
}

impl Fake {
    fn layers(layers: &[u32]) -> (Self, Vec<SourceTensor>) {
        let mut bytes = BTreeMap::new();
        let mut tensors = Vec::new();
        for &layer in layers {
            for (proj, rows, cols) in [
                ("q_proj", WIDTH, HIDDEN),
                ("k_proj", WIDTH, HIDDEN),
                ("v_proj", WIDTH, HIDDEN),
                ("o_proj", HIDDEN, WIDTH),
            ] {
                let name = format!("{layer}.self_attn.{proj}.weight");
                let v: Vec<u8> = (0..rows * cols)
                    .flat_map(|i| {
                        let f = ((i as f32) * 0.017 + layer as f32).sin();
                        ((f.to_bits() >> 16) as u16).to_le_bytes()
                    })
                    .collect();
                bytes.insert(name.clone(), v);
                tensors.push(SourceTensor {
                    name,
                    shape: vec![rows, cols],
                });
            }
        }
        (Self { bytes }, tensors)
    }
}

impl SourceOperands for Fake {
    fn load_stored(&self, operand: &OperandRef) -> Result<StoredOperand, VindexError> {
        Ok(StoredOperand {
            dtype: "BF16".into(),
            bytes: self
                .bytes
                .get(&operand.tensor)
                .ok_or_else(|| VindexError::Parse(format!("no `{}`", operand.tensor)))?
                .clone(),
        })
    }
}

fn map_for(exceptions: Vec<Exception>) -> PrecisionMap {
    PrecisionMap {
        name: "kda-native-test".into(),
        encoding: "BF16".into(),
        roles: vec!["decoder-linear".into()],
        exceptions,
    }
}

fn band(lo: u32, hi: u32, encoding: &str) -> Exception {
    Exception {
        projection: None,
        layers: Some((lo, hi)),
        encoding: Some(encoding.into()),
    }
}

fn index(map: PrecisionMap) -> CandidateIndex {
    CandidateIndex::new(
        "Kimi-Linear-48B-A3B-Instruct",
        SourceDependency {
            identity: SourceIdentity::synthetic(
                "m".repeat(64),
                "g".repeat(64),
                [("target.decoder_stack.bin".into(), "a".repeat(64))],
            ),
            locator_hint: "/somewhere/source.vindex3".into(),
        },
        OBJECT,
        map,
    )
}

/// The four slots land disjoint, in order, at the strides the encoding
/// dictates — and a two-layer bank places the second layer at exactly
/// the first's extent.
#[test]
fn the_four_slots_are_disjoint_and_layers_stack() {
    let map = map_for(vec![band(1, 2, "Q8_0")]);
    let placement = KdaPlacement::resolve(&map, Role::DecoderLinear, &[1, 2], WIDTH, HIDDEN)
        .expect("placement resolves");
    let l1 = placement.layout(1).expect("layer 1 placed");
    let per = LayerBankLayout::matrix_bytes("Q8_0", WIDTH, HIDDEN).unwrap();
    assert_eq!(l1.slot("q_proj").unwrap(), (0, per));
    assert_eq!(l1.slot("k_proj").unwrap(), (per, per));
    assert_eq!(l1.slot("v_proj").unwrap(), (2 * per, per));
    assert_eq!(l1.slot("o_proj").unwrap().0, 3 * per);
    assert!(
        l1.slot("w1").is_err(),
        "an expert spelling has no slot here"
    );
    assert_eq!(placement.layer_base(2).unwrap(), l1.layer_bytes());
}

/// End to end through the real compile path: every compiled projection
/// sealed at its placed offset, bytes at those offsets decode back to
/// the source values within the Q8_0 roundtrip bound, and a rerun
/// resumes every seal instead of recompiling.
#[test]
fn a_kda_bank_compiles_seals_and_resumes() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("kda_bank.bin");
    let (fake, tensors) = Fake::layers(&[1, 2]);
    let mut idx = index(map_for(vec![band(1, 2, "Q8_0")]));

    let outcome = compile_kda_bank(&fake, &tensors, OBJECT, &out, None, &mut idx, &mut |_| {})
        .expect("compiles");
    assert_eq!(outcome.sealed, 8, "4 projections x 2 layers");
    assert_eq!(outcome.resumed, 0);

    // Bytes on disk equal the arena's own answer for one slot.
    let placement =
        KdaPlacement::resolve(&idx.map, Role::DecoderLinear, &[1, 2], WIDTH, HIDDEN).unwrap();
    let l2 = placement.layout(2).unwrap();
    let (o, len) = l2.slot("v_proj").unwrap();
    let base = placement.layer_base(2).unwrap();
    let disk = std::fs::read(&out).unwrap();
    let seal = idx
        .ledger
        .get(OBJECT, "2.self_attn.v_proj.weight")
        .expect("sealed");
    assert_eq!(seal.target_offset, base + o);
    assert_eq!(seal.target_len, len);
    assert_eq!(
        hash_bytes(&disk[(base + o) as usize..(base + o + len) as usize]),
        seal.target_hash,
        "the sealed hash is the hash of the bytes at the sealed offset"
    );

    let again = compile_kda_bank(&fake, &tensors, OBJECT, &out, None, &mut idx, &mut |_| {})
        .expect("recompiles");
    assert_eq!(again.sealed, 0);
    assert_eq!(again.resumed, 8, "a second pass resumes, never rewrites");
}

/// A scope holding anything but the four projections is refused BY
/// NAME — a bank that silently skipped a tensor would leave the caller
/// believing it was compiled.
#[test]
fn a_non_kda_tensor_in_the_scope_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("kda_bank.bin");
    let (fake, mut tensors) = Fake::layers(&[1]);
    tensors.push(SourceTensor {
        name: "1.block_sparse_moe.experts.0.w1.weight".into(),
        shape: vec![WIDTH, HIDDEN],
    });
    let mut idx = index(map_for(vec![band(1, 1, "Q8_0")]));
    let err = compile_kda_bank(&fake, &tensors, OBJECT, &out, None, &mut idx, &mut |_| {})
        .expect_err("must refuse");
    assert!(format!("{err}").contains("w1"), "{err}");
}

/// A transposed projection — right bytes, wrong axes — refuses at the
/// placement check instead of landing plausibly at the wrong extent.
/// The geometry is chosen so the byte COUNT would match: only the
/// per-axis block constraint and stride derivation can catch it.
#[test]
fn a_transposed_projection_is_refused() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("kda_bank.bin");
    let (fake, mut tensors) = Fake::layers(&[1]);
    // Swap q_proj's declared axes.
    let q = tensors
        .iter_mut()
        .find(|t| t.name.ends_with("q_proj.weight"))
        .unwrap();
    q.shape = vec![HIDDEN, WIDTH];
    let mut idx = index(map_for(vec![band(1, 1, "Q6_K")]));
    // Q6_K needs k % 256 — neither axis satisfies it, so the refusal
    // must name the constraint rather than accept the transpose.
    let err = compile_kda_bank(&fake, &tensors, OBJECT, &out, None, &mut idx, &mut |_| {})
        .expect_err("must refuse");
    assert!(format!("{err}").contains("256"), "{err}");
}

/// Per-layer encodings compose in one bank exactly as the expert
/// placement allows — and a layer with MIXED encodings is refused.
#[test]
fn per_layer_encodings_place_and_mixed_within_a_layer_refuses() {
    let map = map_for(vec![band(1, 1, "Q8_0"), band(2, 2, "BF16")]);
    let p = KdaPlacement::resolve(&map, Role::DecoderLinear, &[1, 2], WIDTH, HIDDEN)
        .expect("two encodings, two layers");
    assert_eq!(p.layout(1).unwrap().encoding, "Q8_0");
    assert_eq!(p.layout(2).unwrap().encoding, "BF16");

    let mixed = map_for(vec![
        Exception {
            projection: Some("q_proj".into()),
            layers: Some((1, 1)),
            encoding: Some("Q8_0".into()),
        },
        Exception {
            projection: Some("o_proj".into()),
            layers: Some((1, 1)),
            encoding: Some("BF16".into()),
        },
    ]);
    let err = KdaPlacement::resolve(&mixed, Role::DecoderLinear, &[1], WIDTH, HIDDEN)
        .expect_err("mixed encodings within one layer must refuse");
    assert!(format!("{err}").contains("ONE encoding"), "{err}");
}
