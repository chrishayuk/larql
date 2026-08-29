//! `tests` for [`super`].

use super::*;
use crate::format::vindex3::represent::arena::StoredOperand;
use crate::format::vindex3::represent::compiler::{SourceDependency, SourceIdentity};
use crate::format::vindex3::represent::experiment::RoleScope;
use crate::format::vindex3::represent::map::Exception;
use crate::format::vindex3::represent::selection::{Refusal, Verdict};
use std::collections::BTreeMap;

const HIDDEN: usize = 512;
const INTER: usize = 256;
const EXPERTS: u32 = 4;

struct Fake {
    bytes: BTreeMap<String, Vec<u8>>,
}

impl Fake {
    fn kimi_layer(layer: u32, seed: f32) -> (Self, Vec<SourceTensor>) {
        let mut bytes = BTreeMap::new();
        let mut tensors = Vec::new();
        for e in 0..EXPERTS {
            for (proj, rows, cols) in [
                ("w1", INTER, HIDDEN),
                ("w2", HIDDEN, INTER),
                ("w3", INTER, HIDDEN),
            ] {
                let name = format!("{layer}.block_sparse_moe.experts.{e}.{proj}.weight");
                let v: Vec<u8> = (0..rows * cols)
                    .flat_map(|i| {
                        let f = ((i as f32) * 0.013 + seed + e as f32).sin();
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
        name: "q2-layer1-q6".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions,
    }
}

fn opts<'a>(object: &'a str, experts: u32, out: &'a std::path::Path) -> CompileOptions<'a> {
    CompileOptions {
        object,
        role: Role::ExpertWeight,
        experts,
        out,
        checkpoint: None,
    }
}

fn index(map: PrecisionMap) -> CandidateIndex {
    CandidateIndex::new(
        "Kimi-Linear-48B-A3B-Instruct",
        SourceDependency {
            identity: SourceIdentity {
                manifest_hash: "m".repeat(64),
                graph_hash: "g".repeat(64),
                segments: BTreeMap::from([("target.expert_bank.bin".into(), "a".repeat(64))]),
            },
            locator_hint: "/somewhere/source.vindex3".into(),
        },
        "target.expert_bank",
        map,
    )
}

/// **The driver is generic.** `layers: [1]` is a map, not a code path:
/// the same call compiles layer 1 and leaves layer 2 at source
/// precision, purely because the map says so.
#[test]
fn only_the_scoped_layer_is_compiled() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (l1, mut tensors) = Fake::kimi_layer(1, 0.3);
    let (l2, t2) = Fake::kimi_layer(2, 1.7);
    let mut bytes = l1.bytes;
    bytes.extend(l2.bytes);
    let source = Fake { bytes };
    tensors.extend(t2);

    // Layer 1 at Q6_K; every other layer falls through to source.
    let mut idx = index(map_for(vec![
        Exception {
            projection: None,
            layers: Some((1, 1)),
            encoding: Some("Q6_K".into()),
        },
        Exception {
            projection: None,
            layers: None,
            encoding: None,
        },
    ]));
    let outcome = compile_expert_bank(
        &source,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        |_| {},
    )
    .expect("compiles");

    assert_eq!(outcome.sealed, (EXPERTS * 3) as usize, "all of layer 1");
    assert_eq!(
        outcome.source_precision,
        (EXPERTS * 3) as usize,
        "all of layer 2"
    );
    assert_eq!(outcome.resumed, 0);
    for e in 0..EXPERTS {
        assert!(idx
            .ledger
            .get(
                "target.expert_bank",
                &format!("1.block_sparse_moe.experts.{e}.w2.weight")
            )
            .is_some());
        assert!(
            idx.ledger
                .get(
                    "target.expert_bank",
                    &format!("2.block_sparse_moe.experts.{e}.w2.weight")
                )
                .is_none(),
            "layer 2 must not be sealed — it was never compiled"
        );
    }
    assert!(idx.ledger.overlaps().is_empty(), "banks must not collide");
}

/// **Resume.** A second pass re-does nothing, and the file is not
/// rewritten — which is what makes a multi-hour compile interruptible.
#[test]
fn a_second_pass_resumes_instead_of_recompiling() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (source, tensors) = Fake::kimi_layer(1, 0.3);
    let mut idx = index(map_for(vec![]));

    let first = compile_expert_bank(
        &source,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        |_| {},
    )
    .expect("first pass");
    assert_eq!(first.sealed, (EXPERTS * 3) as usize);
    let after_first = std::fs::metadata(&out).expect("stat").len();

    let second = compile_expert_bank(
        &source,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        |_| {},
    )
    .expect("second pass");
    assert_eq!(second.sealed, 0, "nothing may be recompiled");
    assert_eq!(second.resumed, (EXPERTS * 3) as usize);
    assert_eq!(second.bytes_written, 0);
    assert_eq!(std::fs::metadata(&out).expect("stat").len(), after_first);
}

/// **An interrupted compile resumes from where it stopped.** Simulated
/// by compiling a prefix of the work, then the whole list.
#[test]
fn an_interrupted_compile_finishes_only_the_remainder() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (source, tensors) = Fake::kimi_layer(1, 0.3);
    let mut idx = index(map_for(vec![]));

    let half = tensors.len() / 2;
    let first = compile_expert_bank(
        &source,
        &tensors[..half],
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        |_| {},
    )
    .expect("interrupted pass");
    assert_eq!(first.sealed, half);

    let rest = compile_expert_bank(
        &source,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        |_| {},
    )
    .expect("resumed pass");
    assert_eq!(rest.resumed, half, "the sealed prefix is skipped");
    assert_eq!(
        rest.sealed,
        tensors.len() - half,
        "and only the rest is done"
    );
}

/// A source that changed under a seal is recompiled, not trusted.
#[test]
fn a_moved_source_is_recompiled_on_resume() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (mut source, tensors) = Fake::kimi_layer(1, 0.3);
    let mut idx = index(map_for(vec![]));
    compile_expert_bank(
        &source,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        |_| {},
    )
    .expect("first");

    // One operand's source bytes move.
    let victim = "1.block_sparse_moe.experts.2.w2.weight";
    source.bytes.get_mut(victim).expect("present")[0] ^= 0xFF;
    let again = compile_expert_bank(
        &source,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        |_| {},
    )
    .expect("second");
    assert_eq!(again.sealed, 1, "exactly the moved operand is redone");
    assert_eq!(again.resumed, tensors.len() - 1);
}

/// **Compiled is not selected.** The bytes exist and are addressable;
/// authority stays with the source until a promotion says otherwise.
#[test]
fn compiled_bytes_are_not_authoritative_until_promoted() {
    let mut idx = index(map_for(vec![]));
    assert_eq!(idx.selected_representation, "source");
    assert!(!idx.is_authoritative());
    assert_eq!(idx.can_represent_as, vec!["Q6_K"]);

    // A promotion that refused everything cannot select anything.
    let refused = Promotion {
        map: map_for(vec![]),
        verdicts: vec![Verdict {
            scope: RoleScope::role(Role::ExpertWeight),
            target: "Q6_K".into(),
            outcome: Err(Refusal::QualityUnproven),
        }],
    };
    assert!(idx.apply_promotion(&refused).is_err());
    assert!(
        !idx.is_authoritative(),
        "a refusal must leave source in charge"
    );

    // A promotion of a DIFFERENT encoding cannot select these bytes either.
    let other = Promotion {
        map: map_for(vec![]),
        verdicts: vec![Verdict {
            scope: RoleScope::role(Role::ExpertWeight),
            target: "Q4_K".into(),
            outcome: Ok(()),
        }],
    };
    assert!(idx.apply_promotion(&other).is_err());
    assert!(!idx.is_authoritative());

    // Only a promotion of what was actually compiled.
    let promoted = Promotion {
        map: map_for(vec![]),
        verdicts: vec![Verdict {
            scope: RoleScope::role(Role::ExpertWeight),
            target: "Q6_K".into(),
            outcome: Ok(()),
        }],
    };
    idx.apply_promotion(&promoted).expect("promotes");
    assert_eq!(idx.selected_representation, "Q6_K");
    assert!(idx.is_authoritative());
}

/// The three projections occupy disjoint regions, so a compiled bank
/// can be mmap'd and handed to the grouped kernel with an offset table.
#[test]
fn the_three_projection_banks_do_not_overlap() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (source, tensors) = Fake::kimi_layer(1, 0.3);
    let mut idx = index(map_for(vec![]));
    compile_expert_bank(
        &source,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        |_| {},
    )
    .expect("compiles");
    assert!(idx.ledger.overlaps().is_empty());

    // And each projection's experts are contiguous within their bank.
    for proj in ["w1", "w3", "w2"] {
        let mut offs: Vec<u64> = (0..EXPERTS)
            .map(|e| {
                idx.ledger
                    .get(
                        "target.expert_bank",
                        &format!("1.block_sparse_moe.experts.{e}.{proj}.weight"),
                    )
                    .expect("sealed")
                    .target_offset
            })
            .collect();
        offs.sort_unstable();
        let stride = offs[1] - offs[0];
        for w in offs.windows(2) {
            assert_eq!(w[1] - w[0], stride, "{proj} experts are not evenly spaced");
        }
    }
}

#[test]
fn the_expert_index_is_parsed_from_the_checkpoints_own_naming() {
    assert_eq!(
        expert_of("17.block_sparse_moe.experts.137.w2.weight"),
        Some(137)
    );
    assert_eq!(expert_of("17.self_attn.q_proj.weight"), None);
}

/// **A candidate overlay must not attach to a different source.** Its
/// bytes cover only what the map compiled; everything else comes from
/// the source, so a mismatched one composes a hybrid nobody measured.
#[test]
fn a_candidate_refuses_a_source_it_was_not_compiled_against() {
    let idx = index(map_for(vec![]));
    let good = idx.source.identity.clone();
    idx.source
        .verify(&good)
        .expect("the compiled-against source is accepted");

    // Payloads identical, SEMANTICS different — the case payload hashes
    // alone would wave through.
    let mut other_graph = good.clone();
    other_graph.graph_hash = "z".repeat(64);
    let err = idx.source.verify(&other_graph).expect_err("must refuse");
    assert!(format!("{err}").contains("semantic graph"), "{err}");

    let mut other_manifest = good.clone();
    other_manifest.manifest_hash = "z".repeat(64);
    let err = idx.source.verify(&other_manifest).expect_err("must refuse");
    assert!(format!("{err}").contains("manifest"), "{err}");

    let mut moved_bytes = good.clone();
    moved_bytes
        .segments
        .insert("target.expert_bank.bin".into(), "b".repeat(64));
    let err = idx.source.verify(&moved_bytes).expect_err("must refuse");
    assert!(format!("{err}").contains("DIFFERENT container"), "{err}");

    let mut absent = good.clone();
    absent.segments.clear();
    let err = idx.source.verify(&absent).expect_err("must refuse");
    assert!(format!("{err}").contains("missing segment"), "{err}");
}

/// **Identity is content, not location.** Both artifacts must survive
/// being moved to another disk, so the locator is a hint for finding the
/// source and never what verification resolves on.
#[test]
fn a_moved_source_container_still_verifies() {
    let mut idx = index(map_for(vec![]));
    let identity = idx.source.identity.clone();
    idx.source.locator_hint = "/some/other/disk/Kimi-Linear.vindex3".into();
    idx.source
        .verify(&identity)
        .expect("moving the container must not invalidate the overlay");
}
