//! `tests` for [`super`].

use super::*;
use crate::format::vindex3::represent::arena::StoredOperand;
use crate::format::vindex3::represent::compiler::{
    ContainerArtifactDigest, SourceDependency, SourceIdentity,
};
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
            identity: SourceIdentity::synthetic(
                "m".repeat(64),
                "g".repeat(64),
                [("target.expert_bank.bin".into(), "a".repeat(64))],
            ),
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
        &mut |_| {},
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
        &mut |_| {},
    )
    .expect("first pass");
    assert_eq!(first.sealed, (EXPERTS * 3) as usize);
    let after_first = std::fs::metadata(&out).expect("stat").len();

    let second = compile_expert_bank(
        &source,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
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
        &mut |_| {},
    )
    .expect("interrupted pass");
    assert_eq!(first.sealed, half);

    let rest = compile_expert_bank(
        &source,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
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
        &mut |_| {},
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
        &mut |_| {},
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
        &mut |_| {},
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
    other_graph.semantic.graph_hash = "z".repeat(64);
    let err = idx.source.verify(&other_graph).expect_err("must refuse");
    assert!(format!("{err}").contains("semantic graph"), "{err}");

    let mut moved_bytes = good.clone();
    moved_bytes.semantic.representations[0].payload_sha256 = "b".repeat(64);
    let err = idx.source.verify(&moved_bytes).expect_err("must refuse");
    assert!(format!("{err}").contains("DIFFERENT container"), "{err}");

    let mut absent = good.clone();
    absent.semantic.representations.clear();
    let err = idx.source.verify(&absent).expect_err("must refuse");
    assert!(format!("{err}").contains("missing segment"), "{err}");

    // The graph agrees and every payload agrees, and the container is
    // still not the one this was compiled against: a restated segment
    // header table moves `segment_sha256` and nothing else. No check
    // above can see it, and it is exactly the fact a physical optimiser
    // prices a PROTECTED decision from.
    let mut restated_table = good.clone();
    restated_table.semantic.representations[0].segment_sha256 = "z".repeat(64);
    let err = idx.source.verify(&restated_table).expect_err("must refuse");
    assert!(format!("{err}").contains("catalogue"), "{err}");
}

/// **A byte-different export of the same container is the same
/// source.** The overlay depends on what the container IS; re-writing
/// its index with the same values in a different serialisation changes
/// nothing an operand resolves through, and refusing it would
/// invalidate a candidate over formatting.
#[test]
fn a_reserialised_source_container_still_verifies() {
    let idx = index(map_for(vec![]));
    let mut reserialised = idx.source.identity.clone();
    reserialised.artifact = ContainerArtifactDigest::new("z".repeat(64));
    assert_ne!(
        reserialised.artifact, idx.source.identity.artifact,
        "the exported bytes differ, which is the whole premise"
    );
    idx.source
        .verify(&reserialised)
        .expect("a re-export is the same source");
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

/// **A container's identity is read from its own metadata**, at all
/// three levels, and a container that differs at any level produces a
/// different identity.
#[test]
fn identity_is_read_from_the_containers_own_metadata() {
    use crate::format::vindex3::index::{RepresentationEntry, Vindex3Index};

    // Built through the container's OWN schema rather than as a hand
    // written fragment. `read_source_identity` parses through
    // `Vindex3Index`, so a fragment that the schema would refuse is not
    // a container this test may claim anything about — that asymmetry
    // is what `state::tests::source_identity` closed.
    let entry = |object: &str, segment: &str, payload: &str| RepresentationEntry {
        object: object.into(),
        encoding: "BF16".into(),
        segment: segment.into(),
        tensor_count: 1,
        payload_bytes: 16,
        payload_sha256: payload.into(),
        segment_sha256: format!("file-{payload}"),
        compiled_from: None,
        codec: None,
        source_representation_digest: None,
        encoder: None,
    };
    let index_json = {
        let mut index = Vindex3Index::new("m", "llama", 64, 2, "", BTreeMap::new());
        index.system_graph = Some("system_graph.json".into());
        index.representations = BTreeMap::from([
            (
                "target.expert_bank@BF16".to_string(),
                entry("target.expert_bank", "target.expert_bank", "aaaa"),
            ),
            (
                "target.decoder_stack@BF16".to_string(),
                entry("target.decoder_stack", "target.decoder_stack", "bbbb"),
            ),
        ]);
        serde_json::to_string(&index).expect("index")
    };

    let dir = std::env::temp_dir().join(format!("larql-identity-{}", std::process::id()));
    let write = |root: &std::path::Path, index: &str, graph: &str| {
        std::fs::create_dir_all(root).expect("dir");
        std::fs::write(root.join("index.json"), index).expect("index");
        std::fs::write(root.join("system_graph.json"), graph).expect("graph");
    };
    let a = dir.join("a");
    write(&a, &index_json, r#"{"components":[]}"#);
    let id = read_source_identity(&a).expect("identity");
    assert_eq!(id.segments().len(), 2, "one entry per representation");
    assert_eq!(id.segments()["target.expert_bank"], "aaaa");
    assert!(!id.artifact.as_str().is_empty() && !id.graph_hash().is_empty());

    // Re-reading the same container gives the same identity.
    assert_eq!(read_source_identity(&a).expect("again"), id);

    // A different GRAPH under byte-identical payload hashes is a
    // different model, and the identity says so.
    let b = dir.join("b");
    write(&b, &index_json, r#"{"components":[{"id":"target"}]}"#);
    let other = read_source_identity(&b).expect("identity");
    assert_eq!(other.segments(), id.segments(), "same payload hashes");
    assert_eq!(other.artifact, id.artifact, "same index bytes");
    assert_ne!(other.graph_hash(), id.graph_hash(), "different graph");
    assert_ne!(
        other.semantic_digest(),
        id.semantic_digest(),
        "and a different model"
    );

    // An index naming a non-default graph file is followed.
    let c = dir.join("c");
    std::fs::create_dir_all(&c).expect("dir");
    let renamed = index_json.replace("system_graph.json", "other.json");
    std::fs::write(c.join("index.json"), &renamed).expect("index");
    std::fs::write(c.join("other.json"), "{}").expect("graph");
    read_source_identity(&c).expect("a named graph file is followed");

    // A container with no index at all is refused, not defaulted.
    assert!(read_source_identity(&dir.join("absent")).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

/// The index is persisted atomically: a reader never sees a partial
/// file, and the temporary is not left behind.
#[test]
fn the_index_is_written_atomically() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("index.json");
    let idx = index(map_for(vec![]));
    write_index_atomically(&idx, &path).expect("write");
    let read: CandidateIndex =
        serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("parses");
    assert_eq!(read.model, idx.model);
    assert!(
        !path.with_extension("json.partial").exists(),
        "the temporary must be renamed away, not left beside the index"
    );
    // Overwriting an existing index is the resume path and must work.
    write_index_atomically(&idx, &path).expect("overwrite");
}

/// A tensor the compiler cannot place is refused by name — each of the
/// three ways a name or shape can fail to carry a bank position.
#[test]
fn a_tensor_that_cannot_be_placed_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (fake, _) = Fake::kimi_layer(1, 0.5);
    let mut idx = index(map_for(vec![Exception {
        projection: None,
        layers: Some((1, 1)),
        encoding: Some("Q6_K".into()),
    }]));

    // No layer/projection in the name at all.
    let err = compile_expert_bank(
        &fake,
        &[SourceTensor {
            name: "not_a_layer_tensor".into(),
            shape: vec![INTER, HIDDEN],
        }],
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect_err("an unplaceable tensor must be refused");
    assert!(format!("{err}").contains("no place"), "{err}");

    // A layer/projection but no expert index.
    let err = compile_expert_bank(
        &fake,
        &[SourceTensor {
            name: "1.block_sparse_moe.shared_experts.w1.weight".into(),
            shape: vec![INTER, HIDDEN],
        }],
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect_err("a tensor naming no expert must be refused");
    assert!(format!("{err}").contains("no expert index"), "{err}");

    // A shape that is not a matrix.
    let err = compile_expert_bank(
        &fake,
        &[SourceTensor {
            name: "1.block_sparse_moe.experts.0.w1.weight".into(),
            shape: vec![INTER],
        }],
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect_err("a non-matrix must be refused");
    assert!(format!("{err}").contains("not a matrix"), "{err}");

    // An expert beyond the declared population has no slot.
    let err = compile_expert_bank(
        &fake,
        &[SourceTensor {
            name: format!("1.block_sparse_moe.experts.{EXPERTS}.w1.weight"),
            shape: vec![INTER, HIDDEN],
        }],
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect_err("an out-of-range expert must be refused");
    assert!(format!("{err}").contains("outside"), "{err}");
}

/// `bank_base` places the three projections disjointly and refuses a
/// name that is not an expert projection at all.
#[test]
fn bank_base_covers_both_spellings_and_refuses_anything_else() {
    let layout = LayerBankLayout::new(1, "Q6_K", EXPERTS, HIDDEN, INTER).expect("layout");
    let gate_up = layout.bank_bytes("w1").expect("w1 bank");
    for (canonical, checkpoint) in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")] {
        assert_eq!(
            bank_base(&layout, canonical).expect("canonical"),
            bank_base(&layout, checkpoint).expect("checkpoint"),
            "`{canonical}` and `{checkpoint}` are the same bank"
        );
    }
    assert_eq!(bank_base(&layout, "w1").expect("w1"), 0);
    assert_eq!(bank_base(&layout, "w3").expect("w3"), gate_up);
    assert_eq!(bank_base(&layout, "w2").expect("w2"), 2 * gate_up);
    assert!(bank_base(&layout, "k_proj").is_err());
}

/// **The ledger reaches disk MID-RUN**, which is the whole of what
/// makes a long compilation resumable: an index written only at the end
/// gives a run killed at 60 % nothing to resume from.
#[test]
fn the_ledger_is_checkpointed_during_the_run_not_only_at_the_end() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let index_path = dir.path().join("index.json");
    let (fake, tensors) = Fake::kimi_layer(1, 0.5);
    let mut idx = index(map_for(vec![Exception {
        projection: None,
        layers: Some((1, 1)),
        encoding: Some("Q6_K".into()),
    }]));

    // Checkpoint every second seal, and read the index back DURING the
    // run — the observation only a mid-run write can satisfy.
    let mut seen_mid_run = 0usize;
    let opts = CompileOptions {
        checkpoint: Some((&index_path, 2)),
        ..opts("target.expert_bank", EXPERTS, &out)
    };
    let outcome = compile_expert_bank(&fake, &tensors, &opts, &mut idx, &mut |o| {
        if o.sealed > 0 && o.sealed < tensors.len() && index_path.exists() {
            let partial: CandidateIndex =
                serde_json::from_slice(&std::fs::read(&index_path).expect("read"))
                    .expect("a checkpointed index is always parseable");
            seen_mid_run = seen_mid_run.max(partial.ledger.sealed.len());
        }
    })
    .expect("compiles");
    assert_eq!(outcome.sealed, tensors.len());
    assert!(
        seen_mid_run > 0 && seen_mid_run < tensors.len(),
        "the index must be readable mid-run with a PARTIAL ledger, saw {seen_mid_run} of {}",
        tensors.len()
    );

    // And the final write leaves the complete ledger behind.
    let final_index: CandidateIndex =
        serde_json::from_slice(&std::fs::read(&index_path).expect("read")).expect("parses");
    assert_eq!(final_index.ledger.sealed.len(), tensors.len());
    assert!(final_index.ledger.overlaps().is_empty());
}

/// An encoder whose output does not fill the slot the layout reserved
/// is refused — the bytes would land inside a neighbouring expert.
#[test]
fn an_encoding_that_does_not_fill_its_slot_is_refused() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (mut fake, tensors) = Fake::kimi_layer(1, 0.5);
    // The SOURCE holds fewer values than the tensor's declared shape,
    // so the encoder produces a short block while the layout still
    // reserves a full one.
    let short = tensors[0].name.clone();
    fake.bytes.insert(short.clone(), vec![0u8; 512]);
    let mut idx = index(map_for(vec![Exception {
        projection: None,
        layers: Some((1, 1)),
        encoding: Some("Q6_K".into()),
    }]));
    let err = compile_expert_bank(
        &fake,
        &tensors[..1],
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect_err("a short encoding must be refused");
    let text = format!("{err}");
    assert!(
        text.contains(&short) && text.contains("layout reserved"),
        "{text}"
    );
}

/// An index that cannot be written is reported rather than swallowed —
/// a compilation that believes it checkpointed and did not would resume
/// from nothing.
#[test]
fn an_unwritable_index_path_is_reported() {
    let dir = tempfile::tempdir().expect("tmp");
    // A directory where the index file should be: the write fails, and
    // the failure must surface.
    let path = dir.path().join("index.json");
    std::fs::create_dir(&path).expect("occupy the path with a directory");
    let idx = index(map_for(vec![]));
    assert!(
        write_index_atomically(&idx, &path).is_err(),
        "a failed index write must be reported"
    );
}

/// **A composed map holds several layers in ONE candidate, each at its
/// own encoding, placed disjointly.** L2 at Q8_0 beside L3 at Q6_K:
/// every L2 seal lands inside [0, q8_layer_bytes), every L3 seal at
/// q8_layer_bytes onward, no overlaps, and the ledger's total is the
/// sum of the two encodings' own extents — nothing shared, nothing
/// invented.
#[test]
fn a_composed_map_compiles_two_layers_disjointly_each_at_its_own_encoding() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (fake2, t2) = Fake::kimi_layer(2, 0.3);
    let (fake3, t3) = Fake::kimi_layer(3, 1.7);
    let mut bytes = fake2.bytes;
    bytes.extend(fake3.bytes);
    let fake = Fake { bytes };
    let tensors: Vec<SourceTensor> = t2.into_iter().chain(t3).collect();

    let map = PrecisionMap {
        name: "composed-l2q80-l3q6k".into(),
        encoding: "Q8_0".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: None,
                layers: Some((2, 2)),
                encoding: Some("Q8_0".into()),
            },
            Exception {
                projection: None,
                layers: Some((3, 3)),
                encoding: Some("Q6_K".into()),
            },
            Exception {
                projection: None,
                layers: None,
                encoding: None,
            },
        ],
    };
    let mut idx = index(map);
    assert_eq!(
        idx.can_represent_as,
        vec!["Q8_0".to_string(), "Q6_K".to_string()],
        "a composed candidate states every encoding it can represent"
    );
    let outcome = compile_expert_bank(
        &fake,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect("composed compile");
    assert_eq!(outcome.sealed, 2 * 3 * EXPERTS as usize);
    assert!(idx.ledger.overlaps().is_empty(), "layers must not collide");

    // Per-matrix bytes from the format's own arithmetic.
    let q8_matrix = LayerBankLayout::matrix_bytes("Q8_0", INTER, HIDDEN).expect("q8");
    let q6_matrix = LayerBankLayout::matrix_bytes("Q6_K", INTER, HIDDEN).expect("q6");
    let q8_layer = 3 * q8_matrix * u64::from(EXPERTS);
    let q6_layer = 3 * q6_matrix * u64::from(EXPERTS);
    assert_eq!(idx.ledger.compiled_bytes(), q8_layer + q6_layer);

    for seal in idx.ledger.sealed.values() {
        let layer: u32 = seal.tensor.split('.').next().unwrap().parse().unwrap();
        let end = seal.target_offset + seal.target_len;
        match layer {
            2 => {
                assert_eq!(seal.encoding, "Q8_0", "{}", seal.tensor);
                assert!(end <= q8_layer, "{} spills past its layer", seal.tensor);
            }
            3 => {
                assert_eq!(seal.encoding, "Q6_K", "{}", seal.tensor);
                assert!(
                    seal.target_offset >= q8_layer && end <= q8_layer + q6_layer,
                    "{} is outside its layer's band",
                    seal.tensor
                );
            }
            other => panic!("unexpected layer {other}"),
        }
    }
}

/// **A rerun handed a SUBSET of the composed scope must refuse, never
/// silently relocate.** The placement derives layer bases from the
/// handed scope; re-running with only layer 3's tensors would put its
/// bank at base 0 — on top of layer 2's bytes. The seal-offset guard
/// refuses by name instead.
#[test]
fn a_rerun_over_a_subset_of_the_composed_scope_is_refused() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (fake2, t2) = Fake::kimi_layer(2, 0.3);
    let (fake3, t3) = Fake::kimi_layer(3, 1.7);
    let mut bytes = fake2.bytes;
    bytes.extend(fake3.bytes);
    let fake = Fake { bytes };
    let both: Vec<SourceTensor> = t2.into_iter().chain(t3.iter().cloned()).collect();

    let map = PrecisionMap {
        name: "composed-subset-guard".into(),
        encoding: "Q8_0".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: None,
                layers: Some((2, 3)),
                encoding: Some("Q8_0".into()),
            },
            Exception {
                projection: None,
                layers: None,
                encoding: None,
            },
        ],
    };
    let mut idx = index(map);
    compile_expert_bank(
        &fake,
        &both,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect("full-scope compile");

    let err = compile_expert_bank(
        &fake,
        &t3,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect_err("a subset scope recomputes every base and must refuse");
    let text = format!("{err}");
    assert!(
        text.contains("placement") && text.contains("full layer scope"),
        "the refusal must explain the scope mismatch: {text}"
    );
}

/// `CandidatePlacement` itself: single-layer placement is base 0
/// (byte-compatible with every existing candidate), an unmapped layer
/// and a per-projection-mixed layer are refused by name.
#[test]
fn placement_bases_and_refusals() {
    let single = PrecisionMap {
        name: "one".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![],
    };
    let p = CandidatePlacement::resolve(&single, Role::ExpertWeight, &[7], EXPERTS, HIDDEN, INTER)
        .expect("single layer places");
    assert_eq!(p.layer_base(7).expect("base"), 0, "single layer = base 0");
    assert!(p.layer_base(8).is_err(), "an unplaced layer is refused");

    // Two layers, one encoding: second base = first extent.
    let p2 =
        CandidatePlacement::resolve(&single, Role::ExpertWeight, &[7, 9], EXPERTS, HIDDEN, INTER)
            .expect("two layers place");
    let extent = p2.layout(7).expect("layout").layer_bytes().expect("bytes");
    assert_eq!(p2.layer_base(9).expect("base"), extent);

    // A layer the map does not compile cannot be placed.
    let scoped = PrecisionMap {
        name: "scoped".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: None,
                layers: Some((7, 7)),
                encoding: Some("Q6_K".into()),
            },
            Exception {
                projection: None,
                layers: None,
                encoding: None,
            },
        ],
    };
    let err =
        CandidatePlacement::resolve(&scoped, Role::ExpertWeight, &[8], EXPERTS, HIDDEN, INTER)
            .expect_err("an out-of-scope layer must refuse");
    assert!(format!("{err}").contains("compiles none"), "{err}");

    // Mixed encodings WITHIN one layer are not placeable: identity
    // addressing carries one stride.
    let mixed = PrecisionMap {
        name: "mixed".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: Some("w1".into()),
                layers: Some((7, 7)),
                encoding: Some("Q8_0".into()),
            },
            Exception {
                projection: None,
                layers: Some((7, 7)),
                encoding: Some("Q6_K".into()),
            },
        ],
    };
    let err = CandidatePlacement::resolve(&mixed, Role::ExpertWeight, &[7], EXPERTS, HIDDEN, INTER)
        .expect_err("per-projection mixing within a layer must refuse");
    assert!(format!("{err}").contains("ONE encoding"), "{err}");
}
