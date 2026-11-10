//! What it means to verify a VINDEX3 container.
//!
//! For the shipped generation, `verify` is a checksum sweep: the files are
//! opaque blobs and "intact" means "the bytes are the bytes". A VINDEX3
//! container declares its own structure, so verification can ask a much
//! stronger question — *is this container bindable?* — without executing
//! anything:
//!
//! ```text
//! index parses                        ← Vindex3Container::open
//! manifest parses and validates       ← Vindex3Container::open
//! storage keys resolve to files       ← Vindex3Container::open
//! each segment parses as LYRW v2      ← here
//! declared roles satisfy the programme ← here
//! every entry's regions are in bounds  ← here
//! ```
//!
//! The last three are the ones that turn "the files exist" into "this will
//! bind". A container that passes cannot fail at bind time for a structural
//! reason, which is the whole point of declaring structure up front.
//!
//! Execution parity is deliberately **not** here. It needs weights, an input
//! and a kernel, and folding it in would make routine verification cost a
//! forward pass. It belongs in a deeper mode.

use std::path::Path;

use super::index::Vindex3Index;
use super::read::Vindex3Container;
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use crate::format::lyrw2::region_role::RegionRole;

/// Validate that an on-disk directory is a **complete** VINDEX3
/// container — the "validate as VINDEX3" stage of the pull/registry
/// convergence pipeline (`docs/vindex3-registry-design.md` §10.5),
/// deliberately kept separate from identity resolution and the download
/// itself.
///
/// The container's own `index.json` decides which structural loader
/// applies — never guessed: `moe_manifest` present means a routed-MoE
/// container, opened (and thereby structurally validated) via
/// [`Vindex3Container::open`]; absent means a system-graph container,
/// validated via [`super::inspect::inspect_container`] with payload
/// verification. `Vindex3Container::open` fails closed on anything
/// incomplete or malformed. `inspect_container` does NOT: it returns a
/// hash mismatch, a size mismatch or an incoherent directory as a
/// DEFECT in its inspection, and only I/O and parse failures as errors,
/// so this function refuses any inspection that is not coherent. Before
/// it did, a corrupt segment passed `larql pull` and registry resolution.
pub fn validate_downloaded_container(dir: &Path) -> Result<(), VindexError> {
    check_container_structure(dir, true)
}

/// [`validate_downloaded_container`]'s dispatch with payload hashing made
/// optional: `verify_payloads = false` answers "can this build open this
/// container?" from `index.json`, the graph and segment headers alone —
/// cheap enough for a cache listing (`larql list`), which must say a
/// container is unusable (an old graph schema, a legacy bank shape) rather
/// than list it as if it would run. `true` additionally re-hashes every
/// segment, which is what a download validator owes its caller.
pub fn check_container_structure(dir: &Path, verify_payloads: bool) -> Result<(), VindexError> {
    let index_text = std::fs::read_to_string(dir.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&index_text)
        .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    if index.moe_manifest.is_some() {
        Vindex3Container::open(dir)?;
    } else {
        let inspection = super::inspect::inspect_container(dir, verify_payloads)?;
        if let Some(first) = inspection.defects.first() {
            // The count and the first defect: enough to act on, and one line
            // a listing can show. `vindex3 inspect` prints the full set.
            return Err(VindexError::Parse(format!(
                "container {} is not coherent: {} defect(s), first {first:?}",
                dir.display(),
                inspection.defects.len(),
            )));
        }
    }
    Ok(())
}

/// A structural defect that would stop this container binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainerDefect {
    /// A layer's bank names storage the index does not declare, or that does
    /// not parse as LYRW v2.
    Storage {
        layer: u32,
        key: String,
        why: String,
    },
    /// The manifest names a programme this binary does not implement.
    UnknownProgramme { layer: u32, programme: String },
    /// The bank's regions cannot satisfy any of the programme's acceptable
    /// operand sets.
    UnsatisfiedProgramme {
        layer: u32,
        programme: String,
        missing: Vec<RegionRole>,
    },
    /// A declared region does not resolve for some entry.
    ///
    /// Carries the full coordinate V2-0 asks for — `{layer, bank, role,
    /// segment}` — plus the entry. Layer alone cannot locate a region in a
    /// container where one layer may span several banks and each bank several
    /// segments, and a diagnosis that cannot be navigated to is a diagnosis
    /// someone has to reproduce before they can act on it.
    MissingRegion {
        layer: u32,
        bank: u16,
        entry: u32,
        role: RegionRole,
        segment: String,
    },
}

impl std::fmt::Display for ContainerDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage { layer, key, why } => {
                write!(f, "layer {layer}: storage '{key}' unusable: {why}")
            }
            Self::UnknownProgramme { layer, programme } => write!(
                f,
                "layer {layer}: programme '{programme}' is not implemented by this binary"
            ),
            Self::UnsatisfiedProgramme {
                layer,
                programme,
                missing,
            } => write!(
                f,
                "layer {layer}: programme '{programme}' cannot be satisfied; missing {missing:?}"
            ),
            Self::MissingRegion {
                layer,
                bank,
                entry,
                role,
                segment,
            } => write!(
                f,
                "layer {layer} bank {bank} segment '{segment}': entry {entry} has no {role:?} region"
            ),
        }
    }
}

impl Vindex3Container {
    /// Every structural defect, in layer order. Empty means bindable.
    ///
    /// Returns all of them rather than the first: a container missing three
    /// roles should say so once, not across three re-runs.
    pub fn verify(&self) -> Vec<ContainerDefect> {
        let mut defects = Vec::new();
        for layer in &self.manifest().layers {
            let bank_ref = &layer.routed_bank;
            let Some(programme) = bank_ref.resolve_programme() else {
                defects.push(ContainerDefect::UnknownProgramme {
                    layer: layer.layer,
                    programme: bank_ref.programme.clone(),
                });
                continue;
            };
            let segment = match self.segment(&bank_ref.storage) {
                Ok(s) => s,
                Err(e) => {
                    defects.push(ContainerDefect::Storage {
                        layer: layer.layer,
                        key: bank_ref.storage.clone(),
                        why: e.to_string(),
                    });
                    continue;
                }
            };

            let Some(bank) = segment.banks().first().copied() else {
                defects.push(ContainerDefect::Storage {
                    layer: layer.layer,
                    key: bank_ref.storage.clone(),
                    why: "segment declares no bank".into(),
                });
                continue;
            };
            let schemas = segment.schemas_for(bank.bank_id).unwrap_or(&[]);
            let present: Vec<RegionRole> = schemas.iter().map(|s| s.role).collect();

            if !programme.is_satisfied_by(&present) {
                defects.push(ContainerDefect::UnsatisfiedProgramme {
                    layer: layer.layer,
                    programme: bank_ref.programme.clone(),
                    missing: programme.missing_roles(&present),
                });
                continue;
            }

            // Bounds are checked by resolving every declared region for every
            // entry — the reader refuses an out-of-range offset, so a
            // successful resolve is the bounds check.
            // Every *declared schema*, not every distinct role. A bank whose
            // roles are unique makes these the same walk; a split-scale bank
            // does not — it declares two `Scales` regions, and a role-driven
            // walk both visits one of them twice and never reaches the other.
            // Resolving by schema index also keeps this independent of the
            // ambiguity that role lookup now refuses.
            for entry in 0..bank.num_entries {
                for schema in schemas {
                    let resolved = segment
                        .resolve_by_schema_index(bank.bank_id, entry, schema.schema_index)
                        .ok()
                        .flatten();
                    if resolved.is_none() {
                        defects.push(ContainerDefect::MissingRegion {
                            layer: layer.layer,
                            bank: bank.bank_id,
                            entry,
                            role: schema.role,
                            segment: bank_ref.storage.clone(),
                        });
                    }
                }
            }
        }
        defects
    }

    /// Whether the container is structurally bindable.
    pub fn is_bindable(&self) -> bool {
        self.verify().is_empty()
    }
}

#[cfg(test)]
mod validate_downloaded_container_tests {
    use super::validate_downloaded_container;
    use crate::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};
    use crate::format::vindex3::test_support::fixture_a_spec;
    use crate::format::vindex3::write::write_container;

    #[test]
    fn a_complete_routed_moe_container_validates() {
        let dir = tempfile::tempdir().unwrap();
        write_container(dir.path(), &fixture_a_spec()).expect("write");
        validate_downloaded_container(dir.path()).expect("a well-formed MoE container validates");
    }

    #[test]
    fn a_complete_system_graph_container_validates() {
        let checkpoint = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(
            miniature_glimmer,
            checkpoint.path(),
            container.path(),
            "validate-fixture",
        );
        validate_downloaded_container(container.path())
            .expect("a well-formed system-graph container validates");
    }

    #[test]
    fn a_routed_moe_container_missing_a_segment_file_fails_validation() {
        // The completeness contract this function exists for: an
        // incomplete download (or a truncated/corrupt upload) must not
        // silently pass as "downloaded fine."
        let dir = tempfile::tempdir().unwrap();
        let spec = fixture_a_spec();
        write_container(dir.path(), &spec).expect("write");
        let segment_key = spec.segments[0].key.clone();
        std::fs::remove_file(crate::format::vindex3::segment_path(
            dir.path(),
            &segment_key,
        ))
        .expect("remove segment");

        let err = validate_downloaded_container(dir.path())
            .expect_err("a container missing a declared segment must fail validation");
        assert!(err.to_string().contains(&segment_key), "{err}");
    }

    /// A graph container whose segment bytes no longer match the declared
    /// hash must fail validation. The inspection reported this as a
    /// defect, and the validator used to discard it.
    #[test]
    fn a_system_graph_container_with_a_corrupt_segment_fails_validation() {
        let checkpoint = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(
            miniature_glimmer,
            checkpoint.path(),
            container.path(),
            "validate-fixture",
        );
        let index: crate::format::vindex3::index::Vindex3Index = serde_json::from_str(
            &std::fs::read_to_string(container.path().join(crate::format::filenames::INDEX_JSON))
                .unwrap(),
        )
        .unwrap();
        let entry = index
            .representations
            .values()
            .next()
            .expect("the fixture declares a representation");
        let segment = container.path().join(&entry.segment);
        let mut bytes = std::fs::read(&segment).unwrap();
        // Same length, one bit different: only the hash can see it.
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&segment, &bytes).unwrap();

        let err = validate_downloaded_container(container.path())
            .expect_err("a corrupt segment must fail validation");
        assert!(err.to_string().contains("PayloadCorrupt"), "{err}");
    }

    /// The listing-grade check (`verify_payloads = false`) reads structure
    /// only: a payload flip that only a hash can see passes, while a graph
    /// written at a schema this build does not read is refused by name.
    #[test]
    fn the_structural_check_skips_payload_hashes_but_refuses_an_old_graph_schema() {
        use crate::format::vindex3::graph::GRAPH_SCHEMA;

        let checkpoint = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(
            miniature_glimmer,
            checkpoint.path(),
            container.path(),
            "structure-fixture",
        );
        super::check_container_structure(container.path(), false)
            .expect("a well-formed container is structurally openable");

        let index: crate::format::vindex3::index::Vindex3Index = serde_json::from_str(
            &std::fs::read_to_string(container.path().join(crate::format::filenames::INDEX_JSON))
                .unwrap(),
        )
        .unwrap();
        let graph_path = container
            .path()
            .join(index.system_graph.as_deref().expect("a graph container"));
        let mut graph: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
        let stale = GRAPH_SCHEMA - 1;
        graph["schema"] = serde_json::json!(stale);
        std::fs::write(&graph_path, serde_json::to_string(&graph).unwrap()).unwrap();

        let err = super::check_container_structure(container.path(), false)
            .expect_err("an old graph schema is not openable");
        let msg = err.to_string();
        assert!(msg.contains(&format!("schema v{stale}")), "{msg}");
        assert!(msg.contains(&format!("reads v{GRAPH_SCHEMA}")), "{msg}");
    }

    #[test]
    fn a_directory_with_no_index_json_fails_validation() {
        let dir = tempfile::tempdir().unwrap();
        let err = validate_downloaded_container(dir.path())
            .expect_err("a directory with no index.json cannot validate");
        assert!(matches!(err, super::VindexError::Io(_)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::vindex3::test_support::{fixture_a_fused_spec, fixture_a_spec};
    use crate::format::vindex3::write::write_container;

    fn verified(spec: &crate::format::vindex3::ContainerSpec) -> Vec<ContainerDefect> {
        let dir = std::env::temp_dir().join(format!(
            "vindex3-verify-{}-{:p}",
            std::process::id(),
            spec as *const _
        ));
        let _ = std::fs::remove_dir_all(&dir);
        write_container(&dir, spec).expect("write");
        let defects = Vindex3Container::open(&dir).expect("open").verify();
        let _ = std::fs::remove_dir_all(&dir);
        defects
    }

    #[test]
    fn a_well_formed_container_reports_no_defects() {
        assert_eq!(verified(&fixture_a_spec()), Vec::new());
    }

    #[test]
    fn the_fused_rendering_is_equally_bindable() {
        // Both operand sets satisfy `gated-mlp-v1`, so verification must
        // accept either — otherwise the programme's alternatives are a
        // fiction the verifier does not honour.
        assert_eq!(verified(&fixture_a_fused_spec()), Vec::new());
    }

    #[test]
    fn an_unimplemented_programme_is_refused_at_open_not_left_to_verify() {
        // This test used to accept either outcome — open refusing, or open
        // tolerating and verify catching it. That is what let a false comment
        // survive: `parse` did not in fact refuse, and nothing failed. Assert
        // the guarantee we actually want, so the weaker one cannot come back.
        let mut spec = fixture_a_spec();
        spec.manifest.layers[0].routed_bank.programme = "no-such-programme-v9".into();
        let dir = std::env::temp_dir().join(format!("vindex3-badprog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_container(&dir, &spec).expect("write");
        let opened = Vindex3Container::open(&dir);
        let _ = std::fs::remove_dir_all(&dir);

        let err = opened
            .err()
            .expect("a manifest naming an unimplemented programme must not open");
        assert!(
            format!("{err}").contains("no-such-programme-v9"),
            "the refusal must name the programme: {err}"
        );
    }

    #[test]
    fn an_unimplemented_programme_is_describable_even_though_open_refuses_it() {
        // The two halves of the same guarantee. `open` refuses the container
        // so it can never be served; `open_unchecked` still loads it so
        // `larql verify` can say *why*. Without the second, the refusal is
        // correct and unactionable — and `ContainerDefect::UnknownProgramme`
        // becomes unreachable, which is how this arm lost its coverage.
        let mut spec = fixture_a_spec();
        spec.manifest.layers[0].routed_bank.programme = "no-such-programme-v9".into();
        let dir = std::env::temp_dir().join(format!("vindex3-describe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_container(&dir, &spec).expect("write");

        let refused = Vindex3Container::open(&dir);
        let described = Vindex3Container::open_unchecked(&dir);
        let defects = described.as_ref().map(|c| c.verify());
        let _ = std::fs::remove_dir_all(&dir);

        assert!(refused.is_err(), "serving must refuse it");
        let defects = defects.expect("describing must still load it");
        assert!(
            defects.iter().any(|d| matches!(
                d,
                ContainerDefect::UnknownProgramme { programme, .. }
                    if programme == "no-such-programme-v9"
            )),
            "verify must name the programme: {defects:?}"
        );
    }

    /// Write a container from `spec` into a private dir and verify it.
    fn verify_spec(
        tag: &str,
        spec: &crate::format::vindex3::ContainerSpec,
    ) -> Vec<ContainerDefect> {
        let dir = std::env::temp_dir().join(format!("v3-verify-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_container(&dir, spec).expect("write");
        let defects = Vindex3Container::open(&dir).expect("open").verify();
        let _ = std::fs::remove_dir_all(&dir);
        defects
    }

    #[test]
    fn a_segment_that_does_not_parse_is_a_storage_defect() {
        // `open` reads segment files without parsing them, so an unusable
        // segment survives loading and must be caught here — otherwise it
        // would first surface at bind time as an unexplained failure.
        use crate::format::vindex3::test_support::spec_with_segment_bytes;
        let defects = verify_spec("garbage", &spec_with_segment_bytes(vec![0xAB; 512]));
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ContainerDefect::Storage { .. })),
            "expected a Storage defect, got {defects:?}"
        );
        let msg = defects[0].to_string();
        assert!(msg.contains("routed/layer_000"), "must name the key: {msg}");
    }

    #[test]
    fn a_bank_missing_a_required_role_names_what_is_missing() {
        // Gate + up but no down: well-formed LYRW v2 that no gated-MLP
        // alternative can be satisfied by. The defect must say `Down` rather
        // than "unsatisfied", or the reader has to diff the alternatives by
        // hand to learn what to add.
        use crate::format::vindex3::test_support::{
            gate_only_segment_bytes, spec_with_segment_bytes,
        };
        let defects = verify_spec(
            "noduown",
            &spec_with_segment_bytes(gate_only_segment_bytes()),
        );
        let missing = defects
            .iter()
            .find_map(|d| match d {
                ContainerDefect::UnsatisfiedProgramme { missing, .. } => Some(missing.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("expected UnsatisfiedProgramme, got {defects:?}"));
        assert!(
            missing.contains(&RegionRole::Down),
            "the missing role must be named: {missing:?}"
        );
    }

    #[test]
    fn is_bindable_agrees_with_the_defect_list() {
        use crate::format::vindex3::test_support::spec_with_segment_bytes;
        let dir = std::env::temp_dir().join(format!("v3-bindable-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_container(&dir, &fixture_a_spec()).expect("write good");
        assert!(Vindex3Container::open(&dir).expect("open").is_bindable());
        let _ = std::fs::remove_dir_all(&dir);

        write_container(&dir, &spec_with_segment_bytes(vec![0u8; 64])).expect("write bad");
        assert!(!Vindex3Container::open(&dir).expect("open").is_bindable());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_defect_variant_renders_its_own_coordinates() {
        // Display is the operator-facing surface; a variant that renders
        // without its coordinates is a defect nobody can navigate to.
        let rendered = [
            ContainerDefect::Storage {
                layer: 1,
                key: "routed/layer_001".into(),
                why: "truncated".into(),
            },
            ContainerDefect::UnknownProgramme {
                layer: 2,
                programme: "made-up-v1".into(),
            },
            ContainerDefect::UnsatisfiedProgramme {
                layer: 4,
                programme: "gated-mlp-v1".into(),
                missing: vec![RegionRole::Down],
            },
        ]
        .map(|d| d.to_string());
        assert!(rendered[0].contains("routed/layer_001") && rendered[0].contains("truncated"));
        assert!(rendered[1].contains("made-up-v1") && rendered[1].contains("layer 2"));
        assert!(rendered[2].contains("Down") && rendered[2].contains("gated-mlp-v1"));
    }

    #[test]
    fn a_defect_renders_with_the_coordinates_needed_to_fix_it() {
        // V2-0 asks for {layer, bank, role, segment}; the entry is the fifth
        // coordinate a routed bank needs to be navigable.
        let d = ContainerDefect::MissingRegion {
            layer: 3,
            bank: 1,
            entry: 7,
            role: RegionRole::Down,
            segment: "routed/layer_003".into(),
        };
        let msg = d.to_string();
        assert!(msg.contains("layer 3"), "{msg}");
        assert!(msg.contains("bank 1"), "{msg}");
        assert!(msg.contains("entry 7"), "{msg}");
        assert!(msg.contains("Down"), "{msg}");
        assert!(msg.contains("routed/layer_003"), "{msg}");
    }
}
