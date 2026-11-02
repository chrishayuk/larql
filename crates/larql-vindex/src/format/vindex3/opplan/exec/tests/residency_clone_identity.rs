//! NB6 on a real container: the same encoded container at two paths — the
//! shape of the cold-state method's fresh clone per arm — gives every
//! mapped region the same key, although every segment file sits at a
//! different absolute path.
//!
//! The key alone does not license the comparison. The test first
//! establishes, as its stated precondition, that both copies carry the
//! same container identity (equal `index.json` digests) — the check any
//! procedure pairing records across clones must make — and only then
//! pairs the regions.

use std::collections::BTreeSet;
use std::path::Path;

use super::super::prepared::{ExecutionSlice, PreparedOperands, ResidencyDelta};
use super::super::production::ProductionBackend;
use super::kimi_per_expert_prepared::{open_container, Subject};
use crate::format::checksums::sha256_file;
use crate::format::vindex3::fixtures_kimi::kimi_per_expert_moe_f32_model;

/// A byte copy of the directory tree at `from` into `to`.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

#[test]
fn a_copy_of_a_container_at_another_path_pairs_every_region() {
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let elsewhere = tempfile::tempdir().unwrap();
    let clone = elsewhere.path().join("moved.vindex3");
    copy_tree(subject.container(), &clone);

    // Precondition, checked rather than assumed: one container, two paths.
    let digest = |dir: &Path| sha256_file(&dir.join("index.json")).expect("index digest");
    assert_eq!(
        digest(subject.container()),
        digest(&clone),
        "pairing across clones requires equal container identity"
    );

    let backend = ProductionBackend::new();
    let observe = |dir: &Path| {
        let (plan, store) = open_container(dir);
        let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
        ops.observe_residency()
    };
    let a = observe(subject.container());
    let b = observe(&clone);
    assert!(!a.regions.is_empty(), "the fixture maps its expert bank");
    assert_eq!(a.regions.len(), b.regions.len());

    for (ra, rb) in a.regions.iter().zip(&b.regions) {
        let (sa, sb) = (&a.stores[ra.store], &b.stores[rb.store]);
        assert_ne!(sa.source_path, sb.source_path, "the files really moved");
        // The declared segment is the file's path inside its container.
        let declared = sa.declared_segment.as_deref().expect("declared");
        let path = sa.source_path.as_ref().unwrap();
        assert_eq!(
            path.strip_prefix(subject.container()).unwrap(),
            Path::new(declared)
        );
        let key = a.key_of(ra);
        assert!(key.segment.is_portable());
        assert_eq!(key, b.key_of(rb), "same region, same key, either clone");
    }

    // Distinct regions keep distinct keys: nothing collides.
    let keys: BTreeSet<_> = a.regions.iter().map(|r| a.key_of(r)).collect();
    assert_eq!(keys.len(), a.regions.len());

    let d = ResidencyDelta::between(&a, &b);
    #[cfg(unix)]
    assert_eq!(d.comparable, a.regions.len(), "every region pairs");
    assert_eq!(d.excluded + d.comparable, a.regions.len());
}
