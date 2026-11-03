//! MEM-002 I2b: every payload byte comes through the descriptor the store
//! opened, and that descriptor's identity is what gets reported.
//!
//! The path is deliberately broken after open in several of these: reads
//! that still return the original bytes could not have used it.

use std::path::PathBuf;

use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::{
    FileIdentity, OperandStore, RepresentationSource,
};
use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention, OperandRef};

/// Threads reading one descriptor at once. Enough to interleave
/// seek-then-read, were it still used.
const READERS: usize = 8;
/// Reads per thread.
const READS_PER_READER: usize = 200;
/// A clock step every supported filesystem's timestamps can show.
#[cfg(unix)]
const TIMESTAMP_STEP: std::time::Duration = std::time::Duration::from_millis(50);

struct Dense {
    _src: tempfile::TempDir,
    container: tempfile::TempDir,
}

impl Dense {
    fn build() -> Self {
        let src = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(dense_f32_model, src.path(), container.path(), "dense");
        Self {
            _src: src,
            container,
        }
    }

    /// The store, layer 0's query projection, and its segment file.
    fn open(&self) -> (OperandStore, OperandRef, PathBuf) {
        let inspection = inspect_container(self.container.path(), false).unwrap();
        let plan = plan_component_ops(&inspection, self.container.path(), "target")
            .unwrap()
            .plan
            .expect("the dense fixture plans");
        let LayerAttention::Softmax(op) = &plan.layers[0].attention else {
            panic!("the dense fixture's layer 0 is softmax attention");
        };
        let store = OperandStore::open_for(
            self.container.path(),
            &inspection,
            None,
            RepresentationSource::Auto,
        )
        .unwrap();
        let operand = op.q.clone();
        let path = segment_path(&store, &operand.object);
        (store, operand, path)
    }
}

fn segment_path(store: &OperandStore, object: &str) -> PathBuf {
    store
        .instance_report()
        .segments
        .into_iter()
        .find(|s| s.object == object)
        .expect("the operand's object has a segment")
        .path
}

/// The bytes of `operand` as a separate reader of the file finds them.
fn independently_read(path: &std::path::Path, operand: &OperandRef) -> Vec<u8> {
    let (header, payload_start) = read_segment_header(path).unwrap();
    let tensor = header
        .tensors
        .iter()
        .find(|t| t.name == operand.tensor)
        .expect("the tensor is in the header");
    let start = (payload_start + tensor.offset) as usize;
    std::fs::read(path).unwrap()[start..start + tensor.len as usize].to_vec()
}

#[test]
fn copied_ranged_and_mapped_bytes_are_the_files_bytes() {
    let dense = Dense::build();
    let (store, q, path) = dense.open();
    let expected = independently_read(&path, &q);
    let total = expected.len() as u64;
    assert!(total > 1, "the fixture's operand has bytes to split");

    assert_eq!(store.load_raw(&q).unwrap().bytes, expected);
    let (head, tail) = expected.split_at(expected.len() / 2);
    assert_eq!(
        store
            .load_raw_range(&q, total, 0, head.len() as u64)
            .unwrap()
            .bytes,
        head
    );
    assert_eq!(
        store
            .load_raw_range(&q, total, head.len() as u64, tail.len() as u64)
            .unwrap()
            .bytes,
        tail
    );
    let region = store.map_region(&q, total).unwrap();
    assert_eq!(region.bytes(), &expected[..]);
    assert_eq!(region.source_path(), Some(path.as_path()));
    assert_eq!(store.mapped_regions(), 1);
    assert_eq!(store.mapped_objects(), 1);
}

/// One descriptor, many threads, different offsets. With a shared seek
/// position this would return another reader's bytes and still succeed.
#[test]
fn concurrent_reads_through_one_descriptor_get_their_own_bytes() {
    let dense = Dense::build();
    let (store, q, path) = dense.open();
    let expected = independently_read(&path, &q);
    let total = expected.len() as u64;
    std::thread::scope(|scope| {
        for reader in 0..READERS {
            let (store, q, expected) = (&store, &q, &expected);
            scope.spawn(move || {
                for read in 0..READS_PER_READER {
                    let offset = ((reader * 31 + read * 17) % expected.len()) as u64;
                    let len = (total - offset).min(1 + (read as u64 % 64));
                    let got = store.load_raw_range(q, total, offset, len).unwrap().bytes;
                    let start = offset as usize;
                    assert_eq!(got, expected[start..start + len as usize]);
                }
            });
        }
    });
}

#[test]
fn every_selected_segment_reports_how_execution_used_it() {
    let dense = Dense::build();
    let (store, q, _) = dense.open();
    let before = store.instance_report();
    assert!(!before.segments.is_empty());
    assert_eq!(before.segments.len(), store.selection().len());
    assert!(before.segments.iter().all(|s| !s.mapped && !s.copied));
    assert_eq!(before.load_count, 0);
    assert_eq!(before.bytes_read, 0);
    assert!(before.absent_objects.is_empty());

    let len = store.stored_len(&q).unwrap();
    store.load_raw(&q).unwrap();
    let copied = store.instance_report();
    let used = |report: &crate::format::vindex3::opplan::exec::operands::StoreInstance| {
        report
            .segments
            .iter()
            .find(|s| s.object == q.object)
            .cloned()
            .unwrap()
    };
    assert!(used(&copied).copied);
    assert!(!used(&copied).mapped);
    assert_eq!(copied.load_count, 1);
    assert_eq!(copied.bytes_read, len);

    store.map_region(&q, len).unwrap();
    let both = store.instance_report();
    assert!(used(&both).copied && used(&both).mapped);
    assert_eq!(both.mapped_regions, 1);
    assert_eq!(both.mapped_objects, 1);

    // A byte range is a copied read too.
    let (store, q, _) = dense.open();
    store.load_raw_range(&q, len, 0, 1).unwrap();
    let ranged = store.instance_report();
    assert!(ranged
        .segments
        .iter()
        .any(|s| s.object == q.object && s.copied));
}

#[cfg(unix)]
#[test]
fn nothing_changes_while_nothing_touches_the_file() {
    let dense = Dense::build();
    let (store, q, _) = dense.open();
    store.load_raw(&q).unwrap();
    for segment in store.instance_report().segments {
        assert_eq!(segment.changed_fields(), Vec::<&str>::new());
        assert_eq!(segment.capture_failure(), None);
    }
}

#[test]
fn a_comparison_ignores_the_device_and_names_each_other_field() {
    let base = FileIdentity {
        platform: "test",
        volume_uuid: Some("V".into()),
        volume_uuid_source: None,
        inode: 1,
        size: 2,
        mtime_ns: 3,
        ctime_ns: 4,
        nlink: 1,
        device: 9,
    };
    let device_only = FileIdentity {
        device: 10,
        ..base.clone()
    };
    assert_eq!(device_only.changed_fields(&base), Vec::<&str>::new());
    let all = FileIdentity {
        volume_uuid: None,
        inode: 11,
        size: 12,
        mtime_ns: 13,
        ctime_ns: 14,
        nlink: 0,
        ..base.clone()
    };
    assert_eq!(
        all.changed_fields(&base),
        vec![
            "volume_uuid",
            "inode",
            "size",
            "mtime_ns",
            "ctime_ns",
            "nlink"
        ]
    );
}

/// A failed capture is a failure, never a comparison that found nothing.
#[test]
fn a_failed_capture_is_reported_and_compares_as_nothing() {
    let identity = FileIdentity {
        platform: "test",
        volume_uuid: None,
        volume_uuid_source: None,
        inode: 1,
        size: 2,
        mtime_ns: 3,
        ctime_ns: 4,
        nlink: 1,
        device: 5,
    };
    let dense = Dense::build();
    let (store, _, _) = dense.open();
    let mut segment = store.instance_report().segments.remove(0);
    segment.at_open = Ok(identity.clone());
    segment.now = Err("gone".to_string());
    assert_eq!(segment.capture_failure(), Some("gone"));
    assert_eq!(segment.changed_fields(), Vec::<&str>::new());
    segment.now = Ok(identity);
    segment.at_open = Err("never".to_string());
    assert_eq!(segment.capture_failure(), Some("never"));
    assert_eq!(segment.changed_fields(), Vec::<&str>::new());
}

/// A described segment that is not on disk stays absent, and one that
/// cannot be read stays an error: the same sorting as before the open moved.
#[test]
fn a_missing_segment_is_absent_and_a_broken_one_is_refused() {
    let dense = Dense::build();
    let (store, q, path) = dense.open();
    let object = q.object.clone();
    drop(store);

    let keep = path.with_extension("kept");
    std::fs::rename(&path, &keep).unwrap();
    let (absent_store, _) = {
        let inspection = inspect_container(dense.container.path(), false).unwrap();
        (
            OperandStore::open_for(
                dense.container.path(),
                &inspection,
                None,
                RepresentationSource::Auto,
            )
            .expect("a missing segment does not stop the open"),
            (),
        )
    };
    let report = absent_store.instance_report();
    assert!(report.absent_objects.contains(&object));
    assert!(report.segments.iter().all(|s| s.object != object));
    assert!(absent_store
        .load_raw(&q)
        .err()
        .expect("an absent segment cannot load")
        .to_string()
        .contains("not hydrated"));
    drop(absent_store);
    std::fs::rename(&keep, &path).unwrap();

    // Truncated inside the length prefix: it exists and cannot be read.
    let inspection = inspect_container(dense.container.path(), false).unwrap();
    std::fs::write(&path, [0u8; 3]).unwrap();
    assert!(OperandStore::open_for(
        dense.container.path(),
        &inspection,
        None,
        RepresentationSource::Auto
    )
    .is_err());
}

#[cfg(unix)]
mod unix {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    fn only_segment_identity(store: &OperandStore, object: &str) -> (FileIdentity, FileIdentity) {
        let segment = store
            .instance_report()
            .segments
            .into_iter()
            .find(|s| s.object == object)
            .unwrap();
        (segment.at_open.unwrap(), segment.now.unwrap())
    }

    #[test]
    fn the_identity_is_the_opened_files() {
        let dense = Dense::build();
        let (store, q, path) = dense.open();
        let (at_open, now) = only_segment_identity(&store, &q.object);
        let on_disk = std::fs::metadata(&path).unwrap();
        assert_eq!(at_open, now);
        assert_eq!(at_open.inode, on_disk.ino());
        assert_eq!(at_open.size, on_disk.len());
        assert_eq!(at_open.nlink, on_disk.nlink());
        assert_eq!(at_open.device, on_disk.dev());
        assert_eq!(at_open.platform, std::env::consts::OS);
    }

    /// What a rename-over actually does to a held descriptor: the inode
    /// stays, the ctime moves, the link count falls to 0 and the path names
    /// another file. The reads still come from the opened file.
    #[test]
    fn a_file_replaced_by_rename_is_seen_on_the_descriptor_and_not_read() {
        let dense = Dense::build();
        let (store, q, path) = dense.open();
        let original = independently_read(&path, &q);
        let total = original.len() as u64;
        let (at_open, _) = only_segment_identity(&store, &q.object);

        std::thread::sleep(TIMESTAMP_STEP);
        let replacement = path.with_extension("replacement");
        // The operand's own bytes differ, so reading the path would show.
        let (header, payload_start) = read_segment_header(&path).unwrap();
        let tensor = header.tensors.iter().find(|t| t.name == q.tensor).unwrap();
        let start = (payload_start + tensor.offset) as usize;
        let mut bytes = std::fs::read(&path).unwrap();
        for byte in &mut bytes[start..start + original.len()] {
            *byte = !*byte;
        }
        std::fs::write(&replacement, &bytes).unwrap();
        std::fs::rename(&replacement, &path).unwrap();

        let (_, now) = only_segment_identity(&store, &q.object);
        assert_eq!(now.inode, at_open.inode, "the descriptor keeps its inode");
        assert_eq!(now.nlink, 0, "the replaced file has no name left");
        assert_ne!(now.ctime_ns, at_open.ctime_ns, "the ctime moved");
        assert_ne!(
            std::fs::metadata(&path).unwrap().ino(),
            at_open.inode,
            "the path names a different file"
        );
        let changed = now.changed_fields(&at_open);
        assert!(changed.contains(&"nlink") && changed.contains(&"ctime_ns"));

        assert_eq!(store.load_raw(&q).unwrap().bytes, original);
        assert_eq!(
            store.load_raw_range(&q, total, 0, total).unwrap().bytes,
            original
        );
        assert_eq!(store.map_region(&q, total).unwrap().bytes(), &original[..]);
    }

    #[test]
    fn a_deleted_path_does_not_stop_the_descriptor() {
        let dense = Dense::build();
        let (store, q, path) = dense.open();
        let original = independently_read(&path, &q);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(store.load_raw(&q).unwrap().bytes, original);
        let (_, now) = only_segment_identity(&store, &q.object);
        assert_eq!(now.nlink, 0);
    }

    /// An in-place write is detected, and is also read: a descriptor binds
    /// the file, not its content, and a mapping is not a snapshot.
    #[test]
    fn an_in_place_write_is_detected_and_shows_through() {
        use std::os::unix::fs::FileExt;
        let dense = Dense::build();
        let (store, q, path) = dense.open();
        let original = independently_read(&path, &q);
        let total = original.len() as u64;
        let (at_open, _) = only_segment_identity(&store, &q.object);
        let region = store.map_region(&q, total).unwrap();

        std::thread::sleep(TIMESTAMP_STEP);
        let (header, payload_start) = read_segment_header(&path).unwrap();
        let tensor = header.tensors.iter().find(|t| t.name == q.tensor).unwrap();
        let flipped = !original[0];
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all_at(&[flipped], payload_start + tensor.offset)
            .unwrap();

        let (_, now) = only_segment_identity(&store, &q.object);
        let changed = now.changed_fields(&at_open);
        assert!(changed.contains(&"mtime_ns"), "{changed:?}");
        assert!(changed.contains(&"ctime_ns"), "{changed:?}");
        assert_eq!(now.inode, at_open.inode);
        assert_eq!(now.size, at_open.size);
        assert_eq!(store.load_raw(&q).unwrap().bytes[0], flipped);
        assert_eq!(region.bytes()[0], flipped);
    }

    /// False voids are stated limits: this one moves the ctime with no
    /// change to a single byte.
    #[test]
    fn a_hard_link_moves_the_ctime_with_no_content_change() {
        let dense = Dense::build();
        let (store, q, path) = dense.open();
        let original = independently_read(&path, &q);
        let (at_open, _) = only_segment_identity(&store, &q.object);
        std::thread::sleep(TIMESTAMP_STEP);
        std::fs::hard_link(&path, path.with_extension("link")).unwrap();
        let (_, now) = only_segment_identity(&store, &q.object);
        assert!(now.changed_fields(&at_open).contains(&"nlink"));
        assert_eq!(store.load_raw(&q).unwrap().bytes, original);
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use crate::format::vindex3::opplan::exec::operands::{
        volume_uuid_from_descriptor, volume_uuid_from_mount, VOLUME_UUID_FROM_DESCRIPTOR,
    };

    /// The man page says a descriptor on a regular file gets `EINVAL`. It
    /// works, and this pins that it does and that the fallback agrees.
    #[test]
    fn the_descriptors_volume_uuid_agrees_with_its_mounts() {
        let dense = Dense::build();
        let (_store, _, path) = dense.open();
        let file = std::fs::File::open(&path).unwrap();
        let from_descriptor =
            volume_uuid_from_descriptor(&file).expect("a regular file reports its volume");
        let from_mount = volume_uuid_from_mount(&file).expect("its mount reports its volume");
        assert_eq!(from_descriptor, from_mount);
        assert_eq!(from_descriptor.len(), 36, "{from_descriptor}");
        let identity = FileIdentity::capture(&file).unwrap();
        assert_eq!(identity.volume_uuid.as_deref(), Some(&*from_descriptor));
        assert_eq!(
            identity.volume_uuid_source,
            Some(VOLUME_UUID_FROM_DESCRIPTOR)
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn linux_has_no_volume_uuid_and_names_its_platform() {
    let dense = Dense::build();
    let (_store, _, path) = dense.open();
    let identity = FileIdentity::capture(&std::fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(identity.volume_uuid, None);
    assert_eq!(identity.volume_uuid_source, None);
    assert_eq!(identity.platform, "linux");
}

#[cfg(not(unix))]
#[test]
fn a_platform_with_no_descriptor_identity_fails_the_capture_and_still_reads() {
    let dense = Dense::build();
    let (store, q, path) = dense.open();
    assert!(FileIdentity::capture(&std::fs::File::open(&path).unwrap()).is_err());
    let report = store.instance_report();
    assert!(report
        .segments
        .iter()
        .all(|s| s.capture_failure().is_some()));
    assert_eq!(
        store.load_raw(&q).unwrap().bytes,
        independently_read(&path, &q)
    );
}

// ── A mapped bank, in a real binding ──

mod bound {
    use super::super::kimi_per_expert_prepared::Subject;
    use crate::format::vindex3::fixtures_kimi::kimi_per_expert_moe_f32_model;
    use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
    use crate::format::vindex3::opplan::exec::production::ProductionBackend;

    /// Binding reports both uses: the expert bank mapped, the rest copied,
    /// each through its own one descriptor, with identity intact.
    #[test]
    fn a_real_binding_reports_the_bank_mapped_and_the_rest_copied() {
        let subject = Subject::build(kimi_per_expert_moe_f32_model);
        let (plan, store) = subject.open();
        let ops = PreparedOperands::load(
            &plan,
            &store,
            &ProductionBackend::new(),
            ExecutionSlice::Full,
        )
        .unwrap();
        let report = store.instance_report();
        assert_eq!(report.segments.iter().filter(|s| s.mapped).count(), 1);
        assert!(report.segments.iter().any(|s| s.copied && !s.mapped));
        assert_eq!(report.mapped_objects, store.mapped_objects());
        assert_eq!(report.mapped_regions, store.mapped_regions());
        // The regions an observation reports point into the mapped segment.
        let observed = ops.observe_residency();
        let mapped = report.segments.iter().find(|s| s.mapped).unwrap();
        assert_eq!(observed.regions.len(), store.mapped_regions());
        assert!(observed
            .stores
            .iter()
            .all(|s| s.declared_segment.as_deref() == Some(mapped.declared.as_str())));
    }

    /// The output of a real execution comes from the opened files: a whole
    /// container swapped under the store changes nothing it computes.
    #[cfg(unix)]
    #[test]
    fn execution_after_every_segment_is_replaced_computes_what_the_opened_files_hold() {
        use super::super::kimi_per_expert_prepared::open_container;
        use crate::format::vindex3::opplan::exec::execute_plan;
        use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
        let subject = Subject::build(kimi_per_expert_moe_f32_model);
        let (plan, _) = subject.open();
        let tokens = [1u32, 2];
        let backend = ReferenceBackend;
        let (_, baseline_store) = open_container(subject.container());
        let expected = execute_plan(&plan, &baseline_store, &tokens, &backend)
            .unwrap()
            .logits
            .expect("the plan has a head");

        let (_, store) = open_container(subject.container());
        for segment in store.instance_report().segments {
            let mut bytes = std::fs::read(&segment.path).unwrap();
            let payload =
                crate::format::vindex3::encode::segment::read_segment_header(&segment.path)
                    .unwrap()
                    .1 as usize;
            for byte in &mut bytes[payload..] {
                *byte = 0;
            }
            let replacement = segment.path.with_extension("replacement");
            std::fs::write(&replacement, &bytes).unwrap();
            std::fs::rename(&replacement, &segment.path).unwrap();
        }
        let got = execute_plan(&plan, &store, &tokens, &backend).unwrap();
        assert_eq!(got.logits.as_ref(), Some(&expected));

        // The control: the same container reopened reads the zeros, so the
        // replacement really was different bytes.
        let (_, reopened) = open_container(subject.container());
        let zeroed = execute_plan(&plan, &reopened, &tokens, &backend)
            .expect("zeroed weights still execute")
            .logits;
        assert_ne!(zeroed.as_ref(), Some(&expected));
    }
}
