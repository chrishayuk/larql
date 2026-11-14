use std::path::Path;

use super::super::caches::{
    checks, nearest_existing, physical_size, probe_dir, writable, CacheDir,
};
use super::super::report::Status;
use super::{dir, healthy_caches, statuses, GB};

#[test]
fn healthy_caches_pass_with_paths_from_tilde_and_sizes() {
    let out = checks(&healthy_caches());
    assert_eq!(statuses(&out), [Status::Ok; 3]);
    assert!(out[0].summary.starts_with('~'), "{}", out[0].summary);
    assert!(out[0].summary.contains("2.0 GB"));
    assert_eq!(out[1].name, "Encoded models");
    assert!(out[2].summary.starts_with("100.0 GB free"));
}

#[test]
fn an_env_override_is_named() {
    let mut c = healthy_caches();
    c.encoded.env_override = Some("LARQL_HOME");
    assert!(checks(&c)[1].summary.contains("from $LARQL_HOME"));
}

#[test]
fn a_cache_not_yet_created_or_of_unknown_size_says_so() {
    let mut c = healthy_caches();
    if let Ok(f) = &mut c.hub.facts {
        f.exists = false;
    }
    if let Ok(f) = &mut c.encoded.facts {
        f.size_bytes = None;
    }
    let out = checks(&c);
    assert!(out[0].summary.contains("not created yet"));
    assert!(out[1].summary.contains("size unknown"));
}

#[test]
fn an_unwritable_encoded_cache_fails_but_an_unwritable_hub_warns() {
    let mut c = healthy_caches();
    for cache in [&mut c.hub, &mut c.encoded] {
        if let Ok(f) = &mut cache.facts {
            f.writable = false;
        }
    }
    let out = checks(&c);
    assert_eq!(out[0].status, Status::Warn);
    assert_eq!(out[1].status, Status::Fail);
    assert!(out[1].hint.as_deref().unwrap().contains("LARQL_HOME"));
}

#[test]
fn a_cache_with_no_location_fails() {
    let mut c = healthy_caches();
    c.hub = CacheDir {
        facts: Err("no HOME env var".into()),
        env_override: None,
    };
    c.encoded = CacheDir {
        facts: Err("no HOME env var".into()),
        env_override: None,
    };
    let out = checks(&c);
    // No free-disk line without a location to measure.
    assert_eq!(statuses(&out), [Status::Fail, Status::Fail]);
    assert!(out[0].summary.contains("no HOME"));
}

#[test]
fn too_little_free_disk_fails_and_unknown_free_disk_warns() {
    let mut c = healthy_caches();
    c.encoded.facts = Ok(dir(".cache/larql/encoded", Some(GB)));
    let free = checks(&c).pop().unwrap();
    assert_eq!(free.status, Status::Fail);
    assert!(free.summary.contains("needs about 3.0 GB"));

    c.encoded.facts = Ok(dir(".cache/larql/encoded", None));
    assert_eq!(checks(&c).pop().unwrap().status, Status::Warn);
}

#[test]
fn a_short_hub_volume_adds_a_warning() {
    let mut c = healthy_caches();
    c.hub.facts = Ok(dir(".cache/huggingface/hub", Some(GB)));
    let out = checks(&c);
    assert_eq!(out.len(), 4);
    assert_eq!(out[3].name, "Free disk (hub)");
    assert_eq!(out[3].status, Status::Warn);
}

#[test]
fn nearest_existing_walks_up_to_a_real_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("a/b/c");
    assert_eq!(nearest_existing(&missing), Some(tmp.path()));
    assert_eq!(nearest_existing(tmp.path()), Some(tmp.path()));
}

#[test]
fn a_temp_directory_is_writable_and_a_file_is_not_a_directory() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(writable(tmp.path()));
    let file = tmp.path().join("f");
    std::fs::write(&file, b"x").unwrap();
    assert!(!writable(&file));
    assert!(!writable(Path::new("/definitely/not/a/dir")));
}

#[test]
fn physical_size_counts_files_once_and_skips_symlinks() {
    let tmp = tempfile::tempdir().unwrap();
    let blobs = tmp.path().join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::write(blobs.join("a"), vec![0u8; 100]).unwrap();
    std::fs::write(tmp.path().join("b"), vec![0u8; 20]).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(blobs.join("a"), tmp.path().join("link")).unwrap();
    assert_eq!(physical_size(tmp.path()).unwrap(), 120);
    assert!(physical_size(&tmp.path().join("missing")).is_err());
}

#[test]
fn probing_an_existing_directory_reads_its_facts() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("f"), vec![0u8; 10]).unwrap();
    let facts = probe_dir(tmp.path(), |_| Some(7));
    assert!(facts.exists && facts.writable);
    assert_eq!(facts.size_bytes, Some(10));
    assert_eq!(facts.free_bytes, Some(7));
}

#[test]
fn probing_a_directory_not_yet_created_asks_its_parent() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("encoded");
    let asked = std::cell::RefCell::new(None);
    let facts = probe_dir(&missing, |p| {
        *asked.borrow_mut() = Some(p.to_path_buf());
        Some(1)
    });
    assert!(!facts.exists);
    assert!(facts.writable);
    assert_eq!(facts.size_bytes, None);
    assert_eq!(asked.into_inner().as_deref(), Some(tmp.path()));
}
