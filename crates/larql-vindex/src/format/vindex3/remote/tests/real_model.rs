//! **Selective residency on a real model.**
//!
//! The fixture gates prove the mechanism; this proves the mechanism
//! survives contact with a real checkpoint. `#[ignore]` because it needs
//! a real container — set `LARQL_GRANITE_CONTAINER` to one produced by
//!
//! ```text
//! larql vindex3 encode hf://ibm-granite/granite-4.2-3b --output <dir>
//! ```
//!
//! # Why granite can carry this after all
//!
//! It was assumed granite could not demonstrate selectivity, because a
//! whole-model execution requires all four of its objects. That is true
//! of [`ExecutionSlice::Full`] and false of every other slice: a
//! layer-range shard is a hidden-state transform that needs the decoder
//! stack and nothing else, so three of granite's four objects — 1.03 GB
//! of embedding, head and final norm — can be genuinely absent.
//!
//! What granite still cannot witness is a logical role with no distinct
//! serialised object. That is Qwen3-4B's job, and Qwen3-4B does not
//! encode today: its text-generation closure carries three blocking
//! findings (`max_window_layers`, `use_sliding_window`, and one
//! `sliding_window` execution-semantic rule). So that witness is gated on
//! semantic work, not on a download.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::format::vindex3::encode::SEGMENTS_DIR;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::requirements::required_objects;
use crate::format::vindex3::opplan::plan_component_ops;

/// Environment variable naming a real granite container.
const CONTAINER_ENV: &str = "LARQL_GRANITE_CONTAINER";

const COMPONENT: &str = "target";

/// Build a container root holding only `keep`, by hard-linking segments
/// out of `source`. Hard links cost nothing, so this produces exactly
/// what a selective hydration would have written without copying gigabytes
/// or disturbing the source.
fn hydrated_view(source: &Path, keep: &BTreeSet<String>, into: &Path) -> std::io::Result<u64> {
    std::fs::create_dir_all(into.join(SEGMENTS_DIR))?;
    for name in ["index.json", "system_graph.json"] {
        std::fs::copy(source.join(name), into.join(name))?;
    }
    let mut linked = 0u64;
    for entry in std::fs::read_dir(source.join(SEGMENTS_DIR))? {
        let path = entry?.path();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        let object = file.trim_end_matches(".bin").to_string();
        if !keep.contains(&object) {
            continue;
        }
        linked += std::fs::metadata(&path)?.len();
        std::fs::hard_link(&path, into.join(SEGMENTS_DIR).join(&file))?;
    }
    Ok(linked)
}

fn container() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(CONTAINER_ENV).ok()?);
    path.join("index.json").exists().then_some(path)
}

#[test]
#[ignore = "needs a real granite container; set LARQL_GRANITE_CONTAINER"]
fn a_real_model_prepares_with_most_of_it_absent() {
    let Some(source) = container() else {
        panic!("set {CONTAINER_ENV} to a real granite-4.2-3b container");
    };
    let inspection = inspect_container(&source, false).unwrap();
    let outcome = plan_component_ops(&inspection, &source, COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();

    let slice = ExecutionSlice::LayerRange { start: 0, end: 2 };
    let required = required_objects(&plan, &slice).unwrap();
    let described: BTreeSet<String> = inspection
        .graph
        .objects
        .iter()
        .map(|o| o.id.clone())
        .collect();
    let left_remote: BTreeSet<String> = described.difference(&required).cloned().collect();
    assert!(
        !left_remote.is_empty(),
        "this slice requires the whole model; it cannot witness selectivity"
    );

    // Build the partial view and measure what was NOT moved.
    let view = tempfile::tempdir().unwrap();
    let resident = hydrated_view(&source, &required, view.path()).unwrap();
    let whole: u64 = std::fs::read_dir(source.join(SEGMENTS_DIR))
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();

    for object in &left_remote {
        let segment = view.path().join(SEGMENTS_DIR).join(format!("{object}.bin"));
        assert!(
            !segment.exists(),
            "`{object}` was left remote but is on disk at {}",
            segment.display()
        );
    }

    // The claim: prepare and run against a physically incomplete model.
    let partial = inspect_container(view.path(), false).unwrap();
    let store = OperandStore::open(view.path(), &partial)
        .expect("a partly resident real container must open");
    for object in &left_remote {
        assert!(
            !store.is_resident(object),
            "`{object}` should not be resident"
        );
    }
    let backend = ReferenceBackend;
    PreparedOperands::load(&plan, &store, &backend, slice)
        .expect("preparing a layer range must not need the absent objects");
    assert_eq!(
        store.touched_objects(),
        required,
        "preparation touched something outside its requirement set"
    );

    eprintln!(
        "granite-4.2-3b: prepared {:.2} GB resident of {:.2} GB described; \
         {} object(s) left remote: {left_remote:?}",
        resident as f64 / 1e9,
        whole as f64 / 1e9,
        left_remote.len(),
    );
}
