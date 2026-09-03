//! Operation-plan gates (V3-G5b-1): the plan is generic, placement is
//! explicit, and closure blocks on every shortfall.

mod closure;
pub(crate) mod conv_qkv;
mod coverage_opplan;
mod gemma4_closure;
mod kda_mla_exec;
mod kda_op;
mod kimi_mla_closure;
mod kimi_moe_closure;
pub(crate) mod mamba2;
mod mla_op;
mod plan;
mod post_norm_placement;
mod tied_head_realization;
mod unjudged;

use std::path::PathBuf;

use larql_models::inventory::ArchitectureInventory;

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::{inspect_container, SystemInspection};
use crate::format::vindex3::plan::tests_support::{drafter_shaped, glimmer_shaped_target};

/// The fixture pair, encoded and inspected — everything a planner run
/// needs, sources kept alive.
pub(super) struct PlannedFixture {
    pub _target_dir: tempfile::TempDir,
    pub _drafter_dir: tempfile::TempDir,
    pub container: tempfile::TempDir,
    pub root: PathBuf,
    pub inspection: SystemInspection,
    pub named: Vec<(String, ArchitectureInventory)>,
}

pub(super) fn encoded_fixture() -> PlannedFixture {
    let target_dir = tempfile::tempdir().unwrap();
    let drafter_dir = tempfile::tempdir().unwrap();
    let named = vec![
        (
            "target-artifact".to_string(),
            glimmer_shaped_target(target_dir.path()),
        ),
        (
            "drafter-artifact".to_string(),
            drafter_shaped(drafter_dir.path()),
        ),
    ];
    let container = tempfile::tempdir().unwrap();
    encode_system(&named, container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let root = container.path().to_path_buf();
    PlannedFixture {
        _target_dir: target_dir,
        _drafter_dir: drafter_dir,
        container,
        root,
        inspection,
        named,
    }
}
