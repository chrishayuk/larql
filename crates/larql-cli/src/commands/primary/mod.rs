//! Primary user-facing verbs: `run`, `pull`, `list`, `show`, `rm`.
//!
//! These wrap the lower-level `extraction::*` commands behind a slimmer
//! flag set and ollama-style ergonomics. Research/power-user tooling lives
//! under `larql dev <subcmd>`.

pub mod accuracy_cmd;
pub mod bench;
pub mod cache;
pub mod capabilities_cmd;
pub mod card_cmd;
pub mod dec_bench;
pub mod diag_cmd;
pub mod inspect_hf_cmd;
pub mod k3_ledger;
pub mod link_cmd;
pub mod list_cmd;
pub mod model_cmd;
pub mod publish_cmd;
pub mod pull_cmd;
pub mod recipe_cmd;
pub mod registry_cmd;
pub mod rm_cmd;
pub mod run_cmd;
pub mod run_cmd_image;
pub mod run_cmd_speak;
pub mod run_cmd_vindex3;
pub mod serve_resolve;
pub mod server_capabilities_cmd;
pub mod shannon_cmd;
pub mod shannon_trace;
pub mod show_cmd;
pub mod slice_cmd;
pub mod vindex3_cmd;
