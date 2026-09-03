//! Architecture-detection tests, split by concern.
//!
//! Kept under `detect/tests/` rather than alongside implementation files so
//! the suite can exercise `detect`-private items (`config_io`) through the
//! module tree.

mod declared_scalars;
mod disk_path;
mod families_dense;
mod families_moe;
mod gemma4;
mod gpt2_aliases;
mod kv_recompute;
mod norm_eps;
mod qwen35_hybrid;
mod qwen_moe_shared_expert;
mod real_configs;
mod rope_scaling;
mod routing;
mod support;
