//! Architecture inventory of an HF checkpoint directory (`larql inspect-hf`).
//!
//! The first rung of the vindex3 container programme (G0): before a single
//! byte is converted, produce a machine-readable statement of **what the
//! checkpoint declares** next to **what this build would do with it** —
//! identity, components, per-layer attention policy, tensor objects, and,
//! centrally, every `config.json` key the config parser does *not* read.
//!
//! An unconsumed key is a serving bug in waiting: the parser's silence means
//! some default will answer for it on every forward pass (see
//! `docs/k3-funnel.md` §4.7.8 — rope_scaling, layer_types, rms_norm_eps and
//! attention biases have each already shipped that failure). The inventory
//! makes the gap visible from the config alone, with no forward pass and no
//! reference implementation.
//!
//! Downstream (G1), a planner consumes this inventory and reports what the
//! vindex3 container schema can and cannot represent — so representation
//! gaps surface as design findings before any conversion work starts.

pub mod components;
pub mod config_keys;
pub mod interfaces;
pub mod report;
pub mod representation;
pub mod resolved;
pub mod tensors;

#[cfg(test)]
mod tests;

use std::path::Path;

use crate::detect::ModelError;

pub use components::is_operator_config_section;
pub use report::{
    ArchitectureInventory, AttentionSummary, ConfigKeyFact, Detection, Identity, InterfaceFact,
    KeyStatus, LayerPolicy, ResolvedExecution, ResolvedTopology, TensorFact, TensorGroup,
    TensorInventory, INVENTORY_SCHEMA,
};

/// Filename of the config an HF checkpoint directory carries.
const CONFIG_FILE_NAME: &str = "config.json";

/// Build the full inventory for one checkpoint directory.
///
/// Requires `config.json` to exist and parse — an inventory with no config
/// would have nothing to classify — but tolerates everything else: missing
/// shards, unknown `model_type`, validation failures. Those are *findings*,
/// carried in the report, not reasons to refuse it.
pub fn build_inventory(model_dir: &Path) -> Result<ArchitectureInventory, ModelError> {
    if !model_dir.is_dir() {
        return Err(ModelError::NotADirectory(model_dir.to_path_buf()));
    }
    let config_path = model_dir.join(CONFIG_FILE_NAME);
    if !config_path.exists() {
        return Err(ModelError::ConfigMissing(config_path));
    }
    let text = std::fs::read_to_string(&config_path)?;
    // The judged non-finite boundary: HF configs carry Python's bare
    // `Infinity`/`NaN` literals; they parse here as the strings they
    // spell, and nowhere else (config::nonfinite_json).
    let config: serde_json::Value = crate::config::nonfinite_json::parse_config_json(&text)?;

    let identity = resolved::read_identity(&config);
    // Tensors first: the resolution may need the estate as evidence for a
    // declared ambiguity (`resolve_with_tensor_evidence`, J5).
    let tensor_inventory = tensors::scan_tensors(model_dir)?;
    let (detection, topology) =
        resolved::resolve_with_tensor_evidence(&config, &identity, &tensor_inventory.tensors);
    // Nested components first: their recorded reads feed classification.
    let component_readings = components::read_components(&config);
    let mut recorded_reads: std::collections::BTreeSet<String> = component_readings
        .iter()
        .flat_map(|r| r.consumed_paths.iter().cloned())
        .collect();
    let nested_components = component_readings.into_iter().map(|r| r.topology).collect();
    // The stored-representation reader is the second recorded reader:
    // `quantization_config` is credited only because something read it and
    // stored what it read.
    // The parser reads two leaves of `linear_attn_config` by path; credit
    // exactly those, never the container, so its unread siblings stay
    // honestly unconsumed.
    recorded_reads.extend(config_keys::path_read_leaves(&config));
    let representation_reading = representation::read_stored_representation(&config);
    let stored_representation = representation_reading.map(|r| {
        recorded_reads.extend(r.consumed_paths);
        r.representation
    });
    // The multimodal-interface reader is the third: special-token roles,
    // soft-token counts, declared-absent towers and the bidirectional
    // masking policy are credited only because it read and stored them.
    let interface_reading = interfaces::read_interface(&config);
    let multimodal_interface = interface_reading.map(|r| {
        recorded_reads.extend(r.consumed_paths);
        r.interface
    });
    let config_facts = config_keys::classify_config(&config, &recorded_reads);
    let interfaces = config_keys::find_interfaces(&config_facts);
    // The architecture names its expert banks in its own key namespace
    // (post-strip: `layers.3.mlp.experts`); the graph binds source names.
    // Resolve each bank prefix against the tensors actually present, so
    // the recorded prefix is the one the checkpoint spells.
    let mut topology = topology;
    resolved::bind_expert_banks(&mut topology, &tensor_inventory.tensors);

    Ok(ArchitectureInventory {
        schema: INVENTORY_SCHEMA,
        path: model_dir.display().to_string(),
        identity,
        detection,
        resolved: topology,
        nested_components,
        stored_representation,
        multimodal_interface,
        config_keys: config_facts,
        interfaces,
        tensors: tensor_inventory,
    })
}
