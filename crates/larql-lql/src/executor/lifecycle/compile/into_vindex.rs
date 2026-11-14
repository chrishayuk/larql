//! `COMPILE INTO VINDEX`: bake the patch overlay onto a clean copy of
//! the source vindex so the result is self-contained (no overlay
//! needed at load time).
//!
//! What the artifact must contain (Gate A4):
//!
//! - every metadata edit in the overlay: an inserted or updated record,
//!   and an empty record for a deleted slot;
//! - no vector from a slot that was deleted after it was written;
//! - every weight file the source's manifest names.
//!
//! Anything the bake cannot honour is refused by name before a file is
//! written.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::ast::CompileConflict;
use crate::error::LqlError;
use crate::executor::helpers::{dir_size, format_bytes};
use crate::executor::tuning::{MEMIT_DEFAULT_RIDGE, MEMIT_TARGET_ALPHA};
use crate::executor::Session;
use larql_vindex::format::filenames::{DOWN_META_BIN, DOWN_WEIGHTS_BIN, KNN_STORE_BIN};

use super::atomic::run_atomic_compile;
use super::bake::{
    apply_memit_deltas_to_down_weights, patch_down_weights, patch_gate_vectors, patch_up_weights,
};
use super::collect_memit_facts_with_recording;
use super::source_files::{
    carry_source_files, manifest_weight_files, missing_required_file, refuse_unbakeable_layout,
};

/// How many metadata edits a refusal or a label note lists by slot.
const UNSTORABLE_PREVIEW: usize = 5;

/// The overrides whose slot is still live. A `DELETE` tombstones the
/// slot's metadata but leaves the down and up vectors an earlier
/// `INSERT` wrote on the base, so the bake has to drop them here.
fn live_overrides<'a>(
    overrides: &'a HashMap<(usize, usize), Vec<f32>>,
    patched: &larql_vindex::PatchedVindex,
) -> Cow<'a, HashMap<(usize, usize), Vec<f32>>> {
    if overrides.keys().all(|(l, f)| !patched.is_deleted(*l, *f)) {
        return Cow::Borrowed(overrides);
    }
    Cow::Owned(
        overrides
            .iter()
            .filter(|((l, f), _)| !patched.is_deleted(*l, *f))
            .map(|(slot, vector)| (*slot, vector.clone()))
            .collect(),
    )
}

/// Walk the ordered patch history and return the (layer, feature) slots
/// touched by more than one patch, along with the write count. Used by
/// `COMPILE INTO VINDEX ON CONFLICT` to detect ambiguous bakes.
pub(super) fn collect_compile_collisions(
    patches: &[larql_vindex::VindexPatch],
) -> HashMap<(usize, usize), usize> {
    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
    for patch in patches {
        let mut seen_in_this_patch: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        for op in &patch.operations {
            let key = match op.key() {
                Some(k) => k,
                None => continue, // KNN ops don't collide on (layer, feature)
            };
            if seen_in_this_patch.insert(key) {
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts.retain(|_, n| *n > 1);
    counts
}

impl Session {
    pub(super) fn exec_compile_into_vindex(
        &mut self,
        source_path: &std::path::Path,
        output: &str,
        on_conflict: CompileConflict,
    ) -> Result<Vec<String>, LqlError> {
        // Snapshot the source path before we hand control to the atomic
        // wrapper so the inner closure can re-borrow `self` without aliasing.
        let _ = source_path; // accepted for symmetry; current vindex is the source
        let final_dir = PathBuf::from(output);
        let source_path_owned = {
            let (path, _, _) = self.require_vindex()?;
            path.to_path_buf()
        };

        run_atomic_compile(&final_dir, &source_path_owned, |output_dir| {
            self.bake_compile_into_vindex(output_dir, on_conflict)
        })
    }

    fn bake_compile_into_vindex(
        &mut self,
        output_dir: &std::path::Path,
        on_conflict: CompileConflict,
    ) -> Result<Vec<String>, LqlError> {
        // Load the current vindex with patches applied
        let (path, config, patched) = self.require_vindex()?;

        // ── Refusals, before any file is written ──
        refuse_unbakeable_layout(config)?;
        let manifest_files = manifest_weight_files(path, config)?;
        // The text a loader derives from a token id: one token, decoded
        // with special tokens skipped. `None` without a tokenizer.
        let tokenizer = larql_vindex::load_vindex_tokenizer(path).ok();
        let token_text =
            |id: u32| -> Option<String> { tokenizer.as_ref()?.decode(&[id], true).ok() };
        let unstorable = patched.unstorable_meta_edits();
        let relabelled = patched.labels_stored_as_their_token(&token_text);
        if !unstorable.is_empty() {
            let preview = unstorable
                .iter()
                .take(UNSTORABLE_PREVIEW)
                .map(|edit| edit.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(LqlError::Execution(format!(
                "COMPILE INTO VINDEX cannot store {} metadata edit(s) in {DOWN_META_BIN}: {preview}",
                unstorable.len()
            )));
        }

        // ── Conflict detection across applied patches ──
        //
        // The overlay maps in `PatchedVindex` are already collapsed under
        // last-wins semantics. To honour ON CONFLICT we re-scan the
        // ordered patch history and detect (layer, feature) slots that
        // are written by more than one patch.
        let collisions = collect_compile_collisions(&patched.patches);
        match on_conflict {
            CompileConflict::LastWins => {}
            CompileConflict::Fail => {
                if !collisions.is_empty() {
                    let preview = collisions
                        .iter()
                        .take(5)
                        .map(|((l, f), n)| format!("L{l}/F{f} ({n} writes)"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(LqlError::Execution(format!(
                        "COMPILE INTO VINDEX ON CONFLICT FAIL: {} colliding slot(s): {}",
                        collisions.len(),
                        preview
                    )));
                }
            }
            CompileConflict::HighestConfidence => {
                // Down vectors are baked at INSERT time and stored on the
                // base vindex collapsed under last-wins, so re-resolving
                // them from raw patches would require regenerating the
                // synthesised vectors. We do not currently do that — the
                // strategy is accepted for forward compatibility but
                // behaves like LAST_WINS today. This is reported in the
                // output below so callers know.
            }
        }

        // ── Step 0: MEMIT pass over compose-mode inserts ──
        //
        // Compose-mode INSERT emits PatchOp::Insert, which specifies
        // a free slot and the heuristic install_compiled_slot gate/up/
        // down overlays. Those overlays work at N≤10 per layer but hit
        // a Hopfield cap past that because the per-fact install is a
        // strong, non-orthogonal edit.
        //
        // MEMIT solves for ΔW_down in closed form across ALL inserted
        // facts jointly, routing edits through the null-space of typical
        // activations. The resulting delta scales to 200+ facts per
        // layer (validated Python reference). Baking ΔW_down into the
        // compiled vindex's `down_weights.bin` gives the same quality
        // compilation COMPILE INTO MODEL produces — just in vindex format.
        let recording_ops: Vec<larql_vindex::PatchOp> = self
            .patch_recording
            .as_ref()
            .map(|r| r.operations.clone())
            .unwrap_or_default();
        let collected = collect_memit_facts_with_recording(patched, path, &recording_ops)?;
        let memit_facts = collected.facts;
        let memit_warnings = collected.warnings;
        // Only run MEMIT when model weights are present. Without weights
        // (browse-only vindexes) the compile falls back to the legacy
        // column-replace bake of gate/up/down overlays, matching the
        // pre-MEMIT behaviour used by unit tests that exercise the bake
        // path without shipping a real model.
        // MEMIT is opt-in via `LARQL_MEMIT_ENABLE=1`. It is validated
        // on v11 (200/200) but cross-hijacks natives on Gemma 3-4B at
        // every layer tested: the hourglass plateau (L6-L28) makes
        // template-sharing k_stars indistinguishable, so the closed-
        // form solve cannot separate installs from natives. Pure
        // compose column-replace is the default COMPILE path and is
        // what produces the working Gemma installs.
        // Read through the override-aware `larql_compute::options` helper so
        // tests can toggle this without `std::env::set_var` (which races
        // concurrent `getenv` on the decode path and SIGSEGVs libc).
        let memit_enabled = larql_compute::options::env_value("LARQL_MEMIT_ENABLE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let memit_results = if !memit_facts.is_empty() && config.has_model_weights && memit_enabled
        {
            let mut cb = larql_vindex::SilentLoadCallbacks;
            let weights = larql_vindex::load_model_weights(path, &mut cb)
                .map_err(|e| LqlError::exec("load weights for MEMIT", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("load tokenizer for MEMIT", e))?;
            // `LARQL_MEMIT_TARGET_DELTA=1` switches MEMIT from the
            // `target_alpha × embed(target)` shortcut to the per-fact
            // gradient-optimised delta (Python reference Phase 3 +
            // Phase 4). Slow (60 Adam steps/fact) but unlocks scale.
            // `LARQL_MEMIT_SPREAD=N` distributes each fact across N
            // consecutive layers centred on its install layer.
            // `LARQL_MEMIT_RIDGE=f` overrides the solve's ridge term.
            let use_target_delta = std::env::var("LARQL_MEMIT_TARGET_DELTA")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let spread = std::env::var("LARQL_MEMIT_SPREAD")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1);
            let ridge = std::env::var(larql_inference::forward::memit::MEMIT_RIDGE_ENV)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(MEMIT_DEFAULT_RIDGE);
            let results = if use_target_delta {
                larql_inference::forward::memit::run_memit_with_target_opt_multi(
                    &weights,
                    &memit_facts,
                    ridge,
                    larql_inference::TargetDeltaOpts::default(),
                    &tokenizer,
                    spread,
                )
            } else {
                larql_inference::run_memit(
                    &weights,
                    &memit_facts,
                    ridge,
                    MEMIT_TARGET_ALPHA,
                    &tokenizer,
                )
            };
            let results =
                results.map_err(|e| LqlError::Execution(format!("MEMIT solve failed: {e}")))?;
            Some(results)
        } else {
            None
        };

        // ── Step 1: gate_vectors.bin and down_meta.bin ──
        //
        // Both are written from a clone of the patched base. The clone path
        // produces byte-identical output to the source for unchanged
        // layers, and we deliberately do NOT bake any inserted gate
        // vectors into gate_vectors.bin (see comment further down).
        let baked = patched.base().clone();
        let layer_infos = baked
            .save_gate_vectors_with_config(output_dir, config)
            .map_err(|e| LqlError::exec("failed to save gate vectors", e))?;
        // An unedited `down_meta.bin` is hard-linked from the source (Step 2)
        // rather than saved from this clone: the clone is in mmap mode and
        // its heap-side `down_meta` is empty. When the overlay edits
        // metadata, Step 2b writes the session's effective records instead.
        let dm_count: usize = config.layers.iter().map(|l| l.num_features).sum();

        // ── Step 2: hard-link unchanging weight files from the source ──
        //
        // These files are byte-identical to the source (model weights and
        // related artefacts that INSERT does not touch). Hard-linking is
        // free on APFS — same inode, no disk cost, no copy time.
        //
        // We deliberately do NOT bake the inserted gate vectors into
        // gate_vectors.bin. The dense FFN inference path
        // (`walk_ffn_exact` / `walk_ffn_full_mmap`) reads gate scores
        // from this file and feeds them into the GEGLU activation.
        // Baking a norm-matched (~typical-magnitude) gate at the
        // inserted slot makes its dense activation moderate-to-large,
        // which combined with the override down vector blows up the
        // residual stream. Keeping the source weak gate at the inserted
        // slot keeps the activation small — exactly matching the
        // patched-session math, where the small activation × override
        // down vector accumulates across layers into a meaningful
        // constellation effect.
        //
        // The override is instead baked into `down_weights.bin` further
        // down (see Step 3): the dense FFN reads `W_down[:, slot]` from
        // model weights, and replacing those columns with the override
        // values gives `small_activation × poseidon_vector` per layer,
        // which is the exact behaviour the runtime patch overlay
        // produces.
        // Any change to a gate, up or down vector makes the source's other
        // encodings of those vectors stale; they are left behind, by name.
        let vectors_edited = patched.overrides_gate_iter().next().is_some()
            || !patched.down_overrides().is_empty()
            || !patched.up_overrides().is_empty()
            || patched.num_deleted() > 0
            || memit_results.is_some();
        let left_behind = carry_source_files(path, output_dir, &manifest_files, vectors_edited)?;

        // ── Step 2b: metadata edits ──
        //
        // INSERT, UPDATE and DELETE live in the overlay only. The carried
        // `down_meta.bin` is the source's, so when the overlay edits
        // metadata it is replaced by the session's effective records. The
        // write renames a new file into place, so the source's file,
        // hard-linked above, is not touched.
        let meta_edited = patched.has_meta_edits();
        if meta_edited {
            patched
                .save_effective_down_meta(output_dir, config.down_top_k)
                .map_err(|e| LqlError::exec(format!("failed to write {DOWN_META_BIN}"), e))?;
        }

        // ── Step 3: bake down vector overrides into down_weights.bin ──
        //
        // The dense FFN inference path reads `W_down[:, slot]` from
        // `down_weights.bin` (via `load_model_weights` →
        // `walk_ffn_exact`). Replacing the column at the inserted slot
        // with the override down vector makes the inserted feature fire
        // through the standard FFN path with no runtime overlay needed.
        //
        // This is what makes the compiled vindex truly self-contained
        // and what unblocks `COMPILE INTO MODEL FORMAT safetensors|gguf`
        // — those exporters read the same `down_weights.bin` via
        // `weight_manifest.json` and emit it as the canonical down
        // projection, so the constellation is already in the exported
        // model.
        let mut down_overrides = live_overrides(patched.down_overrides(), patched).into_owned();
        let up_overrides = live_overrides(patched.up_overrides(), patched);
        let up_overrides: &HashMap<(usize, usize), Vec<f32>> = &up_overrides;
        // Collect gate overrides from the patch overlay into an owned
        // HashMap matching the shape `patch_gate_vectors` expects.
        let mut gate_overrides: HashMap<(usize, usize), Vec<f32>> = patched
            .overrides_gate_iter()
            .map(|(l, f, g)| ((l, f), g.to_vec()))
            .collect();
        // A deleted slot contributes nothing in the patched session: the
        // walk gives a tombstoned slot an activation of zero and the gate
        // KNN leaves it out. The artifact has no tombstone to carry, so the
        // slot's gate row and down column are written as zeros, which is the
        // same contribution. The down column needs weights to write into; a
        // browse-only source has none and nothing to run.
        let deleted_slots = patched.deleted_slots();
        let has_down_weights = manifest_files.contains(DOWN_WEIGHTS_BIN);
        for slot in &deleted_slots {
            gate_overrides.insert(*slot, vec![0.0; config.hidden_size]);
            if has_down_weights {
                down_overrides.insert(*slot, vec![0.0; config.hidden_size]);
            }
        }
        let down_overrides: &HashMap<(usize, usize), Vec<f32>> = &down_overrides;

        let mut overrides_applied = 0usize;
        // Column-replace bake of gate/up/down overlays from install_compiled_slot.
        // This is the primary compile path: at N≤10 per layer it
        // produces working retrieval in the compiled vindex.
        //
        // When MEMIT is enabled (LARQL_MEMIT_ENABLE=1) the ΔW_down is
        // applied as an ADDITIONAL layer on top of this bake (see
        // apply_memit_deltas_to_down_weights below). MEMIT is disabled
        // by default because on Gemma it corrupts template-sharing
        // natives; it remains opt-in for v11 where it is validated.
        if down_overrides.is_empty() {
            let src = path.join(DOWN_WEIGHTS_BIN);
            let dst = output_dir.join(DOWN_WEIGHTS_BIN);
            if src.exists() {
                let _ = std::fs::remove_file(&dst);
                // Copy (not hard-link) when MEMIT will edit bytes.
                if memit_results.is_some() {
                    std::fs::copy(&src, &dst)
                        .map_err(|e| LqlError::exec("copy down_weights for MEMIT", e))?;
                } else if std::fs::hard_link(&src, &dst).is_err() {
                    std::fs::copy(&src, &dst)
                        .map_err(|e| LqlError::exec("copy down_weights", e))?;
                }
            } else if manifest_files.contains(DOWN_WEIGHTS_BIN) {
                return Err(missing_required_file(DOWN_WEIGHTS_BIN));
            }
        } else {
            patch_down_weights(path, output_dir, config, down_overrides)?;
            overrides_applied = down_overrides.len();
        }

        // ── Step 3b/3c: bake gate + up overlays into the compiled vindex ──
        //
        // The dense FFN in a freshly-loaded compiled vindex reads
        // gate and up from `gate_vectors.bin` / `up_features.bin`
        // directly (no patch overlay present in a cold session). If
        // we only bake down, the compiled INFER path computes
        // `silu(weak_source_gate · x) * (weak_source_up · x) *
        // baked_down` at our installed slots — a tiny activation
        // times the right down direction — which is invisible on
        // prompts the model already knows (Gemma's Paris beats a
        // weak baked down in that direction).
        //
        // Baking gate + up into the source files produces the same
        // math the patched session's `sparse_ffn_forward_with_full_overrides`
        // runs, turning the compiled vindex into a self-contained
        // copy of the patched state. Validated by `refine_demo`:
        // patched session = 10/10; compiled = 8/10 pre-fix because
        // gate/up were never baked.
        patch_gate_vectors(path, output_dir, config, &gate_overrides)?;
        patch_up_weights(path, output_dir, config, up_overrides)?;

        // ── Step 4.5: apply MEMIT ΔW_down to baked down_weights.bin ──
        //
        // MEMIT produces additive deltas across the full W_down matrix
        // per layer. We read the current layer slab, add ΔW to it, and
        // write it back. This is applied AFTER the column-replace
        // `patch_down_weights` call so both mechanisms can coexist:
        // MEMIT handles compose-mode PatchOp::Insert (the scale path),
        // and column-replace handles any legacy per-slot edits that
        // may have sneaked in via older patches.
        let mut memit_layers_touched = 0usize;
        if let Some(ref results) = memit_results {
            apply_memit_deltas_to_down_weights(output_dir, config, results)?;
            memit_layers_touched = results.len();
        }

        // ── Step 4.6: write updated config ──
        //
        // After every weight file has its final bytes, so the checksums
        // describe the artifact and not a file MEMIT then edited.
        let mut new_config = config.clone();
        // `patch_gate_vectors` replaces the re-saved gate file with a patched
        // copy of the source's, so the file then has the source's layout and
        // the index must describe that one.
        new_config.layers = if gate_overrides.is_empty() {
            layer_infos
        } else {
            config.layers.clone()
        };
        new_config.checksums = Some(
            larql_vindex::format::checksums::compute_checksums(output_dir)
                .map_err(|e| LqlError::exec("failed to checksum the compiled vindex", e))?,
        );
        larql_vindex::VectorIndex::save_config(&new_config, output_dir)
            .map_err(|e| LqlError::exec("failed to save config", e))?;

        // ── Step 5: serialize KNN store (Architecture B) ──
        let knn_count = patched.knn_store.len();
        if knn_count > 0 {
            patched
                .knn_store
                .save(&output_dir.join(KNN_STORE_BIN))
                .map_err(|e| LqlError::exec("failed to save knn_store", e))?;
        }

        let mut out = Vec::new();
        out.push(format!(
            "Compiled {} → {}",
            path.display(),
            output_dir.display()
        ));
        out.push(format!("Features: {}", dm_count));
        if meta_edited {
            out.push(format!("Metadata edits written to {DOWN_META_BIN}"));
        }
        if !relabelled.is_empty() {
            let preview = relabelled
                .iter()
                .take(UNSTORABLE_PREVIEW)
                .map(|label| label.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            out.push(format!(
                "Labels: {} edited record(s) reload with their token's text, because \
                 {DOWN_META_BIN} stores one token id and no text: {preview}",
                relabelled.len()
            ));
        }
        if !left_behind.is_empty() {
            out.push(format!(
                "Not carried: {} (computed from the source's unedited weights; the compiled \
                 vindex runs from the weights written here)",
                left_behind.join(", ")
            ));
        }
        if !deleted_slots.is_empty() {
            out.push(format!(
                "Deleted slots zeroed: {} (gate row{})",
                deleted_slots.len(),
                if has_down_weights {
                    " and down column"
                } else {
                    "; the source has no down weights"
                },
            ));
        }
        out.extend(memit_warnings);
        if !collisions.is_empty() {
            let strategy = match on_conflict {
                CompileConflict::LastWins => "LAST_WINS",
                CompileConflict::HighestConfidence => {
                    "HIGHEST_CONFIDENCE (resolves like LAST_WINS for down vectors — see docs)"
                }
                CompileConflict::Fail => "FAIL",
            };
            out.push(format!(
                "Conflicts: {} slot(s) touched by multiple patches — strategy: {}",
                collisions.len(),
                strategy,
            ));
        }
        if overrides_applied > 0 {
            out.push(format!(
                "Down overrides baked: {} ({} layers touched)",
                overrides_applied,
                down_overrides
                    .keys()
                    .map(|(l, _)| *l)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
            ));
        }
        if let Some(ref results) = memit_results {
            let total_facts: usize = results.iter().map(|r| r.fact_results.len()).sum();
            out.push(format!(
                "MEMIT ΔW_down applied: {total_facts} compose fact(s) across {memit_layers_touched} layer(s)"
            ));
        }
        if knn_count > 0 {
            out.push(format!("KNN store: {} entries", knn_count));
        }
        out.push(format!("Size: {}", format_bytes(dir_size(output_dir))));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    //! `collect_compile_collisions` unit tests.
    use super::*;
    use larql_vindex::{PatchOp, VindexPatch};

    fn make_patch(ops: Vec<PatchOp>) -> VindexPatch {
        VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: None,
            author: None,
            tags: Vec::new(),
            operations: ops,
        }
    }

    fn insert_op(layer: usize, feature: usize) -> PatchOp {
        PatchOp::Insert {
            layer,
            feature,
            relation: None,
            entity: "e".into(),
            target: "t".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }
    }

    #[test]
    fn collisions_empty_when_each_slot_unique() {
        let patches = vec![
            make_patch(vec![insert_op(1, 10)]),
            make_patch(vec![insert_op(2, 20)]),
        ];
        assert!(collect_compile_collisions(&patches).is_empty());
    }

    #[test]
    fn collisions_detect_same_slot_in_two_patches() {
        let patches = vec![
            make_patch(vec![insert_op(1, 10)]),
            make_patch(vec![insert_op(1, 10)]),
        ];
        let c = collect_compile_collisions(&patches);
        assert_eq!(c.get(&(1, 10)), Some(&2));
    }

    #[test]
    fn collisions_ignore_repeats_within_one_patch() {
        let patches = vec![make_patch(vec![insert_op(1, 10), insert_op(1, 10)])];
        assert!(collect_compile_collisions(&patches).is_empty());
    }
}
