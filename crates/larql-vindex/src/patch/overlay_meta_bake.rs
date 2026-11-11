//! Writing the overlay's metadata edits into a compiled `down_meta.bin`.
//!
//! `INSERT`, `UPDATE` and `DELETE` change feature metadata in the overlay
//! only (`overrides_meta` and the tombstone set). A compile that copies
//! the source `down_meta.bin` therefore drops all three. This module is
//! the one place that turns the overlay's effective metadata into the
//! on-disk record, so the compiled artifact answers `feature_meta` the
//! way the patched session did.
//!
//! `down_meta.bin` stores a token id, a score and the top-k ids. It does
//! not store token text: a loader derives the text from the id. So a
//! record reloads as its token's text, whatever text the session showed.
//!
//! - An edit that changes a slot's text and keeps its token id is nothing
//!   but a text change, and the file cannot hold it. Neither can a record
//!   the reader takes for an empty slot, or a slot outside the layer.
//!   [`PatchedVindex::unstorable_meta_edits`] reports these so the caller
//!   can refuse instead of writing a different record.
//! - A record whose text is not exactly its token (a multi-token target is
//!   stored as its first token) is stored, and reloads as the token.
//!   [`PatchedVindex::labels_stored_as_their_token`] lists those so the
//!   caller can say so.

use std::path::Path;

use crate::error::VindexError;
use crate::index::FeatureMeta;

use super::overlay::PatchedVindex;

/// Top-k width used when no stored record carries any top-k entry.
/// Matches the default `VectorIndex::save_down_meta` falls back to.
const DEFAULT_TOP_K_WIDTH: usize = 10;

/// Why one overlay metadata edit cannot be written to `down_meta.bin`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnstorableMetaEdit {
    /// The slot is outside the layer's feature range, so the record file
    /// has no row for it.
    OutOfRange {
        layer: usize,
        feature: usize,
        num_features: usize,
    },
    /// The edit changed the token text but kept the token id. The file
    /// stores the id only, so a reload would show the old text.
    TextWithoutTokenId {
        layer: usize,
        feature: usize,
        stored_text: String,
        edited_text: String,
    },
    /// Token id 0, score 0.0 and no top-k entry is how the file marks an
    /// empty slot, so this record would reload as no record.
    ReadsBackEmpty { layer: usize, feature: usize },
}

impl std::fmt::Display for UnstorableMetaEdit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfRange {
                layer,
                feature,
                num_features,
            } => write!(
                f,
                "L{layer}/F{feature} is outside the layer's {num_features} feature(s)"
            ),
            Self::TextWithoutTokenId {
                layer,
                feature,
                stored_text,
                edited_text,
            } => write!(
                f,
                "L{layer}/F{feature} changes the target text from {stored_text:?} to \
                 {edited_text:?} without changing its token id"
            ),
            Self::ReadsBackEmpty { layer, feature } => write!(
                f,
                "L{layer}/F{feature} has token id 0, score 0 and no top-k entry, which the \
                 file reads back as an empty slot"
            ),
        }
    }
}

/// An edited record that is stored, and reloads with its token's text in
/// place of the text the session showed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelStoredAsToken {
    pub layer: usize,
    pub feature: usize,
    /// What the patched session shows.
    pub session_text: String,
    /// What a loader derives from the stored token id.
    pub stored_text: String,
}

impl std::fmt::Display for LabelStoredAsToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "L{}/F{} {:?} reloads as {:?}",
            self.layer, self.feature, self.session_text, self.stored_text
        )
    }
}

impl PatchedVindex {
    /// Whether the overlay changes any feature's metadata: an insert, an
    /// update or a delete.
    pub fn has_meta_edits(&self) -> bool {
        !self.overrides_meta.is_empty() || !self.deleted.is_empty()
    }

    /// How many slots a `DELETE` has tombstoned and nothing has resurrected.
    pub fn num_deleted(&self) -> usize {
        self.deleted.len()
    }

    /// Whether the overlay carries any slot edit at all: a metadata record,
    /// or a gate, up or down vector.
    pub fn has_vector_or_meta_overrides(&self) -> bool {
        self.has_meta_edits()
            || self.overrides_gate_iter().next().is_some()
            || !self.down_overrides().is_empty()
            || !self.up_overrides().is_empty()
    }

    /// Whether `(layer, feature)` is tombstoned by a `DELETE` that no
    /// later insert or update has resurrected.
    pub fn is_deleted(&self, layer: usize, feature: usize) -> bool {
        self.deleted.contains(&(layer, feature))
    }

    /// The slots a `DELETE` has tombstoned, in `(layer, feature)` order.
    pub fn deleted_slots(&self) -> Vec<(usize, usize)> {
        let mut slots: Vec<(usize, usize)> = self.deleted.iter().copied().collect();
        slots.sort_unstable();
        slots
    }

    /// The slots whose metadata the overlay edits, in `(layer, feature)`
    /// order.
    fn meta_edit_slots(&self) -> Vec<(usize, usize)> {
        let mut slots: Vec<(usize, usize)> = self
            .overrides_meta
            .keys()
            .chain(self.deleted.iter())
            .copied()
            .collect();
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    /// The base slot's text, when `edited` keeps the slot's token id and
    /// shows different text.
    fn text_only_change(
        &self,
        layer: usize,
        feature: usize,
        edited: &FeatureMeta,
    ) -> Option<String> {
        let stored = self.base.feature_meta(layer, feature)?;
        (stored.top_token_id == edited.top_token_id && stored.top_token != edited.top_token)
            .then_some(stored.top_token)
    }

    /// The overlay metadata edits that `down_meta.bin` cannot hold, in
    /// `(layer, feature)` order. Empty means
    /// [`Self::save_effective_down_meta`] writes every edit.
    pub fn unstorable_meta_edits(&self) -> Vec<UnstorableMetaEdit> {
        let mut unstorable = Vec::new();
        for (layer, feature) in self.meta_edit_slots() {
            let num_features = self.base.num_features(layer);
            if feature >= num_features {
                unstorable.push(UnstorableMetaEdit::OutOfRange {
                    layer,
                    feature,
                    num_features,
                });
                continue;
            }
            let Some(Some(edited)) = self.overrides_meta.get(&(layer, feature)) else {
                continue;
            };
            if edited.top_token_id == 0 && edited.c_score == 0.0 && edited.top_k.is_empty() {
                unstorable.push(UnstorableMetaEdit::ReadsBackEmpty { layer, feature });
                continue;
            }
            if let Some(stored_text) = self.text_only_change(layer, feature, edited) {
                unstorable.push(UnstorableMetaEdit::TextWithoutTokenId {
                    layer,
                    feature,
                    stored_text,
                    edited_text: edited.top_token.clone(),
                });
            }
        }
        unstorable
    }

    /// The edited records that are stored and reload with different text,
    /// in `(layer, feature)` order.
    ///
    /// `token_text` gives the text a loader derives from a token id, or
    /// `None` when there is no tokenizer; a token with no text (an unknown
    /// id, a special token) says nothing about the record and is not
    /// listed. Records [`Self::unstorable_meta_edits`] reports are not
    /// listed either.
    pub fn labels_stored_as_their_token(
        &self,
        token_text: &dyn Fn(u32) -> Option<String>,
    ) -> Vec<LabelStoredAsToken> {
        let mut labels = Vec::new();
        for (layer, feature) in self.meta_edit_slots() {
            if feature >= self.base.num_features(layer) {
                continue;
            }
            let Some(Some(edited)) = self.overrides_meta.get(&(layer, feature)) else {
                continue;
            };
            if self.text_only_change(layer, feature, edited).is_some() {
                continue;
            }
            let Some(token) = token_text(edited.top_token_id) else {
                continue;
            };
            let (token, session) = (token.trim(), edited.top_token.trim());
            if !token.is_empty() && token != session {
                labels.push(LabelStoredAsToken {
                    layer,
                    feature,
                    session_text: session.to_string(),
                    stored_text: token.to_string(),
                });
            }
        }
        labels
    }

    /// The metadata every slot has in the patched session: overlay edits
    /// over the base, tombstoned slots empty. One row per base feature,
    /// so the record count stays aligned with `gate_vectors.bin`.
    fn effective_down_meta(&self) -> Vec<Option<Vec<Option<FeatureMeta>>>> {
        (0..self.base.num_layers)
            .map(|layer| {
                let num_features = self.base.num_features(layer);
                if num_features == 0 {
                    return None;
                }
                Some(
                    (0..num_features)
                        .map(|feature| self.feature_meta(layer, feature))
                        .collect(),
                )
            })
            .collect()
    }

    /// Write `down_meta.bin` in `dir` from the patched session's effective
    /// metadata. Returns the number of non-empty records written.
    ///
    /// `min_top_k` is the source's declared top-k width; the file is
    /// written at least that wide and wider if an edited record carries
    /// more entries, so nothing is truncated. The write goes to a
    /// temporary file and is renamed into place, so a hard link at the
    /// destination is replaced and the file it pointed to is not touched.
    pub fn save_effective_down_meta(
        &self,
        dir: &Path,
        min_top_k: usize,
    ) -> Result<usize, VindexError> {
        let down_meta = self.effective_down_meta();
        let widest = down_meta
            .iter()
            .flatten()
            .flat_map(|features| features.iter().flatten())
            .map(|meta| meta.top_k.len())
            .max()
            .unwrap_or(0);
        let top_k_width = match widest.max(min_top_k) {
            0 => DEFAULT_TOP_K_WIDTH,
            width => width,
        };
        crate::format::down_meta::write_binary(dir, &down_meta, top_k_width)
    }
}
