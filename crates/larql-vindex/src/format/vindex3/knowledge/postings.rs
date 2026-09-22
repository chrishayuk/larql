//! GW-3A's deliberately ordinary access path: inverted token postings over
//! VINDEX3's existing `feature_gate` and `feature_down` semantic roles.
//!
//! Building the index is offline work over a frozen source-key vocabulary. For
//! every declared token, source postings are the features with the largest
//! absolute detector response; target postings are the tokens promoted by those
//! features' down directions. A lookup is then a union of the subject token
//! postings, optionally intersected with a relation-derived layer mask. It
//! performs no gate-vector dot products.
//!
//! This is an experimental baseline, not a replacement for exact WALK. It
//! keeps physical feature addresses and accounting explicit so phase-one
//! experiments can compare recall with candidate coverage and bytes touched.

use std::collections::{BTreeMap, BTreeSet};

use super::KnowledgeView;

/// A stable physical address in the current VINDEX3 browse surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeatureAddress {
    pub layer: usize,
    pub feature: usize,
}

/// The result of one postings lookup. Counts are logical payload counts; they
/// deliberately exclude allocator and map overhead so implementations remain
/// comparable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostingLookup {
    pub candidates: Vec<FeatureAddress>,
    pub promoted_token_ids: Vec<u32>,
    pub source_postings_touched: usize,
    pub target_postings_touched: usize,
    pub eligible_features: usize,
    pub total_features: usize,
    pub logical_bytes_touched: usize,
}

impl PostingLookup {
    /// Fraction of the address universe admitted for this lookup. A layer
    /// mask changes both the candidates and this denominator.
    pub fn candidate_fraction(&self) -> f64 {
        if self.eligible_features == 0 {
            0.0
        } else {
            self.candidates.len() as f64 / self.eligible_features as f64
        }
    }
}

/// A non-learned inverted index built only from the executable container's
/// semantic roles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostingIndex {
    source: BTreeMap<u32, Vec<FeatureAddress>>,
    targets: BTreeMap<FeatureAddress, Vec<u32>>,
    features_per_layer: BTreeMap<usize, usize>,
    total_features: usize,
    logical_index_bytes: usize,
    features_per_source: usize,
    target_top_k: usize,
    source_tokens: usize,
}

impl PostingIndex {
    pub fn features_per_source(&self) -> usize {
        self.features_per_source
    }

    pub fn target_top_k(&self) -> usize {
        self.target_top_k
    }

    pub fn source_tokens(&self) -> usize {
        self.source_tokens
    }

    pub fn total_features(&self) -> usize {
        self.total_features
    }

    /// Logical encoded size using `(token_id:u32, layer:u32, feature:u32)` for
    /// source entries and `(layer:u32, feature:u32, token_id:u32)` for target
    /// entries. This is the data-structure-independent quantity GW-7 may later
    /// turn into a physical layout.
    pub fn logical_index_bytes(&self) -> usize {
        self.logical_index_bytes
    }

    /// Logical encoded size of a per-layer prefix of this index. This lets a
    /// frozen maximum-width build expose a complete width curve without
    /// rescanning the gate matrices at every operating point.
    pub fn logical_index_bytes_at_width(&self, width: usize) -> usize {
        let source_entries = self
            .source
            .values()
            .map(|postings| {
                let mut per_layer = BTreeMap::<usize, usize>::new();
                postings
                    .iter()
                    .filter(|address| {
                        let count = per_layer.entry(address.layer).or_default();
                        let keep = *count < width;
                        *count += 1;
                        keep
                    })
                    .count()
            })
            .sum::<usize>();
        let addresses = self
            .source
            .values()
            .flat_map(|postings| {
                let mut per_layer = BTreeMap::<usize, usize>::new();
                postings.iter().copied().filter(move |address| {
                    let count = per_layer.entry(address.layer).or_default();
                    let keep = *count < width;
                    *count += 1;
                    keep
                })
            })
            .collect::<BTreeSet<_>>();
        let target_entries = addresses
            .iter()
            .map(|address| self.promoted_tokens(*address).len())
            .sum::<usize>();
        (source_entries + target_entries) * 12
    }

    pub fn source_postings(&self, token_id: u32) -> &[FeatureAddress] {
        self.source.get(&token_id).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn promoted_tokens(&self, address: FeatureAddress) -> &[u32] {
        self.targets.get(&address).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Resolve subject tokens through the inverted index. `eligible_layers`
    /// is the frozen GW-1 mask; `None` is the unmasked GW-3A control.
    pub fn lookup(
        &self,
        subject_token_ids: &[u32],
        eligible_layers: Option<&[usize]>,
    ) -> PostingLookup {
        self.lookup_at_width(subject_token_ids, eligible_layers, self.features_per_source)
    }

    /// Resolve a prefix of the frozen source postings. `width` applies within
    /// each layer and may not exceed the width used to build the index.
    pub fn lookup_at_width(
        &self,
        subject_token_ids: &[u32],
        eligible_layers: Option<&[usize]>,
        width: usize,
    ) -> PostingLookup {
        let width = width.min(self.features_per_source);
        let layer_mask: Option<BTreeSet<usize>> =
            eligible_layers.map(|layers| layers.iter().copied().collect());
        let layer_is_eligible = |layer: usize| {
            layer_mask
                .as_ref()
                .is_none_or(|eligible| eligible.contains(&layer))
        };

        let mut candidates = BTreeSet::new();
        let mut source_postings_touched = 0usize;
        for token_id in subject_token_ids.iter().copied().collect::<BTreeSet<_>>() {
            let postings = self.source_postings(token_id);
            let mut per_layer = BTreeMap::<usize, usize>::new();
            for address in postings {
                let count = per_layer.entry(address.layer).or_default();
                let in_prefix = *count < width;
                *count += 1;
                if in_prefix && layer_is_eligible(address.layer) {
                    source_postings_touched += 1;
                    candidates.insert(*address);
                }
            }
        }

        let mut promoted = BTreeSet::new();
        let mut target_postings_touched = 0usize;
        for address in &candidates {
            let targets = self.promoted_tokens(*address);
            target_postings_touched += targets.len();
            promoted.extend(targets.iter().copied());
        }

        let eligible_features = self
            .features_per_layer
            .iter()
            .filter(|(layer, _)| layer_is_eligible(**layer))
            .map(|(_, count)| count)
            .sum();
        // A posting payload is three u32 values. Query keys and map metadata
        // are intentionally excluded from both stored and touched bytes.
        let logical_bytes_touched = (source_postings_touched + target_postings_touched) * 12;

        PostingLookup {
            candidates: candidates.into_iter().collect(),
            promoted_token_ids: promoted.into_iter().collect(),
            source_postings_touched,
            target_postings_touched,
            eligible_features,
            total_features: self.total_features,
            logical_bytes_touched,
        }
    }
}

impl KnowledgeView {
    /// Build GW-3A's token postings from `feature_gate`, `feature_down`, and
    /// `embedding`. `source_token_ids` is the key vocabulary frozen before the
    /// evaluation; aliases and held-out subjects may be present, but labels do
    /// not affect construction. Each token uses the exact WALK statistic
    /// (absolute dot product) with stable feature-index tie breaking. Targets
    /// reuse the existing feature annotations, whose ordering is
    /// `embedding · feature_down`.
    pub fn build_posting_index(
        &self,
        layers: &[usize],
        source_token_ids: &[u32],
        features_per_source: usize,
        target_top_k: usize,
    ) -> Result<PostingIndex, crate::error::VindexError> {
        let selected: BTreeSet<usize> = layers.iter().copied().collect();
        let target_top_k = target_top_k.min(super::ANNOTATION_TOP_K);
        let source_tokens: BTreeSet<u32> = source_token_ids.iter().copied().collect();
        if let Some(token_id) = source_tokens
            .iter()
            .find(|&&token_id| token_id as usize >= self.vocab_size)
        {
            return Err(crate::error::VindexError::Parse(format!(
                "GW-3A source token {token_id} is outside vocabulary {}",
                self.vocab_size
            )));
        }
        let mut source = BTreeMap::<u32, Vec<FeatureAddress>>::new();
        let mut targets = BTreeMap::<FeatureAddress, Vec<u32>>::new();
        let mut features_per_layer = BTreeMap::new();
        let mut total_features = 0usize;

        for layer in self.loaded_layers() {
            if !selected.contains(&layer) {
                continue;
            }
            let knowledge = self.layers[layer].as_ref().expect("loaded layer");
            let features = knowledge.gate.nrows();
            features_per_layer.insert(layer, features);
            total_features += features;

            for &token_id in &source_tokens {
                let query = self.embedding.row(token_id as usize).to_owned() * self.embed_scale;
                let scores = knowledge.gate.dot(&query);
                let postings = source.entry(token_id).or_default();
                postings.extend(
                    top_abs_indices(
                        scores.as_slice().expect("contiguous"),
                        features_per_source.min(features),
                    )
                    .into_iter()
                    .map(|feature| FeatureAddress { layer, feature }),
                );
            }
        }

        if target_top_k > 0 {
            let candidates: BTreeSet<FeatureAddress> = source.values().flatten().copied().collect();
            for address in candidates {
                if let Some(meta) = self.feature_meta(address.layer, address.feature) {
                    let token_ids = meta
                        .top_k
                        .iter()
                        .take(target_top_k)
                        .map(|entry| entry.token_id)
                        .collect::<Vec<_>>();
                    if !token_ids.is_empty() {
                        targets.insert(address, token_ids);
                    }
                }
            }
        }

        let source_entries = source.values().map(Vec::len).sum::<usize>();
        let target_entries = targets.values().map(Vec::len).sum::<usize>();
        Ok(PostingIndex {
            source,
            targets,
            features_per_layer,
            total_features,
            logical_index_bytes: (source_entries + target_entries) * 12,
            features_per_source,
            target_top_k,
            source_tokens: source_tokens.len(),
        })
    }
}

fn top_abs_indices(scores: &[f32], top_k: usize) -> Vec<usize> {
    let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
        .into_iter()
        .take(top_k)
        .map(|(index, _)| index)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::top_abs_indices;

    #[test]
    fn absolute_ranking_is_stable_on_token_id() {
        assert_eq!(top_abs_indices(&[1.0, -2.0, 2.0, 0.5], 3), vec![1, 2, 0]);
        assert!(top_abs_indices(&[1.0], 0).is_empty());
    }
}
