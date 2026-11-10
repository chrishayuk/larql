//! `SELECT * FROM EDGES` — the default SELECT verb.
//!
//! Two modes:
//!   1. With both `entity` and `relation` filters, embed the entity
//!      and walk the FFN at every layer to find features whose
//!      relation label matches. Scales the lookup to "capital
//!      features that fire on France" rather than "features whose
//!      top token contains the substring 'France'".
//!   2. Otherwise scan `feature_meta` directly across the requested
//!      layer/feature filters (handles both heap and mmap modes).
//!
//! After collection, optional WHERE-score filter, ORDER BY, and
//! LIMIT are applied before formatting.

use crate::ast::{Condition, Field, NearestClause, OrderBy};
use crate::error::LqlError;
use crate::executor::predicate::{EdgePredicate, PredicateScope, RelationOp, RelationPredicate};
use crate::executor::Session;

use super::format::{also_display, banner, format_also, EDGES_DEFAULT_LIMIT, EDGES_WALK_TOP_K};

/// One edge an `EDGES` predicate addresses: a `(layer, feature)` slot
/// and what SELECT prints for it.
pub(crate) struct EdgeRow {
    pub(crate) layer: usize,
    pub(crate) feature: usize,
    top_token: String,
    also: String,
    relation: String,
    c_score: f32,
}

/// The edges `predicate` addresses, on whichever backend `ctx` binds.
///
/// This is the **only** resolver of an `EDGES` predicate. SELECT prints
/// its result; DELETE and UPDATE act on the slots of the very same rows.
/// A mutation therefore reaches exactly what the matching SELECT lists —
/// same candidates, same matching rules, same score — before SELECT's
/// ORDER BY and LIMIT, which only arrange and page the listing.
///
/// The rules, stated once:
/// - only slots with current metadata are candidates: the overlay's
///   view, so a tombstoned slot matches nothing and a relabelled one
///   matches by its new token;
/// - `entity` alone is a case-insensitive substring of the slot's **top
///   token** (not its other top-k tokens);
/// - `entity` with `relation` embeds the entity and walks the gates at
///   each layer, keeping hits whose relation label matches; the score
///   compared is then the walk's gate score, otherwise the stored
///   `c_score`;
/// - `layer` / `feature` restrict where either mode looks.
pub(crate) fn resolve_edge_rows(
    ctx: &crate::executor::knowledge::BrowseCtx<'_>,
    classifier: Option<&crate::relations::RelationClassifier>,
    predicate: &EdgePredicate<'_>,
) -> Result<Vec<EdgeRow>, LqlError> {
    let scan_layers: Vec<usize> = match predicate.layer {
        Some(layer) => vec![layer],
        None => ctx.source.loaded_layers(),
    };
    let mut rows: Vec<EdgeRow> = Vec::new();
    collect_edges(
        ctx,
        classifier,
        predicate.entity,
        predicate.relation,
        predicate.feature,
        &scan_layers,
        &mut rows,
    )?;
    if let Some(score) = predicate.score {
        rows.retain(|row| score.matches(row.c_score));
    }
    Ok(rows)
}

impl Session {
    pub(crate) fn exec_select(
        &self,
        _fields: &[Field],
        conditions: &[Condition],
        nearest: Option<&NearestClause>,
        order: Option<&OrderBy>,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        // Validate the whole clause before reading anything: a condition
        // SELECT cannot evaluate is refused, not dropped.
        let filters = EdgePredicate::parse(conditions, PredicateScope::EDGES)?;
        let ctx = self.browse()?;

        if let Some(nc) = nearest {
            // NEAREST ranks by embedding distance and applies no row
            // filter; a WHERE beside it would be validated, then ignored.
            if !conditions.is_empty() {
                return Err(LqlError::Execution(
                    "unsupported WHERE clause: SELECT … NEAREST TO does not evaluate WHERE \
                     conditions; drop the clause, or filter with a plain SELECT"
                        .into(),
                ));
            }
            return self.exec_select_nearest(&ctx, nc, limit);
        }

        // With a feature filter the user expects to see that feature at
        // every layer; otherwise the page-size default applies.
        let default_limit = if filters.feature.is_some() {
            ctx.num_layers
        } else {
            EDGES_DEFAULT_LIMIT as usize
        };
        let limit = limit.unwrap_or(default_limit as u32) as usize;

        let classifier = self.relation_classifier();
        let mut rows = resolve_edge_rows(&ctx, classifier, &filters)?;

        // FR3 — synonym-robust relation addressing. If an exact relation filter
        // matched nothing, the word may be a SYNONYM of a known relation
        // ("seat"→capital, "money"→currency). Resolve it by meaning (a trained
        // residual probe, not string/cosine — see relation_resolver) and
        // re-collect against the canonical relation.
        let mut notes: Vec<String> = Vec::new();
        if rows.is_empty() {
            // Only an equality asks "which relation did you mean?"; `!=`
            // and LIKE describe a set and are taken literally.
            let exact = filters
                .relation
                .filter(|r| r.op == RelationOp::Eq)
                .map(|r| r.wanted);
            if let (Some(rel), Some(rc)) = (exact, classifier) {
                let relations = rc.relation_labels();
                let already_exact = relations.iter().any(|r| r.eq_ignore_ascii_case(rel));
                if relations.len() >= 2 && !already_exact {
                    // FR3b two-tier resolve (the FR2 router shape, for relations):
                    // Tier 1 = the cheap residual probe (synonym-robust, cached);
                    // Tier 2 = explicit few-shot classification on probe abstain
                    // (phrasing-robust — the probe is ~chance on unseen phrasings
                    // at its layer — but a full forward, so opt-in). See
                    // docs/diagnoses/fr3-explicit-rewrite.md.
                    let resolved = self
                        .resolve_relation_synonym(ctx.path, relations.clone(), rel)
                        .map(|(c, conf)| (c, conf, "meaning"))
                        .or_else(|| {
                            // Tier 2 candidates = the frequency-ranked relations
                            // (the meaningful ones), bounded like the probe's set.
                            let cands = rc.relation_labels_ranked(
                                crate::executor::relation_resolver::MAX_RELATIONS,
                            );
                            ctx.config.and_then(|config| {
                                self.resolve_relation_explicit(ctx.path, config, &cands, rel)
                                    .map(|(c, conf)| (c, conf, "explicit classification"))
                            })
                        });
                    if let Some((canonical, conf, how)) = resolved {
                        notes.push(format!(
                            "  (relation '{rel}' resolved to '{canonical}' by {how}, confidence {conf:.2})"
                        ));
                        // A reinterpreted relation is SELECT's alone: a
                        // mutation never acts on a relation the user did
                        // not write (see `resolve_mutation_targets`).
                        let resolved = EdgePredicate {
                            relation: Some(RelationPredicate {
                                op: RelationOp::Eq,
                                wanted: canonical.as_str(),
                            }),
                            ..filters.clone()
                        };
                        rows = resolve_edge_rows(&ctx, classifier, &resolved)?;
                    }
                }
            }
        }

        if let Some(ord) = order {
            sort_rows(&mut rows, ord);
        }

        rows.truncate(limit);

        let mut out = notes;
        out.extend(format_rows(&rows, filters.relation.is_some()));
        Ok(out)
    }

    /// FR3 — build (once, cached per vindex path) the relation resolver and
    /// resolve a relation word to a known canonical relation by meaning.
    fn resolve_relation_synonym(
        &self,
        path: &std::path::Path,
        relations: Vec<String>,
        word: &str,
    ) -> Option<(String, f32)> {
        // Cache hit for the active vindex?
        {
            let cache = self.relation_resolver.borrow();
            if let Some((p, resolver)) = cache.as_ref() {
                if p == path {
                    return resolver.as_ref().and_then(|r| r.resolve(word));
                }
            }
        }
        // Build (one-time forward passes), cache, then resolve.
        let built = crate::executor::relation_resolver::RelationResolver::build(path, relations)
            .ok()
            .flatten();
        let result = built.as_ref().and_then(|r| r.resolve(word));
        *self.relation_resolver.borrow_mut() = Some((path.to_path_buf(), built));
        result
    }

    /// FR3b — explicit relation classification (phrasing-robust Tier 2).
    ///
    /// When the cheap residual probe (Tier 1, [`Self::resolve_relation_synonym`])
    /// abstains, ask the model directly: a few-shot `word -> relation` prompt
    /// with a `none` escape, read top-1 from a **full forward** (lm_head). The
    /// probe is synonym-robust but *phrasing*-brittle (≈chance at its layer on
    /// unseen phrasings like "head city" / "legal tender"); the explicit pass
    /// nails both, and the `none` escape stops out-of-domain words ("weather",
    /// "altitude") snapping to the nearest relation — the project's recurring
    /// confident-wrong trap (cf. FR1's verify gate, FR2's fallback). Measured
    /// 12/12 synonyms+phrasings, 0/3 distractor false-fires
    /// (`docs/diagnoses/fr3-explicit-rewrite.md`).
    ///
    /// The resolver only dequantises `0..=probe_layer`, so it cannot run
    /// lm_head; Tier 2 goes through `InferenceWeights` (the same path INFER
    /// uses). Opt-in via `LARQL_FR3_EXPLICIT` because it is a full forward (plus
    /// a model load) per probe-abstain; default off keeps SELECT byte-identical.
    fn resolve_relation_explicit(
        &self,
        path: &std::path::Path,
        config: &larql_vindex::VindexConfig,
        candidates: &[String],
        word: &str,
    ) -> Option<(String, f32)> {
        // Opt-in: absent var → abstain (the `?` short-circuits to `None`).
        std::env::var_os("LARQL_FR3_EXPLICIT")?;
        if candidates.len() < 2 {
            return None;
        }
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path).ok()?;
        let mut iw = larql_inference::InferenceWeights::load(path, config, &mut cb).ok()?;

        // Few-shot frame lifted verbatim from examples/fr3_explicit_rewrite.rs:
        // the examples pin the "word -> relation" task, and the trailing
        // `music -> none` teaches the `none` escape so an out-of-domain word
        // abstains instead of snapping to a relation. `candidates` is the
        // frequency-ranked, bounded relation set (the meaningful relations, not
        // an alphabetical slice — see `relation_labels_ranked`). The
        // demonstration mappings are tuned for the country-facts relation set
        // (the measured scope); a different relation set should re-verify
        // 12/12 + 0/3 before this is load-bearing for it.
        let rel_list = candidates.join(", ");
        let prompt = format!(
            "Map each word to one of: {rel_list}, none.\n\
             city -> capital\ndollar -> currency\ndialect -> language\nmusic -> none\n\
             {word} ->"
        );
        let ids = tokenizer
            .encode(prompt.as_str(), true)
            .ok()?
            .get_ids()
            .to_vec();
        let result = iw.predict_dense(&tokenizer, &ids, 5);
        let (top1, prob) = result.predictions.first()?;
        match_relation_top1(candidates, top1).map(|r| (r, *prob as f32))
    }
}

/// FR3b — `none`-gated prefix match: which canonical relation (if any) does the
/// explicit classifier's top-1 token indicate? `none` and any out-of-domain
/// token match nothing → abstain. A relation may tokenise to a leading
/// sub-word, so prefix-match in either direction (mirrors the harness's
/// `any_rel_top1`).
fn match_relation_top1(relations: &[String], top1: &str) -> Option<String> {
    let t = top1.trim().to_lowercase();
    if t.is_empty() {
        return None;
    }
    relations
        .iter()
        .find(|r| {
            let r = r.to_lowercase();
            r.starts_with(&t) || t.starts_with(&r)
        })
        .cloned()
}

/// Dispatch edge collection: walk-anchored when both entity and relation are
/// given, else a metadata scan. Shared so the FR3 synonym fallback can re-run
/// it against the resolved canonical relation.
#[allow(clippy::too_many_arguments)]
fn collect_edges(
    ctx: &crate::executor::knowledge::BrowseCtx<'_>,
    classifier: Option<&crate::relations::RelationClassifier>,
    entity: Option<&str>,
    relation: Option<RelationPredicate<'_>>,
    feature: Option<usize>,
    scan_layers: &[usize],
    rows: &mut Vec<EdgeRow>,
) -> Result<(), LqlError> {
    if let (Some(entity), Some(rel)) = (entity, relation) {
        collect_via_walk(ctx, classifier, entity, rel, feature, scan_layers, rows)?;
    } else {
        collect_via_scan(
            &ctx.source,
            classifier,
            entity,
            relation,
            feature,
            scan_layers,
            rows,
        );
    }
    Ok(())
}

/// Walk-anchored collection: embed the entity, walk every requested
/// layer, filter hits by the relation label.
#[allow(clippy::too_many_arguments)]
fn collect_via_walk(
    ctx: &crate::executor::knowledge::BrowseCtx<'_>,
    classifier: Option<&crate::relations::RelationClassifier>,
    entity: &str,
    rel: RelationPredicate<'_>,
    feature_filter: Option<usize>,
    scan_layers: &[usize],
    rows: &mut Vec<EdgeRow>,
) -> Result<(), LqlError> {
    let (embed, embed_scale) = ctx.embeddings()?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(ctx.path)
        .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

    let Some(query) =
        crate::executor::helpers::entity_query_vec(&tokenizer, &embed, embed_scale, entity)?
    else {
        return Ok(());
    };

    let trace = ctx.source.walk(&query, scan_layers, EDGES_WALK_TOP_K);

    for (layer_idx, hits) in &trace.layers {
        for hit in hits {
            if let Some(ff) = feature_filter {
                if hit.feature != ff {
                    continue;
                }
            }
            let rel_label = classifier
                .and_then(|rc| rc.label_for_feature(*layer_idx, hit.feature))
                .unwrap_or("")
                .to_string();
            if !rel.matches(&rel_label) {
                continue;
            }
            rows.push(EdgeRow {
                layer: *layer_idx,
                feature: hit.feature,
                top_token: hit.meta.top_token.clone(),
                also: format_also(&hit.meta.top_k),
                relation: rel_label,
                c_score: hit.gate_score,
            });
        }
    }

    Ok(())
}

/// Direct metadata scan: enumerate features at the requested layers
/// and apply optional entity/relation/feature filters.
fn collect_via_scan(
    source: &crate::executor::knowledge::KnowledgeSource<'_>,
    classifier: Option<&crate::relations::RelationClassifier>,
    entity_filter: Option<&str>,
    relation_filter: Option<RelationPredicate<'_>>,
    feature_filter: Option<usize>,
    scan_layers: &[usize],
    rows: &mut Vec<EdgeRow>,
) {
    for layer in scan_layers {
        let nf = source.num_features(*layer);
        for feat_idx in 0..nf {
            if let Some(ff) = feature_filter {
                if feat_idx != ff {
                    continue;
                }
            }
            let Some(meta) = source.feature_meta(*layer, feat_idx) else {
                continue;
            };
            if let Some(ent) = entity_filter {
                if !meta.top_token.to_lowercase().contains(&ent.to_lowercase()) {
                    continue;
                }
            }
            let rel_label = classifier
                .and_then(|rc| rc.label_for_feature(*layer, feat_idx))
                .unwrap_or("")
                .to_string();
            if let Some(rel) = relation_filter {
                if !rel.matches(&rel_label) {
                    continue;
                }
            }
            rows.push(EdgeRow {
                layer: *layer,
                feature: feat_idx,
                top_token: meta.top_token.clone(),
                also: format_also(&meta.top_k),
                relation: rel_label,
                c_score: meta.c_score,
            });
        }
    }
}

fn sort_rows(rows: &mut [EdgeRow], ord: &OrderBy) {
    match ord.field.as_str() {
        "confidence" | "c_score" => {
            rows.sort_by(|a, b| {
                let cmp = a
                    .c_score
                    .partial_cmp(&b.c_score)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if ord.descending {
                    cmp.reverse()
                } else {
                    cmp
                }
            });
        }
        "layer" => {
            rows.sort_by(|a, b| {
                let cmp = a.layer.cmp(&b.layer);
                if ord.descending {
                    cmp.reverse()
                } else {
                    cmp
                }
            });
        }
        _ => {}
    }
}

fn format_rows(rows: &[EdgeRow], explicit_relation_filter: bool) -> Vec<String> {
    let show_relation = explicit_relation_filter || rows.iter().any(|r| !r.relation.is_empty());
    let show_also = rows.iter().any(|r| !r.also.is_empty());

    let mut out = Vec::new();

    let (header, banner_len) = match (show_relation, show_also) {
        (true, true) => (
            format!(
                "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
                "Layer", "Feature", "Token", "Also", "Relation", "Score"
            ),
            86,
        ),
        (true, false) => (
            format!(
                "{:<8} {:<8} {:<20} {:<20} {:>10}",
                "Layer", "Feature", "Token", "Relation", "Score"
            ),
            70,
        ),
        (false, true) => (
            format!(
                "{:<8} {:<8} {:<16} {:<28} {:>8}",
                "Layer", "Feature", "Token", "Also", "Score"
            ),
            72,
        ),
        (false, false) => (
            format!(
                "{:<8} {:<8} {:<20} {:>10}",
                "Layer", "Feature", "Token", "Score"
            ),
            50,
        ),
    };
    out.push(header);
    out.push(banner(banner_len));

    for row in rows {
        let also = also_display(&row.also);
        match (show_relation, show_also) {
            (true, true) => out.push(format!(
                "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                row.layer, row.feature, row.top_token, also, row.relation, row.c_score
            )),
            (true, false) => out.push(format!(
                "L{:<7} F{:<7} {:20} {:20} {:>10.4}",
                row.layer, row.feature, row.top_token, row.relation, row.c_score
            )),
            (false, true) => out.push(format!(
                "L{:<7} F{:<7} {:16} {:28} {:>8.4}",
                row.layer, row.feature, row.top_token, also, row.c_score
            )),
            (false, false) => out.push(format!(
                "L{:<7} F{:<7} {:20} {:>10.4}",
                row.layer, row.feature, row.top_token, row.c_score
            )),
        }
    }

    if rows.is_empty() {
        out.push("  (no matching edges)".into());
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sort_rows_by_layer_descending() {
        let mut rows = vec![
            EdgeRow {
                layer: 1,
                feature: 0,
                top_token: "".into(),
                also: "".into(),
                relation: "".into(),
                c_score: 0.0,
            },
            EdgeRow {
                layer: 5,
                feature: 0,
                top_token: "".into(),
                also: "".into(),
                relation: "".into(),
                c_score: 0.0,
            },
            EdgeRow {
                layer: 3,
                feature: 0,
                top_token: "".into(),
                also: "".into(),
                relation: "".into(),
                c_score: 0.0,
            },
        ];
        sort_rows(
            &mut rows,
            &OrderBy {
                field: "layer".into(),
                descending: true,
            },
        );
        assert_eq!(rows[0].layer, 5);
        assert_eq!(rows[1].layer, 3);
        assert_eq!(rows[2].layer, 1);
    }

    #[test]
    fn format_rows_empty_emits_no_match_line() {
        let out = format_rows(&[], false);
        assert!(out.last().unwrap().contains("no matching edges"));
    }

    #[test]
    fn format_rows_chooses_widest_layout_with_relation_and_also() {
        let row = EdgeRow {
            layer: 1,
            feature: 2,
            top_token: "Paris".into(),
            also: "French, Europe".into(),
            relation: "capital".into(),
            c_score: 0.95,
        };
        let out = format_rows(&[row], true);
        assert!(out[0].contains("Relation"));
        assert!(out[0].contains("Also"));
        assert!(out.iter().any(|l| l.contains("Paris")));
        assert!(out.iter().any(|l| l.contains("[French, Europe]")));
    }

    #[test]
    fn match_relation_top1_accepts_exact_and_subword_relations() {
        let rels = vec![
            "capital".to_string(),
            "currency".to_string(),
            "language".to_string(),
        ];
        // Full-word top-1 (the common case — these tokenise to one token).
        assert_eq!(
            match_relation_top1(&rels, " capital").as_deref(),
            Some("capital")
        );
        assert_eq!(
            match_relation_top1(&rels, "Currency").as_deref(),
            Some("currency")
        );
        // Leading sub-word still resolves (prefix-match in either direction).
        assert_eq!(
            match_relation_top1(&rels, "lang").as_deref(),
            Some("language")
        );
    }

    #[test]
    fn match_relation_top1_abstains_on_none_and_out_of_domain() {
        let rels = vec![
            "capital".to_string(),
            "currency".to_string(),
            "language".to_string(),
        ];
        // The `none` escape: top-1 == none → no relation → abstain.
        assert_eq!(match_relation_top1(&rels, "none"), None);
        // Out-of-domain distractors abstain (the confident-wrong fix).
        assert_eq!(match_relation_top1(&rels, "weather"), None);
        assert_eq!(match_relation_top1(&rels, "banana"), None);
        // Empty / whitespace top-1 abstains rather than panicking.
        assert_eq!(match_relation_top1(&rels, "   "), None);
    }

    #[test]
    fn format_rows_drops_relation_column_when_no_filter_and_no_label() {
        let row = EdgeRow {
            layer: 0,
            feature: 0,
            top_token: "Foo".into(),
            also: "".into(),
            relation: "".into(),
            c_score: 0.5,
        };
        let out = format_rows(&[row], false);
        assert!(!out[0].contains("Relation"));
        assert!(!out[0].contains("Also"));
    }
}
