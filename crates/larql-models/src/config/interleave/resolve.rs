//! Resolving declarations into one kind per layer.
//!
//! The base is **proven from the declaration**, never taken from a family
//! table: the same key names zero-based sets on GLM-5.3-Flash and
//! one-based sets on Kimi Linear, so a table would have to be right about
//! every future checkpoint in advance, and would be silently wrong when it
//! was not.

use super::{
    Declaration, InterleaveEncoding, InterleaveError, InterleaveProvenance, LayerIndexBase,
    LayerKind, Membership, ResolvedInterleave,
};

/// Why one candidate base did not resolve.
///
/// Kept apart from [`InterleaveError`] because a failure under the *wrong*
/// base is expected and says nothing: an overlap only means something once
/// no base worked at all.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AttemptFailure {
    /// An index lands outside the scope — the ordinary way a wrong base
    /// fails, and never worth reporting on its own.
    OutOfRange,
    Overlap {
        layer: usize,
    },
    Uncovered {
        layer: usize,
    },
}

/// Place every declaration under one candidate base.
fn attempt(
    declarations: &[Declaration],
    layer_count: usize,
    base: LayerIndexBase,
) -> Result<Vec<LayerKind>, AttemptFailure> {
    let mut slots: Vec<Option<LayerKind>> = vec![None; layer_count];
    for declaration in declarations {
        let Membership::ExplicitSet(indices) = &declaration.membership else {
            continue;
        };
        for declared in indices {
            let zero_based = declared - base.offset();
            let layer = usize::try_from(zero_based)
                .ok()
                .filter(|l| *l < layer_count)
                .ok_or(AttemptFailure::OutOfRange)?;
            if slots[layer].is_some() {
                return Err(AttemptFailure::Overlap { layer });
            }
            slots[layer] = Some(declaration.kind.clone());
        }
    }
    // The complement takes what is left — and only after the base has
    // placed every explicit index, so a wrong base cannot be rescued by
    // sweeping its mistakes into the complement.
    if let Some(complement) = declarations
        .iter()
        .find(|d| d.membership == Membership::Complement)
    {
        return Ok(slots
            .into_iter()
            .map(|slot| slot.unwrap_or_else(|| complement.kind.clone()))
            .collect());
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(layer, slot)| slot.ok_or(AttemptFailure::Uncovered { layer }))
        .collect()
}

/// Resolve set-based declarations, proving the index base.
///
/// Succeeds only when exactly one base places every declared index inside
/// the scope and leaves every layer with exactly one kind.
pub fn resolve_declarations(
    scope: &str,
    sources: Vec<String>,
    declarations: &[Declaration],
    layer_count: usize,
) -> Result<ResolvedInterleave, InterleaveError> {
    let declared_indices: usize = declarations
        .iter()
        .filter_map(|d| match &d.membership {
            Membership::ExplicitSet(indices) => Some(indices.len()),
            Membership::Complement => None,
        })
        .sum();
    if declared_indices == 0 {
        return Err(InterleaveError::NotDeclared);
    }
    if declarations
        .iter()
        .filter(|d| d.membership == Membership::Complement)
        .count()
        > 1
    {
        return Err(InterleaveError::MultipleComplements);
    }

    let complement_implied = declarations
        .iter()
        .any(|d| d.membership == Membership::Complement);
    let attempts: Vec<(LayerIndexBase, Result<Vec<LayerKind>, AttemptFailure>)> =
        LayerIndexBase::ALL
            .into_iter()
            .map(|base| (base, attempt(declarations, layer_count, base)))
            .collect();

    let mut resolved = attempts.iter().filter(|(_, r)| r.is_ok());
    match (resolved.next(), resolved.next()) {
        (Some((base, Ok(layers))), None) => Ok(ResolvedInterleave {
            layer_count,
            provenance: InterleaveProvenance {
                sources,
                encoding: if complement_implied {
                    InterleaveEncoding::ExplicitSetWithComplement
                } else {
                    InterleaveEncoding::PartitionSets
                },
                resolved_base: Some(*base),
                scope: scope.to_string(),
            },
            layers: layers.clone(),
        }),
        // Both bases place it validly, so the declaration does not
        // determine its own reading. Only reachable with an implied
        // complement — a partition of `0..n` cannot also partition
        // `1..=n`, since one contains 0 and the other does not.
        (Some(_), Some(_)) => Err(InterleaveError::AmbiguousBase { layer_count }),
        _ => {
            // No base worked. An overlap or a hole is worth naming; an
            // out-of-range index is just how a wrong base fails.
            let structural = attempts.iter().find_map(|(_, r)| match r {
                Err(AttemptFailure::Overlap { layer }) => {
                    Some(InterleaveError::Overlap { layer: *layer })
                }
                Err(AttemptFailure::Uncovered { layer }) => {
                    Some(InterleaveError::Uncovered { layer: *layer })
                }
                _ => None,
            });
            Err(structural.unwrap_or(InterleaveError::NoConsistentBase {
                declared_indices,
                layer_count,
            }))
        }
    }
}

/// Resolve a per-layer array, one entry per layer.
///
/// No base to prove — position *is* the layer index — but the length and
/// every entry still have to be right, and an entry with no kind blocks
/// rather than taking a default.
pub fn resolve_per_layer_array(
    scope: &str,
    source: &str,
    entries: &[String],
    layer_count: usize,
    kind_of: impl Fn(&str) -> Option<LayerKind>,
) -> Result<ResolvedInterleave, InterleaveError> {
    if entries.is_empty() {
        return Err(InterleaveError::NotDeclared);
    }
    if entries.len() != layer_count {
        return Err(InterleaveError::LengthMismatch {
            declared: entries.len(),
            layer_count,
        });
    }
    // An entry with no kind becomes `Unexpressed` for THAT layer rather
    // than failing the array: the invariant is one kind per layer, and a
    // spelling this build cannot read is a fact about its layer, not about
    // the 34 beside it that read cleanly.
    let layers: Vec<LayerKind> = entries
        .iter()
        .map(|entry| {
            kind_of(entry).unwrap_or_else(|| LayerKind::Unexpressed {
                declared: entry.clone(),
            })
        })
        .collect();
    Ok(ResolvedInterleave {
        layer_count,
        provenance: InterleaveProvenance {
            sources: vec![source.to_string()],
            encoding: InterleaveEncoding::PerLayerArray,
            resolved_base: None,
            scope: scope.to_string(),
        },
        layers,
    })
}
