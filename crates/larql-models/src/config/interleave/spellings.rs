//! The known spellings, each mapped to the same canonical declarations.
//!
//! One function per spelling, all producing [`Declaration`]s, so adding a
//! checkpoint adds a reader and never a second resolution rule. What each
//! reader must *not* do is decide the index base — that is proven from the
//! declaration in [`super::resolve`], because the same key is zero-based
//! on GLM-5.3-Flash and one-based on Kimi Linear.

use serde_json::Value;

use super::{
    resolve_declarations, resolve_per_layer_array, Declaration, DeclaredInterleave,
    InterleaveError, LayerKind, Membership, RecurrenceFamily,
};
use crate::config::layer_types::{
    LAYER_TYPE_FULL_ATTENTION, LAYER_TYPE_LINEAR_ATTENTION, LAYER_TYPE_SLIDING_ATTENTION,
    LAYER_TYPE_WINDOW_ATTENTION,
};

/// Which layer space a declaration indexes.
///
/// Inkling-Small declares `local_layer_ids` twice — once for its 42-layer
/// decoder, once in `mtp_config` for its 8-layer MTP sub-stack. They index
/// different spaces, so a resolution is only meaningful against its scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterleaveScope {
    /// The primary decoder stack.
    DecoderStack,
    /// A multi-token-prediction sub-stack, with its own layer count.
    MtpStack,
}

impl InterleaveScope {
    /// Stable name recorded in the provenance.
    pub fn name(self) -> &'static str {
        match self {
            Self::DecoderStack => "target.decoder_stack",
            Self::MtpStack => "target.mtp_stack",
        }
    }

    /// Config prefix this scope's keys live under.
    fn prefix(self) -> &'static str {
        match self {
            Self::DecoderStack => "",
            Self::MtpStack => "mtp_config.",
        }
    }
}

/// `layer_types` entry → kind. An unrecognised entry answers `None` so the
/// resolver blocks, rather than resolving to a behavioural default.
fn kind_from_layer_type(entry: &str, window: Option<usize>) -> Option<LayerKind> {
    if entry.eq_ignore_ascii_case(LAYER_TYPE_FULL_ATTENTION) {
        Some(LayerKind::Full)
    } else if entry.eq_ignore_ascii_case(LAYER_TYPE_SLIDING_ATTENTION)
        || entry.eq_ignore_ascii_case(LAYER_TYPE_WINDOW_ATTENTION)
    {
        Some(LayerKind::Sliding { window })
    } else if entry.eq_ignore_ascii_case(LAYER_TYPE_LINEAR_ATTENTION) {
        // The array names a recurrence but never which one; the geometry
        // does that, and it is read elsewhere.
        Some(LayerKind::Recurrent(RecurrenceFamily::Unidentified))
    } else {
        None
    }
}

fn index_list(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_i64).collect())
        .unwrap_or_default()
}

/// Read whichever spelling this config uses, for one scope.
///
/// Spellings are tried most-specific first and the first that declares
/// anything wins; a checkpoint using two would otherwise have its reading
/// depend on evaluation order. Every path consulted is recorded in the
/// resolution's provenance.
pub fn read_declared_interleave(
    config: &Value,
    scope: InterleaveScope,
    layer_count: usize,
    window: Option<usize>,
) -> DeclaredInterleave {
    let outcome = read_layer_types(config, scope, layer_count, window)
        .or_else(|| read_linear_attn_sets(config, scope, layer_count))
        .or_else(|| read_local_layer_ids(config, scope, layer_count, window));
    match outcome {
        None => DeclaredInterleave::Absent,
        Some(Ok(resolved)) => DeclaredInterleave::Resolved(Box::new(resolved)),
        Some(Err(InterleaveError::NotDeclared)) => DeclaredInterleave::Absent,
        Some(Err(other)) => DeclaredInterleave::Unresolved(other),
    }
}

/// Result of one spelling: `None` when this checkpoint does not use it.
type SpellingOutcome = Option<Result<super::ResolvedInterleave, InterleaveError>>;

/// Qwen3.8 and GLM-5.3-Flash: one entry per layer.
fn read_layer_types(
    config: &Value,
    scope: InterleaveScope,
    layer_count: usize,
    window: Option<usize>,
) -> SpellingOutcome {
    let path = format!("{}layer_types", scope.prefix());
    let entries: Vec<String> = path
        .split('.')
        .try_fold(config, |node, seg| node.get(seg))?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    if entries.is_empty() {
        return None;
    }
    Some(resolve_per_layer_array(
        scope.name(),
        &path,
        &entries,
        layer_count,
        |entry| kind_from_layer_type(entry, window),
    ))
}

/// Kimi Linear and GLM-5.3-Flash: two sets that partition the stack.
fn read_linear_attn_sets(
    config: &Value,
    scope: InterleaveScope,
    layer_count: usize,
) -> SpellingOutcome {
    let base = format!("{}linear_attn_config", scope.prefix());
    let section = base
        .split('.')
        .try_fold(config, |node, seg| node.get(seg))?;
    let recurrent = index_list(&section["kda_layers"]);
    let full = index_list(&section["full_attn_layers"]);
    if recurrent.is_empty() && full.is_empty() {
        return None;
    }
    Some(resolve_declarations(
        scope.name(),
        vec![
            format!("{base}.kda_layers"),
            format!("{base}.full_attn_layers"),
        ],
        &[
            Declaration {
                kind: LayerKind::Recurrent(RecurrenceFamily::Kda),
                membership: Membership::ExplicitSet(recurrent),
            },
            Declaration {
                kind: LayerKind::Full,
                membership: Membership::ExplicitSet(full),
            },
        ],
        layer_count,
    ))
}

/// Inkling-Small: one set of sliding layers, the rest implied global.
fn read_local_layer_ids(
    config: &Value,
    scope: InterleaveScope,
    layer_count: usize,
    window: Option<usize>,
) -> SpellingOutcome {
    let path = format!("{}local_layer_ids", scope.prefix());
    let local = index_list(
        path.split('.')
            .try_fold(config, |node, seg| node.get(seg))?,
    );
    if local.is_empty() {
        return None;
    }
    Some(resolve_declarations(
        scope.name(),
        vec![path],
        &[
            Declaration {
                kind: LayerKind::Sliding { window },
                membership: Membership::ExplicitSet(local),
            },
            // Everything not named local is global. The complement is the
            // declaration here — Inkling states only one side.
            Declaration {
                kind: LayerKind::Full,
                membership: Membership::Complement,
            },
        ],
        layer_count,
    ))
}
