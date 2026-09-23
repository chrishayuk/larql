//! **The one rule a provider identity obeys**, whichever plane it names.
//!
//! LOWERING-PLUGIN-1 wrote the rule for lowering providers; C1 of
//! CONTINUATION-PLUGIN-1 (`docs/represent/forecasts/continuation-plugin-1.json`)
//! needs the same rule for continuation providers. Two copies would be two
//! rules the day one of them is edited, so both identities delegate here
//! and differ only in the plane their refusals name.

use crate::error::VindexError;

/// Refuse a `family`/`revision` pair that could not key a registry or
/// name a refusal: an empty or blank family, one that would not survive a
/// file name or a flag (anything outside `[A-Za-z0-9_-]`), or a revision
/// of zero, which is the "unstated" value and never a real one. `plane`
/// prefixes every message (`lowering identity: …`).
pub(crate) fn validate(plane: &str, family: &str, revision: u32) -> Result<(), VindexError> {
    if family.trim().is_empty() {
        return Err(VindexError::Parse(format!(
            "{plane} identity: the family is empty"
        )));
    }
    if let Some(bad) = family
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '-'))
    {
        return Err(VindexError::Parse(format!(
            "{plane} identity: family `{family}` contains `{bad}`; only ASCII letters, digits, \
             `_` and `-` name a provider"
        )));
    }
    if revision == 0 {
        return Err(VindexError::Parse(format!(
            "{plane} identity: `{family}` declares revision 0, which means unstated"
        )));
    }
    Ok(())
}
