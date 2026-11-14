//! Rollback is a capability a provider states by implementing it. The row
//! provider does. Every other shipped provider must refuse, and refusing
//! must change nothing: the window provider has evicted rows a rewind could
//! need, the codec providers do not hold rows in a rewindable form, and the
//! canonical provider is not part of the first rung.

use crate::vindex3::{CanonicalKvState, CodecKvState, CodecRecentKvState, WindowKvState};
use larql_vindex::format::vindex3::opplan::exec::kv::{ContinuationError, KvState};

/// A logical position these providers are told they have reached.
const POSITION: usize = 6;
/// Any codec configuration the providers accept; rollback does not depend on it.
const CODEC_BITS: u8 = 4;
const RECENT_WINDOW: usize = 2;

fn refuses(name: &str, mut provider: impl KvState) {
    provider.set_position(POSITION);
    for target in [0, POSITION / 2, POSITION, POSITION + 1] {
        let error = provider.truncate(target).unwrap_err();
        assert!(
            matches!(error, ContinuationError::TruncateUnsupported { position } if position == target),
            "{name}: {error}"
        );
        assert_eq!(
            provider.position(),
            POSITION,
            "{name}: a refused rewind moved the position"
        );
    }
}

#[test]
fn every_shipped_provider_but_the_row_provider_refuses_a_rewind() {
    refuses("canonical", CanonicalKvState::default());
    refuses("window", WindowKvState::default());
    refuses("codec", CodecKvState::new(CODEC_BITS));
    refuses(
        "codec-recent",
        CodecRecentKvState::new(CODEC_BITS, RECENT_WINDOW),
    );
}
