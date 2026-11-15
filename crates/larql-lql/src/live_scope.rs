//! Which statements compute with a session's pending stored-weight ops, and which read the stored weights
//! whatever is pending. Said once.
//!
//! The engine's replies (`executor::vindex3::live`) and the help catalogue (`help::catalogue`) both take these
//! words from here. A page therefore cannot name other statements than the replies do, and
//! `tests/help/live_effect.rs` holds both to what a session does.
//!
//! The words are macros as well as constants because a catalogue entry is static text assembled with `concat!`,
//! which takes literals only.

/// The statements that compute with pending stored-weight ops (or are refused), as prose.
macro_rules! executed_by {
    () => {
        "INFER and TRACE"
    };
}

/// The statements that read the stored weights whatever is pending, as prose.
macro_rules! reads_stored {
    () => {
        "DESCRIBE, WALK, SELECT and DIFF"
    };
}

pub(crate) use {executed_by, reads_stored};

/// The statements that compute with pending stored-weight ops (or are refused).
pub const EXECUTED_BY: &str = executed_by!();
/// The statements that read the stored weights whatever is pending.
pub const READS_STORED: &str = reads_stored!();
/// How a reply opens the lines that list the tensors a session executes from patched bytes.
pub const IN_EFFECT: &str = concat!("In effect for ", executed_by!(), " in this session");
/// How a reply opens the reason a pending op cannot be executed.
pub const NOT_EXECUTABLE: &str = "NOT executable in this session";
/// How a reply says the last pending op was withdrawn.
pub const NOTHING_IN_EFFECT: &str = "No stored-weight op is in effect";
