//! Shared per-token accumulator for the streaming generation paths.
//!
//! Every OpenAI endpoint that streams (chat V2/V3, responses V2/V3)
//! needs the same three behaviours inside its per-token callback:
//! buffer the decoded text, forward it to an emit action until the
//! client is gone, and halt generation-side work once a client stop
//! string appears in the accumulated text. This type is that logic,
//! extracted so the four call sites cannot drift and the state machine
//! is unit-testable without a generating model (the CPU generation arm
//! never invokes per-token callbacks, so end-to-end tests cannot reach
//! it — see `generate_streaming_runs_against_synthetic_fixture` in
//! larql-inference).

use super::util::contains_any;

/// What a failed emit means for the rest of the stream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum EmitFailure {
    /// Stop everything — no further buffering or emission (the chat
    /// stream has nothing to do once the SSE channel is gone).
    Halt,
    /// Stop emitting but keep buffering (the responses stream still
    /// builds the final envelope for storage after a disconnect).
    StopEmitting,
}

/// Per-token state machine: buffer + emit-until-failed + halt-on-stop.
pub(super) struct TokenTap<'a> {
    stop_strings: &'a [String],
    on_emit_failure: EmitFailure,
    buffered: String,
    emitting: bool,
    halted: bool,
}

impl<'a> TokenTap<'a> {
    pub(super) fn new(stop_strings: &'a [String], on_emit_failure: EmitFailure) -> Self {
        Self {
            stop_strings,
            on_emit_failure,
            buffered: String::new(),
            emitting: true,
            halted: false,
        }
    }

    /// A tap that only buffers and stop-checks, never emits — the chat
    /// tools arm, whose tool-call delta is built after generation.
    pub(super) fn buffering_only(stop_strings: &'a [String]) -> Self {
        Self {
            emitting: false,
            ..Self::new(stop_strings, EmitFailure::StopEmitting)
        }
    }

    /// Feed one decoded token. `emit` runs while emission is active and
    /// returns false when the client stopped listening; the configured
    /// [`EmitFailure`] policy decides what that does to the tap.
    pub(super) fn feed(&mut self, text: &str, emit: impl FnOnce(&str) -> bool) {
        if self.halted {
            return;
        }
        self.buffered.push_str(text);
        // A token can decode to no text (a special token, or the first half
        // of a multi-byte character). There is nothing to send for it, and
        // an empty delta event is noise a client has to filter.
        if text.is_empty() {
            return;
        }
        if self.emitting && !emit(text) {
            self.emitting = false;
            if self.on_emit_failure == EmitFailure::Halt {
                self.halted = true;
                return;
            }
        }
        if !self.stop_strings.is_empty() && contains_any(&self.buffered, self.stop_strings) {
            self.halted = true;
        }
    }

    /// True once a stop string matched or an emit failure halted the
    /// stream — generation-side work should wind down.
    pub(super) fn halted(&self) -> bool {
        self.halted
    }

    /// Everything buffered so far.
    pub(super) fn text(&self) -> &str {
        &self.buffered
    }

    /// Consume the tap, returning the buffered text.
    pub(super) fn into_text(self) -> String {
        self.buffered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn a_token_with_no_text_is_not_emitted() {
        let stop = stops(&[]);
        let mut tap = TokenTap::new(&stop, EmitFailure::Halt);
        let mut emitted = Vec::new();
        for text in ["Par", "", "is"] {
            tap.feed(text, |t| {
                emitted.push(t.to_string());
                true
            });
        }
        assert_eq!(emitted, ["Par", "is"]);
        assert_eq!(tap.text(), "Paris");
    }

    #[test]
    fn feed_buffers_and_emits_each_token() {
        let stop = stops(&[]);
        let mut tap = TokenTap::new(&stop, EmitFailure::Halt);
        let mut emitted = Vec::new();
        tap.feed("Par", |t| {
            emitted.push(t.to_string());
            true
        });
        tap.feed("is", |t| {
            emitted.push(t.to_string());
            true
        });
        assert_eq!(tap.text(), "Paris");
        assert_eq!(emitted, vec!["Par", "is"]);
        assert!(!tap.halted());
    }

    #[test]
    fn stop_string_halts_and_freezes_the_buffer() {
        let stop = stops(&["is"]);
        let mut tap = TokenTap::new(&stop, EmitFailure::Halt);
        tap.feed("Par", |_| true);
        assert!(!tap.halted());
        tap.feed("is", |_| true);
        assert!(tap.halted());
        // Further tokens are ignored entirely.
        tap.feed(" ignored", |_| panic!("must not emit after halt"));
        assert_eq!(tap.into_text(), "Paris");
    }

    #[test]
    fn emit_failure_halt_policy_stops_everything() {
        let stop = stops(&[]);
        let mut tap = TokenTap::new(&stop, EmitFailure::Halt);
        tap.feed("a", |_| false);
        assert!(tap.halted());
        tap.feed("b", |_| panic!("must not emit after halt"));
        assert_eq!(tap.text(), "a");
    }

    #[test]
    fn emit_failure_stop_emitting_policy_keeps_buffering() {
        let stop = stops(&[]);
        let mut tap = TokenTap::new(&stop, EmitFailure::StopEmitting);
        tap.feed("a", |_| false);
        assert!(!tap.halted());
        tap.feed("b", |_| panic!("emission must stay off"));
        assert_eq!(tap.text(), "ab");
    }

    #[test]
    fn stop_string_still_halts_after_emission_stopped() {
        let stop = stops(&["END"]);
        let mut tap = TokenTap::new(&stop, EmitFailure::StopEmitting);
        tap.feed("x", |_| false);
        tap.feed("END", |_| panic!("emission must stay off"));
        assert!(tap.halted());
    }

    #[test]
    fn buffering_only_never_emits() {
        let stop = stops(&["z"]);
        let mut tap = TokenTap::buffering_only(&stop);
        tap.feed("a", |_| panic!("buffering-only tap must not emit"));
        tap.feed("z", |_| panic!("buffering-only tap must not emit"));
        assert!(tap.halted());
        assert_eq!(tap.text(), "az");
    }

    #[test]
    fn stop_string_spanning_two_tokens_is_detected() {
        // The check runs on the accumulated buffer, so a stop string
        // split across token boundaries still matches.
        let stop = stops(&["Paris"]);
        let mut tap = TokenTap::new(&stop, EmitFailure::Halt);
        tap.feed("Par", |_| true);
        assert!(!tap.halted());
        tap.feed("is", |_| true);
        assert!(tap.halted());
    }
}
