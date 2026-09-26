//! Opt-in, calling-thread wall-time traces. No global ledger resets or tensor copies.
use std::{cell::RefCell, marker::PhantomData, rc::Rc, time::Instant};

#[derive(Debug, Default, serde::Serialize)]
pub struct TokenProfile {
    pub position: usize,
    pub token: Option<u32>,
    pub complete: bool,
    pub total_ns: u64,
    pub attention_ns: u64,
    pub ffn_ns: u64,
    pub reentry_ns: u64,
    pub other_ns: u64,
    pub provider_calls: Vec<serde_json::Value>,
}
#[derive(Default)]
struct State {
    tokens: Vec<TokenProfile>,
    calls: Vec<serde_json::Value>,
    layer: Option<usize>,
}
thread_local! { static ACTIVE: RefCell<Option<State>> = const { RefCell::new(None) }; }

/// A capture cannot migrate threads or nest. Dropping it restores the disabled state.
pub struct Capture(PhantomData<Rc<()>>);
impl Capture {
    pub fn start() -> Result<Self, &'static str> {
        ACTIVE.with(|s| {
            let mut s = s.borrow_mut();
            if s.is_some() {
                return Err("a profile capture is already active on this thread");
            }
            *s = Some(State::default());
            Ok(Self(PhantomData))
        })
    }
    pub fn finish(self) -> Vec<TokenProfile> {
        ACTIVE.with(|s| s.borrow_mut().take().expect("active capture").tokens)
    }
    /// Finish a transport-thread capture, which has calls but no decoder tokens.
    /// The coordinator attaches these records to its own current position.
    pub fn finish_provider_calls(self) -> Vec<serde_json::Value> {
        ACTIVE.with(|s| s.borrow_mut().take().expect("active capture").calls)
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.with(|s| {
            s.borrow_mut().take();
        });
    }
}
pub fn enabled() -> bool {
    ACTIVE.with(|s| s.borrow().is_some())
}
pub(super) fn current_layer() -> Option<usize> {
    ACTIVE.with(|s| s.borrow().as_ref().and_then(|s| s.layer))
}
/// Transport-specific details are diagnostics, never execution authority.
pub fn record_provider_call(call: serde_json::Value) {
    ACTIVE.with(|s| {
        if let Some(s) = s.borrow_mut().as_mut() {
            s.calls.push(call);
        }
    });
}
#[derive(Clone, Copy)]
pub(super) enum Phase {
    Attention,
    Ffn,
    Reentry,
    Other,
}
pub(super) struct Token {
    clock: Option<Instant>,
    phase: Phase,
    profile: TokenProfile,
}
impl Token {
    pub(super) fn layer(&mut self, layer: usize) {
        if self.clock.is_some() {
            ACTIVE.with(|s| {
                if let Some(s) = s.borrow_mut().as_mut() {
                    s.layer = Some(layer);
                }
            });
        }
    }
    pub(super) fn start(position: usize, token: Option<u32>) -> Self {
        Self {
            clock: enabled().then(Instant::now),
            phase: Phase::Other,
            profile: TokenProfile {
                position,
                token,
                ..Default::default()
            },
        }
    }
    pub(super) fn phase(&mut self, next: Phase) {
        if let Some(clock) = self.clock.as_mut() {
            let now = Instant::now();
            let ns = now.duration_since(*clock).as_nanos() as u64;
            *clock = now;
            self.profile.total_ns += ns;
            match self.phase {
                Phase::Attention => self.profile.attention_ns += ns,
                Phase::Ffn => self.profile.ffn_ns += ns,
                Phase::Reentry => self.profile.reentry_ns += ns,
                Phase::Other => self.profile.other_ns += ns,
            }
        }
        self.phase = next;
    }
    pub(super) fn complete(&mut self) {
        self.profile.complete = true;
    }
}
impl Drop for Token {
    fn drop(&mut self) {
        if self.clock.is_none() {
            return;
        }
        self.phase(Phase::Other);
        ACTIVE.with(|s| {
            if let Some(s) = s.borrow_mut().as_mut() {
                self.profile.provider_calls = std::mem::take(&mut s.calls);
                s.tokens.push(std::mem::take(&mut self.profile));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn captures_are_thread_local_and_failed_steps_are_named() {
        let capture = Capture::start().unwrap();
        assert!(Capture::start().is_err());
        std::thread::spawn(|| assert!(!enabled())).join().unwrap();
        {
            let mut token = Token::start(3, Some(17));
            token.phase(Phase::Attention);
            record_provider_call(serde_json::json!({"bytes": 41}));
        }
        let rows = capture.finish();
        assert!(!enabled());
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].complete);
        assert_eq!(rows[0].position, 3);
        assert_eq!(rows[0].provider_calls[0]["bytes"], 41);
        assert_eq!(
            rows[0].total_ns,
            rows[0].attention_ns + rows[0].ffn_ns + rows[0].reentry_ns + rows[0].other_ns
        );
    }
    #[test]
    fn dispatch_records_return_to_the_parent_without_sharing_a_capture() {
        let parent = Capture::start().unwrap();
        let calls = std::thread::spawn(|| {
            assert!(!enabled());
            let child = Capture::start().unwrap();
            record_provider_call(serde_json::json!({"kind": "child", "bytes": 123}));
            child.finish_provider_calls()
        })
        .join()
        .unwrap();
        assert!(enabled());
        {
            let mut token = Token::start(0, Some(3));
            token.layer(7);
            assert_eq!(current_layer(), Some(7));
            for call in calls {
                record_provider_call(call);
            }
            token.complete();
        }
        let rows = parent.finish();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].provider_calls[0]["bytes"], 123);
        assert!(!enabled());
    }
}
