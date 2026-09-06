//! PERF-QUAL-1: a deterministic work multiplier for instrument qualification.
//!
//! This exists to build a KNOWN-POSITIVE control for the bench gate. The
//! no-op control (issue #370, run 34035301273) showed the gate reporting
//! 13 regressions on bit-identical code, so before any threshold is
//! chosen the instrument has to be shown capable of detecting a slowdown
//! it was actually given.
//!
//! Off unless `LARQL_BENCH_SLOWDOWN` is set, and `1.0` is exactly the
//! unmodified path — so a normal bench run is bit-identical to one
//! without this module.
//!
//! The multiplier is applied by REPEATING the measured closure, never by
//! sleeping: a spin or sleep would test the harness's timing loop
//! rather than its ability to resolve a change in real kernel work.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;

/// Environment variable naming the multiplier, e.g. `1.25`.
pub const SLOWDOWN_ENV: &str = "LARQL_BENCH_SLOWDOWN";

/// Multiplier meaning "run exactly the work the benchmark describes".
pub const NO_SLOWDOWN: f64 = 1.0;

/// Denominator used to express a fractional multiplier as "one extra run
/// every N iterations". 1.25 -> every 4th, 1.5 -> every 2nd.
const FRACTION_RESOLUTION: f64 = 1.0;

pub struct Slowdown {
    whole: u32,
    extra_period: u32,
    counter: AtomicU32,
}

impl Slowdown {
    /// Reads the multiplier from the environment. Anything absent,
    /// unparseable, or below 1.0 means "no slowdown" — a qualification
    /// knob must never make a benchmark look FASTER by accident.
    pub fn from_env() -> Self {
        let m = std::env::var(SLOWDOWN_ENV)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= NO_SLOWDOWN)
            .unwrap_or(NO_SLOWDOWN);
        Self::from_multiplier(m)
    }

    pub fn from_multiplier(m: f64) -> Self {
        let whole = m.floor().max(NO_SLOWDOWN) as u32;
        let frac = m - f64::from(whole);
        // frac 0.25 -> one extra run every 4 iterations.
        let extra_period = if frac > f64::EPSILON {
            (FRACTION_RESOLUTION / frac).round() as u32
        } else {
            0
        };
        Self {
            whole,
            extra_period,
            counter: AtomicU32::new(0),
        }
    }

    /// Runs `f` enough times to realise the multiplier, returning the last
    /// value so the caller's `black_box` behaviour is unchanged.
    #[inline]
    pub fn run<T>(&self, mut f: impl FnMut() -> T) -> T {
        let mut last = f();
        for _ in 1..self.whole {
            last = f();
        }
        if self.extra_period > 0 {
            let i = self.counter.fetch_add(1, Ordering::Relaxed);
            if i.is_multiple_of(self.extra_period) {
                last = f();
            }
        }
        last
    }
}

/// Process-wide handle, resolved once from the environment.
///
/// A `static` rather than a per-callsite value so wiring a bench into the
/// qualification costs one wrapped call and no signature changes.
pub fn slowdown() -> &'static Slowdown {
    static SLOWDOWN: OnceLock<Slowdown> = OnceLock::new();
    SLOWDOWN.get_or_init(Slowdown::from_env)
}
