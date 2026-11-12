//! Per-IP rate limiting using a token bucket.
//!
//! Module layout:
//!
//! ```text
//! ratelimit/
//! ├── mod.rs        — token bucket + per-IP limiter (pure logic)
//! ├── middleware.rs — axum middleware applying the limiter per request
//! └── tests/        — unit tests (module tests folder)
//! ```
//!
//! The bucket map is bounded by [`RateLimiter::evict_stale`], which the
//! server's [`crate::maintenance`] sweeper calls periodically — an idle
//! client's bucket refills to full and then only occupies memory, so
//! dropping it is lossless (a returning client gets a fresh full bucket).

mod middleware;

#[cfg(test)]
mod tests;

pub use middleware::{rate_limit_middleware, ClientIpSource, LocalTransport, RateLimitState};

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// GC window for [`RateLimiter::evict_stale`]: buckets idle (no refill
/// activity) for longer than this — 5 minutes — are dropped.
const BUCKET_IDLE_GC_SECS: u64 = 300;

/// Token bucket for a single IP.
struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

/// Per-IP rate limiter.
pub struct RateLimiter {
    buckets: Mutex<HashMap<IpAddr, Bucket>>,
    max_tokens: f64,
    refill_per_sec: f64,
}

impl RateLimiter {
    /// Parse a rate limit string like "100/min" or "10/sec".
    pub fn parse(spec: &str) -> Option<Self> {
        let parts: Vec<&str> = spec.split('/').collect();
        if parts.len() != 2 {
            return None;
        }
        let count: f64 = parts[0].trim().parse().ok()?;
        let per_sec = match parts[1].trim() {
            "sec" | "s" | "second" => count,
            "min" | "m" | "minute" => count / 60.0,
            "hour" | "h" => count / 3600.0,
            _ => return None,
        };
        Some(Self {
            buckets: Mutex::new(HashMap::new()),
            max_tokens: count,
            refill_per_sec: per_sec,
        })
    }

    /// Maximum tokens a bucket holds (the burst size).
    pub fn max_tokens(&self) -> f64 {
        self.max_tokens
    }

    /// Steady-state refill rate in tokens per second.
    pub fn refill_per_sec(&self) -> f64 {
        self.refill_per_sec
    }

    /// Check if a request from this IP is allowed. Returns true if allowed.
    pub fn check(&self, ip: IpAddr) -> bool {
        let mut buckets = match self.buckets.lock() {
            Ok(b) => b,
            Err(_) => return true, // Don't block on poisoned mutex.
        };

        let now = Instant::now();
        let bucket = buckets.entry(ip).or_insert(Bucket {
            tokens: self.max_tokens,
            last_refill: now,
        });

        // Refill tokens based on elapsed time.
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * self.refill_per_sec).min(self.max_tokens);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Number of IPs currently holding a bucket.
    pub fn bucket_count(&self) -> usize {
        self.buckets.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// Evict buckets idle longer than the GC window. Returns how many were
    /// removed. Called periodically by the [`crate::maintenance`] sweeper.
    pub fn evict_stale(&self) -> usize {
        self.evict_stale_at(Instant::now())
    }

    /// [`Self::evict_stale`] with an explicit clock, so tests can advance
    /// time without sleeping.
    pub fn evict_stale_at(&self, now: Instant) -> usize {
        let idle_window = Duration::from_secs(BUCKET_IDLE_GC_SECS);
        match self.buckets.lock() {
            Ok(mut buckets) => {
                let before = buckets.len();
                buckets.retain(|_, b| now.duration_since(b.last_refill) < idle_window);
                before - buckets.len()
            }
            Err(_) => 0,
        }
    }
}
