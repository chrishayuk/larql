//! Pluggable KV-cache engines for larql-inference.
//!
//! Each engine implements the full prefill + autoregressive decode loop but
//! manages its persistent inference state differently. Engines are selected
//! via [`EngineKind`] and benched via `larql bench --engine`.
//!
//! Correctness contract: `prefill` and `decode_step` return the pre-lm_head
//! hidden state (shape `[1, hidden_dim]`). The caller applies `final_norm +
//! lm_head` to get logits — see `larql_inference::forward::hidden_to_raw_logits`.

#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "macos",
    target_os = "windows"
))]
extern crate blas_src;

pub mod accuracy;
pub mod accuracy_suite;
pub mod cache;
pub mod engines;
pub mod generation;
pub mod model_walk;
pub mod profiler;
pub mod vindex3;
pub mod vindex_compare;

pub use cache::KvCache;
pub use vindex3::{
    shipped_continuations, CanonicalFactory, CanonicalKvState, WindowFactory, WindowKvState,
};

pub use engines::apollo;
pub use engines::boundary_kv;
pub use engines::boundary_per_layer;
pub use engines::markov_residual;
pub use engines::markov_residual_codec;
pub use engines::no_cache;
pub use engines::semantic_promotion;
pub use engines::standard;
pub use engines::turbo_quant;
pub use engines::windowed_checkpoint;

pub use engines::markov_residual::MarkovResidualEngine;
pub use engines::no_cache::NoCacheEngine;
pub use engines::standard::StandardEngine;
pub use engines::windowed_checkpoint::WindowedCheckpointEngine;

// ─── Trait surface re-exported from larql-inference ──────────────────────────
//
// `KvEngine`, `EngineInfo`, and `DecodeStageSummary` live in
// `larql-inference` so the dispatch loop there can reference them without
// a circular dependency on `larql-kv`. They're re-exported here so external
// callers and engine impls in this crate keep their existing public API:
// `larql_kv::KvEngine` continues to resolve to the same trait.
//
// See `crates/larql-inference/docs/specs/kv-engine-unification.md` §10.4.
pub use larql_inference::kv_engine::{
    AnyEngine, DecodeStageSummary, EngineError, EngineInfo, KvEngine, RetrievalEngine,
};

// ─── EngineKind ───────────────────────────────────────────────────────────────

/// Engine selector. Parse with [`EngineKind::from_name`]; build with [`EngineKind::build`].
#[derive(Debug, Clone)]
pub enum EngineKind {
    /// Production K/V tensor cache. `window_size: None` = unbounded
    /// growth (`--kv-cache standard`); `Some(N)` = sliding window
    /// (`--kv-cache markov-bounded --context-window N`). Default
    /// engine; bit-identical to today's live decode.
    Standard {
        window_size: Option<usize>,
    },
    /// No cache; full re-forward per decode step. O(N²) wall-time.
    /// Correctness fallback only (`--kv-cache none`).
    NoCache,
    MarkovResidual {
        window_size: Option<usize>,
    },
    WindowedCheckpoint {
        window_size: usize,
    },
    TurboQuant {
        bits: u8,
    },
    Apollo {
        injection_layer: usize,
        inject_coefficient: f32,
        top_k: usize,
        /// BOS token id to strip from the front of the query when
        /// assembling the injection context (`bos=N` in the spec).
        /// `None` = strip nothing / defer to the architecture's
        /// structural `bos_token_id()`; there is no hardcoded model
        /// default.
        bos_token_id: Option<u32>,
    },
    /// `BoundaryKvEngine`: Standard semantics + per-chunk
    /// `larql-boundary` frame emission. See
    /// `crates/larql-inference/docs/specs/boundary-kv-engine.md`.
    BoundaryKv {
        window_size: Option<usize>,
        chunk_tokens: usize,
        sequence_id: String,
    },
    /// `MarkovResidualCodecEngine`: MarkovResidualEngine with a codec-encoded
    /// cold tier. v0.1 ships `Bf16` codec only. See
    /// `crates/larql-inference/docs/specs/markov-residual-codec-engine.md`.
    MarkovResidualCodec {
        window_size: Option<usize>,
        codec: markov_residual_codec::ColdResidualCodec,
    },
    /// `BoundaryPerLayerEngine`: per-layer codec policy on the cold tier.
    /// v0.1 ships `Bf16` uniform across layers; the `num_layers` arg
    /// must match `weights.num_layers` at prefill time (construction
    /// errors otherwise). See
    /// `crates/larql-kv/src/engines/boundary_per_layer/`.
    BoundaryPerLayer {
        window_size: Option<usize>,
        num_layers: usize,
    },
    /// `SemanticPromotionEngine`: a semantic-authority policy wrapper
    /// over another engine. `base` is the wrapped engine's own spec, so
    /// `semantic-promotion:base=standard:window=512` composes. See
    /// `crates/larql-kv/docs/specs/semantic-promotion-engine.md`.
    ///
    /// Only `PromotionMode::Observe` is constructible today, and it is
    /// bit-identical to `base` — no engine yet implements the masking
    /// and snapshot hooks the enforcing modes require, and construction
    /// refuses those modes rather than downgrading.
    SemanticPromotion {
        base: Box<EngineKind>,
        mode: semantic_promotion::PromotionMode,
    },
}

impl EngineKind {
    /// Parse a CLI engine spec. Accepts `name` or `name:key=value[,key=value]`.
    ///
    /// Examples:
    /// ```text
    /// standard
    /// standard:window=1024
    /// no-cache
    /// markov-rs
    /// markov-rs:window=1024
    /// windowed-checkpoint:window=256
    /// turbo-quant:bits=3
    /// tq4
    /// apollo:layer=25,coef=8.0,top_k=12,bos=2
    /// ```
    pub fn from_name(spec: &str) -> Option<Self> {
        // Split "name:key=val,key=val" into name + param pairs.
        let (name, params_str) = spec.split_once(':').unwrap_or((spec, ""));
        let params: std::collections::HashMap<&str, &str> = params_str
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|kv| kv.split_once('='))
            .collect();

        let get_usize = |key: &str, default: usize| -> usize {
            params
                .get(key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            params
                .get(key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };

        match name.trim() {
            "standard" | "full" | "fp32" => {
                let window_size = params.get("window").and_then(|v| v.parse().ok());
                Some(EngineKind::Standard { window_size })
            }
            "markov-bounded" | "bounded" | "sliding" => {
                // Legacy `--kv-cache markov-bounded` flag resolves to the
                // sliding-window form of the standard engine. Bit-parity
                // with today's live decode.
                let window_size = params.get("window").and_then(|v| v.parse().ok());
                Some(EngineKind::Standard { window_size })
            }
            "no-cache" | "no_cache" | "none" | "off" => Some(EngineKind::NoCache),
            "markov-rs" | "markov_rs" | "markov-residual" | "markov_residual" => {
                let window_size = params.get("window").and_then(|v| v.parse().ok());
                Some(EngineKind::MarkovResidual { window_size })
            }
            // `unlimited-context` and friends are the pre-2026-08-03 names,
            // kept so existing scripts and baselines keep parsing. The engine
            // was renamed because the old name described a capability
            // (arbitrarily long streams via archive + replay) while reading as
            // a claim about attention — which is exactly the confusion that
            // let issue #200 hide: it reported `window=N` and attended over
            // everything.
            "windowed-checkpoint"
            | "windowed_checkpoint"
            | "unlimited"
            | "unlimited-context"
            | "unlimited_context" => Some(EngineKind::WindowedCheckpoint {
                window_size: get_usize("window", 512),
            }),
            "turbo-quant" | "turbo_quant" | "turboquant" | "tq4" => Some(EngineKind::TurboQuant {
                bits: get_usize("bits", 4) as u8,
            }),
            "tq3" => Some(EngineKind::TurboQuant { bits: 3 }),
            "apollo" => {
                let cfg = apollo::entry::InjectionConfig::default();
                Some(EngineKind::Apollo {
                    injection_layer: get_usize("layer", cfg.injection_layer),
                    inject_coefficient: get_f32("coef", cfg.inject_coefficient),
                    top_k: get_usize("top_k", cfg.top_k),
                    // No hardcoded model-specific BOS default (cfg default
                    // is None): callers name it explicitly via `bos=N`.
                    bos_token_id: params.get("bos").and_then(|v| v.parse().ok()),
                })
            }
            "boundary-kv" | "boundary_kv" | "boundary" => Some(EngineKind::BoundaryKv {
                window_size: params.get("window").and_then(|v| v.parse().ok()),
                chunk_tokens: get_usize("chunk_tokens", 512),
                sequence_id: params
                    .get("sequence_id")
                    .map(|s| (*s).to_string())
                    .unwrap_or_else(|| "default".into()),
            }),
            "markov-rs-codec"
            | "markov_rs_codec"
            | "markov-residual-codec"
            | "markov_residual_codec" => Some(EngineKind::MarkovResidualCodec {
                window_size: params.get("window").and_then(|v| v.parse().ok()),
                // v0.1: bf16 is the only safely-defaultable codec; other
                // ColdResidualCodec variants require explicit per-architecture
                // calibration that does not yet exist in tree.
                codec: markov_residual_codec::ColdResidualCodec::Bf16,
            }),
            "boundary-per-layer" | "boundary_per_layer" | "boundary-pl" => {
                // num_layers defaults to 34 (Gemma 3 4B); override via
                // `layers=N` when benching other architectures. Mismatch
                // against weights.num_layers errors at prefill.
                Some(EngineKind::BoundaryPerLayer {
                    window_size: params.get("window").and_then(|v| v.parse().ok()),
                    num_layers: get_usize("layers", 34),
                })
            }
            "semantic-promotion" | "semantic_promotion" | "promotion" => {
                // `base=<spec>` may itself carry `:key=value` params —
                // the split above takes only the FIRST colon, so
                // `semantic-promotion:base=standard:window=512` parses.
                // A base spec containing commas cannot be nested this
                // way; name such a base without params.
                let base = params
                    .get("base")
                    .map(|s| EngineKind::from_name(s))
                    .unwrap_or(Some(EngineKind::Standard { window_size: None }))?;
                let mode = params
                    .get("mode")
                    .map(|s| semantic_promotion::PromotionMode::from_name(s))
                    .unwrap_or(Some(semantic_promotion::PromotionMode::default()))?;
                Some(EngineKind::SemanticPromotion {
                    base: Box::new(base),
                    mode,
                })
            }
            _ => None,
        }
    }

    /// Split an engine-list string into individual specs.
    ///
    /// Engine specs can carry params (`name:k=v,k=v`), and commas inside
    /// params clash with the legacy comma-separator for engine lists.
    /// The splitter handles both forms:
    ///
    /// - **`;`-separated** (preferred when any engine carries multiple
    ///   params): `"a:x=1,y=2;b:p=3"` → `["a:x=1,y=2", "b:p=3"]`.
    /// - **`,`-separated** (legacy): splits by `,`, then merges
    ///   adjacent pieces back into the previous spec when a piece fails
    ///   to parse via [`Self::from_name`]. This makes
    ///   `"boundary-kv:chunk_tokens=64,sequence_id=demo"` round-trip as
    ///   a single spec under the legacy form, while keeping
    ///   `"standard,markov-rs"` and `"standard:window=512,markov-rs"`
    ///   working.
    ///
    /// Returns owned `String`s because merging requires building new
    /// values from pieces of the input.
    pub fn split_specs(spec: &str) -> Vec<String> {
        // Prefer `;` when present.
        if spec.contains(';') {
            return spec
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        // Legacy `,` path with reparse-driven merge.
        let pieces: Vec<&str> = spec
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        let mut result: Vec<String> = Vec::with_capacity(pieces.len());
        for piece in pieces {
            // A piece that parses on its own starts a new spec.
            if Self::from_name(piece).is_some() {
                result.push(piece.to_string());
                continue;
            }
            // Otherwise it's a continuation of the previous spec's params.
            if let Some(last) = result.last_mut() {
                last.push(',');
                last.push_str(piece);
            } else {
                // First piece doesn't parse — keep it so the caller can
                // surface the parse error rather than silently dropping it.
                result.push(piece.to_string());
            }
        }
        result
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            EngineKind::Standard { .. } => "standard",
            EngineKind::NoCache => "no-cache",
            EngineKind::MarkovResidual { .. } => "markov-rs",
            EngineKind::WindowedCheckpoint { .. } => "windowed-checkpoint",
            EngineKind::TurboQuant { .. } => "turbo-quant",
            EngineKind::Apollo { .. } => "apollo",
            EngineKind::BoundaryKv { .. } => "boundary-kv",
            EngineKind::MarkovResidualCodec { .. } => "markov-rs-codec",
            EngineKind::BoundaryPerLayer { .. } => "boundary-per-layer",
            EngineKind::SemanticPromotion { .. } => "semantic-promotion",
        }
    }

    /// All engine names recognised by [`Self::from_name`] — bare names
    /// only, no parameter syntax. Single source of truth for the
    /// "supported engines" help text in CLI error messages so the list
    /// doesn't drift when a new engine lands.
    ///
    /// The list contains the canonical name for each variant (the one
    /// `display_name` returns). Aliases that `from_name` also accepts
    /// (`full`/`fp32` for `standard`, `none`/`off` for `no-cache`,
    /// etc.) are intentionally omitted — the help text shows users
    /// the recommended spelling, not every accepted spelling.
    ///
    /// Adding a new engine variant **must** add a new entry here; the
    /// `engine_kind_supported_names_covers_every_variant` test in
    /// this module enforces that invariant via the parser.
    pub fn supported_names() -> &'static [&'static str] {
        &[
            "standard",
            "no-cache",
            "markov-rs",
            "markov-rs-codec",
            "windowed-checkpoint",
            "turbo-quant",
            "apollo",
            "boundary-kv",
            "boundary-per-layer",
            "semantic-promotion",
        ]
    }

    /// Specs the criterion microbenchmark (`benches/engine_decode.rs`)
    /// runs, parameterised where a bare name would not build something
    /// meaningful. Single source of truth so the bench cannot silently
    /// drift behind the engine roster — pinned by
    /// `bench_specs_cover_every_benchable_engine`.
    ///
    /// **Apollo is deliberately absent.** It is a [`RetrievalEngine`]
    /// whose `prefill` fails closed with `RetrievalMiss` unless a
    /// boundary store is attached, and the synthetic fixture has none.
    /// Benching it there timed the error return, not the engine: it
    /// reported ~65 ns against ~16 µs for `standard`, reading as a 250x
    /// win in the criterion report. A meaningful Apollo number needs a
    /// real store, which belongs in the CLI bench, not here.
    pub fn bench_specs() -> &'static [&'static str] {
        &[
            "standard",
            "standard:window=4",
            "no-cache",
            "markov-rs",
            "markov-rs:window=4",
            "markov-rs-codec",
            "windowed-checkpoint:window=4",
            "turbo-quant:bits=4",
            "turbo-quant:bits=3",
            "boundary-kv:chunk_tokens=4",
            "boundary-per-layer:layers=2",
        ]
    }

    /// Engines that [`Self::bench_specs`] intentionally omits, with the
    /// reason. Keeping the exclusion explicit means a new engine can't
    /// be dropped from the bench by simply never being added.
    pub fn bench_excluded_names() -> &'static [(&'static str, &'static str)] {
        &[
            (
                "apollo",
                "needs an attached boundary store; without one prefill returns \
                 RetrievalMiss and the bench would time the error path",
            ),
            (
                "semantic-promotion",
                "a policy wrapper, not an engine: it decides what an inner exact \
                 engine retains, so a standalone number would time the wrapper's \
                 bookkeeping against no policy and read as if it were a decode cost",
            ),
        ]
    }

    /// Build a boxed engine, dispatching compute through `backend`.
    pub fn build(self, backend: Box<dyn larql_inference::EngineBackend>) -> AnyEngine {
        self.build_with_profiling(backend, false)
    }

    /// Build a boxed engine with optional per-stage decode profiling.
    ///
    /// Returns [`AnyEngine`] — the dispatch enum that wraps either a
    /// [`KvEngine`] (per-token K/V cache engines: standard, no_cache,
    /// markov_residual, markov_residual_codec, windowed_checkpoint,
    /// turbo_quant, boundary_kv, boundary_per_layer) or a
    /// [`RetrievalEngine`] (Apollo, future Mode 5). Callers branch
    /// once on the enum variant and stay in the variant-specific code
    /// path thereafter. See the `AnyEngine` docs for the rationale.
    ///
    /// Takes [`larql_inference::EngineBackend`] — the umbrella over
    /// `ComputeBackend + KvDispatch` — so migrated engines (Step 3c
    /// of the ComputeBackend redesign) can dispatch through the
    /// trait. Construct via `larql_inference::cpu_engine_backend()` /
    /// `larql_inference::default_engine_backend()`.
    pub fn build_with_profiling(
        self,
        backend: Box<dyn larql_inference::EngineBackend>,
        profiling: bool,
    ) -> AnyEngine {
        // `profiling` is honoured only by engines that implement it
        // (currently MarkovResidual). Other engines accept the flag for
        // a uniform construction API and ignore it.
        let _ = profiling;
        match self {
            EngineKind::Standard { window_size } => AnyEngine::Kv(Box::new(
                standard::StandardEngine::with_backend(window_size, backend),
            )),
            EngineKind::NoCache => {
                AnyEngine::Kv(Box::new(no_cache::NoCacheEngine::with_backend(backend)))
            }
            EngineKind::MarkovResidual { window_size } => AnyEngine::Kv(Box::new(
                markov_residual::MarkovResidualEngine::with_backend(window_size, backend)
                    .with_profiling(profiling),
            )),
            EngineKind::WindowedCheckpoint { window_size } => AnyEngine::Kv(Box::new(
                windowed_checkpoint::WindowedCheckpointEngine::with_backend(window_size, backend)
                    .with_profiling(profiling),
            )),
            EngineKind::TurboQuant { bits } => AnyEngine::Kv(Box::new(
                turbo_quant::TurboQuantEngine::with_backend(bits, backend)
                    .with_profiling(profiling),
            )),
            EngineKind::Apollo {
                injection_layer,
                inject_coefficient,
                top_k,
                bos_token_id,
            } => AnyEngine::Retrieval(Box::new(apollo::ApolloEngine::new(
                apollo::InjectionConfig {
                    injection_layer,
                    inject_coefficient,
                    top_k,
                    // `bos=N` spec param; `None` (unset) defers to the
                    // structural `weights.arch.bos_token_id()` at prefill.
                    bos_token_id,
                },
            ))),
            EngineKind::BoundaryKv {
                window_size,
                chunk_tokens,
                sequence_id,
            } => {
                let identity = boundary_kv::BoundaryModelIdentity::placeholder("boundary-kv-cli");
                let mut config = boundary_kv::BoundaryKvEngineConfig::new(sequence_id, identity);
                config.window_size = window_size;
                config.chunk_tokens = chunk_tokens;
                AnyEngine::Kv(Box::new(boundary_kv::BoundaryKvEngine::with_backend(
                    config, backend,
                )))
            }
            EngineKind::MarkovResidualCodec { window_size, codec } => AnyEngine::Kv(Box::new(
                markov_residual_codec::MarkovResidualCodecEngine::with_backend(
                    window_size,
                    codec,
                    backend,
                )
                .with_profiling(profiling),
            )),
            EngineKind::BoundaryPerLayer {
                window_size,
                num_layers,
            } => {
                // v0.1: uniform Bf16 policy. Calibration store seeded
                // with the trivial bf16 record. Real production use
                // would inject a calibration store populated by the
                // offline sweep harness (per spec §4.7).
                use boundary_per_layer::{
                    BoundaryCalibrationRecord, BoundaryCalibrationStore, BoundaryLayerPolicy,
                    BoundaryPerLayerEngine, InMemoryCalibrationStore,
                };
                let policy = BoundaryLayerPolicy::bf16_uniform("cli", num_layers);
                let cal = InMemoryCalibrationStore::new();
                cal.put(BoundaryCalibrationRecord::bf16_uniform_default(
                    policy.fingerprint(),
                ))
                .expect("calibration store seed failed");
                AnyEngine::Kv(Box::new(
                    BoundaryPerLayerEngine::with_backend(
                        window_size,
                        policy,
                        num_layers,
                        &cal,
                        backend,
                    )
                    .expect("boundary-per-layer construction failed"),
                ))
            }
            EngineKind::SemanticPromotion { base, mode } => {
                let inner = match base.build_with_profiling(backend, profiling) {
                    AnyEngine::Kv(engine) => engine,
                    AnyEngine::Retrieval(engine) => panic!(
                        "semantic-promotion cannot wrap the retrieval engine {:?}: \
                         the promotion protocol appends records through a KV decode path",
                        engine.name()
                    ),
                };
                let config = semantic_promotion::SemanticPromotionConfig::default().with_mode(mode);
                AnyEngine::Kv(Box::new(
                    semantic_promotion::SemanticPromotionEngine::new(inner, config)
                        // Construction refuses any mode the base cannot
                        // back. Surfacing that as a panic here matches
                        // the other build arms; the CLI's own validation
                        // should reject the spec before reaching this.
                        .expect("semantic-promotion construction failed"),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_kind_from_name_roundtrip() {
        for name in &[
            "markov-rs",
            "markov_rs",
            "markov-residual",
            "markov_residual",
        ] {
            assert!(
                matches!(
                    EngineKind::from_name(name),
                    Some(EngineKind::MarkovResidual { .. })
                ),
                "failed to parse {name:?}"
            );
        }
        for name in &[
            "windowed-checkpoint",
            "windowed_checkpoint",
            "unlimited",
            "unlimited-context",
            "unlimited_context",
        ] {
            assert!(
                matches!(
                    EngineKind::from_name(name),
                    Some(EngineKind::WindowedCheckpoint { .. })
                ),
                "failed to parse {name:?}"
            );
        }
        assert!(EngineKind::from_name("unknown").is_none());
        assert!(EngineKind::from_name("").is_none());
    }

    #[test]
    fn engine_kind_from_name_with_params() {
        match EngineKind::from_name("standard") {
            Some(EngineKind::Standard { window_size: None }) => {}
            other => panic!("expected Standard{{window=None}}, got {other:?}"),
        }
        match EngineKind::from_name("standard:window=512") {
            Some(EngineKind::Standard {
                window_size: Some(512),
            }) => {}
            other => panic!("expected Standard{{window=512}}, got {other:?}"),
        }
        match EngineKind::from_name("markov-bounded:window=256") {
            // Legacy flag → Standard{Some(N)}.
            Some(EngineKind::Standard {
                window_size: Some(256),
            }) => {}
            other => panic!("expected Standard{{window=256}}, got {other:?}"),
        }
        match EngineKind::from_name("no-cache") {
            Some(EngineKind::NoCache) => {}
            other => panic!("expected NoCache, got {other:?}"),
        }
        match EngineKind::from_name("none") {
            Some(EngineKind::NoCache) => {}
            other => panic!("expected NoCache from 'none', got {other:?}"),
        }
        match EngineKind::from_name("markov-rs:window=1024") {
            Some(EngineKind::MarkovResidual {
                window_size: Some(1024),
                ..
            }) => {}
            other => panic!("expected MarkovResidual{{window=1024}}, got {other:?}"),
        }
        match EngineKind::from_name("unlimited-context:window=256") {
            Some(EngineKind::WindowedCheckpoint { window_size: 256 }) => {}
            other => panic!("expected WindowedCheckpoint{{window=256}}, got {other:?}"),
        }
        match EngineKind::from_name("turbo-quant:bits=3") {
            Some(EngineKind::TurboQuant { bits: 3 }) => {}
            other => panic!("expected TurboQuant{{bits=3}}, got {other:?}"),
        }
        match EngineKind::from_name("apollo:layer=25,coef=8.0,top_k=12") {
            Some(EngineKind::Apollo {
                injection_layer: 25,
                top_k: 12,
                // No `bos=` param → None: never a hardcoded model default.
                bos_token_id: None,
                ..
            }) => {}
            other => panic!("expected Apollo{{layer=25,top_k=12,bos=None}}, got {other:?}"),
        }
        match EngineKind::from_name("apollo:layer=25,bos=2") {
            Some(EngineKind::Apollo {
                injection_layer: 25,
                bos_token_id: Some(2),
                ..
            }) => {}
            other => panic!("expected Apollo{{layer=25,bos=Some(2)}}, got {other:?}"),
        }
        match EngineKind::from_name("markov-rs:unknown=999") {
            Some(EngineKind::MarkovResidual {
                window_size: None, ..
            }) => {}
            other => panic!("expected MarkovResidual{{window=None}}, got {other:?}"),
        }
    }

    // ── BoundaryKv parsing ───────────────────────────────────────────────

    #[test]
    fn engine_kind_from_name_boundary_kv_aliases() {
        for name in &["boundary-kv", "boundary_kv", "boundary"] {
            assert!(
                matches!(
                    EngineKind::from_name(name),
                    Some(EngineKind::BoundaryKv { .. })
                ),
                "failed to parse {name:?}"
            );
        }
    }

    #[test]
    fn engine_kind_from_name_boundary_kv_with_params() {
        match EngineKind::from_name("boundary-kv:chunk_tokens=64,sequence_id=demo") {
            Some(EngineKind::BoundaryKv {
                chunk_tokens: 64,
                sequence_id,
                window_size: None,
            }) => assert_eq!(sequence_id, "demo"),
            other => panic!("expected BoundaryKv with custom params, got {other:?}"),
        }
    }

    #[test]
    fn engine_kind_from_name_boundary_kv_defaults() {
        match EngineKind::from_name("boundary-kv") {
            Some(EngineKind::BoundaryKv {
                chunk_tokens: 512,
                sequence_id,
                window_size: None,
            }) => assert_eq!(sequence_id, "default"),
            other => panic!("expected default BoundaryKv, got {other:?}"),
        }
    }

    #[test]
    fn engine_kind_from_name_boundary_kv_with_window() {
        match EngineKind::from_name("boundary-kv:window=128,chunk_tokens=32") {
            Some(EngineKind::BoundaryKv {
                window_size: Some(128),
                chunk_tokens: 32,
                ..
            }) => {}
            other => panic!("expected BoundaryKv{{window=128,chunk=32}}, got {other:?}"),
        }
    }

    // ── BoundaryPerLayer parsing ─────────────────────────────────────────

    #[test]
    fn engine_kind_from_name_boundary_per_layer_aliases() {
        for name in &["boundary-per-layer", "boundary_per_layer", "boundary-pl"] {
            assert!(
                matches!(
                    EngineKind::from_name(name),
                    Some(EngineKind::BoundaryPerLayer { .. })
                ),
                "failed to parse {name:?}"
            );
        }
    }

    #[test]
    fn engine_kind_from_name_boundary_per_layer_defaults_to_34_layers() {
        match EngineKind::from_name("boundary-per-layer") {
            Some(EngineKind::BoundaryPerLayer {
                window_size: None,
                num_layers: 34,
            }) => {}
            other => panic!("expected BoundaryPerLayer{{layers=34}}, got {other:?}"),
        }
    }

    #[test]
    fn engine_kind_from_name_boundary_per_layer_with_window_and_layers() {
        match EngineKind::from_name("boundary-per-layer:window=256,layers=12") {
            Some(EngineKind::BoundaryPerLayer {
                window_size: Some(256),
                num_layers: 12,
            }) => {}
            other => panic!("expected BoundaryPerLayer{{window=256,layers=12}}, got {other:?}"),
        }
    }

    // ── MarkovResidualCodec parsing ──────────────────────────────────────

    #[test]
    fn engine_kind_from_name_markov_rs_codec_aliases() {
        for name in &[
            "markov-rs-codec",
            "markov_rs_codec",
            "markov-residual-codec",
            "markov_residual_codec",
        ] {
            assert!(
                matches!(
                    EngineKind::from_name(name),
                    Some(EngineKind::MarkovResidualCodec {
                        codec: markov_residual_codec::ColdResidualCodec::Bf16,
                        ..
                    })
                ),
                "failed to parse {name:?}"
            );
        }
    }

    #[test]
    fn engine_kind_from_name_markov_rs_codec_with_window() {
        match EngineKind::from_name("markov-rs-codec:window=256") {
            Some(EngineKind::MarkovResidualCodec {
                window_size: Some(256),
                codec: markov_residual_codec::ColdResidualCodec::Bf16,
                ..
            }) => {}
            other => panic!("expected MarkovResidualCodec{{window=256,codec=Bf16}}, got {other:?}"),
        }
    }

    // ── display_name for new variants ────────────────────────────────────

    #[test]
    fn engine_kind_display_name_covers_new_variants() {
        let kinds = [
            EngineKind::BoundaryKv {
                window_size: None,
                chunk_tokens: 512,
                sequence_id: "x".into(),
            },
            EngineKind::MarkovResidualCodec {
                window_size: None,
                codec: markov_residual_codec::ColdResidualCodec::Bf16,
            },
        ];
        let expected = ["boundary-kv", "markov-rs-codec"];
        for (k, name) in kinds.into_iter().zip(expected) {
            assert_eq!(k.display_name(), name);
        }
    }

    // ── build() for new variants ─────────────────────────────────────────

    #[test]
    fn engine_kind_build_boundary_kv_returns_engine() {
        let kind = EngineKind::BoundaryKv {
            window_size: None,
            chunk_tokens: 16,
            sequence_id: "test".into(),
        };
        let engine = kind.build(larql_inference::cpu_engine_backend());
        assert_eq!(engine.name(), "boundary-kv");
    }

    #[test]
    fn engine_kind_build_markov_rs_codec_returns_engine() {
        let kind = EngineKind::MarkovResidualCodec {
            window_size: Some(32),
            codec: markov_residual_codec::ColdResidualCodec::Bf16,
        };
        let engine = kind.build(larql_inference::cpu_engine_backend());
        assert_eq!(engine.name(), "markov-rs-codec");
    }

    // ── split_specs edge: first piece doesn't parse ───────────────────────

    #[test]
    fn split_specs_first_piece_unparseable_is_preserved() {
        // The first comma piece doesn't match a known engine name. The
        // splitter keeps it so the caller can surface a parse error rather
        // than silently dropping it (lines 239-242).
        let v = EngineKind::split_specs("garbage_engine,standard");
        // garbage_engine doesn't parse → kept as the first spec; then
        // 'standard' parses → becomes second spec.
        assert_eq!(v, vec!["garbage_engine", "standard"]);
        // The caller's `from_name` will then fail on "garbage_engine".
        assert!(EngineKind::from_name(&v[0]).is_none());
        assert!(EngineKind::from_name(&v[1]).is_some());
    }

    #[test]
    fn engine_info_summary_with_config() {
        let info = EngineInfo {
            name: "markov-rs".into(),
            description: "residual KV".into(),
            backend: "cpu".into(),
            config: "window=512".into(),
        };
        let s = info.summary();
        assert!(s.contains("markov-rs"));
        assert!(s.contains("cpu"));
        assert!(s.contains("window=512"));
    }

    #[test]
    fn engine_info_summary_no_config() {
        let info = EngineInfo {
            name: "test".into(),
            description: "desc".into(),
            backend: "metal".into(),
            config: String::new(),
        };
        let s = info.summary();
        assert!(!s.contains("()"));
    }
}

// ─── Cross-engine trait compliance ───────────────────────────────────────────

#[cfg(test)]
mod compliance_tests {
    use super::*;
    use larql_compute::cpu_backend;
    use larql_inference::{cpu_engine_backend, ModelWeights};
    use ndarray::Array2;

    fn all_kinds() -> Vec<EngineKind> {
        vec![
            EngineKind::Standard { window_size: None },
            EngineKind::Standard {
                window_size: Some(64),
            },
            EngineKind::NoCache,
            EngineKind::MarkovResidual { window_size: None },
            EngineKind::MarkovResidual {
                window_size: Some(32),
            },
            EngineKind::WindowedCheckpoint { window_size: 64 },
            EngineKind::TurboQuant { bits: 4 },
            EngineKind::TurboQuant { bits: 3 },
            EngineKind::Apollo {
                injection_layer: 30,
                inject_coefficient: 10.0,
                top_k: 8,
                bos_token_id: None,
            },
        ]
    }

    #[test]
    fn all_engines_memory_zero_before_prefill() {
        for kind in all_kinds() {
            let engine = kind.clone().build(cpu_engine_backend());
            assert_eq!(
                engine.memory_bytes(),
                0,
                "{} should have 0 memory before prefill",
                kind.display_name()
            );
        }
    }

    #[test]
    fn all_engines_have_valid_name() {
        let expected = [
            "standard",
            "standard",
            "no-cache",
            "markov-rs",
            "markov-rs",
            "windowed-checkpoint",
            "turbo-quant",
            "turbo-quant",
            "apollo",
        ];
        for (kind, expected_name) in all_kinds().into_iter().zip(expected.iter()) {
            let engine = kind.build(cpu_engine_backend());
            assert_eq!(engine.name(), *expected_name);
        }
    }

    #[test]
    fn all_engines_info_has_nonempty_fields() {
        for kind in all_kinds() {
            let name = kind.display_name();
            let engine = kind.build(cpu_engine_backend());
            let info = engine.info();
            assert!(!info.name.is_empty(), "{name}: empty name");
            assert!(!info.backend.is_empty(), "{name}: empty backend");
        }
    }

    #[test]
    fn all_engines_window_tokens_zero_before_prefill() {
        for kind in all_kinds() {
            let engine = kind.clone().build(cpu_engine_backend());
            assert_eq!(
                engine.window_tokens(),
                0,
                "{} window_tokens should be 0 before prefill",
                kind.display_name()
            );
        }
    }

    #[test]
    fn all_engines_cold_bytes_zero_before_prefill() {
        for kind in all_kinds() {
            let engine = kind.clone().build(cpu_engine_backend());
            assert_eq!(
                engine.cold_bytes(),
                0,
                "{} cold_bytes should be 0 before prefill",
                kind.display_name()
            );
        }
    }

    #[test]
    fn all_engines_stage_summary_none_before_decode() {
        for kind in all_kinds() {
            let engine = kind
                .clone()
                .build_with_profiling(cpu_engine_backend(), true);
            assert!(
                engine.stage_summary().is_none(),
                "{} stage_summary should be None before decode",
                kind.display_name()
            );
        }
    }

    #[test]
    fn from_name_unknown_param_ignored_defaults_apply() {
        match EngineKind::from_name("unlimited-context:unknown=42") {
            Some(EngineKind::WindowedCheckpoint { window_size: 512 }) => {}
            other => panic!("unknown param should use default, got {other:?}"),
        }
    }

    #[test]
    fn supported_names_every_entry_parses_back_via_from_name() {
        // Every name `supported_names()` advertises must be a name the
        // parser actually accepts. Catches the failure mode where
        // someone adds a variant to one side without the other.
        for name in EngineKind::supported_names() {
            let kind = EngineKind::from_name(name).unwrap_or_else(|| {
                panic!("supported_names lists {name:?} but from_name rejected it")
            });
            assert_eq!(
                kind.display_name(),
                *name,
                "supported_names entry {name:?} parses to a different display_name"
            );
        }
    }

    #[test]
    fn supported_names_covers_every_engine_kind_variant() {
        // Build one of every variant via the parser, collect their
        // canonical names, and verify supported_names() lists each.
        // This is the test the doc comment refers to; adding a new
        // EngineKind variant without adding a supported_names entry
        // makes this test fail with a useful diff.
        let one_of_each: Vec<&'static str> = [
            "standard",
            "no-cache",
            "markov-rs",
            "markov-rs-codec",
            "unlimited-context",
            "turbo-quant",
            "apollo",
            "boundary-kv",
            "boundary-per-layer",
            "semantic-promotion",
        ]
        .iter()
        .map(|s| {
            EngineKind::from_name(s)
                .unwrap_or_else(|| panic!("test fixture {s:?} failed to parse"))
                .display_name()
        })
        .collect();
        for name in &one_of_each {
            assert!(
                EngineKind::supported_names().contains(name),
                "EngineKind variant with display_name {name:?} is missing from supported_names"
            );
        }
        assert_eq!(
            EngineKind::supported_names().len(),
            one_of_each.len(),
            "supported_names and the variant set are out of sync"
        );
    }

    /// The criterion bench must cover every engine that can be benched
    /// on the synthetic fixture. It previously listed 7 of 9 — the three
    /// engines this PR touches most (`markov-rs-codec`, `boundary-kv`,
    /// `boundary-per-layer`) had no microbenchmark at all, and nothing
    /// failed when they were added. Now an engine is either in
    /// `bench_specs` or explicitly in `bench_excluded_names` with a
    /// reason; there is no third, silent option.
    #[test]
    fn bench_specs_cover_every_benchable_engine() {
        let excluded: Vec<&str> = EngineKind::bench_excluded_names()
            .iter()
            .map(|(n, _)| *n)
            .collect();

        let benched: Vec<&'static str> = EngineKind::bench_specs()
            .iter()
            .map(|s| {
                EngineKind::from_name(s)
                    .unwrap_or_else(|| panic!("bench_specs entry {s:?} no longer parses"))
                    .display_name()
            })
            .collect();

        for name in EngineKind::supported_names() {
            if excluded.contains(name) {
                assert!(
                    !benched.contains(name),
                    "{name:?} is listed as excluded but also appears in bench_specs"
                );
                continue;
            }
            assert!(
                benched.contains(name),
                "engine {name:?} has no criterion bench arm — add a spec to \
                 EngineKind::bench_specs, or name it in bench_excluded_names \
                 with the reason it cannot be benched"
            );
        }
    }

    #[test]
    fn bench_excluded_names_carry_a_reason_and_are_real_engines() {
        for (name, reason) in EngineKind::bench_excluded_names() {
            assert!(
                EngineKind::supported_names().contains(name),
                "bench_excluded_names lists {name:?}, which is not a supported engine"
            );
            assert!(
                reason.len() > 20,
                "exclusion of {name:?} needs a real reason, got {reason:?}"
            );
        }
    }

    #[test]
    fn from_name_all_engines_parseable() {
        let specs = [
            ("standard", "standard"),
            ("standard:window=128", "standard"),
            ("markov-bounded", "standard"),
            ("no-cache", "no-cache"),
            ("none", "no-cache"),
            ("markov-rs", "markov-rs"),
            ("windowed-checkpoint", "windowed-checkpoint"),
            // Pre-rename spellings still parse, and normalise to the new
            // canonical name rather than echoing themselves back.
            ("unlimited-context", "windowed-checkpoint"),
            ("unlimited", "windowed-checkpoint"),
            ("turbo-quant", "turbo-quant"),
            ("tq3", "turbo-quant"),
            ("apollo", "apollo"),
            ("semantic-promotion", "semantic-promotion"),
            ("semantic-promotion:base=markov-rs", "semantic-promotion"),
        ];
        for (spec, expected_display) in specs {
            let kind =
                EngineKind::from_name(spec).unwrap_or_else(|| panic!("{spec:?} failed to parse"));
            assert_eq!(
                kind.display_name(),
                expected_display,
                "{spec} parsed to wrong display_name"
            );
        }
    }

    #[test]
    fn semantic_promotion_defaults_to_an_unbounded_standard_base_in_observe_mode() {
        let kind = EngineKind::from_name("semantic-promotion").unwrap();
        let EngineKind::SemanticPromotion { base, mode } = kind else {
            panic!("expected a SemanticPromotion variant");
        };
        assert!(matches!(*base, EngineKind::Standard { window_size: None }));
        assert_eq!(mode, semantic_promotion::PromotionMode::Observe);
    }

    #[test]
    fn semantic_promotion_nests_a_parameterised_base_spec() {
        // The outer split takes only the first colon, so the base spec
        // keeps its own `:key=value` tail.
        let kind = EngineKind::from_name("semantic-promotion:base=standard:window=512").unwrap();
        let EngineKind::SemanticPromotion { base, .. } = kind else {
            panic!("expected a SemanticPromotion variant");
        };
        assert!(matches!(
            *base,
            EngineKind::Standard {
                window_size: Some(512)
            }
        ));
    }

    #[test]
    fn semantic_promotion_rejects_an_unknown_base_or_mode() {
        assert!(EngineKind::from_name("semantic-promotion:base=nonsuch").is_none());
        assert!(EngineKind::from_name("semantic-promotion:mode=delete-everything").is_none());
    }

    #[test]
    fn semantic_promotion_builds_and_wraps_its_base() {
        let engine = EngineKind::from_name("semantic-promotion")
            .unwrap()
            .build(larql_inference::cpu_engine_backend());
        assert_eq!(engine.name(), "semantic-promotion(standard)");
        assert!(engine.is_kv());
    }

    #[test]
    fn semantic_promotion_refuses_to_build_an_enforcing_mode() {
        // No base engine implements the masking or snapshot hooks yet,
        // so the enforcing modes must not construct — the build arm
        // surfaces that as a panic rather than downgrading to Observe.
        let kind = EngineKind::from_name("semantic-promotion:mode=enforce").unwrap();
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            kind.build(larql_inference::cpu_engine_backend())
        }));
        assert!(built.is_err(), "enforcing mode must not construct today");
    }

    /// Synthetic engine that does not override `prefill_quant` /
    /// `decode_step_quant`. Exercises the default trait methods that route to
    /// the f32 fallback — every shipped engine overrides these, so without
    /// this fixture they sit at 0% line coverage.
    struct DefaultMethodsEngine {
        /// Counts calls to `prefill` to confirm the q4k → prefill fallback
        /// path actually dispatches through the f32 method.
        prefill_calls: usize,
        decode_calls: usize,
    }

    impl KvEngine for DefaultMethodsEngine {
        fn name(&self) -> &str {
            "default-methods-test"
        }
        fn info(&self) -> EngineInfo {
            EngineInfo {
                name: self.name().into(),
                description: "test fixture".into(),
                backend: "cpu".into(),
                config: String::new(),
            }
        }
        fn prefill(
            &mut self,
            _weights: &ModelWeights,
            _ffn: &dyn larql_inference::ffn::FfnBackend,
            _token_ids: &[u32],
        ) -> Result<Array2<f32>, EngineError> {
            self.prefill_calls += 1;
            Ok(Array2::zeros((1, 4)))
        }
        fn decode_step(
            &mut self,
            _weights: &ModelWeights,
            _ffn: &dyn larql_inference::ffn::FfnBackend,
            _token_id: u32,
        ) -> Result<Array2<f32>, EngineError> {
            self.decode_calls += 1;
            Ok(Array2::zeros((1, 4)))
        }
        fn memory_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn default_q4k_methods_fallback_to_f32() {
        use larql_inference::ffn::WeightFfn;
        let weights = larql_inference::test_utils::make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = cpu_backend();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = DefaultMethodsEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };

        // Build a separate &mut binding for the `prefill_quant` call.
        let weights_for_q4k = larql_inference::test_utils::make_test_weights();
        let out = engine.prefill_quant(&weights_for_q4k, &ffn, &index, &[1, 2, 3], &*backend);
        assert!(out.is_ok());
        assert_eq!(
            engine.prefill_calls, 1,
            "default prefill_quant must call prefill"
        );

        let out = engine.decode_step_quant(&weights_for_q4k, &ffn, &index, 4, &*backend);
        assert!(out.is_ok());
        assert_eq!(
            engine.decode_calls, 1,
            "default decode_step_quant must call decode_step"
        );
    }

    #[test]
    fn default_window_tokens_and_cold_bytes_are_zero() {
        // Both have default impls returning 0; exercises the trait defaults
        // for an engine that doesn't override them.
        let engine = DefaultMethodsEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };
        assert_eq!(engine.window_tokens(), 0);
        assert_eq!(engine.cold_bytes(), 0);
        assert!(engine.stage_summary().is_none());
        assert_eq!(engine.name(), "default-methods-test");
    }

    // ── split_specs ──────────────────────────────────────────────────────────

    #[test]
    fn split_specs_legacy_comma_for_simple_engines() {
        // No colons → no param-comma ambiguity. Comma is the legacy separator.
        let v = EngineKind::split_specs("standard,markov-rs,no-cache");
        assert_eq!(v, vec!["standard", "markov-rs", "no-cache"]);
    }

    #[test]
    fn split_specs_legacy_comma_with_single_param_each() {
        // Single param per engine is unambiguous under comma split: each
        // engine's `name:key=value` doesn't contain a comma.
        let v = EngineKind::split_specs("standard:window=512,markov-rs:window=256");
        assert_eq!(v, vec!["standard:window=512", "markov-rs:window=256"]);
    }

    #[test]
    fn split_specs_semicolon_separator_for_multi_param_engines() {
        // Multi-param engines need ';' as the list separator to avoid
        // colliding with their param commas.
        let v = EngineKind::split_specs(
            "boundary-kv:chunk_tokens=64,sequence_id=demo;markov-rs:window=256",
        );
        assert_eq!(
            v,
            vec![
                "boundary-kv:chunk_tokens=64,sequence_id=demo",
                "markov-rs:window=256",
            ]
        );
    }

    #[test]
    fn split_specs_trims_whitespace() {
        let v = EngineKind::split_specs(" standard , markov-rs ");
        assert_eq!(v, vec!["standard", "markov-rs"]);
    }

    #[test]
    fn split_specs_drops_empty_entries() {
        let v = EngineKind::split_specs(",,standard,,markov-rs,");
        assert_eq!(v, vec!["standard", "markov-rs"]);
    }

    #[test]
    fn split_specs_semicolon_drops_empties_and_trims() {
        let v = EngineKind::split_specs(" ; standard ;; markov-rs ; ");
        assert_eq!(v, vec!["standard", "markov-rs"]);
    }

    #[test]
    fn split_specs_single_engine_returns_one_entry() {
        assert_eq!(EngineKind::split_specs("standard"), vec!["standard"]);
        assert_eq!(
            EngineKind::split_specs("boundary-kv:chunk_tokens=64,sequence_id=demo"),
            vec!["boundary-kv:chunk_tokens=64,sequence_id=demo"]
        );
    }

    #[test]
    fn split_specs_empty_returns_empty_vec() {
        assert!(EngineKind::split_specs("").is_empty());
        assert!(EngineKind::split_specs(" ").is_empty());
        assert!(EngineKind::split_specs(",,,").is_empty());
        assert!(EngineKind::split_specs(";;;").is_empty());
    }

    #[test]
    fn split_specs_round_trips_with_from_name() {
        // Each split entry must round-trip through EngineKind::from_name.
        let input = "standard;markov-rs:window=512;boundary-kv:chunk_tokens=64,sequence_id=demo";
        let specs = EngineKind::split_specs(input);
        for s in &specs {
            assert!(
                EngineKind::from_name(s).is_some(),
                "spec {s:?} should parse"
            );
        }
    }
}
