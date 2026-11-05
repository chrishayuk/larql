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
    shipped_continuations, CanonicalFactory, CanonicalKvState, CodecFactory, CodecKvState,
    CodecRecentFactory, CodecRecentKvState, WindowFactory, WindowKvState,
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
        /// The model depth the uniform policy is built for. `None` adopts
        /// the served model's own depth at prefill; `Some(n)` is a
        /// declaration the engine checks against the model.
        num_layers: Option<usize>,
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

        // A value that is present but does not parse rejects the spec
        // (`None`) rather than silently becoming the default: a mistyped
        // `window=` must not mean "unbounded".
        fn parsed<T: std::str::FromStr>(
            params: &std::collections::HashMap<&str, &str>,
            key: &str,
        ) -> Option<Option<T>> {
            match params.get(key) {
                None => Some(None),
                Some(v) => v.parse().ok().map(Some),
            }
        }
        let get_usize = |key: &str, default: usize| -> Option<usize> {
            Some(parsed(&params, key)?.unwrap_or(default))
        };
        let get_f32 = |key: &str, default: f32| -> Option<f32> {
            Some(parsed(&params, key)?.unwrap_or(default))
        };
        let opt_usize = |key: &str| -> Option<Option<usize>> { parsed(&params, key) };

        match name.trim() {
            "standard" | "full" | "fp32" => {
                let window_size = opt_usize("window")?;
                Some(EngineKind::Standard { window_size })
            }
            "markov-bounded" | "bounded" | "sliding" => {
                // Legacy `--kv-cache markov-bounded` flag resolves to the
                // sliding-window form of the standard engine. Bit-parity
                // with today's live decode.
                let window_size = opt_usize("window")?;
                Some(EngineKind::Standard { window_size })
            }
            "no-cache" | "no_cache" | "none" | "off" => Some(EngineKind::NoCache),
            "markov-rs" | "markov_rs" | "markov-residual" | "markov_residual" => {
                let window_size = opt_usize("window")?;
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
                window_size: get_usize("window", 512)?,
            }),
            "turbo-quant" | "turbo_quant" | "turboquant" | "tq4" => Some(EngineKind::TurboQuant {
                bits: u8::try_from(get_usize("bits", 4)?).ok()?,
            }),
            "tq3" => Some(EngineKind::TurboQuant { bits: 3 }),
            "apollo" => {
                let cfg = apollo::entry::InjectionConfig::default();
                Some(EngineKind::Apollo {
                    injection_layer: get_usize("layer", cfg.injection_layer)?,
                    inject_coefficient: get_f32("coef", cfg.inject_coefficient)?,
                    top_k: get_usize("top_k", cfg.top_k)?,
                    // No hardcoded model-specific BOS default (cfg default
                    // is None): callers name it explicitly via `bos=N`.
                    bos_token_id: parsed(&params, "bos")?,
                })
            }
            "boundary-kv" | "boundary_kv" | "boundary" => Some(EngineKind::BoundaryKv {
                window_size: opt_usize("window")?,
                chunk_tokens: get_usize("chunk_tokens", 512)?,
                sequence_id: params
                    .get("sequence_id")
                    .map(|s| (*s).to_string())
                    .unwrap_or_else(|| "default".into()),
            }),
            "markov-rs-codec"
            | "markov_rs_codec"
            | "markov-residual-codec"
            | "markov_residual_codec" => Some(EngineKind::MarkovResidualCodec {
                window_size: opt_usize("window")?,
                // v0.1: bf16 is the only safely-defaultable codec; other
                // ColdResidualCodec variants require explicit per-architecture
                // calibration that does not yet exist in tree.
                codec: markov_residual_codec::ColdResidualCodec::Bf16,
            }),
            "boundary-per-layer" | "boundary_per_layer" | "boundary-pl" => {
                // Without `layers=N` the policy takes the served model's
                // depth at prefill; with it, a mismatch is refused there.
                Some(EngineKind::BoundaryPerLayer {
                    window_size: opt_usize("window")?,
                    num_layers: opt_usize("layers")?,
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
                let Some(num_layers) = num_layers else {
                    return AnyEngine::Kv(Box::new(BoundaryPerLayerEngine::adopting_model_depth(
                        window_size,
                        backend,
                    )));
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
mod tests;

// ─── Cross-engine trait compliance ───────────────────────────────────────────

#[cfg(test)]
mod compliance_tests;
