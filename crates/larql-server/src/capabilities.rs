//! What this server will and will not do, answered by the server.
//!
//! The Explorer contract's premise (`docs/runtime-lifecycle-design.md`
//! and the Explorer programme's step 3) is that a client must never
//! infer a server's powers from its hostname — "localhost means I may
//! encode, https means read-only" is a guess, and a guess that is
//! wrong in the permissive direction offers a button that fails. So
//! the server says.
//!
//! The risk in saying it is that the answer becomes a second,
//! hand-maintained list beside the router: a table that claims
//! `components: true` while `/v1/components` was never mounted moves
//! the guess from the client into the server rather than removing it.
//!
//! So no boolean here is written down. Every route-backed capability
//! is derived from [`MountedRoutes`] — the ledger `routes::mod`
//! records *while it builds the router* — through the one table
//! [`ROUTE_CAPABILITIES`], which names, for each advertised key, the
//! route whose presence **is** that capability. Consequences worth
//! stating:
//!
//! - A capability cannot be advertised without the route existing.
//! - A route this server does not mount reports `false` on every
//!   profile, with no "unsupported" list to maintain.
//! - The rung that mounts `/v1/plan` flips `sources.plan.*` to `true`
//!   without editing this file.
//! - `tests/test_capabilities.rs` re-reads the same question through
//!   axum's own matcher (a method-mismatch probe: 405 means the path
//!   is mounted, 404 means it is not) and asserts the two readings
//!   agree, per profile. The ledger is checked, not trusted.
//!
//! Two facts here are *not* route-derived, and are marked as such:
//! the accepted source reference forms (read from the resolver's own
//! classifier, [`crate::bootstrap::classify_source`]) and the
//! execution backends ([`V3_BACKENDS`]).

use std::collections::BTreeSet;

use crate::bootstrap::SourceKind;
use crate::routes::paths;

/// The shape of this report. A client that does not recognise the
/// schema must refuse the document rather than read the keys it
/// happens to know — the same discipline `SystemPlan::parse` applies
/// to plan schema 4: an unattributed answer is refused, not guessed.
pub const CAPABILITIES_SCHEMA: u32 = 1;

/// The plan-document schema this build's `/v1/plan` emits, re-exported
/// from the planner so a test asserting the wire shape reads the
/// planner's own number rather than a copy of it.
pub const PLAN_SCHEMA_EXPECTED: u32 = larql_vindex::format::vindex3::plan::PLAN_SCHEMA;

/// The execution backends this server can actually bind a VINDEX3
/// container to.
///
/// `crate::vindex3::load_v3_model` opens every V3 container with
/// `ProductionBackend` — the CPU executor — so this is `["cpu"]` even
/// on a binary built with `metal-experts`. That feature drives
/// VINDEX2 MoE expert dispatch; it is not on the V3 execution path,
/// and reporting it here would tell the Explorer it can offer a GPU
/// run this server has no way to perform. `larql run --metal` is a
/// CLI capability, not a server one. When the server gains a Metal V3
/// binding, this list grows with it — and
/// `capabilities_backends_match_the_v3_binding` fails until it does.
pub const V3_BACKENDS: &[&str] = &["cpu"];

/// Which router `bootstrap::serve` built. Distinct from
/// [`crate::state::RouterTopology`], which answers a narrower
/// question (may the bound model count mutate?) and deliberately
/// cannot see the public surface: `--public-explorer` freezes
/// `SingleModel` topology while building an entirely different route
/// table. Conflating them would report a public deployment as an
/// ordinary single-model server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerProfile {
    /// `routes::public_explorer_router` — the read surface plus
    /// `POST /v1/query`. No inference, no lifecycle, no mutation.
    PublicExplorer,
    /// `routes::single_model_router`.
    SingleModel,
    /// `routes::multi_model_router`.
    MultiModel,
}

impl ServerProfile {
    /// The wire name. Stable: a client keys behaviour off it.
    pub const fn as_str(self) -> &'static str {
        match self {
            ServerProfile::PublicExplorer => "public_explorer",
            ServerProfile::SingleModel => "single_model",
            ServerProfile::MultiModel => "multi_model",
        }
    }
}

/// A reference form a client can hand to a source-taking verb.
/// Reported separately from the verb's route because the two can fail
/// independently: `/v1/runtime/model` being mounted says the *verb*
/// exists, and the resolver's classifier says which *reference forms*
/// it understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceScheme {
    /// A filesystem path.
    Local,
    /// An `hf://owner/repo` reference.
    Hf,
}

impl SourceScheme {
    /// A reference of this form, used to ask the resolver's own
    /// classifier whether it recognises the form — rather than
    /// restating the resolver's branches here.
    const fn exemplar(self) -> &'static str {
        match self {
            SourceScheme::Local => "/var/lib/larql/example.vindex3",
            SourceScheme::Hf => "hf://owner/repo",
        }
    }

    const fn expected_kind(self) -> SourceKind {
        match self {
            SourceScheme::Local => SourceKind::Local,
            SourceScheme::Hf => SourceKind::Hf,
        }
    }

    /// Whether `load_artifact`'s resolver actually classifies this
    /// form as itself. Positive evidence: if `is_hf_path` stopped
    /// recognising `hf://`, the exemplar would classify as `Local`
    /// and `sources.load.hf` would go false on its own.
    fn accepted(self) -> bool {
        crate::bootstrap::classify_source(self.exemplar()) == self.expected_kind()
    }
}

/// One advertised capability, and the route whose presence is it.
pub struct RouteCapability {
    /// Where this lands in the report, as a JSON pointer. Also what
    /// the conformance test reads back.
    pub key: &'static str,
    /// The route whose presence **is** this capability. Reported
    /// `true` iff `routes::mod` mounted exactly this path.
    pub route: &'static str,
    /// An extra conjunct for source-taking verbs. `None` for
    /// capabilities that are purely a question of the route existing.
    pub source: Option<SourceGate>,
}

/// The second half of a source-taking capability: what, besides the
/// route existing, has to be true for the server to accept this
/// reference form.
///
/// Each variant names the authority that decides it, and the handler
/// for that verb asks the *same* authority — so what the report
/// advertises and what the endpoint does are one decision read twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceGate {
    /// `load` — `bootstrap::classify_source` must recognise the form.
    Resolver(SourceScheme),
    /// `plan` — this profile must be willing to plan that form.
    /// `routes::plan` refuses through [`plans_source`].
    PlanPolicy(SourceScheme),
}

/// Whether `profile` will plan a source of this form.
///
/// The public surface plans `hf://` references and nothing else. A
/// local path accepted from an internet caller is a filesystem probe:
/// even when it plans nothing, the refusals distinguish "no such
/// directory" from "not a checkpoint", and that difference maps the
/// host. A localhost server has no such caller and plans either.
///
/// `POST /v1/plan` refuses through this function, and
/// `sources.plan.{local,hf}` is advertised through it — one decision,
/// two readings, checked against each other in
/// `tests/test_plan_route.rs`.
pub const fn plans_source(profile: ServerProfile, scheme: SourceScheme) -> bool {
    !matches!(
        (profile, scheme),
        (ServerProfile::PublicExplorer, SourceScheme::Local)
    )
}

/// Every capability this server advertises. The report's whole
/// route-backed content is generated from this table plus the
/// mounted-route ledger — see the module doc.
pub const ROUTE_CAPABILITIES: &[RouteCapability] = &[
    // ── sources: what a client may hand this server, and as what ──
    // `load` binds an ALREADY-ENCODED container. `plan` and `encode`
    // take a raw checkpoint. Keeping them apart matters: the server
    // resolves `hf://` today, but only as a published *container*
    // (`resolve_hf_vindex` fetches `index.json`). Reporting one
    // "hf: true" would let the Explorer offer `hf://Qwen/Qwen3-0.6B`
    // — a raw checkpoint — to a verb that can only take a container.
    RouteCapability {
        key: "/sources/load/local",
        route: paths::RUNTIME_MODEL,
        source: Some(SourceGate::Resolver(SourceScheme::Local)),
    },
    RouteCapability {
        key: "/sources/load/hf",
        route: paths::RUNTIME_MODEL,
        source: Some(SourceGate::Resolver(SourceScheme::Hf)),
    },
    RouteCapability {
        key: "/sources/plan/local",
        route: paths::PLAN,
        source: Some(SourceGate::PlanPolicy(SourceScheme::Local)),
    },
    RouteCapability {
        key: "/sources/plan/hf",
        route: paths::PLAN,
        source: Some(SourceGate::PlanPolicy(SourceScheme::Hf)),
    },
    RouteCapability {
        key: "/sources/encode/local",
        route: paths::ENCODE,
        // No gate: encode-by-source has no policy yet, and the route is
        // mounted by nobody, so this is `false` on the route alone. The
        // rung that mounts it brings its own `SourceGate`.
        source: None,
    },
    RouteCapability {
        key: "/sources/encode/hf",
        route: paths::ENCODE,
        source: None,
    },
    // ── explorer: the read surface over a bound container ──
    RouteCapability {
        key: "/explorer/models",
        route: paths::MODELS,
        source: None,
    },
    RouteCapability {
        key: "/explorer/describe",
        route: paths::DESCRIBE,
        source: None,
    },
    RouteCapability {
        key: "/explorer/walk",
        route: paths::WALK,
        source: None,
    },
    RouteCapability {
        key: "/explorer/relations",
        route: paths::RELATIONS,
        source: None,
    },
    RouteCapability {
        key: "/explorer/stats",
        route: paths::STATS,
        source: None,
    },
    RouteCapability {
        key: "/explorer/components",
        route: paths::COMPONENTS,
        source: None,
    },
    RouteCapability {
        key: "/explorer/representations",
        route: paths::REPRESENTATIONS,
        source: None,
    },
    RouteCapability {
        key: "/explorer/provenance",
        route: paths::PROVENANCE,
        source: None,
    },
    RouteCapability {
        key: "/explorer/authority",
        route: paths::AUTHORITY,
        source: None,
    },
    RouteCapability {
        key: "/explorer/query",
        route: paths::QUERY,
        source: None,
    },
    RouteCapability {
        key: "/explorer/residency",
        route: paths::RESIDENCY,
        source: None,
    },
    // ── runtime: what this process will do with the bound model ──
    RouteCapability {
        key: "/runtime/introspect",
        route: paths::RUNTIME,
        source: None,
    },
    RouteCapability {
        key: "/runtime/execute",
        route: paths::INFER,
        source: None,
    },
    RouteCapability {
        key: "/runtime/lifecycle",
        route: paths::RUNTIME_MODEL,
        source: None,
    },
];

/// The paths `routes::mod` actually handed to axum, recorded as it
/// built the router. The only thing that decides a route-backed
/// capability.
#[derive(Debug, Default, Clone)]
pub struct MountedRoutes(BTreeSet<&'static str>);

impl MountedRoutes {
    /// Record a mount. Returns whether the path was new — a
    /// double-mount is a router bug (axum silently keeps the last
    /// one), and the caller asserts on it.
    pub fn record(&mut self, path: &'static str) -> bool {
        self.0.insert(path)
    }

    pub fn contains(&self, path: &str) -> bool {
        self.0.contains(path)
    }

    /// Every mounted path, sorted. Reported verbatim so a client can
    /// see the surface it is talking to without probing for it.
    pub fn paths(&self) -> impl Iterator<Item = &&'static str> {
        self.0.iter()
    }
}

/// The answer to `GET /v1/capabilities`, computed once at router
/// build time — the route table cannot change afterwards, so neither
/// can this.
#[derive(Debug, Clone)]
pub struct Capabilities {
    profile: ServerProfile,
    mounted: MountedRoutes,
}

impl Capabilities {
    pub fn derive(profile: ServerProfile, mounted: MountedRoutes) -> Self {
        Self { profile, mounted }
    }

    pub fn profile(&self) -> ServerProfile {
        self.profile
    }

    /// Whether `cap` is offered: its route was mounted, and — for a
    /// source-taking verb — the resolver accepts that reference form.
    pub fn advertises(&self, cap: &RouteCapability) -> bool {
        self.mounted.contains(cap.route)
            && match cap.source {
                None => true,
                Some(SourceGate::Resolver(scheme)) => scheme.accepted(),
                Some(SourceGate::PlanPolicy(scheme)) => plans_source(self.profile, scheme),
            }
    }

    /// The report. The `sources` / `explorer` / `runtime` blocks are
    /// generated from [`ROUTE_CAPABILITIES`]; nothing else writes a
    /// boolean into them.
    pub fn to_json(&self) -> serde_json::Value {
        let mut report = serde_json::json!({
            "object": "capabilities",
            "schema": CAPABILITIES_SCHEMA,
            "profile": self.profile.as_str(),
            "server": {
                "name": env!("CARGO_PKG_NAME"),
                "version": env!("CARGO_PKG_VERSION"),
            },
            "runtime": { "backends": V3_BACKENDS },
            "routes": self.mounted.paths().collect::<Vec<_>>(),
        });

        for cap in ROUTE_CAPABILITIES {
            insert_at(&mut report, cap.key, self.advertises(cap).into());
        }
        report
    }
}

/// Insert `value` at a `/`-separated JSON pointer, creating the
/// intermediate objects. Small and local on purpose: `serde_json`'s
/// `pointer_mut` will not vivify a missing branch, and the alternative
/// — hand-building the nested literal — is the second list this
/// module exists to avoid.
fn insert_at(root: &mut serde_json::Value, pointer: &str, value: serde_json::Value) {
    let mut node = root;
    let mut segments = pointer.trim_start_matches('/').split('/').peekable();
    while let Some(segment) = segments.next() {
        if segments.peek().is_none() {
            node[segment] = value;
            return;
        }
        if !node[segment].is_object() {
            node[segment] = serde_json::json!({});
        }
        node = &mut node[segment];
    }
}
