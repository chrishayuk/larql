//! Container generation detection and dispatch (format spec §12.1).
//!
//! One larql binary supports both vindex generations, indefinitely, for reading
//! and serving. `index.json`'s `version` field is the **sole** discriminator —
//! no filename sniffing, no directory-shape heuristics:
//!
//! | `index.json.version` | generation | layer format |
//! | -------------------- | ---------- | ------------ |
//! | 1                    | VINDEX2    | LYRW `format_version` 1 |
//! | 2                    | VINDEX2    | LYRW `format_version` 1 |
//! | 3                    | VINDEX3    | pre-`RegionLayout` legacy schema, still read |
//! | 4                    | VINDEX3    | current ([`V3_CURRENT_SCHEMA`]); LYRW `format_version` 2 banks or tensor-table segments |
//!
//! The generation is named for its *current* `index.json.version`. An earlier
//! draft called the shipped generation "VINDEX1" while its version was already
//! 2, which put a permanent off-by-one between the name and the discriminator
//! — the single most likely way to mis-detect a directory. Both were renamed
//! so the two agree.
//!
//! **The mapping is not a bijection below 2.** `index.json.version` 1 is a
//! *legacy schema of the same shipped generation*, not a pre-generation
//! artifact: such indexes exist in the wild and the loader reads them by
//! filling absent fields with defaults. Refusing them would break VINDEX2
//! compatibility, which is the one thing dual-generation support exists to
//! protect. This was caught by the E0 preservation matrix, which is why that
//! matrix runs on every commit.
//!
//! The layer format keeps its own sequence and is deliberately *not* aligned to
//! either: LYRW is a different artifact with a different lifetime, and its own
//! numbering was already correct. A VINDEX3 container holds LYRW v2 files.
//!
//! Detection fails closed. An unknown version names the version found and the
//! versions this binary supports, before any weight byte is read — a loader
//! that guesses produces a served model with wrong weights, not an error.

use std::path::Path;

use crate::format::filenames::INDEX_JSON;
use crate::VindexError;

/// An `index.json` schema revision.
///
/// A newtype so no API can accept a bare `u32` and call it a generation. The
/// regression that motivated this compiled perfectly while conflating the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexSchemaVersion(pub u32);

impl IndexSchemaVersion {
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for IndexSchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Oldest `index.json` schema still recognised as the shipped generation.
///
/// Schema 1 predates several fields and loads with defaults. It is the same
/// container generation, not an older one.
pub const V2_MIN_SCHEMA: u32 = 1;
/// What a fresh extraction of the shipped generation writes.
pub const V2_CURRENT_SCHEMA: u32 = 2;
pub const V3_MIN_SCHEMA: u32 = 3;
/// What a fresh VINDEX3 extraction writes.
///
/// Bumped 3 → 4 when `RegionSchema` claimed its reserved u16 as
/// [`RegionLayout`](super::lyrw2::region_layout::RegionLayout). The wire
/// size did not change — `REGION_SCHEMA_BYTES` is still 20 — and that is
/// precisely why the bump is needed rather than why it is not.
///
/// A reader built before the field existed sees a nonzero value at bytes
/// 10..12 and ignores it. If it could then bind the region it would read a
/// fused gate/up operand as contiguous halves when the container said
/// interleaved: two branches silently mixed, plausible output, no error.
/// **Versioning tracks changed meaning, not changed byte count.**
///
/// Admission is at the container: an old binary supports `3..=3` and so
/// refuses a schema-4 `index.json` outright, never reaching the LYRW
/// files. (A caller invoking `Lyrw2Reader::parse` on a stray `.weights`
/// file bypasses that gate, but the container is the unit of
/// distribution and the only path the loaders take.)
pub const V3_CURRENT_SCHEMA: u32 = 4;

/// Which container generation a directory holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerGeneration {
    /// The shipped generation: `index.json` schemas 1-2, LYRW `format_version` 1.
    V2,
    /// The successor: `index.json` schema 3, LYRW `format_version` 2.
    V3,
}

impl ContainerGeneration {
    /// The schema a fresh extraction of this generation writes.
    pub const fn current_schema_version(self) -> IndexSchemaVersion {
        match self {
            Self::V2 => IndexSchemaVersion(V2_CURRENT_SCHEMA),
            Self::V3 => IndexSchemaVersion(V3_CURRENT_SCHEMA),
        }
    }

    /// Every schema this generation can read. Many-to-one, not a bijection.
    pub const fn supported_schema_versions(self) -> std::ops::RangeInclusive<u32> {
        match self {
            Self::V2 => V2_MIN_SCHEMA..=V2_CURRENT_SCHEMA,
            Self::V3 => V3_MIN_SCHEMA..=V3_CURRENT_SCHEMA,
        }
    }

    pub fn reads_schema(self, version: IndexSchemaVersion) -> bool {
        self.supported_schema_versions().contains(&version.get())
    }

    /// The LYRW `format_version` this generation's layer files carry.
    pub const fn lyrw_format_version(self) -> u32 {
        match self {
            Self::V2 => 1,
            Self::V3 => 2,
        }
    }

    /// Short label for tables and API fields — "v2" / "v3".
    pub const fn schema_label(self) -> &'static str {
        match self {
            Self::V2 => "v2",
            Self::V3 => "v3",
        }
    }

    /// The generation number, for APIs that report it numerically.
    pub const fn number(self) -> u32 {
        match self {
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }

    /// Human-readable name used in diagnostics — "VINDEX2" / "VINDEX3".
    pub const fn name(self) -> &'static str {
        match self {
            Self::V2 => "VINDEX2",
            Self::V3 => "VINDEX3",
        }
    }

    /// Map a LYRW `format_version` to the generation that writes it.
    pub fn from_lyrw_format_version(found: u32) -> Result<Self, VindexError> {
        match found {
            1 => Ok(Self::V2),
            2 => Ok(Self::V3),
            other => Err(VindexError::UnknownContainerGeneration {
                found: other,
                supported: "1 (VINDEX2), 2 (VINDEX3)".into(),
            }),
        }
    }

    /// Refuse if this is not the generation the caller's path handles.
    pub fn require(self, required: Self) -> Result<(), VindexError> {
        if self == required {
            return Ok(());
        }
        Err(VindexError::WrongContainerGeneration {
            found: self.name(),
            required: required.name(),
        })
    }
}

/// Every generation this binary implements, oldest first.
pub const ALL_GENERATIONS: [ContainerGeneration; 2] =
    [ContainerGeneration::V2, ContainerGeneration::V3];

/// Which container generation a caller **asked** a fresh extraction to
/// write.
///
/// Deliberately a different type from [`ContainerGeneration`] (what the
/// policy *decided*), for the same reason `ExtractionRequest` is not
/// `ExtractionTarget`: "V3 was requested and refused" and "V3 was never
/// requested" must never collapse into the same value. Every extraction
/// surface (LQL `EXTRACT`, `larql extract`, factory recipes) resolves its
/// caller's intent to this type and passes it through
/// [`admit_extraction_generation`] — none of them carries its own default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationRequest {
    /// No preference; [`DEFAULT_EXTRACTION_GENERATION`] applies.
    Auto,
    /// The named generation, explicitly. Refuses if the surface cannot
    /// produce it — never downgrades.
    Explicit(ContainerGeneration),
}

/// **The default-flip gate.** The generation `Auto` resolves to — i.e.
/// what a fresh extraction writes when the caller expressed no preference.
///
/// Flipping this constant to `V3` IS the "VINDEX3 becomes the primary
/// generation" decision. It is made here, once, together with the pinned
/// test in `generation_tests.rs` that names the decision — never as a
/// side effect of a CLI default, a recipe template, or a surface-local
/// fallback. Until the flip: V3 stays the explicitly-requested
/// generation, V2 the default; after it: V2 becomes the explicitly-
/// requested compatibility generation.
pub const DEFAULT_EXTRACTION_GENERATION: ContainerGeneration = ContainerGeneration::V2;

/// Resolve a caller's extraction-generation request to a decision.
///
/// This is deliberately the only place `Auto` gains a meaning. It cannot
/// refuse — whether the *surface* can actually produce the decided
/// generation is that surface's own admission step, and its refusal must
/// name the request rather than fall back.
pub fn admit_extraction_generation(request: GenerationRequest) -> ContainerGeneration {
    match request {
        GenerationRequest::Auto => DEFAULT_EXTRACTION_GENERATION,
        GenerationRequest::Explicit(generation) => generation,
    }
}

/// Map a schema revision to its owning container generation.
///
/// `index.json.version` remains the **sole** dispatch input — no filename
/// sniffing, no directory-shape heuristics — but it is a schema discriminator,
/// not a generation identifier. The loader maps supported schema revisions to
/// the generation that owns them, and that mapping is many-to-one.
pub fn generation_for_schema(
    version: IndexSchemaVersion,
) -> Result<ContainerGeneration, VindexError> {
    ALL_GENERATIONS
        .into_iter()
        .find(|g| g.reads_schema(version))
        .ok_or_else(|| VindexError::UnknownContainerGeneration {
            found: version.get(),
            supported: supported_schema_summary(),
        })
}

/// One generation's supported range, as `"3-4 (VINDEX3)"` or `"3 (VINDEX3)"`.
///
/// Split out so both renderings stay testable on their own. Every
/// generation begins its life spanning a single schema and widens as it
/// gains one — VINDEX3 was a singleton until `RegionLayout` — so the
/// singleton branch is a state the *next* generation will occupy, not dead
/// code. Testing it through whichever generation happens to be a singleton
/// today is what left it uncovered the moment that stopped being true.
pub fn schema_range_label(range: std::ops::RangeInclusive<u32>, name: &str) -> String {
    if range.start() == range.end() {
        format!("{} ({name})", range.start())
    } else {
        format!("{}-{} ({name})", range.start(), range.end())
    }
}

/// "1-2 (VINDEX2), 3-4 (VINDEX3)" — every schema this binary reads.
pub fn supported_schema_summary() -> String {
    ALL_GENERATIONS
        .into_iter()
        .map(|g| schema_range_label(g.supported_schema_versions(), g.name()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Read `index.json` and report which generation the directory holds.
///
/// Deliberately parses only the `version` field: a VINDEX3 `index.json` carries
/// keys the VINDEX2 config struct does not model, and full deserialisation
/// would fail on shape before it could report the far more useful "wrong
/// generation".
pub fn detect_generation(dir: &Path) -> Result<ContainerGeneration, VindexError> {
    let path = dir.join(INDEX_JSON);
    let text = std::fs::read_to_string(&path)?;
    let probe: VersionProbe =
        serde_json::from_str(&text).map_err(|e| VindexError::Parse(e.to_string()))?;
    match probe.version {
        Some(v) => generation_for_schema(IndexSchemaVersion(v)),
        None => Err(VindexError::UnknownContainerGeneration {
            found: 0,
            supported: format!(
                "{}; index.json declared no version field",
                supported_schema_summary()
            ),
        }),
    }
}

/// The refusal a consumer owes a container generation it does not
/// implement.
///
/// Consumer readiness has three allowed states — supports, explicitly
/// refuses, or is not reached — and "silently does nothing" is not one
/// of them. A V2-only verb that meets a V3 container must say which
/// verb, which container, and which generation, so the user can act;
/// "not found" and an empty listing are both failures of this contract.
///
/// Prefer this over hand-rolled strings: one wording means one thing to
/// grep for when the flip lands and these refusals start turning into
/// implementations.
pub fn unsupported_generation(op: &str, dir: &Path, found: ContainerGeneration) -> VindexError {
    VindexError::Parse(format!(
        "{op} does not support {} containers yet; {} is generation {}",
        found.name(),
        dir.display(),
        found.schema_label(),
    ))
}

/// A container's identity, readable without knowing its generation.
///
/// The consumer-readiness rule (`docs/vindex-generation-policy.md`) is
/// that no VINDEX3 artifact may enter the system and then silently
/// disappear from a listing. Listing surfaces therefore need one fact
/// source that answers for both generations — otherwise each surface
/// grows its own `match generation` and V3 falls out of whichever one
/// nobody updated.
#[derive(Debug, Clone)]
pub struct ContainerSummary {
    pub generation: ContainerGeneration,
    /// The model identity the container names itself by.
    pub model: String,
    pub num_layers: usize,
}

/// Read one container's identity, whichever generation it holds.
///
/// Both generations record model and layer count in `index.json`, so
/// this stays a single small read — cheap enough for a directory scan,
/// and it never opens segments or builds a plan.
pub fn summarize_container(dir: &Path) -> Result<ContainerSummary, VindexError> {
    let generation = detect_generation(dir)?;
    let text = std::fs::read_to_string(dir.join(INDEX_JSON))?;
    let probe: IdentityProbe =
        serde_json::from_str(&text).map_err(|e| VindexError::Parse(e.to_string()))?;
    Ok(ContainerSummary {
        generation,
        model: probe.model.unwrap_or_default(),
        num_layers: probe.num_layers.unwrap_or(0),
    })
}

/// The identity fields both generations spell the same way in
/// `index.json`. Optional throughout: a container missing one is
/// listed with the field blank, never hidden.
#[derive(serde::Deserialize)]
struct IdentityProbe {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    num_layers: Option<usize>,
}

/// Minimal view over `index.json` — the version field and nothing else.
#[derive(serde::Deserialize)]
struct VersionProbe {
    version: Option<u32>,
}
