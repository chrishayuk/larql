//! `index.json` for a VINDEX3 container — the sole root authority (spec §12).
//!
//! Deliberately a **different type** from [`VindexConfig`](crate::config::index::VindexConfig),
//! not a superset of it. The shipped generation's index describes a dense
//! Gemma-shaped extraction: `layers`, `down_top_k`, `intermediate_size`. A
//! VINDEX3 index describes a *catalogue* — which segments exist, which
//! manifest interprets them, which profiles may be selected. Growing one
//! struct to cover both would give every field an "unless version 3" caveat,
//! and the loader would sniff fields to decide which half it is looking at,
//! which is exactly the heuristic dispatch §12.1 forbids.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::profile::{Profile, ProfileSelectionError, ResolvedProfile};
use super::variants::VariantCatalogue;
use crate::format::generation::V3_CURRENT_SCHEMA;

/// One encoded representation's physical location and integrity facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationEntry {
    /// Logical object this representation materialises (a graph object id).
    pub object: String,
    /// Storage encoding, as the graph declares it (`BF16`, …).
    pub encoding: String,
    /// Segment file path relative to the container root.
    pub segment: String,
    /// Number of tensors in the segment's table.
    pub tensor_count: usize,
    /// Payload bytes (excluding the segment header framing).
    pub payload_bytes: u64,
    /// SHA-256 over the payload region, computed while copying from the
    /// source — G4's byte-equivalence anchor on the source side.
    pub payload_sha256: String,
    /// SHA-256 over the whole segment file as written — detects
    /// post-write corruption independently of source access.
    pub segment_sha256: String,
    /// Representation id this one was compiled from, when it was compiled
    /// rather than encoded from a source checkpoint.
    ///
    /// Provenance a compiled pack cannot otherwise state: its bytes are
    /// derived, and a reader deciding whether to trust them needs to know
    /// what they are derived *from*. `None` on a representation taken
    /// straight from the source — there is no earlier container-side
    /// authority to name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiled_from: Option<String>,
    /// The representation ABI this pack was compiled against.
    ///
    /// `encoding` names a family; this names the contract. Without it a
    /// container's bytes mean "whatever the encoder did that day", and an
    /// improved encoder would silently redefine every artifact already on
    /// disk. Readers refuse a revision they do not implement. `None` on a
    /// representation taken from a source checkpoint — its bytes are the
    /// source's, not this compiler's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec: Option<crate::format::vindex3::represent::nvfp4_pack::CodecIdentity>,
    /// Digest of the representation this was compiled from, so a derived
    /// pack can be tied to the exact bytes it derives from even after it
    /// is copied out of the container that holds them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_representation_digest: Option<String>,
    /// Which compilation algorithm chose these values.
    ///
    /// Separate from [`Self::codec`] on purpose: the codec is the decode
    /// contract, this is the encode recipe. Nearest-rounding and a
    /// Hessian-aware encoder emit the same ABI and different numbers, so
    /// only this field decides whether a pack is byte-reproducible by the
    /// build reading it. A mismatch is never a refusal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder: Option<crate::format::vindex3::represent::nvfp4_pack::EncoderRecipe>,
}

/// Whether a container carries the authority for its own bytes.
///
/// A canonical container holds bit-authoritative source bytes and can
/// recompile any derived representation from them. A deployment image
/// holds only what execution needs: compiled representations plus the
/// surfaces a precision policy protected, with the authority named by
/// digest rather than carried.
///
/// The distinction is not "smaller" — it is what the bytes *claim*. A
/// derived image cannot recompile itself, and saying so is the difference
/// between an artifact that is missing something and one that never
/// promised it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContainerAuthority {
    /// Source bytes present; derived representations can be recompiled.
    #[default]
    Canonical,
    /// Executable, not re-compilable without the source it names.
    Derived,
}

pub use super::profile::PROFILE_EXACT;

/// `index.json` as a VINDEX3 container writes it.
///
/// `segments` maps a segment *key* to the number of physical files under it.
/// A key is a path stem relative to the container root (`routed/layer_000`),
/// so the loader composes a filename rather than globbing a directory —
/// filename sniffing is what §12.1 rules out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vindex3Index {
    /// Always [`V3_CURRENT_SCHEMA`]. The sole generation discriminator.
    pub version: u32,
    /// Identity, carried so a container names itself without a sidecar.
    pub model: String,
    pub family: String,
    /// Residual width the model operates at.
    pub hidden_size: usize,
    pub num_layers: usize,
    /// Filename of the MoE programme manifest, relative to the root.
    /// `None` when no routed programme is declared; inventing an empty one
    /// would be a fabricated authority.
    ///
    /// Absence is **not** evidence the model is dense. This manifest is an
    /// artifact of the import path; a graph-encoded container carries none
    /// whether or not it routes — `gpt-oss-20b.vindex3` is a routed MoE with a
    /// `target.expert_bank` representation and `moe_manifest: None`. Read
    /// [`Self::representations`] to learn what a container actually holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_manifest: Option<String>,
    /// Filename of the system-graph manifest
    /// ([`super::graph::SystemGraph`]), relative to the root. Absent on
    /// containers written before the graph existed — absence means "no
    /// graph recorded", never "single-component assumed".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_graph: Option<String>,
    /// Filename of the auxiliary reference table
    /// ([`super::auxiliary_references::AuxiliaryReferences`]), relative to
    /// the root: which represented object stands for each codec's named
    /// dependency. Absent on every container that declares none — absence
    /// means "no dependency is declared", never "the dependencies follow
    /// from a naming rule". A table this names and the container does not
    /// hold is a refusal, not an empty table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auxiliary_references: Option<String>,
    /// Filename of the representation attestation table
    /// ([`super::representation_attestations::RepresentationAttestations`]),
    /// relative to the root: what each represented operand instance
    /// ACHIEVED at an extent, as measured, as opposed to what its codec's
    /// scheme guarantees. Absent on every container that attests nothing
    /// — and absence means the guarantee is UNAVAILABLE, never that the
    /// error is zero. A table this names and the container does not hold
    /// is a refusal, not an empty table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation_attestations: Option<String>,
    /// Representation directory: representation id
    /// (`{object_id}@{encoding}`) → where its bytes physically live.
    /// Graph edges and objects never reference source tensor names — once
    /// encoded, the source checkpoint disappears as an authority, and this
    /// directory is the only route from a logical representation to bytes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub representations: BTreeMap<String, RepresentationEntry>,
    /// Profiles this container declares. Never empty: a container with no
    /// selectable profile cannot be served, and discovering that at bind time
    /// rather than at load time is the failure mode profiles exist to prevent.
    ///
    /// A profile selects variants (§9.1); it is not just a name. See
    /// [`Profile`].
    pub profiles: Vec<Profile>,
    /// Region set → the variants physically present for it (§9.1).
    ///
    /// Defaulted, because a container with no alternative packs — every one
    /// written so far — catalogues nothing and is served entirely from its
    /// baselines. An absent catalogue means "no packs", never "unchecked".
    #[serde(default, skip_serializing_if = "VariantCatalogue::is_empty")]
    pub variants: VariantCatalogue,
    /// Segment key → physical file count.
    pub segments: BTreeMap<String, u32>,
    /// Whether this container carries the authority for its own bytes.
    ///
    /// Defaulted to [`ContainerAuthority::Canonical`], because every
    /// container written before deployment images existed carries its
    /// source. An absent field means "canonical", never "unknown".
    #[serde(default, skip_serializing_if = "is_canonical")]
    pub authority: ContainerAuthority,
    /// The precision program this container's representations were
    /// compiled under.
    ///
    /// The authority, not a description. `stored` checks its pack conforms
    /// to this; `transient` manufactures exactly what this says is
    /// represented, reading no pack; `auto` fills in what is missing. A
    /// compiled artifact cannot be the authority for its own correctness,
    /// which is what reading the decisions back out of a pack's tensor
    /// table amounted to.
    ///
    /// `None` on containers with no compiled representation, and on those
    /// written before the map was explicit — for which the pack's tensor
    /// table remains the only available statement of what was done.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision_map: Option<crate::format::vindex3::represent::map::PrecisionMap>,
    /// Identity of the model this image derives from, when derived.
    ///
    /// A deployment image's representations name the digests they were
    /// compiled from, but those digests resolve against a container that is
    /// no longer present — so the image also names the model itself, which
    /// is what an operator needs to find the authority again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_from_model: Option<String>,
    /// Fields this build does not understand, carried verbatim.
    ///
    /// The compatibility rules (candidate spec §5.7) bind the writer, not
    /// just the reader: additive 3.x vocabulary must survive a rewrite.
    /// Ontology-drill finding F13 was exactly this loss — COMPILE
    /// round-tripped the index through this struct and dropped newer
    /// fields, while COMPACT (which carries the file byte-identically)
    /// preserved them. The flatten map closes that asymmetry for every
    /// struct round-trip at once.
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, serde_json::Value>,
}

fn is_canonical(a: &ContainerAuthority) -> bool {
    matches!(a, ContainerAuthority::Canonical)
}

impl Vindex3Index {
    /// A single-profile container over the given segments.
    pub fn new(
        model: impl Into<String>,
        family: impl Into<String>,
        hidden_size: usize,
        num_layers: usize,
        moe_manifest: impl Into<String>,
        segments: BTreeMap<String, u32>,
    ) -> Self {
        Self {
            version: V3_CURRENT_SCHEMA,
            model: model.into(),
            family: family.into(),
            hidden_size,
            num_layers,
            moe_manifest: Some(moe_manifest.into()),
            system_graph: None,
            auxiliary_references: None,
            representation_attestations: None,
            representations: BTreeMap::new(),
            profiles: vec![Profile::exact()],
            variants: VariantCatalogue::new(),
            segments,
            authority: ContainerAuthority::Canonical,
            precision_map: None,
            derived_from_model: None,
            extra: BTreeMap::new(),
        }
    }

    /// Whether `profile` is one this container declares by name.
    ///
    /// A name check, and *only* a name check — it says nothing about whether
    /// the profile's selections can be honoured. Use [`Self::select_profile`]
    /// before binding; §9.1's requirement is that an absent variant is refused
    /// before a byte is read, and a name cannot carry that.
    pub fn declares_profile(&self, profile: &str) -> bool {
        self.profile(profile).is_some()
    }

    /// The declared profile called `name`.
    pub fn profile(&self, name: &str) -> Option<&Profile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// Resolve `name` against the variants this container physically carries.
    ///
    /// The §9.1 gate — see [`super::profile::select`].
    pub fn select_profile(&self, name: &str) -> Result<ResolvedProfile<'_>, ProfileSelectionError> {
        super::profile::select(self, name)
    }

    /// Declared profile names, in declaration order.
    pub fn profile_names(&self) -> Vec<&str> {
        self.profiles.iter().map(|p| p.name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments() -> BTreeMap<String, u32> {
        BTreeMap::from([("routed/layer_000".to_string(), 1)])
    }

    fn index() -> Vindex3Index {
        Vindex3Index::new(
            "fixture-a",
            "direct-moe",
            256,
            1,
            "moe_manifest.json",
            segments(),
        )
    }

    #[test]
    fn a_fresh_index_declares_the_successor_schema() {
        // The whole dispatch turns on this one number being 3 and nothing else.
        assert_eq!(index().version, V3_CURRENT_SCHEMA);
    }

    #[test]
    fn a_fresh_index_is_servable_because_it_declares_a_profile() {
        let i = index();
        assert!(i.declares_profile(PROFILE_EXACT));
        assert!(
            !i.profiles.is_empty(),
            "a profile-less container cannot serve"
        );
    }

    #[test]
    fn an_undeclared_profile_is_refused_rather_than_assumed() {
        assert!(!index().declares_profile("browse"));
    }

    #[test]
    fn the_index_round_trips_through_json() {
        // It is the root authority; if it cannot survive its own serialisation
        // nothing below it is reachable.
        let before = index();
        let json = serde_json::to_string(&before).unwrap();
        let after: Vindex3Index = serde_json::from_str(&json).unwrap();
        assert_eq!(before, after);
    }

    // ── Profile selection (§9.1) ────────────────────────────────────────

    use super::super::variants::{RegionSetVariants, StoredVariant};
    use crate::format::capability::authority::Fidelity;

    const GATE_UP: &str = "layer.0.routed.gate_up";
    const Q6K: &str = "exact-q6k";
    const MXFP4: &str = "native-mxfp4";

    /// An index carrying one region set with an MXFP4 pack beside its baseline,
    /// and a profile that selects the pack.
    fn packed() -> Vindex3Index {
        let mut i = index();
        i.variants = VariantCatalogue::new().with_set(
            GATE_UP,
            RegionSetVariants::single(
                Q6K,
                StoredVariant::new("routed/layer_000.q6k", Fidelity::SourceEquivalent),
            )
            .with_variant(
                MXFP4,
                StoredVariant::new("routed/layer_000.mxfp4", Fidelity::SourceExact),
            ),
        );
        i.profiles
            .push(Profile::new("routed-mxfp4").selecting(GATE_UP, MXFP4));
        i
    }

    #[test]
    fn a_packless_index_still_resolves_its_canonical_profile() {
        let i = index();
        let resolved = i.select_profile(PROFILE_EXACT).unwrap();
        assert_eq!(resolved.name(), PROFILE_EXACT);
        assert!(resolved.is_empty(), "nothing catalogued, nothing selected");
    }

    #[test]
    fn a_profile_resolves_to_the_variant_it_selected() {
        let i = packed();
        let exact = i.select_profile(PROFILE_EXACT).unwrap();
        assert_eq!(
            exact.variant_for(GATE_UP).unwrap().storage,
            "routed/layer_000.q6k",
            "the canonical profile takes the baseline"
        );
        let packed_profile = i.select_profile("routed-mxfp4").unwrap();
        assert_eq!(
            packed_profile.variant_for(GATE_UP).unwrap().storage,
            "routed/layer_000.mxfp4",
            "selecting a pack must change which bytes are named"
        );
    }

    #[test]
    fn an_undeclared_profile_names_the_ones_that_are_declared() {
        let err = packed().select_profile("browse").unwrap_err();
        match &err {
            ProfileSelectionError::UndeclaredProfile {
                requested,
                declared,
            } => {
                assert_eq!(requested, "browse");
                assert_eq!(
                    declared,
                    &vec![PROFILE_EXACT.to_string(), "routed-mxfp4".to_string()]
                );
            }
            other => panic!("wrong error: {other:?}"),
        }
        let msg = err.to_string();
        assert!(msg.contains("browse"), "{msg}");
        assert!(msg.contains("routed-mxfp4"), "{msg}");
    }

    #[test]
    fn a_declared_profile_selecting_an_absent_variant_is_unsatisfiable() {
        // Declared-but-unservable is the case `declares_profile` cannot see,
        // and the whole reason selection exists beside it.
        let mut i = packed();
        i.profiles
            .push(Profile::new("ghost").selecting(GATE_UP, "nvfp4"));
        assert!(
            i.declares_profile("ghost"),
            "the name check passes — that is the point"
        );
        let err = i.select_profile("ghost").unwrap_err();
        match &err {
            ProfileSelectionError::Unsatisfiable { profile, defects } => {
                assert_eq!(profile, "ghost");
                assert_eq!(defects.len(), 1);
            }
            other => panic!("wrong error: {other:?}"),
        }
        // The message has to carry all three of §9.1's names.
        let msg = err.to_string();
        assert!(msg.contains("ghost"), "profile: {msg}");
        assert!(msg.contains(GATE_UP), "region set: {msg}");
        assert!(msg.contains("nvfp4"), "requested: {msg}");
        assert!(msg.contains(MXFP4), "present: {msg}");
    }

    #[test]
    fn profile_lookup_and_names_agree() {
        let i = packed();
        assert_eq!(i.profile_names(), vec![PROFILE_EXACT, "routed-mxfp4"]);
        assert_eq!(i.profile("routed-mxfp4").unwrap().name, "routed-mxfp4");
        assert!(i.profile("browse").is_none());
        assert!(i.declares_profile("routed-mxfp4"));
    }

    #[test]
    fn a_packed_index_round_trips_with_its_catalogue_and_selections() {
        let before = packed();
        let after: Vindex3Index =
            serde_json::from_str(&serde_json::to_string(&before).unwrap()).unwrap();
        assert_eq!(before, after);
        assert_eq!(
            after.select_profile("routed-mxfp4").unwrap().storage_keys(),
            vec!["routed/layer_000.mxfp4"]
        );
    }

    #[test]
    fn a_packless_index_omits_the_catalogue_on_the_wire() {
        // Every container written so far has no packs; an empty object in each
        // of their index.json files would be noise that reads as meaningful.
        let json = serde_json::to_value(index()).unwrap();
        assert!(json.get("variants").is_none(), "{json}");
    }

    #[test]
    fn the_serialised_form_carries_the_version_a_detector_reads() {
        // `detect_generation` parses the raw JSON, not this struct, so the
        // wire key matters independently of the field name.
        let json: serde_json::Value = serde_json::to_value(index()).unwrap();
        assert_eq!(json["version"], serde_json::json!(V3_CURRENT_SCHEMA));
    }
}
