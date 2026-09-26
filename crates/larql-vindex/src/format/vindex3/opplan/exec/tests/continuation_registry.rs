//! **C2 of CONTINUATION-PLUGIN-1: continuation providers as a registry
//! of factories, selected against the plan before any prefill.**
//!
//! The forecast (`docs/represent/forecasts/continuation-plugin-1.json`)
//! predicts a registry carried as a value, holding factories that state
//! identity and regions, validate configuration and build per-conversation
//! state; a configuration digest stable across key order; and refusal at
//! selection — naming the first layer and region — rather than mid-prefill
//! (F5). The pinned digests below were computed OUTSIDE this crate
//! (Python `hashlib.sha256` over the documented canonical form), so they
//! witness the form rather than echo the implementation.

use larql_models::inventory::report::RecurrentStateDtype;

use super::super::continuation::{
    LayerContinuationGeometry, LayerLatentKvGeometry, RecurrentBufferGeometry, RecurrentGeometry,
    StateInitialization,
};
use super::super::continuation_authority::{
    ConfigDigest, ContinuationAuthority, ContinuationConfig,
};
use super::super::continuation_identity::ContinuationIdentity;
use super::super::continuation_registry::{
    BoxedContinuation, ContinuationFactory, ContinuationRegion, ContinuationRegistry,
    ContinuationRegistryError,
};
use super::super::kv::{HistoryRange, LayerKvGeometry, RowFactory, RowKvState};

// ---- fixtures -------------------------------------------------------------

fn kv() -> LayerContinuationGeometry {
    LayerContinuationGeometry::Kv(LayerKvGeometry {
        kv_dim: 4,
        window: None,
        history: HistoryRange::Full,
    })
}

fn recurrence() -> RecurrentGeometry {
    RecurrentGeometry::single(RecurrentBufferGeometry {
        shape: vec![2, 2],
        dtype: RecurrentStateDtype::Float32,
        initialization: StateInitialization::Zeros,
    })
}

fn recurrent() -> LayerContinuationGeometry {
    LayerContinuationGeometry::Recurrent(recurrence())
}

fn latent() -> LayerContinuationGeometry {
    LayerContinuationGeometry::LatentKv(LayerLatentKvGeometry { width: 3 })
}

fn conv_qkv() -> LayerContinuationGeometry {
    LayerContinuationGeometry::KvAndRecurrent {
        kv: LayerKvGeometry {
            kv_dim: 4,
            window: Some(2),
            history: HistoryRange::Trailing(2),
        },
        recurrent: recurrence(),
    }
}

/// A provider no production source names, declaring only what it is
/// told to — row storage underneath, which is all C2 needs of it.
struct Declaring {
    identity: ContinuationIdentity,
    regions: Vec<ContinuationRegion>,
}

impl Declaring {
    fn new(family: &str, revision: u32, regions: &[ContinuationRegion]) -> Box<Self> {
        Box::new(Self {
            identity: ContinuationIdentity::new(family, revision),
            regions: regions.to_vec(),
        })
    }
}

impl ContinuationFactory for Declaring {
    fn identity(&self) -> ContinuationIdentity {
        self.identity.clone()
    }

    fn regions(&self) -> &[ContinuationRegion] {
        &self.regions
    }

    fn validate_config(&self, config: &ContinuationConfig) -> Result<(), String> {
        match config.keys().find(|key| *key != "bits") {
            Some(key) => Err(format!("unknown option `{key}`")),
            None => Ok(()),
        }
    }

    fn build(&self, _config: &ContinuationConfig) -> BoxedContinuation {
        Box::new(RowKvState::default())
    }
}

fn hostile() -> ContinuationIdentity {
    ContinuationIdentity::new("hostile-test-provider", 77)
}

// ---- configuration --------------------------------------------------------

#[test]
fn configuration_is_held_in_key_order() {
    let written = ContinuationConfig::parse(&["mode=a", "bits=4"]).unwrap();
    let sorted = ContinuationConfig::parse(&["bits=4", "mode=a"]).unwrap();
    assert_eq!(written, sorted);
    assert_eq!(written.canonical_form(), "bits=4\nmode=a\n");
    assert_eq!(written.get("bits"), Some("4"));
    assert_eq!(written.get("absent"), None);
    assert_eq!(written.keys().collect::<Vec<_>>(), ["bits", "mode"]);
    assert!(ContinuationConfig::empty().is_empty());
    assert_eq!(ContinuationConfig::empty().canonical_form(), "");
    // `=` in a value is data, not a second separator.
    let eq = ContinuationConfig::parse(&["expr=a=b"]).unwrap();
    assert_eq!(eq.get("expr"), Some("a=b"));
}

#[test]
fn a_malformed_configuration_is_refused_by_shape() {
    let cases: &[(&[&str], &str)] = &[
        (&["bits"], "is not `key=value`"),
        (&["=4"], "has an empty key"),
        (&["bi ts=4"], "contains ` `"),
        (&["bits=4\n"], "has a newline"),
        (&["bits=4", "bits=3"], "is given twice"),
    ];
    for (options, reason) in cases {
        let err = ContinuationConfig::parse(options).unwrap_err();
        assert!(err.contains(reason), "{options:?}: {err}");
    }
}

// ---- digest and authority -------------------------------------------------

#[test]
fn the_digest_is_the_documented_canonical_form() {
    assert_eq!(
        ConfigDigest::of(&RowKvState::identity(), &ContinuationConfig::empty()).as_str(),
        "sha256:bd24a22a1998984de1eb53f6bce9934a4f85155e5a13ac00b2335ac77c85908a"
    );
    let config = ContinuationConfig::parse(&["mode=a", "bits=4"]).unwrap();
    assert_eq!(
        ConfigDigest::of(&hostile(), &config).to_string(),
        "sha256:269bb1c5efadef6160ae87a6635aa40078a2bc853d1ec7a6a668ae0ee73069aa"
    );
}

#[test]
fn the_digest_moves_with_every_input_but_key_order() {
    let base = ContinuationConfig::parse(&["bits=4"]).unwrap();
    let digest = |id: &ContinuationIdentity, options: &[&str]| {
        ConfigDigest::of(id, &ContinuationConfig::parse(options).unwrap())
    };
    let reference = ConfigDigest::of(&hostile(), &base);
    assert_eq!(reference, digest(&hostile(), &["bits=4"]));
    assert_ne!(reference, digest(&hostile(), &["bits=3"]));
    assert_ne!(reference, digest(&hostile(), &["bits=4", "mode=a"]));
    assert_ne!(
        reference,
        digest(
            &ContinuationIdentity::new("hostile-test-provider", 78),
            &["bits=4"]
        )
    );
    assert_ne!(reference, digest(&RowKvState::identity(), &["bits=4"]));
}

#[test]
fn authority_is_identity_and_digest_compared_separately() {
    let a = ContinuationAuthority::new(hostile(), &ContinuationConfig::parse(&["bits=4"]).unwrap());
    let b = ContinuationAuthority::new(hostile(), &ContinuationConfig::parse(&["bits=3"]).unwrap());
    assert_eq!(a.identity, b.identity, "same implementation…");
    assert_ne!(
        a.config_digest, b.config_digest,
        "…different interpretation"
    );
    assert_ne!(a, b);
    assert!(
        a.to_string()
            .starts_with("hostile-test-provider/v77 (sha256:"),
        "{a}"
    );
    let json = serde_json::to_string(&a).unwrap();
    assert_eq!(
        serde_json::from_str::<ContinuationAuthority>(&json).unwrap(),
        a
    );
}

// ---- registry -------------------------------------------------------------

#[test]
fn registration_refuses_invalid_and_duplicate_identities() {
    let mut registry = ContinuationRegistry::new();
    assert!(registry.is_empty());
    registry.register(Box::new(RowFactory)).unwrap();
    let dup = registry.register(Box::new(RowFactory)).unwrap_err();
    assert!(
        matches!(dup, ContinuationRegistryError::Duplicate { .. }),
        "{dup}"
    );
    assert_eq!(
        dup.to_string(),
        "continuation provider `row/v1` is already registered"
    );
    let invalid = registry
        .register(Declaring::new("row/v2", 1, &ContinuationRegion::ALL))
        .unwrap_err();
    assert!(
        invalid.to_string().contains("continuation identity: "),
        "{invalid}"
    );
    // Another revision of a registered family is a different provider.
    registry
        .register(Declaring::new("row", 2, &ContinuationRegion::ALL))
        .unwrap();
    assert_eq!(
        registry.identities(),
        [RowKvState::identity(), ContinuationIdentity::new("row", 2)]
    );
    assert_eq!(registry.len(), 2);
    assert!(format!("{registry:?}").contains("row"));
}

#[test]
fn an_unregistered_identity_is_refused_naming_what_is_registered() {
    let mut registry = ContinuationRegistry::new();
    let empty = registry
        .select(&hostile(), &ContinuationConfig::empty(), &[kv()])
        .unwrap_err();
    assert!(empty.to_string().ends_with("registered: none"), "{empty}");
    registry.register(Box::new(RowFactory)).unwrap();
    let err = registry
        .select(&hostile(), &ContinuationConfig::empty(), &[kv()])
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "no continuation provider `hostile-test-provider/v77` is registered; registered: row/v1"
    );
}

#[test]
fn a_provider_refuses_configuration_it_does_not_read() {
    let mut registry = ContinuationRegistry::new();
    registry.register(Box::new(RowFactory)).unwrap();
    let bits = ContinuationConfig::parse(&["bits=4", "mode=a"]).unwrap();
    let err = registry
        .select(&RowKvState::identity(), &bits, &[kv()])
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "continuation provider `row/v1` refused its configuration: takes no options; given \
         `bits`, `mode`"
    );
    // A provider that reads `bits` takes it — and refuses anything else.
    registry
        .register(Declaring::new(
            "hostile-test-provider",
            77,
            &ContinuationRegion::ALL,
        ))
        .unwrap();
    let four = ContinuationConfig::parse(&["bits=4"]).unwrap();
    registry.select(&hostile(), &four, &[kv()]).unwrap();
    let err = registry.select(&hostile(), &bits, &[kv()]).unwrap_err();
    assert!(err.to_string().ends_with("unknown option `mode`"), "{err}");
}

/// F5's witness at the registry: a KV-only provider against a hybrid plan
/// is refused at selection — no provider built, the first layer it
/// cannot hold named with the plan's own vocabulary.
#[test]
fn a_hybrid_plan_is_refused_at_selection_naming_the_layer() {
    let mut registry = ContinuationRegistry::new();
    registry
        .register(Declaring::new(
            "hostile-test-provider",
            77,
            &[ContinuationRegion::Kv],
        ))
        .unwrap();
    let plan = [kv(), kv(), recurrent(), latent()];
    let err = registry
        .select(&hostile(), &ContinuationConfig::empty(), &plan)
        .unwrap_err();
    match &err {
        ContinuationRegistryError::Unsupported {
            layer,
            region,
            declared,
            ..
        } => {
            assert_eq!(*layer, 2);
            assert_eq!(*region, ContinuationRegion::Recurrent);
            assert_eq!(declared, &[ContinuationRegion::Kv]);
        }
        other => panic!("expected Unsupported, got {other}"),
    }
    assert_eq!(
        err.to_string(),
        "continuation provider `hostile-test-provider/v77` cannot hold layer 2, which keeps \
         recurrent continuation state; it declares: kv"
    );
}

/// Holding each half is not holding the layer: a conv-QKV layer needs
/// its own capability.
#[test]
fn kv_and_recurrent_separately_do_not_hold_a_conv_qkv_layer() {
    let mut registry = ContinuationRegistry::new();
    let halves = [ContinuationRegion::Kv, ContinuationRegion::Recurrent];
    registry
        .register(Declaring::new("hostile-test-provider", 77, &halves))
        .unwrap();
    let err = registry
        .select(
            &hostile(),
            &ContinuationConfig::empty(),
            &[kv(), conv_qkv()],
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            ContinuationRegistryError::Unsupported {
                layer: 1,
                region: ContinuationRegion::KvAndRecurrent,
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn a_stateless_layer_asks_nothing_of_a_provider() {
    let mut registry = ContinuationRegistry::new();
    registry
        .register(Declaring::new("hostile-test-provider", 77, &[]))
        .unwrap();
    registry
        .select(
            &hostile(),
            &ContinuationConfig::empty(),
            &[LayerContinuationGeometry::Stateless],
        )
        .unwrap();
    assert_eq!(
        ContinuationRegion::required_by(&LayerContinuationGeometry::Stateless),
        None
    );
    let names: Vec<String> = ContinuationRegion::ALL
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(names, ["kv", "latent-kv", "recurrent", "kv-and-recurrent"]);
}

/// Factories, not instances: every build is a fresh conversation's state,
/// and the built provider — a trait object — runs the full hybrid
/// continuation contract.
#[test]
fn every_build_is_a_fresh_provider() {
    let mut registry = ContinuationRegistry::new();
    registry.register(Box::new(RowFactory)).unwrap();
    let plan = [kv(), recurrent(), latent(), conv_qkv()];
    let selected = registry
        .select(&RowKvState::identity(), &ContinuationConfig::empty(), &plan)
        .unwrap();
    assert_eq!(
        selected.authority(),
        &ContinuationAuthority::new(RowKvState::identity(), &ContinuationConfig::empty())
    );
    assert!(format!("{selected:?}").contains("row"));

    let mut first = selected.build();
    first.prepare_continuation(&plan).unwrap();
    first.append(0, vec![1.0; 4], vec![2.0; 4]);
    first.set_position(1);
    first.recurrent_state(1).unwrap();
    first.latent_state(2).unwrap();

    let mut second = selected.clone().build();
    second.prepare_continuation(&plan).unwrap();
    assert_eq!(first.keys(0).len(), 1);
    assert!(
        second.keys(0).is_empty(),
        "a build shared state with another"
    );
    assert_eq!(second.position(), 0);
}
