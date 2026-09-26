//! **C6 of CONTINUATION-PLUGIN-1: the C5 provider, loaded as a plugin,
//! through the CLI's own loader and selection.**
//!
//! Frozen in `docs/represent/forecasts/continuation-plugin-1-notes.json`
//! before this file existed. The fixture dylib is BUILT by cargo and
//! `dlopen`ed by `Plugins::load` (the real loader: ABI stamp, exported
//! register symbol, `PluginRegistrar::continuation`); nothing of it is
//! linked into this binary. Selection is the CLI's one resolver,
//! `select_in`, over the registry the loader composed. The journey is the
//! C5 gate unchanged: bit-identical to `canonical/v1` in logits, ids,
//! position and every K/V row. The `larql` binary itself is exercised by
//! `tests/test_continuation_plugin_cli.rs`.

#[cfg(unix)]
#[path = "../../../../../tests/support/continuation_fixture.rs"]
mod fixture;

#[cfg(unix)]
mod unix {
    use std::path::Path;

    use larql_vindex::format::vindex3::fixtures::G_TOKENS;
    use larql_vindex::format::vindex3::inspect::inspect_container;
    use larql_vindex::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
    use larql_vindex::format::vindex3::opplan::exec::continuation_handoff::ContinuationHandoff;
    use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
    use larql_vindex::format::vindex3::opplan::exec::continuation_registry::SelectedContinuation;
    use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
    use larql_vindex::format::vindex3::opplan::exec::kv::KvState;
    use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
    use larql_vindex::format::vindex3::opplan::exec::prefill_plan;
    use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
    use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

    use crate::commands::primary::continuation::{select_in, ContinuationChoice};
    use crate::commands::primary::vindex3_cmd::plugins::{PluginArgs, Plugins};

    use super::fixture;
    use fixture::{fixture_dylib, glimmer_container, FIXTURE_PACKAGE, HOSTILE};

    /// Decode steps before the handoff is saved, and again after resume.
    const STEPS: usize = 3;

    struct Program {
        _root: tempfile::TempDir,
        plan: ComponentOpPlan,
        store: OperandStore,
    }

    fn glimmer() -> Program {
        let root = tempfile::tempdir().unwrap();
        let container = glimmer_container(root.path());
        let inspection = inspect_container(&container, false).unwrap();
        let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
        assert!(outcome.closed(), "{:?}", outcome.defects);
        let plan = outcome.plan.unwrap();
        let store = OperandStore::open(&container, &inspection).unwrap();
        Program {
            _root: root,
            plan,
            store,
        }
    }

    fn loaded(paths: &[&Path]) -> Result<Plugins, String> {
        Plugins::load(&PluginArgs {
            plugins: paths.iter().map(|p| p.to_path_buf()).collect(),
            ..Default::default()
        })
        .map_err(|e| e.to_string())
    }

    fn named(identity: &str) -> ContinuationChoice<'_> {
        ContinuationChoice {
            identity: Some(identity),
            ..ContinuationChoice::default()
        }
    }

    // ---- the journey (the C5 gate) ------------------------------------

    #[derive(Debug, PartialEq)]
    struct Record {
        prefill_logits: Vec<f32>,
        step_logits: Vec<Vec<f32>>,
        ids: Vec<u32>,
        position: usize,
        keys: Vec<Vec<Vec<f32>>>,
        values: Vec<Vec<Vec<f32>>>,
    }

    fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                if v > best.1 {
                    (i, v)
                } else {
                    best
                }
            })
            .0 as u32
    }

    fn decode(
        p: &Program,
        state: &mut dyn KvState,
        ids: &mut Vec<u32>,
        logits: &mut Vec<Vec<f32>>,
    ) {
        let backend = ReferenceBackend::new();
        let mut session = DecodeSession::with_kv_state(&p.plan, &p.store, &backend, state).unwrap();
        for _ in 0..STEPS {
            let next = session.step(*ids.last().unwrap()).unwrap().logits.unwrap();
            ids.push(argmax(&next));
            logits.push(next);
        }
    }

    /// Prefill and decode under `writer`, hold the sealed handoff apart
    /// from every session, resume under `reader`'s authority, decode on.
    fn journey(
        p: &Program,
        writer: &SelectedContinuation,
        reader: &SelectedContinuation,
    ) -> Record {
        let mut handoff = writer.begin();
        let prefill_logits = prefill_plan(
            &p.plan,
            &p.store,
            &G_TOKENS,
            &ReferenceBackend::new(),
            handoff.state_mut(),
        )
        .unwrap()
        .logits
        .unwrap();
        let mut ids = vec![argmax(&prefill_logits)];
        let mut step_logits = Vec::new();
        decode(p, handoff.state_mut(), &mut ids, &mut step_logits);
        let saved: Vec<ContinuationHandoff> = vec![handoff];
        let mut resumed = saved
            .into_iter()
            .next()
            .unwrap()
            .resume(reader.authority())
            .expect("the authority that wrote the state resumes it");
        decode(p, resumed.state_mut(), &mut ids, &mut step_logits);
        let state = resumed.state();
        let layers = 0..plan_continuation_geometry(&p.plan).unwrap().len();
        Record {
            prefill_logits,
            step_logits,
            ids,
            position: state.position(),
            keys: layers
                .clone()
                .map(|l| state.rows(l).to_owned_rows().0.to_vec())
                .collect(),
            values: layers
                .map(|l| state.rows(l).to_owned_rows().1.to_vec())
                .collect(),
        }
    }

    #[test]
    fn a_loaded_provider_carries_a_conversation_bit_identical_to_canonical() {
        let program = glimmer();
        let plugins = loaded(&[&fixture_dylib()]).unwrap();
        let writer = select_in(&plugins.continuations, &program.plan, &named(HOSTILE)).unwrap();
        assert_eq!(writer.authority().identity.to_string(), HOSTILE);
        // Resume under an authority re-selected through the same path.
        let reader = select_in(&plugins.continuations, &program.plan, &named(HOSTILE)).unwrap();
        let record = journey(&program, &writer, &reader);

        let canonical = select_in(
            &plugins.continuations,
            &program.plan,
            &ContinuationChoice::engine(Some("standard")),
        )
        .unwrap();
        let reference = journey(&program, &canonical, &canonical);
        assert!(
            record == reference,
            "the loaded provider's conversation diverges from canonical/v1"
        );
        assert_eq!(record.ids.len(), 1 + 2 * STEPS);
    }

    // ---- overlay, not replacement --------------------------------------

    #[test]
    fn loaded_factories_overlay_the_shipped_set() {
        let program = glimmer();
        let plugins = loaded(&[&fixture_dylib()]).unwrap();
        let shipped = larql_kv::shipped_continuations().identities();
        let mut expected = shipped.clone();
        expected.push(ContinuationIdentity::new("hostile-test-provider", 77));
        assert_eq!(plugins.continuations.identities(), expected);
        // The shipped providers still select from the composed registry.
        for identity in &shipped {
            let spec = identity.to_string();
            let selected = select_in(&plugins.continuations, &program.plan, &named(&spec)).unwrap();
            assert_eq!(&selected.authority().identity, identity);
        }
    }

    #[test]
    fn control_without_the_plugin_the_identity_is_unregistered() {
        // The provider exists only in the dylib: with no --plugin, the same
        // resolver refuses it and names what IS registered.
        let program = glimmer();
        let err = select_in(
            &Plugins::none().continuations,
            &program.plan,
            &named(HOSTILE),
        )
        .unwrap_err();
        assert!(err.contains("hostile-test-provider/v77"), "{err}");
        assert!(
            err.contains("row/v1") && err.contains("canonical/v1"),
            "{err}"
        );
    }

    #[test]
    fn loading_the_plugin_twice_is_refused_as_a_duplicate_not_a_replacement() {
        let dylib = fixture_dylib();
        let err = loaded(&[&dylib, &dylib]).map(|_| ()).unwrap_err();
        assert!(err.contains("already registered"), "{err}");
        assert!(err.contains(&dylib.display().to_string()), "{err}");
    }

    #[test]
    fn the_loaded_provider_is_refused_on_a_hybrid_at_selection() {
        // F5 through the plugin path: the dylib's factory declares Kv only.
        use larql_vindex::format::vindex3::fixtures::encode_fixture_container;
        use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
        let root = tempfile::tempdir().unwrap();
        let (checkpoint, container) = (root.path().join("c"), root.path().join("v"));
        std::fs::create_dir_all(&checkpoint).unwrap();
        std::fs::create_dir_all(&container).unwrap();
        encode_fixture_container(
            hybrid_kda_mla_f32_model,
            &checkpoint,
            &container,
            "c6-hybrid",
        );
        let inspection = inspect_container(&container, false).unwrap();
        let plan = plan_component_ops(&inspection, &container, "target")
            .unwrap()
            .plan
            .unwrap();
        let plugins = loaded(&[&fixture_dylib()]).unwrap();
        let err = select_in(&plugins.continuations, &plan, &named(HOSTILE)).unwrap_err();
        assert!(err.contains("cannot hold layer"), "{err}");
    }

    // ---- choice refusals -----------------------------------------------

    #[test]
    fn continuation_and_engine_together_are_refused() {
        let program = glimmer();
        let err = select_in(
            &Plugins::none().continuations,
            &program.plan,
            &ContinuationChoice {
                engine: Some("row"),
                identity: Some(HOSTILE),
                options: &[],
            },
        )
        .unwrap_err();
        assert!(
            err.contains("--continuation") && err.contains("--engine"),
            "{err}"
        );
    }

    #[test]
    fn a_malformed_identity_or_option_is_refused_before_selection() {
        let program = glimmer();
        let none = Plugins::none();
        for spec in ["hostile", "hostile/77", "hostile/vx", "/v1"] {
            let err = select_in(&none.continuations, &program.plan, &named(spec)).unwrap_err();
            assert!(
                err.contains(spec) || err.contains("family"),
                "{spec}: {err}"
            );
        }
        let options = ["bits".to_string()];
        let err = select_in(
            &none.continuations,
            &program.plan,
            &ContinuationChoice {
                identity: Some("row/v1"),
                options: &options,
                ..ContinuationChoice::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("key=value"), "{err}");
    }

    #[test]
    fn a_loaded_providers_options_reach_it() {
        let program = glimmer();
        let plugins = loaded(&[&fixture_dylib()]).unwrap();
        let bits = ["bits=4".to_string()];
        let unknown = ["mode=x".to_string()];
        let choice = |options| ContinuationChoice {
            identity: Some(HOSTILE),
            options,
            ..ContinuationChoice::default()
        };
        assert!(select_in(&plugins.continuations, &program.plan, &choice(&bits)).is_ok());
        let err = select_in(&plugins.continuations, &program.plan, &choice(&unknown)).unwrap_err();
        assert!(err.contains("mode"), "{err}");
    }

    // ---- anti-cheat: nothing of the fixture is linked -------------------

    #[test]
    fn the_fixture_is_not_a_dependency_of_the_cli() {
        let output = std::process::Command::new(std::env::var("CARGO").unwrap_or("cargo".into()))
            .current_dir(fixture::workspace_root())
            .args(["metadata", "--format-version", "1", "--no-deps", "--locked"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let package = |name: &str| {
            metadata["packages"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["name"] == name)
                .cloned()
                .unwrap_or_else(|| panic!("{name} is not a workspace package"))
        };
        // Positive control: the metadata does describe the fixture, as a cdylib.
        let fixture = package(FIXTURE_PACKAGE);
        assert_eq!(
            fixture["targets"][0]["crate_types"],
            serde_json::json!(["cdylib"])
        );
        let cli = package("larql-cli");
        let deps: Vec<&str> = cli["dependencies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["name"].as_str().unwrap())
            .collect();
        assert!(
            deps.contains(&"larql-vindex"),
            "control: the scan reads real dependencies"
        );
        assert!(
            !deps.contains(&FIXTURE_PACKAGE),
            "larql-cli depends on the fixture in some table: {deps:?}"
        );
    }

    #[test]
    fn no_cli_source_names_the_fixtures_types() {
        // Spelled in pieces so this file does not match its own scan.
        let needles = [
            concat!("Hostile", "Rows"),
            concat!("Hostile", "Factory"),
            concat!("larql_continuation", "_fixture"),
        ];
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut stack = vec![src];
        let mut scanned = 0usize;
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    scanned += 1;
                    let text = std::fs::read_to_string(&path).unwrap();
                    for needle in needles {
                        assert!(
                            !text.contains(needle),
                            "{} names `{needle}`",
                            path.display()
                        );
                    }
                }
            }
        }
        assert!(scanned > 50, "scanned only {scanned} files");
    }
}

/// The loader is unix `dlopen` only; elsewhere `--plugin` is refused by
/// name, before anything is opened. C6 does not change that.
#[cfg(not(unix))]
#[test]
fn plugins_are_refused_by_name_off_unix() {
    use crate::commands::primary::vindex3_cmd::plugins::{PluginArgs, Plugins};
    let path = std::path::PathBuf::from("larql_continuation_plugin.dll");
    let err = Plugins::load(&PluginArgs {
        plugins: vec![path.clone()],
        ..Default::default()
    })
    .map(|_| ())
    .unwrap_err()
    .to_string();
    assert!(err.contains("unix dlopen only"), "{err}");
    assert!(err.contains(&path.display().to_string()), "{err}");
}
