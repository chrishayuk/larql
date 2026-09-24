use super::*;
use crate::vindex3::{
    dense_ffn::{self, DenseFfnSession, FfnTransport},
    LogitsSession,
};
use larql_router_protocol::vindex3_ffn::{Binding, Response};
use larql_vindex::format::vindex3::opplan::{
    exec::{
        self,
        prepared::{ExecutionSlice, PreparedOperands},
        Plane, PlaneEvent,
    },
    ComponentOpPlan,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Workers {
    plan: ComponentOpPlan,
    backend: ProductionBackend,
    images: Vec<PreparedOperands>,
    bindings: Vec<Binding>,
    fault: Arc<AtomicUsize>,
}
impl FfnTransport for Workers {
    fn bindings(&self) -> Vec<Binding> {
        self.bindings.clone()
    }
    fn forward(&self, shard: usize, layer: usize, row: &[f32]) -> Result<Response, String> {
        // Fail after attention and the first layer have already advanced.
        if layer == 1 && self.fault.load(Ordering::SeqCst) == 1 {
            return Err("injected timeout".into());
        }
        let mut r = dense_ffn::forward(
            &self.plan,
            &self.images[shard],
            &self.backend,
            &self.bindings[shard],
            layer,
            row,
        )
        .map_err(|e| e.to_string())?;
        if layer == 1 {
            match self.fault.load(Ordering::SeqCst) {
                2 => {
                    r.row.pop();
                }
                3 => r.binding.program.artifact = "0".repeat(64),
                4 => r.layer = 0,
                5 => r.row[0] = f32::NAN,
                _ => {}
            }
        }
        Ok(r)
    }
}
fn workers(
    runtime: &Vindex3Runtime<ProductionBackend>,
    path: &std::path::Path,
    split: bool,
) -> Workers {
    let worker_runtime = Vindex3Runtime::open(path, COMPONENT, ProductionBackend::new()).unwrap();
    assert_eq!(
        worker_runtime.plan().layers.len(),
        runtime.plan().layers.len()
    );
    let runtime = &worker_runtime;
    let ranges = if split {
        vec![(0, 1), (1, 2)]
    } else {
        vec![(0, 2)]
    };
    let images: Vec<_> = ranges
        .into_iter()
        .map(|(start, end)| {
            PreparedOperands::load(
                runtime.plan(),
                runtime.operands(),
                runtime.backend(),
                ExecutionSlice::DenseFfns { start, end },
            )
            .unwrap()
        })
        .collect();
    let bindings = images
        .iter()
        .map(|ops| dense_ffn::binding(path, runtime.plan(), ops).unwrap())
        .collect();
    Workers {
        plan: runtime.plan().clone(),
        backend: ProductionBackend::new(),
        images,
        bindings,
        fault: Arc::new(AtomicUsize::new(0)),
    }
}
fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn dense_ffn_placement_matches_local_prefill_layers_logits_and_decode() {
    let dir = container_with(miniature_glimmer);
    for split in [false, true] {
        let runtime =
            Vindex3Runtime::open(dir.path(), COMPONENT, ProductionBackend::new()).unwrap();
        let control =
            Vindex3Runtime::open(dir.path(), COMPONENT, ProductionBackend::new()).unwrap();
        let local = PreparedOperands::load(
            control.plan(),
            control.operands(),
            control.backend(),
            ExecutionSlice::Full,
        )
        .unwrap();
        let remote = dense_ffn::prepare_coordinator(
            dir.path(),
            runtime.plan(),
            runtime.operands(),
            runtime.backend(),
            workers(&runtime, dir.path(), split),
        )
        .unwrap();
        assert_eq!(remote.residency_census().ffn.total(), 0);
        assert_eq!(
            runtime.operands().store().touched_objects(),
            exec::requirements::required_objects(
                runtime.plan(),
                &ExecutionSlice::DenseFfnCoordinator
            )
            .unwrap()
        );
        let mut kv = RowKvState::default();
        let mut reference =
            Vindex3Session::over_prepared(runtime.plan(), &local, runtime.backend(), &mut kv)
                .unwrap();
        let mut session = DenseFfnSession::new(runtime.plan(), &remote, runtime.backend()).unwrap();
        assert_eq!(
            bits(&reference.prefill(&G_TOKENS).unwrap()),
            bits(&session.prefill(&G_TOKENS).unwrap())
        );
        // Window is three. This checks continuing across eviction boundaries.
        for id in G_TOKENS.iter().cycle().take(14) {
            assert_eq!(
                bits(&reference.step(*id).unwrap()),
                bits(&session.step(*id).unwrap())
            );
        }
        let trace = |ops: &PreparedOperands| {
            let mut layers = Vec::new();
            let out = exec::execute_prepared_streaming(
                runtime.plan(),
                ops,
                &G_TOKENS,
                runtime.backend(),
                None,
                &mut |event| {
                    if let PlaneEvent::Layer { trace, .. } = event {
                        if let Plane::Rows(rows) = trace.post_layer {
                            layers.push(rows.into_iter().map(|r| bits(&r)).collect::<Vec<_>>());
                        }
                    }
                    Ok(())
                },
            )
            .unwrap();
            (layers, bits(&out.logits.unwrap()))
        };
        assert_eq!(trace(&local), trace(&remote));
    }
}

#[test]
fn dense_ffn_binding_refuses_each_mismatch_before_execution() {
    let dir = container_with(miniature_glimmer);
    let runtime = Vindex3Runtime::open(dir.path(), COMPONENT, ProductionBackend::new()).unwrap();
    for (case, want) in [
        (0, "incomplete"),
        (1, "overlapping"),
        (2, "artifact"),
        (3, "numerical provider"),
        (4, "dimensions"),
        (5, "representation"),
    ] {
        let mut w = workers(&runtime, dir.path(), true);
        match case {
            0 => {
                w.bindings.pop();
            }
            1 => w.bindings.push(w.bindings[0].clone()),
            2 => w.bindings[0].program.artifact = "0".repeat(64),
            3 => w.bindings[0].program.lowering = "cpu-production/v999".into(),
            4 => w.bindings[0].program.hidden += 1,
            5 => w.bindings[0].operands[0].realization.push_str("wrong"),
            _ => unreachable!(),
        }
        let error = dense_ffn::prepare_coordinator(
            dir.path(),
            runtime.plan(),
            runtime.operands(),
            runtime.backend(),
            w,
        )
        .err()
        .expect("refuses")
        .to_string();
        assert!(error.contains(want), "{case}: {error}");
    }
}

#[test]
fn dense_ffn_worker_is_stateless_and_refuses_unowned_layers() {
    let dir = container_with(miniature_glimmer);
    let runtime = Vindex3Runtime::open(dir.path(), COMPONENT, ProductionBackend::new()).unwrap();
    let w = workers(&runtime, dir.path(), true);
    let a = vec![0.2; w.images[0].hidden()];
    let b = vec![0.7; a.len()];
    let first = w.forward(0, 0, &a).unwrap();
    w.forward(0, 0, &b).unwrap();
    assert_eq!(bits(&first.row), bits(&w.forward(0, 0, &a).unwrap().row));
    assert!(w.forward(0, 1, &a).unwrap_err().contains("outside"));
    let mut kv = RowKvState::default();
    assert!(Vindex3Session::over_prepared(
        runtime.plan(),
        &w.images[0],
        runtime.backend(),
        &mut kv
    )
    .is_err());
}

#[test]
fn dense_ffn_failure_invalidates_session_and_fresh_replay_recovers() {
    let dir = container_with(miniature_glimmer);
    let runtime = Vindex3Runtime::open(dir.path(), COMPONENT, ProductionBackend::new()).unwrap();
    for mode in 1..=5 {
        let w = workers(&runtime, dir.path(), true);
        let fault = w.fault.clone();
        let remote = dense_ffn::prepare_coordinator(
            dir.path(),
            runtime.plan(),
            runtime.operands(),
            runtime.backend(),
            w,
        )
        .unwrap();
        let mut session = DenseFfnSession::new(runtime.plan(), &remote, runtime.backend()).unwrap();
        let initial = session.step(G_TOKENS[0]).unwrap();
        fault.store(mode, Ordering::SeqCst);
        let capture = dense_ffn::profile::Capture::start().unwrap();
        assert!(session.step(G_TOKENS[1]).is_err());
        let rows = capture.finish();
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].complete);
        assert_eq!(rows[0].position, 1);
        assert_eq!(session.position(), 1);
        fault.store(0, Ordering::SeqCst);
        assert!(session
            .step(G_TOKENS[1])
            .unwrap_err()
            .to_string()
            .contains("invalid"));
        let mut fresh = DenseFfnSession::new(runtime.plan(), &remote, runtime.backend()).unwrap();
        assert_eq!(bits(&initial), bits(&fresh.step(G_TOKENS[0]).unwrap()));
        let mut reference = runtime.session().unwrap();
        reference.step(G_TOKENS[0]).unwrap();
        assert_eq!(
            bits(&reference.step(G_TOKENS[1]).unwrap()),
            bits(&fresh.step(G_TOKENS[1]).unwrap())
        );
    }
}
