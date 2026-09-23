use super::*;
use crate::vindex3::{
    distributed::{self, DistributedSession, ShardTransport},
    input::{CachedInputSession, InputPosition, ReplaySession},
};
use larql_router_protocol::vindex3::{Binding, Response};
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn external_rows_and_replay_match_tokens_beyond_the_sliding_window() {
    let dir = container_with(miniature_glimmer);
    let runtime = Vindex3Runtime::open(dir.path(), COMPONENT, ProductionBackend::new())
        .unwrap()
        .prepare()
        .unwrap();
    let (plan, ops, backend) = (runtime.plan(), runtime.operands(), runtime.backend());
    let mut token_kv = RowKvState::default();
    let mut mixed_kv = RowKvState::default();
    let mut tokens = CachedInputSession::new(plan, ops, backend, &mut token_kv).unwrap();
    let mut mixed = CachedInputSession::new(plan, ops, backend, &mut mixed_kv).unwrap();
    let mut replay = ReplaySession::new(plan, ops, backend);
    for (position, id) in G_TOKENS.iter().cycle().take(14).enumerate() {
        let input = if position % 2 == 0 {
            InputPosition::Embedding(ops.embed_token(plan, backend, *id).unwrap())
        } else {
            InputPosition::Token(*id)
        };
        let expected = tokens.step(*id).unwrap();
        assert_eq!(
            bits(&expected),
            bits(&mixed.extend_inputs(std::slice::from_ref(&input)).unwrap()),
            "external row at {position}"
        );
        assert_eq!(
            bits(&expected),
            bits(&replay.extend_inputs(&[input]).unwrap()),
            "replay at {position}"
        );
    }
    // A genuinely changed image row changes evidence, so parity is meaningful.
    let a = mixed
        .extend_inputs(&[InputPosition::Embedding(vec![0.7; ops.hidden()])])
        .unwrap();
    let b = tokens.step(1).unwrap();
    assert_ne!(bits(&a), bits(&b));
}

#[test]
fn invalid_prompt_is_rejected_before_either_provider_advances() {
    let dir = container_with(miniature_glimmer);
    let runtime = Vindex3Runtime::open(dir.path(), COMPONENT, ProductionBackend::new())
        .unwrap()
        .prepare()
        .unwrap();
    let (plan, ops, backend) = (runtime.plan(), runtime.operands(), runtime.backend());
    let mut kv = RowKvState::default();
    let mut cached = CachedInputSession::new(plan, ops, backend, &mut kv).unwrap();
    let mut replay = ReplaySession::new(plan, ops, backend);
    for bad in [
        InputPosition::Token(u32::MAX),
        InputPosition::Embedding(vec![0.0; ops.hidden() + 1]),
        InputPosition::Embedding(vec![f32::NAN; ops.hidden()]),
    ] {
        let prompt = [InputPosition::Token(1), bad];
        assert!(cached.extend_inputs(&prompt).is_err());
        assert!(replay.extend_inputs(&prompt).is_err());
        assert_eq!(cached.position(), 0);
        assert_eq!(replay.position(), 0);
    }
    assert_eq!(
        bits(&cached.prefill(&G_TOKENS).unwrap()),
        bits(&replay.prefill(&G_TOKENS).unwrap())
    );
}

struct LocalShards<'a> {
    plan: &'a larql_vindex::format::vindex3::opplan::ComponentOpPlan,
    backend: &'a ProductionBackend,
    images: Vec<PreparedOperands>,
    bindings: Vec<Binding>,
    corrupt: bool,
}
impl ShardTransport for LocalShards<'_> {
    fn bindings(&self) -> Vec<Binding> {
        self.bindings.clone()
    }
    fn forward(&self, index: usize, rows: Vec<Vec<f32>>) -> Result<Response, String> {
        let mut response = distributed::forward(
            self.plan,
            &self.images[index],
            self.backend,
            &self.bindings[index],
            rows,
        )
        .map_err(|e| e.to_string())?;
        if self.corrupt {
            response.binding.artifact = "a".repeat(64);
        }
        Ok(response)
    }
}
#[test]
fn distributed_prefix_matches_local_and_refuses_incomplete_or_changed_bindings() {
    let dir = container_with(miniature_glimmer);
    let runtime = Vindex3Runtime::open(dir.path(), COMPONENT, ProductionBackend::new()).unwrap();
    let plan = runtime.plan();
    let backend = runtime.backend();
    let endpoints =
        PreparedOperands::load(plan, runtime.operands(), backend, ExecutionSlice::Endpoints)
            .unwrap();
    assert_eq!(endpoints.layer_count(), 0);
    assert!(endpoints.has_output());
    let mut kv = RowKvState::default();
    assert!(DecodeSession::over_prepared(plan, &endpoints, backend, &mut kv).is_err());
    assert!(
        larql_vindex::format::vindex3::opplan::exec::prefill_prepared(
            plan,
            &endpoints,
            &[1],
            backend,
            &mut kv
        )
        .is_err()
    );
    assert_eq!(kv.position(), 0);
    let full =
        PreparedOperands::load(plan, runtime.operands(), backend, ExecutionSlice::Full).unwrap();
    let mut local = CachedInputSession::new(plan, &full, backend, &mut kv).unwrap();
    let identity = distributed::artifact_identity(dir.path(), plan).unwrap();
    let make = || {
        let images: Vec<_> = (0..plan.layers.len())
            .map(|i| {
                PreparedOperands::load(
                    plan,
                    runtime.operands(),
                    backend,
                    ExecutionSlice::LayerRange {
                        start: i,
                        end: i + 1,
                    },
                )
                .unwrap()
            })
            .collect();
        let bindings = images
            .iter()
            .map(|ops| distributed::binding(dir.path(), plan, ops).unwrap())
            .collect();
        LocalShards {
            plan,
            backend,
            images,
            bindings,
            corrupt: false,
        }
    };
    let mut remote = DistributedSession::new(plan, &endpoints, backend, &identity, make()).unwrap();
    for (position, id) in G_TOKENS.iter().cycle().take(12).enumerate() {
        let input = if position == 2 {
            InputPosition::Embedding(full.embed_token(plan, backend, *id).unwrap())
        } else {
            InputPosition::Token(*id)
        };
        let a = local.extend_inputs(std::slice::from_ref(&input)).unwrap();
        let b = remote.extend_inputs(&[input]).unwrap();
        let max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-5, "position {position}: max logit delta {max}");
    }
    for defect in 0..5 {
        let mut transport = make();
        match defect {
            0 => {
                transport.bindings.pop();
            }
            1 => transport.bindings[1].start = 0,
            2 => transport.bindings[1].artifact = "b".repeat(64),
            3 => transport.bindings[1].backend = "metal".into(),
            _ => transport.bindings[1].lowering = "cpu@999".into(),
        }
        assert!(DistributedSession::new(plan, &endpoints, backend, &identity, transport).is_err());
    }
    let mut transport = make();
    transport.corrupt = true;
    let mut broken =
        DistributedSession::new(plan, &endpoints, backend, &identity, transport).unwrap();
    assert!(broken.prefill(&G_TOKENS).is_err());
    assert_eq!(broken.position(), 0);
    let too_long = vec![
        InputPosition::Embedding(vec![0.0; endpoints.hidden()]);
        larql_router_protocol::vindex3::MAX_POSITIONS + 1
    ];
    assert!(broken
        .extend_inputs(&too_long)
        .unwrap_err()
        .to_string()
        .contains("4096"));
    assert_eq!(broken.position(), 0);
}
