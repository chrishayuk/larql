//! The VINDEX3 transport seams: `FfnTransport::forward_bound`'s default
//! authentication, and that each trait is implementable from this crate's
//! wire types alone.

use larql_router_protocol::vindex3::{self, SCHEMA};
use larql_router_protocol::vindex3_experts;
use larql_router_protocol::vindex3_ffn::{self, OperandIdentity};
use larql_router_protocol::vindex3_transport::{
    ExpertOutput, ExpertTransport, FfnTransport, ShardTransport,
};

const HIDDEN: usize = 4;
const LAYER: usize = 2;

fn program() -> vindex3::Binding {
    vindex3::Binding {
        schema: SCHEMA,
        artifact: "a".repeat(64),
        execution_identity: "e".repeat(64),
        backend: "cpu".into(),
        lowering: "cpu-production/v1".into(),
        start: 0,
        end: 1,
        layers: 1,
        hidden: HIDDEN,
    }
}

fn ffn_binding(tag: &str) -> vindex3_ffn::Binding {
    vindex3_ffn::Binding {
        program: program(),
        operands: vec![OperandIdentity {
            layer: 0,
            operand: tag.into(),
            representation: "f32".into(),
            codec: "raw".into(),
            realization: "dense".into(),
            extent: "full".into(),
            dependencies: String::new(),
        }],
    }
}

/// A transport answering with a fixed binding and layer offset.
struct Ffn {
    binding: vindex3_ffn::Binding,
    layer_offset: usize,
    fail: bool,
}

impl FfnTransport for Ffn {
    fn bindings(&self) -> Vec<vindex3_ffn::Binding> {
        vec![self.binding.clone()]
    }
    fn forward(
        &self,
        _shard: usize,
        layer: usize,
        normalized: &[f32],
    ) -> Result<vindex3_ffn::Response, String> {
        if self.fail {
            return Err("worker down".into());
        }
        Ok(vindex3_ffn::Response {
            binding: self.binding.clone(),
            layer: layer + self.layer_offset,
            row: normalized.iter().map(|x| x * 2.0).collect(),
        })
    }
}

#[test]
fn forward_bound_returns_the_row_for_the_admitted_binding() {
    let binding = ffn_binding("w");
    let t = Ffn {
        binding: binding.clone(),
        layer_offset: 0,
        fail: false,
    };
    let row = t.forward_bound(0, LAYER, &[1.0; HIDDEN], &binding).unwrap();
    assert_eq!(row, vec![2.0; HIDDEN]);
    assert_eq!(t.bindings(), vec![binding]);
}

#[test]
fn forward_bound_refuses_a_changed_binding_or_layer() {
    let admitted = ffn_binding("w");
    let swapped = Ffn {
        binding: ffn_binding("other"),
        layer_offset: 0,
        fail: false,
    };
    let err = swapped
        .forward_bound(0, LAYER, &[1.0; HIDDEN], &admitted)
        .unwrap_err();
    assert!(err.contains("changed binding or layer"), "got: {err}");

    let shifted = Ffn {
        binding: admitted.clone(),
        layer_offset: 1,
        fail: false,
    };
    assert!(shifted
        .forward_bound(0, LAYER, &[1.0; HIDDEN], &admitted)
        .is_err());
}

#[test]
fn forward_bound_passes_transport_errors_through() {
    let binding = ffn_binding("w");
    let t = Ffn {
        binding: binding.clone(),
        layer_offset: 0,
        fail: true,
    };
    assert_eq!(
        t.forward_bound(0, LAYER, &[1.0; HIDDEN], &binding),
        Err("worker down".to_string())
    );
}

struct Shards;

impl ShardTransport for Shards {
    fn bindings(&self) -> Vec<vindex3::Binding> {
        vec![program()]
    }
    fn forward(&self, _shard: usize, rows: Vec<Vec<f32>>) -> Result<vindex3::Response, String> {
        Ok(vindex3::Response {
            binding: program(),
            rows,
        })
    }
}

struct Experts;

impl ExpertTransport for Experts {
    fn bindings(&self) -> Vec<vindex3_experts::Binding> {
        Vec::new()
    }
    fn forward(
        &self,
        _shard: usize,
        _layer: usize,
        experts: &[usize],
        row: &[f32],
    ) -> Result<Vec<ExpertOutput>, String> {
        Ok(experts
            .iter()
            .map(|&expert| ExpertOutput {
                expert,
                row: row.to_vec(),
            })
            .collect())
    }
    fn forward_rows(
        &self,
        _shard: usize,
        _layer: usize,
        rows: &[larql_router_protocol::vindex3_expert_rows::Row],
    ) -> Result<Vec<larql_router_protocol::vindex3_expert_rows::Output>, String> {
        Ok(rows
            .iter()
            .flat_map(|row| {
                row.experts.iter().map(|&expert| {
                    larql_router_protocol::vindex3_expert_rows::Output {
                        row: row.id,
                        expert,
                        values: row.input.clone(),
                    }
                })
            })
            .collect())
    }
}

#[test]
fn shard_and_expert_transports_implement_over_wire_types() {
    let rows = vec![vec![0.5; HIDDEN]];
    let response = Shards.forward(0, rows.clone()).unwrap();
    assert_eq!(response.rows, rows);
    assert_eq!(Shards.bindings(), vec![program()]);

    assert!(Experts.bindings().is_empty());
    let out = Experts.forward(0, LAYER, &[3, 1], &[1.0; HIDDEN]).unwrap();
    assert_eq!(
        out,
        vec![
            ExpertOutput {
                expert: 3,
                row: vec![1.0; HIDDEN]
            },
            ExpertOutput {
                expert: 1,
                row: vec![1.0; HIDDEN]
            },
        ]
    );
}
