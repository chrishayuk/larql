# Inference engines and runtime composition

**Class: CURRENT.** [Stack architecture](architecture-stack.md) ·
[compute substrate](compute-substrate.md) · [interface map](runtime-surfaces.md).

`larql-inference` owns execution orchestration: opening artifacts, sessions,
generation, tokenization/chat composition, FFN routing and VINDEX3 records.
CPU forward math and dispatch traits live in `larql-compute`; Metal kernels
and lowering live in `larql-compute-metal`. Compatibility re-exports do not
move implementation ownership back into inference.

## Two execution paths

The model-weight/V2 path composes `ModelWeights`, architecture facts, an index
where needed and a numerical backend. The layer graph, FFN policy and engine
state select and execute supported dense, sparse or distributed operations.
The `KvEngine` trait lives in inference; its implementations live in `larql-kv`.

The V3 path opens an artifact's own component plan and representations.
`Vindex3Runtime`/`PreparedVindex3` bind the plan, operand policy and numerical
realization. `Vindex3Session` advances the canonical interpreter's state;
`LogitsSession` exposes logits and logical position to generation. It does not
translate the artifact into a V2 `ModelWeights` object to make it executable.

| Owner | Source |
|---|---|
| Engine orchestration and generation | [layer_graph](../crates/larql-inference/src/layer_graph/) |
| FFN routing policy | [ffn](../crates/larql-inference/src/ffn/) and [ffn_policy](../crates/larql-inference/src/ffn_policy/) |
| Model-weight engine contract | [kv_engine](../crates/larql-inference/src/kv_engine/) |
| V3 runtime/session/record composition | [vindex3](../crates/larql-inference/src/vindex3/) |
| CPU intent and math | [larql-compute](../crates/larql-compute/README.md) |
| Metal implementation and controls | [larql-compute-metal](../crates/larql-compute-metal/README.md) |
| State implementations | [larql-kv](../crates/larql-kv/README.md) |

## Evidence and capabilities

Canonical V3 decode exposes actual carrier-write boundaries and scoped head
observation/intervention. The recorder binds provenance and receipts; lenses
read the prepared image under their own contract. Generation, observation,
descriptive attribution and controlled intervention are different operations.
See [the observation guide](vindex3/observation-and-intervention.md).

Backend constructors, lowering support and available operands determine what
executes. CPU fallback must be explicit; an artifact binding defect must not
be hidden as a backend preference. The shared
[refusal vocabulary](../crates/larql-execution/README.md) distinguishes missing
residency, unsupported execution and invalid bindings.

Use [the V3 runtime guide](vindex3-runtime.md) for state/serving details,
[FFN documentation](ffn/README.md) for routing, and
[the KV state policy](../crates/larql-kv/docs/state-policy.md) for continuation
contracts. The [archived engine guide](archive/README.md) preserves prior
walkthroughs and measured results without presenting them as today's defaults.
