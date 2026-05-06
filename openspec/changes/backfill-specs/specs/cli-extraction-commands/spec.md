## ADDED Requirements

### Requirement: `larql extract` builds vindexes at the requested level

`larql extract` (and its `extract-index` alias) SHALL build a vindex
from a HuggingFace-shaped model directory or repo, MUST honour the
`--level browse|inference|all` flag exactly as documented (browse is
gate-only, inference adds projections, all is the full extract), and
SHALL accept either safetensors or GGUF inputs through the same code
path. The runner MUST snapshot HuggingFace metadata (chat template,
special tokens, generation config) into the output vindex so that
downstream `larql run` does not have to re-read the source.

#### Scenario: Browse-level extract carves the gate-only footprint
- **WHEN** `larql extract MODEL --level browse -o OUT` is invoked on a synthetic model
- **THEN** the resulting vindex SHALL contain only the browse-tier files (gate vectors + tokenizer + index)
<!-- unbacked: integration covered indirectly via larql_vindex extract pipeline -->

#### Scenario: Inference-level extract carves projections + gate
- **WHEN** `--level inference` is supplied
- **THEN** the vindex SHALL additionally contain attention/ffn projection weights so `larql run --metal` can execute on it
<!-- unbacked -->

#### Scenario: Full-level extract carries the entire weight set
- **WHEN** `--level all` is supplied
- **THEN** the vindex SHALL be a complete weight snapshot including residual artefacts
<!-- unbacked -->

#### Scenario: GGUF input flows through the same extract pipeline
- **WHEN** the source path points at a `.gguf` file with a sibling `tokenizer.json`
- **THEN** the runner SHALL load the GGUF, normalise its config, and emit the same vindex layout as the safetensors path
<!-- unbacked -->

### Requirement: `larql compile` writes patches into model weights

`larql compile` SHALL run in one of four modes — `single`, `patch`,
`menu`, or `edge` — and MUST write the rewritten weights into a new
model directory (or vindex, depending on `INTO MODEL` vs `INTO
VINDEX`) without mutating the source. The `edge` mode MUST install
each (trigger, write) pair into the FFN gate/up/down trio of the
target layer using the documented row-zero convention. The compiler
SHALL preserve magnitude (`||gate_row||` matches the reported
`g_norm * α_norm`) and SHALL surface explicit errors for missing
tensors and zero-trigger inputs.

#### Scenario: Edge install writes magnitude-preserved rows into slot zero
- **WHEN** `install_edge` is invoked with a unit trigger, a unit write vector, and α_norm = 30.0
- **THEN** `gate[0, 0]` SHALL equal `stats.g_norm * 30.0` within 1e-5
<!-- test: larql_cli::commands::extraction::compile_cmd::edge::tests::install_writes_into_slot_zero -->

#### Scenario: Magnitude is preserved across input scales
- **WHEN** `install_edge` is run with triggers scaled by 0.1×, 1×, and 100×
- **THEN** the resulting `||gate_row||` SHALL equal `stats.g_norm * 30.0` within 1e-5 relative error for every scale
<!-- test: larql_cli::commands::extraction::compile_cmd::edge::tests::magnitude_preservation_invariant -->

#### Scenario: Down-projection alpha matches the reported stats
- **WHEN** `install_edge` is invoked with a non-unit write vector
- **THEN** every `down[j, 0]` row SHALL equal `write[j] * stats.alpha` within 1e-5
<!-- test: larql_cli::commands::extraction::compile_cmd::edge::tests::write_down_alpha_matches_stats -->

#### Scenario: Alpha multiplier scales the write vector linearly
- **WHEN** the same trigger/write pair is installed with α_mul = 1.0 and α_mul = 5.0
- **THEN** the ratio `s2.alpha / s1.alpha` SHALL equal 5.0 within 1e-5
<!-- test: larql_cli::commands::extraction::compile_cmd::edge::tests::alpha_mul_scales_write_linearly -->

#### Scenario: Zero trigger is rejected
- **WHEN** `install_edge` is called with an all-zero trigger vector
- **THEN** it SHALL return `EdgeError::ZeroTrigger`
<!-- test: larql_cli::commands::extraction::compile_cmd::edge::tests::zero_trigger_rejected -->

#### Scenario: Missing target tensor is reported by key name
- **WHEN** `install_edge` is called with a non-existent gate tensor key
- **THEN** it SHALL return `EdgeError::MissingTensor(k)` containing the missing key
<!-- test: larql_cli::commands::extraction::compile_cmd::edge::tests::missing_tensor_reports_key -->

#### Scenario: Trigger shorter than hidden dim does not panic
- **WHEN** the trigger vector is shorter than `hidden`
- **THEN** `install_edge` SHALL run successfully, leaving untouched gate columns at their pre-existing values
<!-- test: larql_cli::commands::extraction::compile_cmd::edge::tests::shorter_trigger_does_not_panic -->

### Requirement: Weight and attention walk research commands extract circuits without inference

`larql dev weight-extract` and `larql dev attention-extract` SHALL
read a vindex's projection weights, MUST decompose the QK / OV
products via SVD, and SHALL emit edge / circuit data without running
a forward pass. `larql dev vector-extract` SHALL stream full
projection vectors to NDJSON for downstream walks. These commands
MUST be safe to run on a Browse-level vindex even though their
output is only meaningful at Inference level or above.

#### Scenario: weight-extract refuses to run on a missing-FFN vindex
- **WHEN** `larql dev weight-extract --index <browse-vindex>` is invoked
- **THEN** the command SHALL exit non-zero with a clear error pointing the user at `--level inference`
<!-- unbacked -->

#### Scenario: attention-extract emits edges scoped to the requested layer range
- **WHEN** `larql dev attention-extract --index X --layers 0-3` is run
- **THEN** the output edges SHALL all carry `layer ∈ [0, 3]`
<!-- unbacked -->

#### Scenario: vector-extract NDJSON is suitable input for `extract-index`
- **WHEN** `larql dev vector-extract` writes NDJSON, then `larql extract-index --vectors NDJSON` consumes it
- **THEN** the resulting vindex SHALL be queryable via `larql run`
<!-- unbacked -->

### Requirement: QK template / rank / mode / OV-gate commands probe attention circuits

`larql dev qk-templates` SHALL extract attention template circuits
from QK-weight SVD decomposition; `larql dev qk-rank` SHALL report
the SVD-rank ladder of attention QK products; `larql dev qk-modes`
SHALL extract interpretable modes from low-rank QK heads via gate
projection; `larql dev ov-gate` SHALL map attention OV circuits to
FFN gate features. All four commands MUST accept a layer range and
SHALL emit JSON or NDJSON for downstream tooling.

#### Scenario: `qk-templates --layers 0-1` emits one record per (layer, head)
- **WHEN** the command is run on a 2-layer model
- **THEN** the output SHALL contain at least 2 × num_heads records
<!-- unbacked -->

#### Scenario: `qk-rank` reports a non-decreasing singular-value ladder
- **WHEN** the command is run against a single attention head
- **THEN** the emitted singular values SHALL be sorted in non-increasing order
<!-- unbacked -->

#### Scenario: `qk-modes` requires a gate index
- **WHEN** the command is run without a built `index_gates`
- **THEN** it SHALL exit non-zero with a hint to run `larql dev index-gates` first
<!-- unbacked -->

#### Scenario: `ov-gate` maps each OV head to a gate feature ID
- **WHEN** the command runs on a vindex with a built gate index
- **THEN** each emitted record SHALL carry an `(ov_head, gate_feature)` pair
<!-- unbacked -->

### Requirement: Bottleneck and projection-test commands isolate ablation effects

The bottleneck family of dev subcommands SHALL each run a
forward-pass-style ablation across `larql dev attn-bottleneck`,
`larql dev ffn-bottleneck`, `larql dev bottleneck-test`, `larql dev
projection-test`, and `larql dev ffn-overlap`, MUST diff the
residual stream before and after the ablation, and MUST emit a
per-layer (and per-head, where applicable) score so the user can
compare ablation effects across components. The commands MUST fail
loudly when given a Browse-level vindex with no projection weights.

#### Scenario: attn-bottleneck reports per-head delta KL
- **WHEN** `larql dev attn-bottleneck --index INF --prompt "..."` runs on an Inference-level vindex
- **THEN** each emitted record SHALL include `(layer, head, kl_delta)`
<!-- unbacked -->

#### Scenario: ffn-bottleneck reports per-feature impact
- **WHEN** the command runs against an Inference-level vindex
- **THEN** each record SHALL carry `(layer, feature_id, impact)`
<!-- unbacked -->

#### Scenario: bottleneck-test substitutes early layers with rule-based logic
- **WHEN** `larql dev bottleneck-test --rules RULES.json` runs
- **THEN** the test SHALL report a head-to-head accuracy comparison vs the unmodified model
<!-- unbacked -->

#### Scenario: projection-test exercises rank-k residual flow
- **WHEN** `larql dev projection-test --rank K` runs
- **THEN** the output SHALL include the rank-k reconstruction error per layer
<!-- unbacked -->

#### Scenario: ffn-overlap measures entity-vs-truth overlap
- **WHEN** the command runs on a vindex with both entity and ground-truth gate features
- **THEN** the output SHALL include precision/recall against the ground-truth feature set
<!-- unbacked -->

### Requirement: Predict, walk, and circuit-discover commands compose into research pipelines

`larql dev predict` SHALL run a full forward pass and emit the
next-token distribution; `larql dev walk` SHALL traverse the model
as a local vector index using gate KNN + down-token lookup;
`larql dev circuit-discover` SHALL identify attention → FFN
circuits from weight decomposition. These commands MUST share the
same model-resolution machinery as `larql run` (cache shorthand,
`hf://`, `owner/name`, local path) and SHALL print timing
diagnostics that distinguish load time from compute time.

#### Scenario: `predict --top-k K` emits exactly K candidates
- **WHEN** `larql dev predict --index X --prompt "..." --top-k 5` runs
- **THEN** stdout SHALL list five candidate tokens with normalised probabilities
<!-- unbacked -->

#### Scenario: `walk --predict` reports the gate-KNN trajectory
- **WHEN** `larql dev walk --index X --prompt "..." --predict` runs
- **THEN** stdout SHALL show the per-layer KNN residue and the eventual prediction
<!-- unbacked -->

#### Scenario: `circuit-discover` emits per-circuit JSON
- **WHEN** the command is run against an Inference-level vindex
- **THEN** each circuit record SHALL include the source attention head, target FFN feature, and a similarity score
<!-- unbacked -->

#### Scenario: Cache shorthand resolves the same way as `larql run`
- **WHEN** any dev subcommand is invoked with `gemma-3-4b-it-vindex`
- **THEN** the model SHALL resolve to the same vindex path that `larql run gemma-3-4b-it-vindex` would
<!-- unbacked -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_cli::commands::extraction::compile_cmd::edge::tests::**::* -->
