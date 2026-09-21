# docs/

**Class: CURRENT — index.** Start with the maintained VINDEX3 overview below.
Linked specifications and research records retain their own dates and status;
indexing a proposal does not make it an implemented capability.

| Current guide | Purpose |
|---|---|
| [What is VINDEX3?](vindex3/what-is-vindex3.md) | Model artifact, executable semantics and evidence |
| [Architecture](vindex3/architecture.md) | Ownership, plans, representations and backends |
| [Execution](vindex3/execution.md) | Build, encode, execute, record and serve |
| [Representation](vindex3/representation.md) | Compilation, candidate authority and evidence/search contracts |
| [Observation and intervention](vindex3/observation-and-intervention.md) | Carrier/head evidence, lenses and counterfactual boundaries |
| [Status](vindex3/status.md) | Capability boundaries and current research |
| [Generated facts](generated/current-facts.md) | Versions, schemas, extraction default and CLI inventories |
| [Documentation policy](documentation-policy.md) | CURRENT, NORMATIVE, RECORD and ARCHIVE authority |

Index of the top-level documentation. One line per file; specs that live
with their crate are indexed in [specs.md](specs.md). Subdirectories:
[adr/](adr/) (architecture decision records), [audits/](audits/) (review
reports), [diagnoses/](diagnoses/) (root-cause write-ups),
[ffn/](ffn/README.md) (FFN backend docs — weight, sparse, walk,
distributed).

## Formats and specs

| Doc | One line |
|---|---|
| [format.md](format.md) | LARQL graph format specification (v0.1.0) |
| [vindex3-format.md](vindex3-format.md) | VINDEX3 model-system container format — implementation guide (plan/encode/verify semantics), companion to the [3.0 Candidate Specification](../crates/larql-vindex/docs/vindex3-format-spec.md) |
| [vindex3-runtime.md](vindex3-runtime.md) | VINDEX3 runtime stack — `Vindex3Runtime`, `LogitsSession`, the KV seam, V3 serving over `/v1/completions`, `/v1/chat/completions`, `/v1/responses` |
| [observatory.md](observatory.md) | LARQL Observatory v0.1 product proposal — HAUSE instrument, local/hosted execution, coordinated views, replay and real-model acceptance |
| [vindex3-observation-contract.md](vindex3-observation-contract.md) | Draft observation contract — canonical taps, event identity, bounded capture, loss accounting, transport, privacy and receipts |
| [head-obs-1-per-head-observation.md](head-obs-1-per-head-observation.md) | HEAD-OBS-1 — pre-registered per-head observation rung: head-sum / source / parity / accounting laws, emergence and precursor witnesses on the INSTRUMENT-1 bank, no causal labels |
| [instrument-1-calibration.md](instrument-1-calibration.md) | INSTRUMENT-1 — pre-registered calibration of `vindex3 observe` readers, Anatomist DLA/lens and Observatory against the sealed EDGE-1 bank on Gemma 3 4B (12B replay); case set, readers, questions and forecasts frozen |
| [instrument-1b-lens.md](instrument-1b-lens.md) | INSTRUMENT-1b — pre-registered: the V3-LENS-1 lens pointed at the emergence write on the INSTRUMENT-1a cases (no new selection); rank/top-1 before, at and after L23 (L35 on 12B) |
| [v3-obs-1-carrier-observation.md](v3-obs-1-carrier-observation.md) | Frozen executor-side carrier observation rung for the VINDEX3 Observatory contract: `leave_site` tap, parity/reconstruction properties, witness map, and capture-cost protocol |
| [v3-head-obs-1-per-head-observation.md](v3-head-obs-1-per-head-observation.md) | Frozen V3-HEAD-OBS-1 rung: per-head distribution and pre-projection output from the one softmax kernel, per-layer coverage on the receipt, the per-head and per-source split identities, and the first HEAD-2 reading |
| [v3-intervene-1-carrier-intervention.md](v3-intervene-1-carrier-intervention.md) | Frozen V3-INTERVENE-1 rung: Zero/Add/Replace on one declared carrier address, vector provenance, the no-op law, receipt fields, and the first causal arm (HEAD-1 CARRIED) |
| [v3-intervene-2-head-intervention.md](v3-intervene-2-head-intervention.md) | Closed V3-INTERVENE-2 rung: in-kernel softmax head Zero/Scale/Replace, provenance and the additive-versus-counterfactual comparison |
| [v3-attr-1d-descriptive-support.md](v3-attr-1d-descriptive-support.md) | ATTR-1D normalization contract for descriptive head/source support |
| [vindex3-experiments.md](vindex3-experiments.md) | Pre-registered VINDEX3 experimental programme (the V2-0..V2-4 gates) |
| [vindex3-ontology-drill.md](vindex3-ontology-drill.md) | The four-architecture ontology drill (candidate §17.4) — run 2026-08-30, findings F1–F16 |
| [lyrw-v2.md](lyrw-v2.md) | LYRW v2 — the K3 routed-layer physical-layout gate (storage half of K3) |
| [specs.md](specs.md) | Pointer page: which spec lives with which crate |
| [knowledge-pipeline.md](knowledge-pipeline.md) | Stub — placeholder for the knowledge pipeline spec |

## CLI, language, bindings

| Doc | One line |
|---|---|
| [cli.md](cli.md) | Broader `larql` CLI guide; VINDEX3 command inventory is in generated facts |
| [lql-guide.md](lql-guide.md) | LQL quick-start guide |
| [larql-python.md](larql-python.md) | Python bindings for the vindex |

## Engine and runtime

| Doc | One line |
|---|---|
| [inference-engine.md](inference-engine.md) | Inference engine — compute substrate (ADR-0022 layout), attention, FFN backends |
| [ffn-graph-layer.md](ffn-graph-layer.md) | FFN graph layer — mmap walk faster than dense (517 ms vs 535 ms) |
| [ffn-cache.md](ffn-cache.md) | FFN activation cache — skip recomputation of repeated feature sets |
| [ffn/README.md](ffn/README.md) | FFN backend family — WeightFfn, SparseFfn, WalkFfn, distributed sharding |
| [kv-residency-contract.md](kv-residency-contract.md) | The KV residency contract — window vs storage vs residency, disentangled |
| [kv-attention-scaling.md](kv-attention-scaling.md) | KV attention scaling — measurement schema + run hygiene rules |
| [metal-kernel-capabilities.md](metal-kernel-capabilities.md) | Metal kernel capability table (Phase B ground truth audit) |
| [mech-interp.md](mech-interp.md) | Mechanistic-interp surface — hooks, lens, ablation, steering, patching |
| [residual-trace.md](residual-trace.md) | Residual stream trace — decomposition, storage, tiered context |
| [multi-modal.md](multi-modal.md) | Multi-modal support — Phase 0–2 shipped, phases 3–6 design-only |
| [virtual-experts-dispatch.md](virtual-experts-dispatch.md) | Virtual experts — bounded routing into typed, sandboxed WASM compute units |
| [confidence.md](confidence.md) | Confidence scoring for query results |

## Extraction and knowledge

| Doc | One line |
|---|---|
| [weight-extraction.md](weight-extraction.md) | Weight extraction pipeline — model weights → vindex, no bulk forward passes |
| [training-free-insert.md](training-free-insert.md) | Training-free knowledge insertion — residual capture + feature writes |
| [validation.md](validation.md) | Graph validation — extraction faithfulness checks |
| [circuit-types.md](circuit-types.md) | Circuit type analysis — gate/down cosine classifies feature roles |
| [findings.md](findings.md) | Research findings from querying Gemma 3 4B weight vectors |
| [walk-boundary-sweep.md](walk-boundary-sweep.md) | Walk boundary sweep — correctness across all layer boundaries |

## Programmes and funnels

| Doc | One line |
|---|---|
| [vindex-factory.md](vindex-factory.md) | Vindex Factory — recipe-driven, verified, remote-executed builds |
| [model-publishing.md](model-publishing.md) | Republishing models — the 2026-08 manual recovery and the recipes it demands |
| [k3-funnel.md](k3-funnel.md) | K3 adapter ladder — GPT-OSS-20B → Kimi Linear → K3 |
| [glm5-flash-funnel.md](glm5-flash-funnel.md) | GLM-5.3-Flash funnel — admission, census and the two tracks (321 B, KDA + DSA + mHC) |
| [dec-funnel.md](dec-funnel.md) | DEC funnel (v0.5, current) — decoupled attention/weights serving |
| [dec-funnel-v0.4.md](dec-funnel-v0.4.md) | DEC funnel v0.4.1 — superseded by dec-funnel.md |
| [dec-funnel-v0.2.md](dec-funnel-v0.2.md) | DEC funnel v0.2 — archived; control plane and gates inherited by reference |
| [tts-funnel.md](tts-funnel.md) | TTS funnel — audio-token output (MOSS-TTS-Realtime), ~1.9× realtime CPU |
| [quant-obs.md](quant-obs.md) | Quant-Obs — observer-metric ladder for quantisation error allocation |
| [fleet-routing-extensions.md](fleet-routing-extensions.md) | Fleet routing extensions FR1–FR4 — spec + frozen pre-registrations |
| [fhg.md](fhg.md) | FHG — Fourier heuristic graph programme (behavioural, model-agnostic) |
| [authority-control-plane.md](authority-control-plane.md) | Authority control plane (EXP-26..38) — layer-mechanism branch closed |
| [kimi-precision-topology.md](kimi-precision-topology.md) | Kimi Linear 48B — the PRECISION-1 topology, the first complete REPRESENT chain |
| [represent-optimizer-mcp.md](represent-optimizer-mcp.md) | REPRESENT as a queryable optimiser — map DAG, MCP surface, and the search ladder to MCTS (design) |

## Positioning

| Doc | One line |
|---|---|
| [positioning.md](positioning.md) | LARQL vs ollama, vLLM, llama.cpp — what it is and is not |

- [`v3-observatory-head-capture.md`](v3-observatory-head-capture.md) — opt-in CPU softmax head tap, real Granite recording, raw direction analysis, and parity/reconstruction gates.
