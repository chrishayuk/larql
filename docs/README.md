# docs/

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
| [vindex3-format.md](vindex3-format.md) | VINDEX3 model-system container format — the actively updated living spec (plan/encode/verify semantics) |
| [vindex3-runtime.md](vindex3-runtime.md) | VINDEX3 runtime stack — `Vindex3Runtime`, `LogitsSession`, the KV seam, V3 serving over `/v1/completions` |
| [vindex3-experiments.md](vindex3-experiments.md) | Pre-registered VINDEX3 experimental programme (the V2-0..V2-4 gates) |
| [lyrw-v2.md](lyrw-v2.md) | LYRW v2 — the K3 routed-layer physical-layout gate (storage half of K3) |
| [specs.md](specs.md) | Pointer page: which spec lives with which crate |
| [knowledge-pipeline.md](knowledge-pipeline.md) | Stub — placeholder for the knowledge pipeline spec |

## CLI, language, bindings

| Doc | One line |
|---|---|
| [cli.md](cli.md) | Full `larql` CLI reference |
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

## Positioning

| Doc | One line |
|---|---|
| [positioning.md](positioning.md) | LARQL vs ollama, vLLM, llama.cpp — what it is and is not |
