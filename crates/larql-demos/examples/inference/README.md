# Inference engine demos

Run any of these from a larql checkout:

```sh
cargo run -p larql-demos --example attention_demo
```

| demo | what it shows | run |
|---|---|---|
| `attention_demo` | The fused online-softmax attention kernel in action. | weight-free · 0.2s |
| `ave_demo` | Arithmetic Virtual Expert against a real Q4_K vindex. Writes `bench/aim-validation/`. | needs a vindex · 51s |
| `backend_demo` | Auto-calibrated hybrid CPU/Metal dispatch. | weight-free · 0.4s |
| `chat_demo` | Multi-turn conversation through `ChatSession`. | needs a vindex · 16s |
| `clustering_demo` | Clustering and relation discovery. | weight-free · 0.2s |
| `detok_demo` | Preserve word spacing across streamed tokens. | weight-free · 0.2s |
| `eos_demo` | The EOS detector halting generation correctly. | `--vindex PATH` · 10s |
| `experts_demo` | WASM expert registry — structured op+args calls across all experts. | needs the wasm build, below |
| `ffn_cache_demo` | FFN L1 cache behaviour, hit/miss stats, patch safety. | `--model ID --vindex PATH` |
| `inference_demo` | Forward pass from safetensors weights. | needs weights · 6s |
| `mech_interp_demo` | Capture, lens, neighbours, ablate, steer, patch. | weight-free · 0.1s |
| `pair_matching_demo` | Pair-based relation matching. | weight-free · 0.2s |
| `observatory_record` | Canonical VINDEX3 Standard JSON capture + separate observed/unobserved bit-parity witness. | `CONTAINER OUTPUT.json PROMPT [PROBE_TEXT ...]` · real model |
| `observatory_record --gw0-batch` | Manifest-bound GW-0 final-position carrier census with per-row parity and content-addressed before/after/delta artifacts. | `CONTAINER MANIFEST INPUT OUTPUT_DIR [START [LIMIT]]` · real model |
| `observatory_record --gw0-walk` | Frozen GW-0 exact dense V3 gate-ranking reference, cached by subject. | `CONTAINER MANIFEST INPUT OUTPUT.jsonl` · real model |
| `observatory_record --gw0b-attribute` | Reconstruct sealed candidate FFN writes as ranked `(layer, feature)` contributions; refuses failed sum checks. | `CONTAINER SEALED_MANIFEST OUTPUT_DIR [START_LAYER [LIMIT]]` · sealed artifacts + real model |
| `observatory_record --gw0b-promote` | Batch-annotate exact-WALK and optional attributed feature addresses with frozen promoted targets. | `CONTAINER SEALED_MANIFEST OUTPUT.jsonl [ATTRIBUTION_DIR]` · real model |
| `observatory_record --gw3af-postings` | Build the frozen GW-3A-F width ladder once and emit subject postings plus matched exact-WALK controls. | `CONTAINER SEALED_MANIFEST PREREGISTRATION OUTPUT.jsonl` · real model |
| `observatory_record --gwsup1-readout` | Read the frozen GW-SUP-1 candidate vocabulary through every sealed post-write carrier using the prepared output head. | `CONTAINER PREREG CANDIDATES SEALED_MANIFEST OUTPUT_DIR` · sealed artifacts + real model |
| `observatory_record --gwconv1-readout` | Read the same frozen candidates through every sealed pre-write carrier for before/after convergence localization. | `CONTAINER PREREG CANDIDATES SEALED_MANIFEST OUTPUT_DIR` · sealed artifacts + real model |
| `observatory_record --gwhead1-capture` | Re-execute the frozen cohort, capture natural L24 head values, and prove effective-W_O/post-norm reconstruction before subset search. | `CONTAINER PREREG CANDIDATES INPUT_MANIFEST OUTPUT_DIR` · frozen inputs + real model |
| `observatory_record --gwhead1-train-search` | Build leave-one-edge-out contribution-matched references and replay all 256 L24 head subsets on train only. | `CONTAINER PREREG CAPTURE_MANIFEST OUTPUT_DIR` · frozen natural capture + real model |
| `observatory_record --gwhead1-heldout` | Execute frozen global/relation necessity and sufficiency arms plus all singleton and zero controls on validation/test, including canonical downstream carrier interventions. | `CONTAINER PREREG SELECTION CAPTURE_MANIFEST OUTPUT_DIR` · frozen selection + real model |
| `observatory_record --gwkey1-capture` | Capture frozen L24H1 natural Q and every conditioned source K/V vector with bit-exact source replay. | `CONTAINER PREREG HEAD_CAPTURE_MANIFEST OUTPUT_DIR` · frozen roles + real model |
| `observatory_record --gwkey1-train-search` | Build leave-one-semantic-edge-out source K/V references and directly execute all 4,096 K/V role-mask pairs on train only. | `CONTAINER PREREG SOURCE_CAPTURE HEAD_CAPTURE OUTPUT_DIR` · frozen roles + real model |
| `observatory_record --gwkey1-heldout` | Execute the frozen K, V, and joint necessity/sufficiency arms on validation/test and transport each carrier through the canonical later model. | `CONTAINER PREREG SELECTION SOURCE_CAPTURE HEAD_CAPTURE OUTPUT_DIR` · frozen selection + real model |
| `sampling_demo` | Greedy vs temperature vs top-p on one prompt. | `--vindex PATH` · 9s |
| `streaming_demo` | Print each token as the model emits it. | needs a vindex · 5s |

6 of these need no model weights and run in well under a second, so they are the quickest way to see the surface working.

## `experts_demo` needs a build first

It loads WASM experts from `crates/larql-experts/target/wasm32-wasip1/release`,
which is not produced by a normal workspace build:

```sh
cd crates/larql-experts && cargo build --target wasm32-wasip1 --release
```

Without it the demo exits naming the directory it looked in.

---

Demos that need a vindex take `--vindex PATH`; point them at any model you have under `output/`. They fail by name if the path is missing rather than surfacing a bare `NotFound`.
