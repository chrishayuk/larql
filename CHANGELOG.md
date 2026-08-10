# Changelog

Completed work, newest first. Extracted from `ROADMAP.md` on 2026-08-10,
adopting at the root the convention the crates have used all along (10 crates
carry both a `ROADMAP.md` and a `CHANGELOG.md`).

**Where things live**

| Document | Holds |
|---|---|
| `ROADMAP.md` | sequencing and priority — what happens next |
| `CHANGELOG.md` (this file) | what happened |
| `ROADMAP_STATUS.md` | live status board — active sequence, P0/P1 boundaries, drift checks |
| `docs/*-funnel.md` | gates and evidence for one live programme |
| `docs/hardening-backlog.md` | open remediation items from code reviews |
| `AGENTS.md` | what is always true |
| `docs/adr/` | decisions |

The rule that keeps `ROADMAP.md` from regrowing: **an item leaves the roadmap
the moment it acquires a gate log.**

---

## K3 / GPT-OSS serving (2026-08)

Write-ups remain in [`docs/k3-funnel.md`](docs/k3-funnel.md); these are the
roadmap-side summaries as they were recorded.

### K3 R1 P4/P5 CLOSED — GPT-OSS served from its vindex; Metal decode 10.2 → 59.8 tok/s (2026-08-09/10)

Write-up [`docs/k3-funnel.md`](docs/k3-funnel.md) §4.11. Registry
`k3r1-gptoss-pipeline`. Branch `feat/k3-r1-p5-moe-bias-gate-policy`, commits
04addd27…f557cec9.

**Serve is real on both backends.** `larql run gpt-oss-20b-q4k.vindex`
generates coherent harmony-format output on CPU at ~60 ms/token, and `--metal`
produces the **identical 32-token greedy trajectory at 16.7 ms/token
(59.8 tok/s)** — parity re-verified after every rung. Correctness took six
stacked hidden%256 defects (MoE norm topology, lm_head widths ×2, Metal QKV/O
stored-row width, missing Metal attention biases, dense-FFN encode over empty
pure-MoE slices); the fixes are all generic — `QuantWeight::stored_cols`
(byte count is the width authority), bias threading via `build_arch_params`,
`has_dense_ffn()` as a representation fact.

**The decode ladder, each rung parity-gated:** 97.8 ms (staged expert
memcpys, ~2.1 GB/token) → 25.9 (zero-copy mmap regions:
`BufferCache::register_region` + experts as byte offsets) → 22.7 (K3a grouped
expert kernels, η 0.64→0.90 as measured) → 22.6 (fused no-QK-norm attention +
folded QKV biases — attention GPU 8→3.3 ms with the wall unmoved, isolating
the sync term) → **16.7** (GPU `moe_weighted_combine`; a layer's experts and
the next layer's attention share one command buffer — one wait/layer).
Gemma 26B A4B hybrid rode the first two rungs free (91.3 → ~30 ms/tok).

**Benchmark framing (pinned):** oMLX's ~85-90 tok/s reference is **native
MXFP4** (~4.25 bpw experts); LARQL carries the lossless Q6_K transcode at
6.56 bpw — ~1.54× the expert bytes. 59.8 × 1.54 ≈ 92 byte-normalised: same
conventional-efficiency territory; the residual is representation, not
runtime. The MXFP4-native experiment (shaders exist) now measures how far
*past* the reference the engine goes.

**Remaining budget at 16.7 ms:** ~11.5 MoE (bandwidth), ~3.3 attention GPU,
~2 sync residue (24 waits; GPU routing + device-side offset tables — which
the grouped kernels already read — take it near zero).

**Standing lesson earned:** faster kernels ≠ faster decode when
command-buffer structure dominates the wall; measure the wall against the
CB-window sum before optimising a kernel.

---

### K3 expert transport codec — CLOSED, cross-expert redundancy is nil (2026-08-10)

Registry: `dec8-13-cross-expert-conditional-census` (programme `dec`), completed
and refuted. Rule **R15** added to [`docs/dec-funnel.md`](docs/dec-funnel.md) §1.

The previous rung measured each expert's MXFP4 symbols *on their own* (3.7525 of
4.0 bits — near-optimal). It conditioned on nothing, leaving open whether the 896
experts in a layer share structure that per-expert coding discards. They do not.
`larql k3-ledger cond-census` conditions one expert's **untouched packed
nibbles** on a leave-one-out bank prototype, an out-of-sample-selected coding
parent, a random parent, and a permutation-invariant per-input-channel profile:
**mutual information 0.0000 bits on all three GLU branches**, every arm sitting
on top of its own marginal-preserving shuffled null. Agreement is 0.086–0.109
against a 1/16 chance floor, so the runnable match/escape code prices at
**4.56–4.66 bpw — worse than shipping the raw 4.25**.

**The controls are the result.** A self control reads exactly 0.0000 with
agreement 1.0000 on the same bytes; the marginal reproduces the banked 3.7525 to
1e-4; and an adjacent-row control *within one expert* also reads 3.7527 — so
there is nothing aligned to find, not merely no correspondence between experts.
Mechanism: MXFP4's per-32 e8m0 scale already absorbs per-channel magnitude,
leaving residual nibbles near-i.i.d. A format that spends its bits well leaves
no cross-object redundancy for a dictionary to collect.

**Closed with it:** entropy coding MXFP4 symbols, dictionary coding aligned
expert symbols, resident-parent lossless delta coding, adaptive symbol
alphabets, and **ETC-0B** (a specific parent adds 0.0007 bits over the
prototype; selection beats random by −0.0000, so there are no edge weights to
route around and the DEC-0 traces need not be opened for coding parents). R4 had
capped the whole idea at **1.87×** before the fetch anyway; the measured floor is
worth **1.06×** end-to-end.

**Still open, deliberately deprioritised:** a permutation-*aligned* comparison
(assignment over 3072 rows — poor prior from the adjacent-row control), and
lossy value-space decomposition, which is **approximate expert factorisation,
not compression** — approximate lane, scored on induced bits/token and route
stability, and this lossless result must not be cited for or against it.

Consequence: the routing graph and the compression graph are different objects.
Effort returns to access structure — residency, owner grouping, prefetch,
avoidance — where the route-aware hot-cache rung already measured **1.80×** from
grouping work by physical owner.

### K3 serving-format ladder + efficiency re-bank (2026-08-01)

Two rungs closed and one measurement corrected. Registry: `dec8-11`, `dec8-12`
(programme `dec`); rules R7/R8 added to [`docs/dec-funnel.md`](docs/dec-funnel.md) §1.

**The exact-format search is finished.** K3's experts are MXFP4, so a group
reconstructs at most 15 distinct values — 4 payload bits is the floor by
counting, not by search, and MXFP4 already spends exactly it. Doubled, the
alphabet is not an arithmetic progression, so the smallest affine grid
containing it needs 25 levels: **Q4_K can never be exact** (9 levels short),
Q5_K can but is dominated, and **Q6_K is the cheapest exact container that can
actually serve today**. The variable-rate loophole is closed too — measured
entropy 3.75 bits over 7.86 M real weights, and **0.0000%** of tiles hold ≤8
symbols at any block size ≥64, so palettes and escape codes are dead. Exact
floor **4.06731 bpw**. `larql k3-ledger formats` / `symbol-census`.

**MXFP4's low kernel efficiency is the container, not a defect.** Seven crossed
arms at the real expert shape decomposed the winner into skeleton 76% / fp4
decode 22% / input gather 2%, with the skeleton already streaming at 0.95 of
attainable bandwidth. Four decoders tried; the ordering is monotone in table
size and a table-free bit-manipulation decoder is worst by 37%. **The
expert-side kernel line closes at single-token width.**

**Numbers, and they moved down twice — both times because a measurement got
honest, never because anything got slower:**

| claim | status |
|---|---|
| **3.02 tok/s** | controlled healthy-regime exact-Q6_K composed ceiling |
| **2.79–3.18** | observed, composed **paired per run** over 7 accepted runs |
| **3.65 tok/s** | + grouped routed experts — clean measurement, integration still required |
| **4.15 tok/s** | + routed MXFP4 — a **kernel projection**, maturity `Grouped`, below `is_servable()` |
| **5.49 tok/s** | density-only **upper bound**; reuses Q6_K efficiencies at MXFP4 density, which R7 forbids |
| *unmeasured* | **sustained** laptop throughput under the degradation regime below |

**Two harness bugs, both silent, both now guarded.** `BufferCache::get_bytes`
keys on `(pointer, length)`, so same-length *temporaries* aliased and returned
each other's buffers — which meant the cold-rotation loop feeding every
efficiency figure was handing back **one buffer eight times**. The composed
ledger survived it (3.70 → 3.68), because the dominant term is also the
steadiest. And a 16-run promotion campaign found that **more repeats make it
worse**: 9 runs were unusable as the machine degraded under sustained load and
the attention control fell 0.89 → 0.06. Runs are a time series, not
exchangeable draws.

1. **Run the sustained end-to-end decode, and name the degradation.** The nine
   rejected runs are a second scoreboard nobody has measured: report throughput
   by time window (startup / healthy / late / steady-state floor) over 20–30
   minutes with system telemetry. Thermal, power management, memory pressure
   and paging are all still live candidates. **The demo number is this one, not
   3.02** — and it may be lower.
   [larql-compute-metal]

2. **Promote `gate/up` and the ungrouped expert shape across independent
   cool-start sessions.** Both sit at 2.2–2.4% relative standard error against a
   1% bar, and both feed DEC-8.7b's target row — which is the only live
   throughput rung now that kernel efficiency is closed as a lever. The R4 lever
   ordering *refuses to print* until they clear. Not another same-session
   campaign; that reproduces the artifact. Check the histogram for bimodality
   before banking a mean.
   [larql-cli]

3. **Finish the grouped-down integration A/B on a loaded model.** DEC-8.9's
   kernel risk is retired and its `next_action` carries the six-step order;
   this is what converts 3.02 → 3.65 from projection into result, and it is the
   nearest end-to-end milestone.
   [larql-compute-metal, larql-inference]

4. **Resolve `A_log` before any KDA numerics.** K3's checkpoint ships `[128]`
   where the reference module allocates `num_heads` = `[96]`, and two readings
   of the geometry each explain the large tensors while breaking one small one
   — **shapes cannot decide it**. `kda_a_log` fails closed and ships a
   deliberately rectangular discriminating fixture, because the two readings
   coincide on the diagonal and a square fixture would pass vacuously.
   [larql-cli, larql-models]

5. **Build a sentinel with a working set ≥ the largest class it gates.**
   Attention is currently both a banked class and the control, so its 0.876 is
   self-selected and biased upward. The obvious cheap fix is *worse*: a 21 MB
   sentinel admitted two runs where the 72 MB attention cell had already
   collapsed. Degradation is size-dependent; the K2 weights-only probe (89.5 MB)
   is the candidate.
   [larql-compute-metal]

6. **`prefill_q4_seq4_synthetic_smoke` is flaky at ~3–5%, all-NaN output.**
   Found by the new commit gate, which runs `--all-targets` rather than the
   `--lib` subset. Failure mode is the *entire* prefill output NaN, not a
   drifted value. **Bisect did not resolve it and n=16 per commit was
   underpowered**: pooled 3 failures in 88 runs, with the failures landing on
   two non-adjacent commits and 0/16 on the commits between them — at an 8%
   true rate, `P(0 in 16) = 0.26`, so a clean 16 proves nothing and ~36 runs
   per candidate are needed. Not attributable to any one change on the
   evidence available. Same family as the threadgroup-scratch reuse race fixed
   earlier in fused attention, so treat it as a real race rather than noise;
   localising it wants a proper campaign, not another bisect.
   [larql-compute-metal]

7. **Attention E/F ceiling probes — parked, bar pre-registered.** R7 means
   attention's 0.87–0.89 may describe its container rather than a fixable
   kernel. Same harness, needs Q6_K variants. **If the skeleton returns ≥ 0.93
   the class is closed and no decoder work is licensed.** Run it when preparing
   dense-format work or DEC-8.7b, not before the integration above.
   [larql-compute-metal]

---

### K3 R1 Gate B — forward parity closed on CPU, open on Metal (2026-08-04/05)

Write-up [`docs/k3-funnel.md`](docs/k3-funnel.md) §4.8–4.10. Registry
`k3r1-gptoss-pipeline` (programme `k3`). **P2 is closed on the CPU f32 path for
both R1-class models; the remaining work is Metal and the P3–P6 phases.**

**Closed.** GB's missing half — the layer-by-layer diff — is built
(`larql shannon layer-dump` / `layer-diff` + `scripts/dump_layers_hf.py`) and
immediately closed two models. OLMoE: `rms_norm_eps` class default 1e-5 with
the field absent from the checkpoint, and a QK-norm applied over the whole
projection rather than per head (cos 0.890 → 0.991 → **1.000000000**; bits/char
0.435 vs the reference's 0.4348). GPT-OSS: `rope_type: "yarn"` parsed and then
ignored because the only scaling hook was an `Option<Llama3RopeScaling>`, a type
that could not express it — 23 of 32 rotary dims at the wrong frequency and
every cos/sin 34.7 % small (cos 0.9777 → **1.000000000** at layer 0). Sliding-
window attention, absent from the dense path entirely, now exists as one
`AttentionSpan` shared by prefill and decode; verified at 511 tokens (4× the
window) with layer 0 — a sliding layer — at cos 1.000000000. The leftover
residual is measured, not assumed: a **four-token tie-break cascade** carrying
98.17 % of the final squared residual, seeded by one exact tie.

| # | Item | Crate | Status |
|---|---|---|---|
| M1 | **Metal decode ignored every RoPE scaling family.** Prefill roped on the host and honoured llama3 / YaRN / Gemma 3's linear divisor; decode roped in-shader from `rope_base` alone and honoured none — live on `gemma-3-4b/12b-it` and `Llama-3.2-1B`. **FIXED**: `RopeFreqPlan` computed once by the same `rope_freq_plan` the CPU uses, bound as a buffer + amplitude. | larql-compute-metal | **done** |
| M2 | **All four rope-bearing shaders converted atomically** — `rope` (4 kernels), `qk_norm_rope_fused`, `attn_fused`, `fused_attention`; eight rotation sites, zero `pow(rope_base, …)` left. `stages::rope_freq` owns the binding and checks the table width against the layer's geometry. **551 Metal tests green; the suite caught 16 binding mistakes**, each surfacing as `cos = 0.0` rather than a compile error, since Metal bindings are untyped. | larql-compute-metal | **done** |
| M3 | **A decode-pass diff exists. DONE (2026-08-06), re-homed 2026-08-07 as `larql shannon decode-diff`.** The example it originally landed in was deleted by main's examples reorganisation, so the pass now lives in the CLI beside `layer-dump`/`layer-diff`, driving `residual_diff::ResidualCapture` rather than reimplementing it. It is a *different axis* from `layer-diff` and the doc says so: `layer-diff` compares this engine to an external HF reference over a prefill, which by construction cannot see a decode-only defect. Verified on `gemma-3-4b-it`, 34/34 layers, `--steps 2`. **Original entry:** The example now runs a fourth section: Metal `prefill(N-1) + decode_token(N)` against CPU `prefill(N)` projected to its last row, per layer, reusing `residual_diff::ResidualCapture` rather than re-spelling the dump plumbing. Note the finding along the way: the *library* already had `metal_decode` / `metal_decode_steps` and `tests/test_decode_consistency.rs` already compared them against a CPU reference — the gap was only in the interactive tool, so "the decode diff does not exist" was too strong. | larql-inference | **done** |
| M4 | **Metal prefill now honours the sliding window. DONE (2026-08-07).** The defect, measured before fixing: Metal *decode* windowed correctly but Metal *prefill* took no window at all — `stages::attention::encode` had no such argument — while CPU prefill windowed via `effective_attention_window_for_layer`. So every sliding layer attended the whole prefix on GPU (Gemma 3: 29 of 34, window 1024; GPT-OSS: 12). **The M1 asymmetry inverted.** **Fix:** `fused_attention` gains `window_size` at buffer 17 and a `k_start` that mirrors the CPU rule (`causal_len.saturating_sub(w)`) exactly; the score, softmax and V-weighted loops all start there, and the two threadgroup reductions now count `active_len` rather than `causal_len`. The per-layer window was already resolved and already on `FullPipelineLayer` — `build_pipeline_layers` computes it through the shared rule with `0` as the no-window sentinel — so global layers arrive as 0 and stay unwindowed. **Evidence:** `tests/test_prefill_sliding_window.rs` compares Metal prefill against the production CPU `gqa_attention_windowed` (not a hand-rolled reference — the claim is that the two *backends* agree). `seq_len=48, window=8` puts 40 of 48 queries outside the window, and a fixture-adequacy test asserts windowed and unwindowed CPU actually differ on it, so the suite cannot pass vacuously. Verified discriminating: with `k_start` forced to 0 exactly one test fails and the no-window control still passes. Full Metal suite green; real-model decode-consistency green on gemma3-4b/llama2-7b/mistral-7b. **Left open:** a model-level long-prompt parity fixture. The existing suites still prompt with ~16 tokens against a 1024 window, so they remain blind to this class — the kernel test is what guards it today. | larql-compute-metal | **done** |
| M5 | **Prove the M1 fix end to end on `gemma-3-4b-it`. DONE (2026-08-06), and it took a detour.** First run passed at cos 1.000000 across all 34 layers — but the vindex the test loads has **no `rope_scaling` at all**, so `rope_position_divisor_for_layer` returned 1.0 on every layer and the 8× divisor M5 names was never exercised. That is a gate–claim congruence failure, not a result. Re-run with `LARQL_ROPE_POS_DIVISOR_GLOBAL=8`, which drives the same `effective_rope_position_divisor_for_layer` → `rope_freq_plan` both backends read: **34/34 layers at cos 1.000000, 1 and 2 decode steps**. Knob verified to bite, not silently no-op: outputs are identical for L00–L04 and diverge from **L05 — the first global layer** — with final ‖h‖ 21424.07 (divisor 1) vs 21878.96 (divisor 8). | — | **done** |
| M6 | **Every Metal Q6_K and Q4_0 kernel decoded a private nibble layout. FIXED (2026-08-07).** PR #207 moved the CPU side of both formats onto ggml's planar layout and changed **no file** under `larql-compute-metal`, so the two halves silently disagreed and `main` went red. Q6_K planar packs a super-block as two 128-element halves where one `l` column yields four elements at *stride 32* from three bytes (`ql[64h+l]`, `ql[64h+l+32]`, `qh[32h+l]`); Q4_0 packs byte `j` as elements `j` and `j+16`. The shaders read the pre-ggml `ql[i/2]`/`qh[i/4]` and `2j`/`2j+1` forms. **Six Q6_K kernels** (`q6k_matvec`, `q6k_matvec_8sg`, `q6k_grouped_experts`, `q4k_q6k_qkv_proj` ×2 kernels, `q6k_geglu_down` ×2, `q6k_geglu_gelu_tanh_down_cached`) and **four Q4_0 kernels** (`q4_matvec_v4`, `q4_f32_matvec`, `q4_vecmat`, `q4_sparse_matvec`) converted. This is a **served-model** defect, not just a test one: #207 also moved `larql-models`' GGUF readers, so Q4_0/Q6_K weights loaded from disk decoded wrong on GPU. `q6k_grouped_experts` is K3's expert-dispatch kernel. | larql-compute-metal | **done** |
| M7 | **Only one test in the tree could see M6, and the others were blind by construction.** `stage_quant_matvec_routes_format_to_correct_shader` caught it because it compares against a **true f32 gemv**. The rest did not: `q6k_matvec_8sg_matches_4sg_bit_equal` compares two shaders *to each other*, so a shared defect keeps it green; and the five `q6k_geglu_down` parity tests did compare against the planar CPU backend but their fixture was `cos(seed + 0.001·i) + 0.3·sin(i >> 8)` — a super-block spanned 0.26 rad of a smooth curve, and the second term was **constant across a whole super-block so it survived any permutation exactly**. A layout error permutes elements; a fixture too smooth to notice a permutation is an *absent* test, the same class as §4.9.1's 85-token window and §4.7.3's `out_features = 2`. **Fixed:** the generator is now hash-decorrelated, `q6k_matvec_both_geometries_match_cpu_reference` anchors both TG geometries to `CpuBackend`, and `fixture_can_distinguish_planar_from_interleaved_layout` asserts the *counterfactual* — that decoding this fixture the old way breaches the very threshold the parity tests enforce — so the property cannot silently regress. All verified discriminating by reverting each shader and confirming the tests fail. | larql-compute-metal | **done** |
| M8 | **The x86 Q4_0 reference test carried the same stale layout.** `tests/test_q4_x86_correctness.rs`'s `dequantize_q4_0_row` still wrote `2j`/`2j+1` while #207 moved `csrc/q4_dot.c` to planar. It is `heavy_tests`-gated so it never ran in the failing CI job, and it is x86-only so an aarch64 box does not reach it by default. Fixed and verified discriminating: 2 of its 3 tests fail against the old reference. | larql-compute | **done** |

**Phases.** R1/P1 (audit) and P2 (adapter) are closed. **P3 harvest, P4 extract,
P5 serve, P6 shrink are not started** — but P5's named blocker is gone.

**P5 expert-store blocker CLEARED (2026-08-07).** The diagnosis in §4.7.10 was
half the story: GPT-OSS fell through *both* writers, not one.
`write_per_layer_moe_kquant` requires `PackedBF16` and
`write_per_layer_moe_per_expert` required `PerExpert`, so `PackedMxfp4` matched
neither and no expert store was written at all — extraction reporting success,
checksums verifying, and the model unservable. That is the **third** appearance
of the silent-0-byte expert store this file's lineage has documented, and each
time the cause was a gate testing an *enum value* rather than the *capability*
the writer needs. The gate is now `arch.is_moe() && arch.expert_ffn_gate_key(0,
0).is_some()` — "does this arch expose per-expert tensors", which is exactly
what the writer consumes. Packed models still decline correctly, because the
trait default for that key is `None` and Gemma 4 does not override it.

**Format:** MXFP4 experts transcode to **Q6_K, not Q4_K.** An MXFP4 group
reconstructs at most 15 distinct values and Q6_K represents every one exactly,
so the transcode is lossless — the K3 kernel-ladder result, applied. Q4_K would
re-quantise an already-quantised tensor and discard the checkpoint's own values
for no benefit the serving path can use.

**Still to verify:** this is pinned by unit tests on a synthetic GPT-OSS-shaped
source (both verified to fail against the old gate), *not* by a real extraction.
`openai/gpt-oss-20b` is present locally; running P4 extract against it end to end
and then serving it is the next step, and until that is done "GPT-OSS is
servable" remains a claim about the writer, not about the model. Item 14 (routed
`FfnBackend`) is still the other half.

**Standing rules earned here** (see [`AGENTS.md`](AGENTS.md)): diff the forward
before theorising about it; a fixture too small to distinguish the candidate
behaviours is an *absent* test, not a weak one; a config fact belongs in the
trait default, not in one architecture; and a threshold chosen without
calibrating it against the quantity it bounds is a guess wearing a number.

---

---

## BitNet b1.58 (2026-06-20)

### Completed — hardening pass (2026-06-20)

The quick-win review items landed; all touched crates build clean and
clippy-clean (`--all-targets`), tests green (compute ternary 19/19,
inference ternary 28/28 incl. the FFN A8-vs-f32 parity gate, models detect
59/59):

1. ✅ **[larql-models] Killed the silent `GenericArch` fallback** — explicit
   `bitnet-*` recognition → thin named `BitnetArch`; `norm_eps` honoured;
   `test_detect_bitnet_is_explicit_not_generic`. *(was P1)*
2. ✅ **[larql-compute] Reconciled the `ternary_matvec.rs` docstring** — no
   longer implies the path routes through `FormatRoute`; states that dispatch
   integration is the open item and the kernel is reached by direct call.
3. ✅ **[larql-inference] Reuse one activation quant** — Q/K/V and gate/up
   quantise the shared activation once (`quantize_activation_i8` +
   `matvec_i2s_a8_into`) across all five forward sites. Bit-exact (parity
   tests unchanged), saves the repeat int8 quantise per projection.
4. ✅ **[larql-inference] Refreshed the `ternary.rs` header comment** — the
   "fold in once the quant-activation kernel exists" precondition is now met;
   the comment frames the fold as live roadmap work, not a missing dependency.
5. ✅ **[larql-compute] x86_64 gap documented** — verified already clear at
   the dispatch entry (`matvec_i2s_a8_into`: "scalar int8 elsewhere — AVX2
   twin is the x86_64 follow-up") and the status block.

Owed back to the user (not a code change):

6. **[git hygiene] Split the `pipeline_layer.rs` refactor** — the
   `attn_str_to_format`/`ffn_str_to_format` → `from_registry_tag` dedup is a
   sound single-source-of-truth cleanup but is **orthogonal to BitNet**
   (BitNet never flows through `resolve_ffn_weights`). Land it as its own
   "refactor: dedupe tag→format mapping" commit, not inside the feature.

---

## Shipped through 2026-05-16

Recorded as "Current state (2026-05-16)" in the roadmap; preserved verbatim.


- **~960 tests passing** across the workspace (server 292 lib + 447 integration = 739, router 169 lib + 50 integration = 220 with `--features http3`), 0 build errors.
- **Primary CLI verbs** in place: `run`, `chat`, `pull`, `list`, `show`, `rm`, `link`, `serve`, `bench`.
- **Gemma 3 4B Metal**: **88 tok/s** (Ollama steady: ~103). **Gap: 1.17×** (was 1.18× pre QKV defuse, 1.30× pre 2026-05-02 dispatch-geometry fix). **Acceptance criterion (~85 tok/s, 1.16×) met.**
- **Gemma 4 26B A4B Metal**: **19.4 tok/s** (was 5.1 — bug-locked under the same dispatch-geometry mismatch; correct multilingual output now).
- **Cross-arch coverage validated** (2026-05-09): Gemma 3, Gemma 4 31B dense, Llama 2 7B, Mistral 7B all dispatch correctly through Metal. Gemma 4 E2B falls back to CPU (deliberate — Metal doesn't yet implement Per-Layer Embeddings; diagnosed and tracked as D-METAL-PLE).
- **Grid (CPU MoE on remote shards)**: 18.3 tok/s 1-shard / 17.3 tok/s 2-shard local-loopback. Multi-host LAN/cross-region scaling unblocked.
- **Remote FFN (dense)**: `larql run --ffn URL` + `larql serve --ffn-only` wired end-to-end.
- **gRPC grid**: 2-shard self-assembling grid live-validated on 26B A4B.
- **4 KV-cache engines**: MarkovRS (287×), WindowedCheckpoint (254×), TurboQuant (4×), Apollo (20,000×) — all at ~95 tok/s on Gemma 3 4B Metal.
- **Wire format negotiation** (2026-05-07): f16 is now the default for all grid traffic (50% bandwidth reduction). i8 symmetric quantised residuals available opt-in (`LARQL_I8_WIRE=1`, 75% reduction). Content-type negotiation via `Accept` header; f32 fallback for non-grid clients.
- **Per-layer latency routing** (2026-05-07): `HeartbeatMsg.layer_stats` carries EMA avg_ms + p99_ms per layer; router routes to the server with lowest per-layer latency (falls back to requests_in_flight when no data yet).
- **WebSocket token streaming** (2026-05-07): `WS /v1/stream` now supports `{"type":"generate","prompt":"...","max_tokens":N}` command with per-token frames and cancel support. SSE streaming on `/v1/chat/completions` was already fully wired.
- **Criterion benchmarks** (2026-05-07): `make bench-wire` (wire codec encode/decode MB/s) and `make bench-routing` (route/heartbeat/rebuild ns/op). `larql-router` now has a library crate (`larql_router::grid`) for test/bench use.
- **Dynamic rebalancing** (2026-05-08): `rebalancer.rs` background task with configurable threshold (--rebalance-interval, --rebalance-threshold). Router detects sustained per-layer latency imbalance and sends `UnassignMsg` to the slow shard; server drains in-flight requests (up to 30s), sends `DroppingMsg`, and re-enters available pool. Real `requests_in_flight` counter wired into heartbeats via `RifGuard` in walk_ffn handler.
- **CI regression gate** (2026-05-08): `scripts/bench-grid-regress.sh` + `scripts/bench_compare.py` + `bench/baselines/`. First run auto-saves baseline; subsequent runs fail if tok/s drops >5% or p99 rises >10%.
- **Shannon arc closed** (2026-05-08): Exps 42–44 prove cross-entropy is a real wire format (Exp 42: 2.0 bits/char vs 6.3 gzip), residual stream is compressible (Exp 43: int8-clip3σ, 98.7% top-1, KL=2.0 nats), gate calibrated at threshold=2.16 (Exp 44: accept=68.9%, early-div=4.8%).
- **`larql-boundary` crate shipped** (2026-05-08): Phases 1–3 of BOUNDARY_REF_PROTOCOL. int8-clip3σ + bf16 codec, per-boundary confidence metadata, calibrated confidence gate. 100% function coverage, CI on Linux/Windows/macOS, 3 examples (encode_decode, gate_decision, accuracy). Phase 4 (server integration) not started.
- **QKV defuse + cleanup pass** (2026-05-09): default flipped from fused `q4k_q6k_qkv_proj_normed` to separate `rms_norm` + non-fused `q4k_q6k_qkv_proj` (+1.6–1.8 tok/s on Gemma 3 4B, +0.4 tok/s on Gemma 4 26B A4B post-thermal-cooldown cross-arch validation, ADR-016). Cross-arch bench captured for 4 model families. Shader inventory survey (47 shaders) + retention rationale doc-blocks added to opt-in shaders. New ADRs: [017 — shader retention under model agnosticity](crates/larql-compute/docs/adr/017-shader-retention-model-agnosticity.md), [018 — architecture → shader routing](crates/larql-compute/docs/adr/018-architecture-shader-routing.md). New docs: [shader-inventory](crates/larql-compute/docs/shader-inventory.md), [architecture-shader-map](crates/larql-compute/docs/architecture-shader-map.md), [llama-cpp-comparison](crates/larql-compute/docs/llama-cpp-comparison.md). One verifiable orphan deleted (`q4k_qkv_proj_v2`).
- **`make bench-cross-arch` shipped** (2026-05-09): runs `larql bench` across the model matrix (Gemma 3 4B, Gemma 4 31B dense, Gemma 4 26B A4B MoE, Llama 2 7B, Mistral 7B). `--save-baseline` / `--compare` modes; `bench/baselines/cross-arch/`. Operationalises ADR-017 model-agnosticity check; multi-arch sweep surfaces thermal artifacts as "every arch regresses simultaneously." Run on a cool machine before saving baselines.
- **D-RMS-FUSE Phase 1 implemented + falsified end-to-end** (2026-05-09): fused post-FFN `residual_add` + next-layer input rms_norm via `residual_norm_store` for the non-Gemma path. Bit-identical parity across Llama 2 7B, Mistral 7B, Gemma 3 4B (Gemma untouched — already triple-fused). End-to-end null vs drift on Llama 2 / Mistral. Kept opt-in `LARQL_FUSED_PRELAYER_NORM=1` per ADR-017 retention. Predicted ~0.2 ms/tok savings collapsed to zero — ADR-015 magnitude-compression at the extreme. Lesson: dispatch-overhead estimates (~7 µs/dispatch) over-predict savings when the kernel being skipped is also short.
- **Gemma 4 E2B 30× anomaly diagnosed** (2026-05-09): root cause = Per-Layer Embeddings (PLE) not implemented in Metal; `gpu.rs:372-374` deliberately routes E2B to CPU. Tracked as **D-METAL-PLE** (1-2 day Metal port of `forward/ple.rs`, 80-150× expected speedup for E2B; unlocks future PLE-using arches like Gemma 4 E4B).
- **larql-compute coverage audit + improvement** (2026-05-09): `cargo llvm-cov` reports **56.03% → 64.81% line coverage** (+8.78 pp; 2,575 newly-covered lines, 22.2% reduction in uncovered LoC). Three rounds: (1) deleted `metal/prefill.rs` (591 LoC of `#[allow(dead_code)]` orphan); (2) targeted tests on small helpers — `tg_width` math (qk_norm 0% → 23%), `scale_vector` dispatch (layer_scalar 12% → 97%), `residual_norm_store` shader parity for D-RMS-FUSE; (3) synthetic end-to-end Metal decode tests (`tests/test_metal_decode_synthetic.rs`, NEW) covering Llama-style + Gemma-3-style + D-RMS-FUSE off-vs-on parity, which lifted `decode/mod.rs` 7% → 61%, `encode_attn` 0% → 46%, `encode_post_ffn` 0% → 83%, `encode_qkv` 0% → 30%, `encode_ffn` 0% → 23%. Coverage policy (`coverage-policy.json`) targets 90% per-file / 93.5% total — current is below but no longer a wide gulf. Largest remaining gaps: `metal/trait_impl/decode.rs` (627 LoC at 21% — MoE / split-profile trait methods), `metal/decode/encode_ffn.rs` (1008 LoC at 23% — Q4_KF / MoE branches), `metal/diag/*.rs` (~3000 LoC at 0% — diagnostic / dev-only).
- **Positioning vs ollama / vLLM / llama.cpp documented** (2026-05-09): [docs/positioning.md](docs/positioning.md). Three-category framing (local single-user / batched serving / research+edit); feature matrix; per-competitor gap analysis; surfaces missing items now tracked under P2 § "Competitive parity" below.
- **Google released Gemma 4 MTP drafters** (2026-05-05, 4 days ago): `google/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant` — every Gemma 4 variant LARQL supports. 0.4B BF16 ~4-layer drafter for the 26B-A4B target. Architecture: shared input embeddings + shared KV cache + target last-layer activations concatenated with token embeddings then down-projected to drafter dimension. Measured **2.2× decode speedup on Apple Silicon at speculative batch 4–8** (Google blog), up to 3× generally. Apache 2.0 / CC-BY-4.0. Supported engines: HF Transformers, MLX, vLLM, SGLang, **Ollama**, LiteRT-LM (notably not llama.cpp). Competitive implication: the LARQL gap on Gemma 4 widens from 1.17× to ~2.6× as users adopt MTP on Ollama. Red Hat AI also released an EAGLE-3 speculator for `gemma-4-26B-A4B-it` (0.9B drafter). MTP1 promoted from P2 to **P1** — see new section below.
- **ADR-019 resolved** (2026-05-09): substrate-primary is **Gemma 4 31B dense + vindex**; MoE coverage retained at single-machine scale (Gemma 4 26B-A4B for cross-arch validation, virtual-expert work). Multi-machine MoE grid (C9 productionisation, critical-path items 5–10) demoted from P0 to P2 — substantial production-engineering work with no current experiment requiring "model spans 4 consumer machines" beyond what single-machine sharding already demonstrates. C1 (CPU MoE forward pass) stays P0 because V1/V2 cross-arch sweep on 26B-A4B requires it. See full resolution in "ADR-019" section below.
- **Engine ↔ Backend unification PR shippable** (2026-05-16): three specs landed in `crates/larql-inference/docs/specs/` — (1) [`kv-engine-unification.md`](crates/larql-inference/docs/specs/kv-engine-unification.md) (Steps 1-7 implemented, all parity tests green); (2) [`compute-backend-redesign.md`](crates/larql-inference/docs/specs/compute-backend-redesign.md) (Steps 1-4 implemented — `KvDispatch` sibling trait in larql-inference, `EngineBackend` umbrella, `CpuBackend`/`MetalBackend` scaffolding, `StandardEngine` migrated to dispatch through trait); (3) [`async-compute-backend.md`](crates/larql-inference/docs/specs/async-compute-backend.md) (trait surface locked, 6 open questions resolved; A1 trait + handles, A2 `CpuBackend`, A3 `MetalBackend` scaffold, and A5 `StandardEngine` opt-in landed 2026-05-16 — A3's Metal-feature validation gate is blocked on a parallel `larql-compute-metal` extraction). Honest finding from Step 5 discovery: per-layer Metal kernels at the sync trait's granularity are *slower* than today's fused decode path because each per-layer call forces a separate GPU command-buffer commit — `AsyncComputeBackend` (intent-collector pattern, deferred dispatch) is the prerequisite for any tok/s win. That work is 6-12 months end-to-end (see new "P0 — Engine ↔ Backend unification" section below). The unification PR ships the foundation; tok/s wins land in A4 (real Metal deferred dispatch) and the multi-step Metal kernel work that compounds on top.
- **Cross-engine forward-pass correctness gate** (2026-05-16): `larql shannon verify` orchestrates LARQL Rust forward against HF/PyTorch + MLX reference scorers (subprocesses) on a shared corpus and prints a bits/char delta table. First serious application surfaced **four config-loading bugs in larql-models** — all closed in the loader (no env-var workarounds in production): (1) `rms_norm_eps` from config.json was never read by the trait default; (2) Gemma 3's per-layer-type `rope_scaling` structured form (`{full_attention: {rope_type: linear, factor: 8}, sliding_attention: {rope_type: default}}`) wasn't honoured; (3) `rope_scaling = llama3` (wavelength-dependent per-channel `inv_freq` adjustment) wasn't implemented; (4) `norm_epsilon` alias (StarCoder2's name for `rms_norm_eps`) wasn't recognised. Post-fix, all four affected models match HF F32 to <0.06% bits/char with zero env vars. `scripts/diagnose_models.py` (multi-arch sweep) reports 7/9 PASS. CI gate at `.github/workflows/shannon-verify.yml` runs SmolLM2-135M verify on every PR. Diagnostic doc: [`docs/diagnoses/shannon-cross-engine-divergence.md`](docs/diagnoses/shannon-cross-engine-divergence.md). Plus GPT-2 legacy config-key aliases (`n_embd`/`n_layer`/`n_head`/`n_inner`) parsed via new alias-list machinery in `detect/config_io.rs`.
- **larql-compute-metal coverage push closed** (2026-05-16): post-ADR-019 split, the Metal backend now lives in its own crate with **97.28% line coverage, 59/59 files at the 90% per-file floor, zero debt baselines**. Up from 75.69% (50/59 files clearing 90%, 9 debt baselines) at session start. Key techniques: (1) `MetalBackend::with_options` to bypass the env-snapshot caching that silently no-op'd flag-toggling tests on `decode_one_token_with_env`, opening the `fused_attn` / `fused_qk_norm_rope` / `fused_kv_append_attend` / `fused_post_attn_norm` branches in `decode/encode_attn.rs` (68.78% → 99.53%); (2) per-format prefill split-phase tests (Q4_K / Q4_KF / Q4_0 × gated / non-gated, `LARQL_PROFILE_SPLIT=1`) for `decode/encode_ffn.rs` (61.43% → 92.86%); (3) direct calls to the public `run_experts_prestaged_metal` / `run_experts_preselected_metal` / `run_dense_ffn_q4k` paths plus a real-MoE-layer `decode_token_q4k_moe` end-to-end test for `moe_dispatch.rs` (38.91% → 95.25%); (4) `decode_attention_layer` integration tests covering V-norm, post-norms, and `wo.format` Q4_KF/Q6_K branches for `decode_hybrid.rs` (0% baseline → 94.41%); (5) dead-code deletion of `MetalBackend::full_pipeline` (108 lines, no callers, doc said "old benchmark entry point") to clear `pipeline.rs` to 100%; (6) `Config::from_args` + JSON helper + Smoke-profile end-to-end coverage for `diag/shader_bench.rs` (4.25% → 99.36%) and `diag/kernel_profile.rs` (0% → 97.12%) — the diag scripts now smoke-run real GPU dispatches in unit tests; (7) a dedicated `tests/test_decode_diag.rs` integration binary (fresh process, fresh `CALL_COUNT`) that hits the previously-believed-structural cap on `decode/diag.rs` (85.23% → 93.75%). Coverage-policy file now an empty-baseline gate: any regression on any file breaks CI.
- **larql-router self-healing + HTTP/3 + hedged-dispatch phase** (2026-05-16): MoE expert routing (ADR-0018, per-(layer, expert-range) replication keys), Prometheus `/metrics` (ADR-0017), Phase 4 HTTP/3 shard transport behind `--http3-shards` / `--http3-port` (ADR-0019, h3 0.0.8 + h3-quinn 0.0.10 + h3-axum 0.2), hot-shard hysteresis (ADR-0014 amendment, `--hot-shard-demote-ratio` default 0.8), backpressure tier (ADR-0020 — `--saturation-ceiling N` filter in `route()` / `route_expert()`, dispatcher distinguishes 503 saturation from 400 no-owner via `has_owners_for()`, emits `Retry-After: 0.5`, bumps `larql_router_route_saturation_total`), long-running chaos test (`tests/test_grid_chaos.rs`, 5,000 random ticks × 2 variants, asserts ledger consistency + coverage floor + no `route()` panic), hedged dispatch (ADR-0021 — opt-in via `--hedge-after-ms M`, new `route_with_rank` / `route_expert_with_rank` grid APIs, `hedged_post_json` racing helper, dense + MoE fan-outs wired, `route_hedge_fires_total` / `route_hedge_wins_total` counters; supersedes the original "speculative next-layer prefetch" P1 framing — an audit falsified that framing since the router sees one batched call per token against a single input residual, so hedge-the-slow-primary is the legitimate router-layer optimisation). Concurrent-route bench (`bench_route_concurrent`, 2026-05-16) surfaced lock-contention plateau: pre-swap 1 = 5.6 → 4 = 8.7 → 8 = **4.0** → 16 = 3.6 Melem/s (8 workers *worse* than 1 — pathological). **Lock primitive swap** (2026-05-16): `tokio::sync::RwLock<GridState>` → `parking_lot::RwLock<GridState>` across larql-router and tests. Every grid critical section is short and sync (no `await` held under the lock), so synchronous is semantically correct and the compiler enforces it (parking_lot guards are `!Send`). Post-swap: 1 = 6.4 / 4 = 11.1 / 8 = 7.2 / 16 = 6.1 Melem/s — **+14% / +28% / +80% / +70%**, pathological 8-worker collapse eliminated. 220 tests still pass. Saturation-filter cost on the happy path: ~108 ns vs ~113 ns baseline (in noise); all-saturated short-circuit ~57 ns. Router test surface: 169 lib + 50 integration = **219 tests** (220 with `--features http3`). Coverage **~93%**. Five examples (`embed_grid`, `static_shards_server`, `admin_client`, `fanout_dispatch`, `saturation_backpressure`); criterion benches cover dense + MoE + saturation + concurrent-route. Multi-host deployment runbook at [`crates/larql-router/docs/multi-host-demo.md`](crates/larql-router/docs/multi-host-demo.md). Server-side `GET /v1/shard/{model}/{start}-{end}` audited + documented in [`crates/larql-server/docs/router-spec.md`](crates/larql-server/docs/router-spec.md) §4. ADRs: [0017](docs/adr/0017-router-metrics.md), [0018](docs/adr/0018-moe-expert-routing.md), [0019](docs/adr/0019-http3-shard-transport.md), [0020](docs/adr/0020-route-backpressure-tier.md), [0021](docs/adr/0021-hedged-dispatch.md).
- **Whole-codebase review** (2026-05-28): multi-agent deep review (17 crates, ~415K LOC; per-crate reader + adversarial verification). Clippy clean (2 trivial nits); exposure concentrated and thematic. ~7 verified high/medium items now tracked under "Codebase hardening (review 2026-05-28)" below and mirrored into crate-local roadmaps. Top two confirmed by hand: infallible `FfnBackend::forward` aborts serving on remote-shard blips; Metal KV append has no `pos<max_seq` clamp (GPU OOB past 4096 rows). Record: [`docs/audits/codebase-review-2026-05-28.md`](docs/audits/codebase-review-2026-05-28.md).
- **Follow-up codebase review** (2026-06-12): working-tree diff review (C10 residency + FR3) plus fresh whole-workspace sweep with adversarial verification. Numeric core verified clean (asm kernels, int8 attention, GGUF loader overflow claims all refuted); verified exposure at the edges: `model_id` path traversal in shard loader, zero GPU-error checking across 77 Metal `wait_until_completed` sites, dispatch-geometry duplication back at 2 sites despite `KernelHandle`, corrupt-vindex panics (2026-05-28 item 1 still open), GIL never released in larql-python, 145 env flags / ~18 documented. Tracked under "Follow-up review (2026-06-12)" below; maintenance-debt recommendations under "Cleanup / consolidation track (added 2026-06-12)". Record: [`docs/audits/codebase-review-2026-06-12.md`](docs/audits/codebase-review-2026-06-12.md).
- **Tagged release binaries + first tag `v0.1.0`** (2026-07-24/25, [ADR-0026](docs/adr/0026-tagged-release-binaries.md)): larql had no distribution artifact — every host started from `git clone` + a cold `cargo build --release`. `.github/workflows/release.yml` now cross-builds `larql` + `larql-server` for macOS-aarch64 / Linux-x86_64 / Windows-x86_64 on `v*` tags under a new `release-dist` profile (stripped, no line tables; the profiling-friendly `release` profile is untouched) and publishes one archive per platform to a GitHub Release. `needs: build` gates publication on all three legs, so a partial release is not reachable. **`v0.1.0` cut 2026-07-25; the workflow went green on its first run** — archives verified to contain both binaries, and the macOS one smoke-run (`larql 0.1.0`, `dec-bench` present, strip confirmed). The driver is a **hard policy, not an optimisation: GPU-provisioned hosts never build from source** — a cold build is 20–40 min of pure CPU work with the GPU idle, and the DEC funnel runs ~10 stages on ephemeral rented hosts. `scripts/lib/larql-binaries.sh` enforces it for the stage drivers (operator-supplied → reuse → fetch → build, with the build refusing and exiting non-zero when `nvidia-smi` is present unless explicitly overridden). DEC-0.5 keeps compiling only the criterion kernel bench — a bench target is not a shippable binary and that kernel is the stage's measurement object. Separately, all **18** workspace crate names are claimed on crates.io as `0.0.0` placeholders (17 + `larql-experts`, a nested workspace invisible to the `[workspace] members` sweep), verified against the registry API; this is squatting-prevention, **not** the crates.io publishing ADR-0026 still declines. (**19th name, `larql-factory`, claimed identically on 2026-07-29** — see the ADR-0026 addendum and the entry directly below.)
- **Vindex Factory G0 slice + `larql recipe estimate`** (2026-07-29, [`docs/vindex-factory.md`](docs/vindex-factory.md)): new `larql-factory` crate — recipe schema (§4), `build_id` canonicaliser (§5), a structural validator covering every §6.1 PR-check gate that doesn't need network I/O, `larql capabilities` (§15.2, sourced from a new declarative architecture registry in `larql-models` rather than a hand-duplicated list — cross-checked by a test against the real `detect_from_json` dispatch), `larql card render` (§9), and `larql recipe estimate` (§6.1 step 4, the crate's first network I/O — upstream size + a coarse per-output byte model + an executor recommendation + a cost band priced against `docs/dec-funnel-v0.2.md` §7's existing rate basis rather than a fabricated duration prediction). Also lands previously-uncommitted OLMoE + GraniteMoE architecture support (a real prerequisite for the capability registry's claims about those two families). Every source file at or above the 90% coverage floor. Full detail in [`ROADMAP_STATUS.md`](ROADMAP_STATUS.md)'s "Recently shipped" entries. PR #192.
- **`larql recipe build` — PREFLIGHT→RELEASE build driver** (2026-07-29, same spec, §7): the `larql-factory::build` module orchestrates FETCH → EXTRACT → SLICE → MANIFEST → VERIFY → PUBLISH → RELEASE as subprocess calls into this same `larql` binary, behind a `CommandRunner` trait (`SubprocessRunner` for real builds, a `MockRunner` in tests) so the whole pipeline's stage ordering and failure handling is unit-tested without spawning a process or touching credentials. FETCH scopes `HF_HUB_CACHE` per build — `resolve_model_path`'s cache lookup doesn't disambiguate by revision, so a shared cache could otherwise let EXTRACT silently build from the wrong commit. PUBLISH always goes `--private` first; RELEASE only flips a repo public once every output has verified (§8's "nothing goes public unverified"). Always returns a `BuildRecord` — JSON-printable whether the build passed or a specific stage failed, matching `dec-bench`'s `--output-file` pattern. **Scope, decided deliberately after tracing what actually exists in this codebase, not what the spec assumed**: MIRROR (R2) and REGISTER (chuk-experiments-server) aren't implemented — no R2/S3 client exists anywhere, MCP tools aren't callable from compiled Rust, and the spec's own text assumes both are the rig worker's job; `BuildRecord` is the hand-off point for an external wrapper, the way `dec0-loopback.sh` already wraps `dec-bench`'s JSON. VERIFY here is checksum integrity only (`larql verify`) — the numeric reconstruction/logit-match checks in §8.1 need per-architecture tensor-naming knowledge that isn't validatable without real model weights. Extended `larql-vindex`'s publish path with a `private: bool` option and a new `set_repo_visibility` capability (verified against the real HF OpenAPI spec) to make the private-then-public two-phase publish possible; new `larql hf visibility <repo> --public|--private` command. Wired in as `larql recipe build <FILE> [--scratch-dir DIR]`. Every new source file at or above the 90% coverage floor.
