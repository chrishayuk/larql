# The TTS funnel — audio-token output as a forcing function

Status: **steps 0-5 green**; speaks at **~1.9x realtime on CPU**
(p50 31 ms/frame), incremental and batch token-identical,
zero-pre-buffer stable. First-turn **TTFA ~1.25 s** (from 2.0 s at
step-5 close) with the remaining budget 99%-accounted by phase split;
the <500 ms gate's scope is the FFN block + attention projections on
Metal. Every replanning step so far has stayed token-exact through
the step-4 dump oracle. Branch `worktree-tts-audio-tokens`, started
2026-08-09.

Gate log:

- **Step 0 PASS** (2026-08-09) — reference dump harness at
  `jarvis-voice/.engines/moss_parity_dump.py`; fixture long-23 with the
  aru-12 voice-clone splice; cpu / fp32 / eager / greedy / repetition
  penalty off. 138 frames (11.0 s audio) in ~34 s, EOS reached, prefill
  343 rows. Two independent process runs **bit-for-bit identical** across
  all eleven arrays (prompt matrix, prefill embeds/hiddens, per-step ids /
  summed embeds / backbone hiddens, 138×16×1027 local logits, frames,
  emitted tokens). Dumps in
  `jarvis-voice/renders/moss-realtime/parity-dump/` (run manifest records
  versions and hashes). Steps 2–4 compare against `run1.npz`.

- **Step 1 PASS** (2026-08-09) — MOSS loads in LARQL with full tensor
  accounting. `MossTtsRealtimeArch` (detection via nested
  `language_config`, `language_model.` prefix strip, `has_lm_head=false` —
  a new arch capability, since the backbone genuinely has no output
  projection and the untied-but-missing error is a text-LM invariant);
  auxiliary weights (per-codebook embedding tables + depth transformer +
  16 heads) side-load via `larql_models::speech::moss_tts_realtime`,
  every count config-derived. The coverage audit is bidirectional and
  derives its expected sets from the arch's own key methods: on the real
  403-tensor inventory it reports 310 backbone / 92 aux / 1 justified
  skip (the byte-identical text-table duplicate, asserted at load) /
  0 unexplained / 0 missing. Real-checkpoint aux load passes in ~6 s with
  the pad-row-zero and duplication assertions running on real bytes
  (`real_checkpoint_aux_load`, ignored in CI).

- **Step 2 PASS** (2026-08-09) — the first non-text input algebra executes
  with parity. `embed_tables_sum` (larql-compute, generic multi-table
  summed embedding — one id column per table, ascending-order sum, no
  scaling) over the reference's exact `[343, 17]` prefill matrix is
  **bit-identical** to the reference's summed input embeddings: max |Δ|
  = 0 across all 343×2048 values. Feeding those embeddings through the
  UNCHANGED Qwen3 path (`StandardEngine::prefill_from_hidden` + final
  norm) reproduces the reference backbone hidden at the last position to
  max |Δ| 7.6e-6, cosine 1.00000000 — fp32 accumulation-order noise.
  Gate test: `larql-kv/tests/moss_prefill_parity.rs` (ignored in CI;
  reads the step-0 dump's flat-binary export). The significance: LARQL's
  Qwen3 execution is now demonstrably independent of where its input
  embeddings came from — the boundary between "LLM engine" and
  "generative-model engine" is crossed at prefill.

- **Step 3 PASS** (2026-08-09) — nested generation executes with no new
  runtime machinery. The depth transformer is re-expressed as an ordinary
  4-layer Qwen3 `ModelWeights` (`depth_transformer_model`; per-codebook
  tables and heads stay outside as the audio-domain boundary) and each
  teacher-forced micro-step prefix runs through the same
  `prefill_from_hidden` the backbone used — causal attention makes the
  prefix forward equal to the reference's cached micro-step decode.
  Against the dump's 138×16×1027 logits: worst |Δ| = 1.75e-4, and all
  2,208 greedy codebook argmaxes identical
  (`larql-kv/tests/moss_depth_parity.rs`, ignored in CI). The finding:
  "one outer generation step contains an inner autoregressive programme"
  reduces, in LARQL terms, to *the inner programme is another model* —
  no nested-decode concept was needed at parity level. What remains
  step-4 work is the loop itself: sampling, EOS on codebook 0, the typed
  output seam, and cached (rather than recomputed-prefix) micro-steps
  for performance.

- **Step 4 PASS** (2026-08-09) — **Milestone A: the first complete
  self-generated utterance of audio tokens.** The full greedy loop
  (`larql-inference/src/speech/moss_realtime.rs::generate_frames_greedy`)
  chains everything with no teacher forcing: prefill from the reference's
  exact prompt, then per frame one backbone `decode_step_from_hidden` —
  the new `KvEngine` method closing the decode-time seam ADR-0023
  deferred — followed by *cached* depth micro-steps (per-frame handle
  set, one appended position per micro-step, exactly the reference's
  frame-lifetime cache), audio EOS detected on codebook 0 via the
  config-derived id. No tokenizer exists anywhere in the loop: ids in,
  ids out — the output-adapter invariant realised in the first non-text
  generation path. Gate: all 138 frames × 16 codebooks identical to the
  dump, EOS on the reference's final frame, emitted crop identical
  (`larql-kv/tests/moss_full_loop_parity.rs`; a single divergent
  codebook anywhere would cascade, so exact equality closes the chain).
  Remaining from the step-4 list: sampled mode (the non-standard top-p),
  and retrofitting the ids-preserved seam into the text decode paths.

- **Step 5 FIRST LIGHT** (2026-08-09) — `larql run <model> --speak "text"`
  speaks: prompt built from scratch in Rust
  (`speech/moss_prompt.rs` — system prompt byte-exact, voice-clone
  splice, 12-token lead), streaming per-frame callback on the driver,
  external codec via `--codec-cmd` / `$LARQL_MOSS_CODEC_CMD`
  (`jarvis-voice/.engines/moss_codec_cli.py`), `--play` for audio out.
  Chain validation at the strongest level: the CLI's emitted tokens for
  the parity fixture are **identical** to the reference dump's — prompt
  construction through generation, from raw text + reference codes, all
  in LARQL. First measured benchmark (cpu/fp32, wholly unoptimised):
  5.0 frames/s on the fixture (12.5 needed), RTF 0.40, TTFA 1.34 s
  (343-row prefill included), p50 frame 189 ms against the 80 ms budget
  — ~2.5x short of realtime with Metal, quantisation, and the fused
  depth kernel all untouched. **Gating discovery:** greedy decoding does
  not terminate on novel text (hit the 1500-frame cap; 120 s of audio
  for sixteen words) — the reference's operating mode is sampled
  (temperature 0.8, top-p 0.6, repetition penalty 1.1 per codebook), so
  sampled mode is required for arbitrary-text TTS, not optional. Still
  open for the full step-5 gate: sampled mode, the incremental
  push-text session (batch-vs-incremental token equality), and the
  buffer/underrun measurements.

- **Step 5 sampled mode + stage split** (2026-08-09) — the reference's
  sampler replicated literally (`speech/moss_sampling.rs`: ascending-sort
  top-p, strict-less top-k ties, once-per-unique-id repetition penalty
  per codebook, never on the prefill frame), unit-tested against fixture
  outputs produced by the reference's own functions. Sampled is now the
  CLI default (`--greedy` for parity, `--seed` for reproducibility) and
  the novel text that ran to the frame cap under greedy terminates
  cleanly (86 frames, EOS). The benchmark table gained a per-frame
  backbone/depth split, which corrected the performance picture twice:
  the clean CPU-build baseline is p50 190 ms/frame = backbone 65 ms +
  depth 125 ms (gap to the 80 ms budget ~2.4x, depth-dominated exactly
  as the bandwidth math predicts — 16 micro-steps re-read the full 1.06
  GB depth weights, ~17 GB/frame vs the backbone's 6.8 GB), and the
  earlier 508 ms figure came from a gpu-featured build whose CPU path is
  ~2.7x slower across both stages — a real regression to file
  separately. Remaining for the step-5 gate: the incremental push-text
  session and buffer/underrun measurements; then the perf phase (Metal,
  quantisation, fused depth frame) attacks a 2.4x gap, not 6x.

- **Q4 frame prediction** (2026-08-09,
  `larql-kv/tests/moss_q4_frame_bench.rs`) — the falsifiable bandwidth
  experiment, run on the real checkpoint's per-frame op mix (240
  matrices, 18.7 GB fp32 → 2.6 GB Q4_K). Three kernels tell the story:
  the naive dequant-dot is 24x SLOWER than fp32 BLAS (3441 ms —
  compute-bound, kernel not physics); the production Q4K·Q8K integer
  dot recovers it (103 ms single-core); and row-parallel it lands at
  **15.8 ms/frame of GEMV, 9.8x over fp32**. Substituted into the
  measured frame band, predicted p50 = **51–91 ms against the 80 ms
  budget**: quantisation alone is borderline-realtime, and any second
  lever (parallel/Metal attention for the non-GEMV remainder, fused
  depth locality) clears it with margin. Also measured while pinning
  the oracle: CPU-build run-to-run drift is real (p50 190–230 ms across
  runs; backbone 65–85, depth 125–148), so all comparisons quote the
  band; the gpu-featured build's 508 ms CPU reading sits far outside it
  and still needs a like-for-like A/B (the saved binary predated the
  sampling flags). Next: wire Q4 into the speech path for real (not
  just the kernel bench) and verify the quantised model still sounds
  like Jarvis before trusting any speed number.

- **Step 5 PASS** (2026-08-09) — the incremental push-text session
  closes the step-5 gate. `MossSession`
  (`larql-inference/src/speech/moss_session.rs`) is the reference's
  streaming contract restated in ids: pending text queue, prefill at
  `DELAY_TOKENS_LEN` pending ids (or fewer when the text has ended),
  one frame per pushed id, `<|text_pad|>` drain on `finish` — and the
  batch entry points are now wrappers over it, so batch and incremental
  share one loop rather than two kept in sync. Prompt construction split
  to match (`build_turn` + `append_text_lead`; `build_prompt` is their
  composition), which is what makes the two paths' prefill matrices
  identical by construction. Gate, both halves green: CI on a synthetic
  miniature speech model (chunk sizes 1/3/all + greedy + short-text
  cases, `larql-inference/tests/test_moss_session.rs`), and the real
  checkpoint (`larql-kv/tests/moss_incremental_parity.rs`, ignored):
  sampled seed 0, novel text — batch 72 frames with EOS, incremental
  **identical at chunk sizes 1, 5, and all-at-once**; greedy chunk 7
  identical over the 150-frame cap. The step-4 full-loop dump parity
  re-verified bit-exact after the refactor (138 frames). Buffer
  measurements landed with it (`speech/stream_timing.rs`, pure and
  unit-tested; reported by `--speak`): on the parity fixture,
  Q4, clean CPU build — 19.9 frames/s (RTF 1.58, within the 1.6-1.7x
  run band), p50 35 ms / p95 37 ms / max 93 ms, TTFA 2.0 s
  (prefill-dominated: prefill 1.97 s), and the streaming verdict:
  **min stable pre-buffer 0 ms, cold-start underruns 0**, buffer
  growing +1.25 s audio per second of playback. Not in scope here,
  deliberately: an actual audio-device ring buffer (the §5 runtime
  shape) and the incremental codec — the session emits frames through
  the same `on_frame` seam either will consume.
  *Correction (2026-08-10):* this entry originally reported the
  gpu-featured build's CPU path ~3.8x slower on the same command; the
  #242 investigation falsified that — see the entry below.

This document plays the role `k3-funnel.md` plays for sparse execution: it
names one model that cannot run, inventories exactly why, and commits to
removing blockers one at a time. No abstraction gets built speculatively;
every change must move the target model closer to parity.

**Target:** LARQL executes the generative portion of
`OpenMOSS-Team/MOSS-TTS-Realtime` — text tokens in, RVQ audio tokens out —
with per-step logit/token parity against the reference implementation.

**Non-goals (initially):** audio playback, vocoding/codec decode inside
LARQL, STT, persona/orchestration, "designing multimodal LARQL". The codec
(`MOSS-Audio-Tokenizer`) stays external: LARQL emits audio tokens, an
external decoder turns them into PCM. The codec is itself a causal
transformer, so pulling it in later is a plausible second act — after
parity, not before.

Why this matters beyond Jarvis: K3 forced routing, residency, containers.
TTS forces the orthogonal axis — typed token domains, multimodal *output*,
multi-codebook generation, persistent conversational state, and eventually
a continuous realtime deadline (§5). `docs/multi-modal.md` declares audio
output a non-goal ("No diffusion head, no audio decoder"); this funnel
deletes that line, one blocker at a time.

Provenance: everything below was read out of the locally cached checkpoint
(`models--OpenMOSS-Team--MOSS-TTS-Realtime`, rev `7568278`, 4.3 GB bf16
safetensors) and the reference source at
`project-jarvis/jarvis-voice/.engines/moss-src` (OpenMOSS/MOSS-TTS
@ `58b20a0`), 2026-08-09. Vendor performance figures (180 ms TTFB,
RTF 0.51 on an L20) are claims, not measurements.

---

## 1. The model

### 1.1 Backbone: stock Qwen3

The backbone is literally `transformers.models.qwen3.Qwen3Model`
(`modeling_mossttsrealtime.py:97`) — not a fork. hidden 2048, 28 layers
(all full attention), 16 heads / 8 KV heads (GQA 2:1), head_dim 128
(explicit), intermediate 6144, SwiGLU, RMSNorm eps 1e-6 with Qwen3
q_norm/k_norm, rope theta 1e6, text vocab 151936.

Two load-time facts:

- **No text lm_head exists.** The backbone never predicts text; its only
  output is `last_hidden_state`. Text is always supplied externally.
- **`language_model.embed_tokens.weight` is dead weight** — a byte-identical
  duplicate of `embed_tokens.0.weight` (the model always runs on
  `inputs_embeds`). 311 MB skippable at load.

Param split: 2.332 B total; backbone 1.721 B (incl. dead embed), depth
transformer 266.5 M, top-level embedding tables 344.8 M.

### 1.2 Audio token scheme

- 16 RVQ codebooks (of the codec's 32; only the first 16 used), codebook
  vocab 1024, model-side `audio_vocab_size` 1027 (+3 specials).
- Frame rate 12.5 Hz: 1 frame = 80 ms = 1920 samples at 24 kHz.
- **No codebook delay pattern** — all 16 codebooks of a frame come from one
  backbone step. (The sibling `moss_tts_delay` package *does* delay; do not
  cross-reference it.)
- Audio specials: 1024 pad (all channels), 1025 BOS / 1026 EOS
  (**channel 1 / codebook 0 only**). Text-side specials include
  `<|audio_pad|>` 151654 (reference splice placeholder) and
  `<|text_pad|>` 151655 (text-exhausted filler).
- Checkpoint property, assert at load: backbone codebook embedding row 1024
  is exactly zero in all 16 tables, rows 1025/1026 zero except table 1 —
  so summing pad channels is a no-op and 16 lookups can be skipped on
  text-only positions. A property of this checkpoint, not the architecture.

### 1.3 One generation step

Input layout is a `[T, 17]` integer matrix: column 0 text, columns 1–16
the codebooks.

```text
step_ids = [ next_text_token | 16 codebook tokens sampled last step ]   [B,1,17]
        │
        ▼
h = Σ over 17 embedding tables (plain sum, no projection, no scaling)
        │
        ▼
backbone forward, 1 position, persistent KV  →  hidden [B,1,2048]
        │
        ▼
depth transformer (4 Qwen3-shaped layers, hidden 2048, fresh 16-slot
static cache each frame — no cross-frame state):
  micro-step 0 :  input = raw backbone hidden, pos 0 → head 0 → c0
  micro-step i :  input = embed of c_{i-1} via table i-1, pos i → head i → c_i
        │
        ▼
frame [B,16] emitted; c0 == 1026 marks stop; frame feeds next step's input
```

There is no backbone↔depth projection (hidden sizes match by design). The
16 output heads (`local_lm_heads.0..15`, each [1027, 2048], no bias) map
depth position `i` to codebook `i`. Hidden state at position t−1 predicts
the frame at position t.

### 1.4 The streaming contract

The entire incremental protocol is: **the text channel leads the audio
channels by 12 positions** (`delay_tokens_len = 12`), and generation
consumes exactly one text token per 80 ms frame. Prefill waits for 12
pending text tokens; audio BOS (1025) sits on channel 1 of the *last*
prefill text token; when text is exhausted the model is fed
`<|text_pad|>` until EOS. The lead is enforced in three separate places in
the reference; an off-by-one desynchronises text and audio silently —
fluent but wrong-timed speech, not a crash.

Voice cloning is a **multi-channel splice, no speaker encoder**: the
system prompt contains N `<|audio_pad|>` text tokens, and channels 1–16 of
exactly those rows are overwritten with codec-encoded reference audio.

Multi-turn is **KV continuation, not re-prefill**: turn 0 prefls the
system prompt; later turns prefill only the new user turn on top of the
retained cache (32 K max context ≈ 40 min).

### 1.5 Sampling

Identical across all 16 heads: temperature 0.8, top_k 30, top_p 0.6,
repetition penalty 1.1 over a 50-frame per-codebook window. No CFG.
Greedy mode (`do_sample=False`) takes an argmax shortcut that bypasses
temperature/top-k/top-p entirely.

Parity traps, verified in source:

1. `apply_top_p` is non-standard: ascending sort, remove where
   `cumsum(softmax) <= 1 - top_p`. Replicate literally; HF's warper
   tie-breaks differently.
2. `apply_repetition_penalty` mutates the logit tensor **in place through a
   view** and changes the return shape `[B,1,V] → [B,V]`. Disable it while
   establishing baseline parity (the reference itself forces it off for the
   prefill frame).
3. Reference runs bf16 with `torch.compile` on the depth loop and
   attention-backend-dependent cache types. For reference dumps: eager,
   fp32, no compile.

### 1.6 Parity hooks in the reference

Per-step dump points (file:line in `moss-src/mossttsrealtime/`):
summed input embedding `modeling_mossttsrealtime.py:101-109`; backbone
hidden `modeling_mossttsrealtime.py:134`; per-codebook logits
`modeling_mossttsrealtime_local.py:437`; the 16-micro-step loop
`streaming_mossttsrealtime.py:378-411`; sampled frame
`streaming_mossttsrealtime.py:285-301`; prefill frame 0
`streaming_mossttsrealtime.py:221-238`; non-streaming reference loop
`inferencer.py:195-303`.

Compare logits, not sampled tokens (`torch.multinomial` RNG will never
match across engines). Greedy end-to-end, logit-level for the samplers.

---

## 2. What LARQL lacks — blocker inventory

From a full workspace survey (2026-08-09). The input side is genuinely
ready; the output side has no seam.

**Ready and reusable as-is:**

- `QwenArch` loads Qwen3-shaped checkpoints (QK-norms included) — the
  backbone is the easy case.
- `Sampler` (`layer_graph/generate/sampling.rs`) is pure
  `&[f32] → Option<u32>`; callable 16× per step with zero modification.
- The multimodal input seam: `EmbeddingPlan` / `EmbeddingChunk`
  (`larql-compute/src/forward/embedding_plan.rs`), `ModalEncoder` /
  `Connector` / `MultiModalProtocol` (`larql-models/src/multimodal.rs` —
  the `Audio` variants already exist, unimplemented), and
  `KvEngine::prefill_from_hidden` (ADR-0023).
- `KvCache.next_position` is already decoupled from row index (ADR-0023
  pinned this) — correct for prefills whose rows aren't text tokens.
- Side-loading non-backbone weights from safetensors is proven by the
  vision tower (`--mm-weights`); the PLE sidecar
  (`format/weights/ple_sidecar.rs`) is the pattern for a vindex-carried
  auxiliary tensor group later.
- `shannon layer-dump` / `layer-diff` — the parity tool. It closed OLMoE
  and GPT-OSS; it is the instrument for step 2 below.

**Tier 1 — structural, no seam exists:**

1. Every decode loop carries a scalar `current_token_id: u32`; ~25
   `generate*` entry points assume one token from one vocabulary. MOSS
   needs `[u32; 16]` + text per step.
2. Detokenization is unconditional inside the sampling step
   (`gpu/sampling_step.rs:37`); `run_decode_loop` takes non-optional
   `&Tokenizer` + `&mut Detokenizer`. `GenerateResult.tokens` is
   `Vec<(String, f64)>` — ids are discarded entirely.
3. One `lm_head` / one `vocab_size` in five places (`ModelWeights`,
   `VectorIndex`, `ModelConfig`, `WeightSource`, `VindexConfig`). MOSS has
   zero text heads and 16 audio heads.
4. `embed_plan` **concatenates** rows; MOSS **sums** 17 embeddings into one
   row, per decode step — exactly what ADR-0023 scoped out ("decode is
   text-out by definition").
5. No `decode_step_from_hidden` peer to `prefill_from_hidden` on
   `KvEngine`; decode never passes through the multimodal seam.
6. EOS is string-shaped (`EosConfig.stop_strings`, literal matching in
   `larql-kv`). Audio EOS is "codebook-0 id == 1026" — no representation,
   and numeric collisions with text special ids would spuriously halt.

**Tier 2 — format & policy:**

7. The vindex extractor actively drops audio towers
   (`extract/coverage/rules.rs::NON_TEXT_TOWER`). Route around it first:
   side-load depth transformer + embedding tables the way SigLIP does;
   revisit the vindex format (sidecar pattern) once parity exists.

**Tier 3 — surface & state:**

8. Both streaming surfaces discard the token id and send text-only frames;
   no binary/typed frame exists.
9. The server holds no engine or KV across requests; `ChatSession` is
   text-token-only. MOSS's persistent acoustic history (backbone KV +
   frame history across turns) has no home. The `WS_CMD_CANCEL` path is a
   useful barge-in precedent.

---

## 3. The funnel — steps and gates

Discipline: each step has a falsifiable gate. No step starts before the
previous gate is green. Abstractions (a `TokenStep` enum, a multi-head
output type) are introduced at the step that needs them, shaped by what
that step needs, not before.

**Step 0 — reference dump harness (Python, in the jarvis-voice venv).**
Instrument the reference at the §1.6 hooks: fixed prompt, greedy, eager,
fp32, repetition penalty off. Dump per-step summed embeddings, backbone
hiddens, per-codebook logits, sampled frames to disk.
*Gate: the dump is reproducible bit-for-bit across two runs.*

**Step 1 — weights load.** MOSS detection arm + config fields
(`num_codebooks`, `audio_vocab_size`, depth dims); backbone through the
existing Qwen3 path; embedding tables 0–16 + depth transformer + 16 heads
side-loaded from safetensors (vision-tower mechanism, not the vindex).
Assert the pad-row-zero checkpoint property at load; skip the dead
311 MB embed.
*Gate: every tensor accounted for against the checkpoint inventory —
the GPT-OSS silent-drop lesson applies here with 34 auxiliary tables.*

**Step 2 — backbone step parity.** Summed-embedding primitive
(`embed_step` peer to `embed_plan`); one backbone forward from a `[T,17]`
prompt matrix.
*Gate: `layer-diff` vs the step-0 dump — first-drifting-capture empty,
final hidden within fp32 tolerance.*

**Step 3 — depth transformer parity.** The 16-micro-step inner loop +
16 heads, greedy.
*Gate: per-codebook logits match the dump within tolerance; greedy frames
identical.*

**Step 4 — full decode loop.** The architectural invariant this step
installs is more fundamental than an audio token type: **core generation
produces model-domain ids/state; interpretation — detokenisation included —
belongs to an output adapter.** The present defect is not "LARQL lacks
`AudioToken`"; it is that generated ids are destroyed by unconditional
text detokenisation inside the sampling step and `GenerateResult` retains
only strings. Fix that seam first (ids preserved end-to-end, the text
detokeniser demoted to one adapter among possible others); then a decode
path carrying `(text_token, [u32;16])` per step — with audio EOS on
codebook 0 — is merely the first non-text consumer of the seam, not the
definition of it.
*Gate: full-utterance greedy audio-token sequence identical to the
reference for the canonical fixture — long enough to exercise the KV
cache well past attention-window degeneracies (the short-fixture lesson).*

**Step 5 — streaming.** The 12-token-lead protocol: incremental text in,
frames out, `<|text_pad|>` drain, voice-clone splice in the prefill.
External codec decodes the emitted tokens for listening checks; measure
real TTFA (first frame emitted after 12th text token) — the first genuine
streaming TTFA figure in the whole Jarvis evaluation.
*Gate: token stream from incremental feeding is identical to batch
generation on the same text.* The gate is token-based deliberately: if
audio sounds wrong while the RVQ sequence matches, LARQL has done its job
and the fault is downstream. A listening check (external codec decode of
the emitted tokens; is the aru-12 clone audibly the same speaker as the
reference implementation's output?) is recorded as a separate integration
observation, never as parity evidence.

**Step 6 — VINDEX3-native MOSS** (gated on steps 4–5). The safetensors
side-load is the oracle implementation, not the destination: once the
execution semantics stop moving, the model moves into the vindex whole.
Gate, brutally simple: *delete the original-safetensors requirement —
open only the vindex artefact and reproduce the already-green step 2–5
parity tests.* No side-load flag, no HF-directory dependency. The format
lesson MOSS teaches is not "support TTS fields": it is that a model is
**named executable tensor regions with explicit architectural roles**
(K3's expert banks, MOSS's embedding banks / depth transformer / sixteen
heads, the vision tower — regions, not exceptions). Three ownerships stay
separate: VINDEX3 describes and carries the model (regions, domains,
vocabularies, capabilities); the architecture owns the algorithm (sum 17
tables, run backbone, 16 micro-steps, sample heads); the runtime owns
temporal policy (12-token lead, 80 ms clock, turn KV, barge-in). Temporal
policy never enters the storage format.

**Later, in rough order, each gated on the above:** session/turn state
(KV continuation across turns — server-side home for a persistent
engine); realtime axis (§5); the voice bank (`larql voice clone` —
voices as first-class model-agnostic data; design in `ROADMAP.md`
"Voice bank", gated on step 5, now open); codec-in-LARQL (§6); the
FUSE ladder (LLM→speech via token piping, then latent composition —
[`fuse-funnel.md`](fuse-funnel.md), which consumes this funnel's step-4
dump as its oracle).

**The perf phase's next gate (set 2026-08-09, after step 5):** first-turn
TTFA **below 500 ms** without regressing the ~1.6x realtime steady-state
class. The 2.0 s TTFA is prefill-dominated and the Q4 prefill is a
row-oriented path (343 sequential GEMVs re-reading the packed weights
per row) where the shape wants GEMM — so the levers are quantized GEMM
/ tiled dequant+BLAS / Metal prefill, chosen by a falsifiable prefill
benchmark before any implementation.

**The prefill prediction** (2026-08-09,
`larql-kv/tests/moss_prefill_bench.rs`; real backbone matrices, seq 343,
0.97 TFLOP, best-of-3): row path 1396 ms (693 GFLOPS — matches the
measured 1.97 s prefill minus ~570 ms non-GEMM overhead); **the
amortised `q4k_matmul_into` is FALSIFIED as the lever — 4555 ms, 3.3x
slower than the row path it was meant to replace** (its inner loop
strides across all 343 activation rows per decoded super-block, and its
f32-FMA loop is no match for the tuned Q8K integer kernels, let alone
AMX); fp32 BLAS GEMM 456 ms (2.1 TFLOPS); dequant-whole+BLAS 922 ms.
Consequences: (1) the best CPU prefill lands ≈ 1.0 s (BLAS + overhead,
needing an fp32 shadow ~5 GB) or ≈ 1.5 s allocation-free
(dequant-per-layer + BLAS) — CPU wiring halves TTFA but cannot reach
the 500 ms gate; **the gate's lever is Metal prefill**. (2) Spillover:
the vindex serving path prefills dense models through this same
amortised matmul — if it reads 212 GFLOPS there too, that path carries
the same defect; check separately.

**The #242 verdict** (2026-08-10) — the "gpu-build CPU regression" is
**falsified as a build effect**. Evidence, in strength order: the
frame-shape kernel bench under a gpu-featured build reads 16.8 ms/frame
against the cpu build's 15.8-16.8 band (no feature effect on the
kernels); the *same* gpu-featured CLI binary ran the identical sampled
command at 4.6 fps (prefill 12.8 s) and 19.1 fps (prefill 2.4 s)
minutes apart; four interleaved A/B pairs put slow episodes on **both**
binaries (cpu-only build: 11.8 fps with a 707 ms max frame; gpu build:
6.3 fps with an 883 ms max frame) with clean-run medians
indistinguishable (~±10%); no pageouts, no thermal warnings. The
original evidence was contaminated: the first "2.7x" reading came from
a stale saved binary, and both of today's slow readings ran immediately
after multi-GB cargo builds (post-build host churn — plausibly
indexing). Two real lessons replace the phantom: (a) **never trust a
single benchmark run adjacent to a build** — interleave A/B trials on
a quiet machine (the run-band discipline this doc already applies to
frame times applies doubly across binaries); (b) the host serves
**episodic multi-hundred-ms stalls** (p95 spiking 34 → 503 ms) that a
realtime speech runtime must survive — that is a §5 deadline-robustness
requirement (QoS pinning, buffer sizing against measured host jitter),
not a build defect. Metal prefill is unblocked.

**The Metal prefill probe** (2026-08-10,
`larql-compute-metal/examples/moss_prefill_gemm_probe.rs`; the four
distinct MOSS projection shapes at seq 343, cache-warm, best-of-5) —
both existing Metal kernels are **falsified for the <500 ms gate**:
the 32×32-tile f32 `sgemm` totals 738 ms (1.26 TFLOPS — loses to CPU
AMX's 456 ms), `q4k_matmul` totals 1327 ms (698 GFLOPS; consistent
with its two Gemma-prefill falsifications, whose diagnosis already
prescribed the fix: **a K-tiled / `simdgroup_matrix` GEMM kernel**,
llama.cpp's `mul_mm` class, predicted 150-250 ms at these shapes).
Supporting findings from the same session: the CPU row path's floor is
partly *synchronisation*, not arithmetic — ~67k spin-pool fork/joins
per prefill (one per activation row per matrix), so a blocked CPU
GEMM (single parallel region per matrix, inner loop over all 343
activations, amortising both the super-block unpack and the pool
round-trip) is the CPU-side counterpart worth building regardless;
`VECLIB_MAXIMUM_THREADS=1` changed nothing (Accelerate thread
contention falsified); a `sample`-instrumented run showed a cvwait
storm but the sampler itself perturbs a spin-pool process — the stall
episodes need in-process phase timing to be caught honestly. The TTFA
attack is therefore two bounded kernel tasks: (1) blocked Q4K×Q8K CPU
GEMM — cuts the 1.4 s row-path term and the sync waste; (2) the
simdgroup-matrix Metal GEMM — the gate-crosser, also the fix the
vindex dense-prefill path was already diagnosed as needing.

**CPU prefill re-planned: dequant + BLAS wired, integer GEMM falsified**
(2026-08-10). The blocked integer GEMM was built and lost in all three
loop structures at the real shapes (per-pair dots 2202 ms /
weight-chunk + transpose 2095 ms / activation-parallel 1499 ms vs the
row path's 1390 ms — the Q4K·Q8K kernel is issue-bound at ~650-700
GFLOPS whatever the granularity, and the 67k-dispatch hypothesis was
wrong: the spin pool's dispatch cost is negligible on a quiet machine).
Kept as recorded negative evidence (`q4k_q8k_dot::gemm`, the
`q4k_matmul` shader precedent). What DID win: **dequant-whole + fp32
AMX BLAS at 863 ms — 1.6x over the row path** — the decode doctrine
("dequant-then-BLAS is dramatically slower") is a one-row truth that
inverts at prefill shape. `Q4kFfn::forward` now routes multi-row
inputs through dequant + BLAS (single-row decode unchanged;
batch ≡ incremental unaffected — both sessions share the path).
Measured end-to-end on the fixture: **prefill 1.97 → 1.50 s, TTFA
2006 → 1530 ms**, steady-state 23.3 fps / RTF 1.85 (clean run), p50
31 ms. TTFA budget now: ~0.86 s GEMM (Metal simdgroup target
~0.2 s) + ~0.64 s non-GEMM overhead — **dissected to a third
shape-mismatch** (2026-08-10): the production attention prefill
(`attention/gqa`) runs per-head × per-query-position loops — one BLAS
gemv per position for Q·K (≈154k gemv calls per 343-row prefill),
per-element f64-`exp` softmax (≈26M exps, the `exp` frames the stall
profile showed), a second gemv plus scalar accumulate for A·V.
Arithmetic puts that at ~0.4-0.7 s, matching the bucket. The batched
form (per layer × head: one [S,d]·[d,S] scores GEMM, triangular mask,
row-vectorised f32 softmax, one [S,S]·[S,d] output GEMM — ~27 GFLOP
at BLAS rates ≈ tens of ms) is the next implementation. Caution
pinned in advance: any accumulation-order change must re-pass the
step-4 dump gate (token-exact greedy parity), so the batched path
lands behind that verification, not alongside it.

**Batched attention prefill landed, token-exact** (2026-08-10,
`attention/gqa::gqa_attention_batched`). Per head: one scores GEMM,
causal/windowed row-wise softmax (the *same* f64 softmax on the same
slices — parity-first, per the contract ladder), one output GEMM;
sliding windows mask rows of the same GEMM, which keeps the SWA bit
contract (a windowed and unwindowed prefill share one physical plan —
the two SWA bit-identity tests caught exactly this before the window
support was added, and pass with it). Capture/sink/decode shapes keep
the loop. Gates: full attention suite (162), full crate suites, and
**the step-4 dump gate token-exact — all 138 frames identical**
through the GEMM reordering; no ladder-rung retreat needed. Measured
(three quiet-machine runs after the first set proved contaminated by
concurrent user work — the discipline pays again): prefill 1.50 →
**1.21-1.25 s**, TTFA 1530 → **~1.25 s**, steady-state p50 31 ms /
RTF 1.91-1.96. The surviving
term inside attention is the f64-`exp` softmax kept for parity
(~26M exps ≈ 0.3 s): the next rung is a vectorised f32 softmax,
admitted only through another dump-gate re-pass — if exactness breaks
there, the contract ladder (logit tolerance + same greedy choices)
gets its first real decision. Also re-read honestly: yesterday's
86.4 s dump-gate wall time was a contaminated reading — today's
clean 28.9 s is the fp32 loop's true rate (decode untouched by this
change). TTFA budget now: ~0.86 s FFN GEMM (Metal simdgroup target
~0.2 s) + ~0.3 s f64 softmax (f32 vectorisation next) + ~0.15 s
remainder.

**f32 softmax: admitted token-exact, banked ~nothing — the cost model
was wrong** (2026-08-10, `softmax_in_place_f32`: Accelerate `vvexpf`
+ f64 denominator, batched prefill path only; decode/capture keep the
f64 reference). The 138-frame dump replay stayed **token-exact through
both physical-plan changes** — the contract ladder's strongest rung
has now admitted a GEMM reordering and an f32 exponential without
retreat. But TTFA measured 1232-1290 ms vs 1245-1286 before: the
"~0.3 s of f64 exp" estimate was wrong (real exp throughput puts it
nearer ~0.1 s, and much of it overlapped BLAS threads). Kept anyway —
same tokens, strictly cheaper plan. Corrected budget, from measurement
not estimate: prefill ≈ 1.22 s = ~0.66 s FFN dequant+BLAS + ~0.11 s
attention-projection BLAS + ~0.36 s unprofiled per-layer machinery
(rope, QK-norm, embeds, KV handle writes, norms) + small. Consequence
for the gate: Metal on the FFN alone projects only ~0.76 s TTFA —
**crossing 500 ms needs Metal to take the attention projections too,
and/or the 0.36 s remainder to shrink**. Next move is an instrumented
prefill phase split — measure, don't estimate; the last two rungs both
said so.

**The phase split** (2026-08-10, `larql_compute::phase_timing`, env
`LARQL_PHASE_TIMING=1`, reported by `--speak` at first frame): 1243 ms
of the ~1250 ms TTFA region accounted — `layer.ffn_block` **904 ms**;
attention projections 82 + 49 ms (qkv, o+residual); `attn.core` 84 ms
(the batching held); rope 68; norms 49; embed 5; KV write 2. Two
corrections fall out: (a) the hypothesised "360 ms glue" bucket
**dissolved** — real glue is ~124 ms and there is no fourth
shape-mismatch hiding in it; (b) the FFN block costs 904 ms in
production against ~660 ms bench-predicted — the gap is suspected
allocation churn (`forward_block` allocates three fresh ~50 MB dequant
buffers per layer, ~4 GB of fresh pages per prefill; a reusable
scratch + `dequantize_into` is the cheap next fix, hypothesis to
falsify by measurement). Gate arithmetic, now from a fully accounted
budget: Metal FFN (904 → ~200) alone ⇒ TTFA ≈ 550 ms; plus the
attention projections (131 → ~40) ⇒ **≈ 460 ms — under the gate with
rope/norms untouched**. The Metal kernel's scope is exactly these GEMM
sites, nothing else. Bonus
diagnostic, twice confirmed: during stall episodes the BLAS-phase
prefill holds (1.58 s) while spin-pool decode collapses (9.4 fps, max
frame 1008 ms) — the episodic interference selectively degrades
spin-pool execution, pointing at scheduling/QoS of the spin threads
(E-core placement suspected), never AMX. That is now the sharpest
lead for the §5 deadline-robustness work.

---

## 4. The voice-as-data ladder (EXP-V, runs in parallel)

Lives in `chris-experiments/voice/`, not here — it probes the *reference*
implementations, needs no LARQL changes, and can start immediately. Full
plan there; summary of what scoping established (from the installed
`qwen-tts` 0.1.1 package and cached checkpoints):

- Qwen3-TTS Base carries a jointly-checkpointed ECAPA-TDNN speaker encoder
  (mel in, no BatchNorm/Dropout — deterministic forward), whose output
  dimension **is** the talker hidden size: 1024 (0.6B) / 2048 (1.7B), no
  adapter. Per-model encoders; VoiceDesign/CustomVoice ship none.
- `x_vector_only_mode` is real but is an **ICL ablation**, not
  "embedding-only vs something else": the embedding conditions both modes;
  the flag removes reference codes + transcript. Cleaner for EXP-V1 than
  assumed.
- The embedding enters as **one summed sequence position** in the prefill
  (index 7 under the harness defaults) — layer-0 input only, nothing
  re-injects downstream. Layer-k insertion experiments are forward-hook
  work on `talker.model.layers[k]`.
- Extraction is one call (`extract_speaker_embedding`,
  `modeling_qwen3_tts.py:1941`); injection needs no monkeypatching
  (`VoiceClonePromptItem` + the shipped save/load prompt precedent in
  `cli/demo.py:501/526`).
- Consequence for the ladder: same-family transplant is confined to
  same-size checkpoints (1024 vs 2048 are dimensionally incompatible, and
  it's architectural). Cross-model portability starts at zero shared
  space — which is the research question, not a disappointment.
- MOSS contrast worth exploiting: MOSS has **no speaker encoder at all** —
  it conditions on spliced audio tokens, where Qwen3-TTS conditions on a
  pooled vector. Two unrelated serialisations of the same identity
  (aru-12) that both cause a transformer to instantiate the same perceived
  speaker. The endpoint of the ladder is therefore not "which embedding
  format wins" but whether both conditioning mechanisms converge onto an
  equivalent downstream residual state — an `I_voice` that is neither the
  x-vector nor the acoustic prompt, just induced by both. If that object
  exists, *it* is the portable representation, and "voice as data" stops
  being a metaphor.

---

## 5. The realtime axis (design note, gated on §3 step 5)

What K3 doesn't have: a continuous external deadline. The DAC needs a
frame every 80 ms forever; a page fault at the wrong moment is audible.

The objective is not tokens/sec:

```text
minimise    time_to_first_playable_frame
subject to  sustained_generation_rate >= playback_consumption_rate
            no_buffer_underrun
            bounded_memory
```

Runtime shape (all hot-path, no Python, no subprocesses, no WAV files):
inference thread → audio-token queue → codec worker → PCM ring buffer →
audio callback. The callback never waits on the model; it reads
already-prepared PCM from a bounded ring or plays silence. Barge-in
(RFC-0001 §7's 100 ms cancellation budget) maps to: stop reading the ring,
cancel in-flight synthesis — the `WS_CMD_CANCEL` select-loop is the
precedent.

The interesting eventual coupling: the residency scheduler working
backwards from the audio clock ("buffer below 120 ms — promote the next
block now"). That turns predictive residency from a throughput
optimisation into deadline satisfaction. Design only until step 5 emits
real frames; then the jarvis-voice t0–t4 timeline (token commit → commit
latency → TTS TTFA → first playable frame → audible) becomes the
measurement surface, and backlog-vs-drain becomes measurable instead of
inferred from batch RTF.

The realtime unit is friendly: one frame = 80 ms, so sustainability is
12.5 frames/s — and at ~2.3 B parameters the whole working set can
plausibly stay resident on this hardware, which inverts K3's problem
("predict what comes off SSD next" becomes "keep the speech machine hot
and make every 80 ms deadline"). The depth loop is a specialisation
target the reference cannot exploit: sequence ≤ 16, fixed hidden, cache
lifetime exactly one frame, known head per position, one new row per
micro-step — `moss_depth_frame(hidden) → [16 ids]` as a fused kernel
eventually. The step-5 benchmark table (all measured, no vendor claims):
acoustic frames/s; realtime factor; first audio-token frame (ms); first
decoded PCM (ms); steady-state buffer growth (ms/s); p99 frame compute
(ms); underruns per 60 s. The bar that matters is p99 frame compute
< 80 ms; comfortably past it, buffering works *for* us, and multiples of
realtime open speculative synthesis (likely acknowledgements prepared
before they are needed) — predictive execution, speech-flavoured.

## 6. Codec-in-LARQL (explicitly deferred)

MOSS-Audio-Tokenizer is a fully causal transformer codec (no conv stacks
in the transform path; strided patching + RoPE transformer stages,
RLFQ quantiser, streaming via ring KV with 10 s context, fp32, ~6.7 GB).
That makes "the whole path in one engine" genuinely plausible — text →
speech LM → RVQ tokens → causal-transformer decode → PCM. It stays out of
scope until §3 is green end-to-end; the external decoder is the oracle the
in-engine one would be judged against.
