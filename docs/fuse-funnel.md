# The FUSE funnel — model-to-model fusion as a runtime primitive

Status: **two results in, and together they reframed the programme.**
FUSE-2.5 failed its gate informatively — the 2048-d seam is specific to
layer 27's output, so "truncate and hope" is dead. The weight diff then
showed MOSS's backbone is Qwen3-*shaped* but independently trained
(projection matrices orthogonal to precision), so there is no second
general-purpose Qwen doing redundant language work inside the mouth.

**The objective is therefore no longer "amputate Qwen".** It is:

> Can Jarvis and its speech system share useful computation and state?

Eliminating the 28-layer backbone is a possible outcome, not the success
criterion — `Jarvis → small latent binding → MOSS speech-state backbone →
depth` is a good result if the binding buys shared intent, prosody
control or latency.

Sequence from here (revised 2026-08-10):

```text
done   FUSE-2.5 ──► weight diff
        │
next   FUSE-1A    paired-state corpus, one voice, Qwen3-1.7B surrogate S0,
        │         CKA/SVCCA + ridge & low-rank B_L maps, held-out semantics
        ▼
       FUSE-1B    behavioural substitution using the best same-space B_L
        ▼
       FUSE-1.5   teacher/student additive injection; learn B_in through
        │         the frozen backbone; α sweep. Gate 0 first: prove the
        │         teacher signal exists before building the corpus.
        ▼
       FUSE-2     raw foreign residual null — record it, expect failure
        ▼
       FUSE-3     serious learned bridge
```

Opened 2026-08-09 as a `ROADMAP.md` section, given its own funnel doc
2026-08-10 once the ladder acquired per-rung gates. Speech is the first
proving ground — the abstraction is model-to-model fusion, not "Qwen
token sharing".

Companion docs: [`tts-funnel.md`](tts-funnel.md) owns the MOSS port and
supplies this programme's exact oracle (the step-4 138-frame dump). The
voice ladder — `chris-experiments/voice/V0_PLAN.md`, outside this repo —
owns identity portability, which FUSE-3/4 depend on.

Gate log:

- **Weight diff — MOSS's backbone is Qwen3-*shaped*, not Qwen3**
  (2026-08-10). Harness `jarvis-voice/.engines/moss_qwen_weight_diff.py`,
  results `renders/moss-realtime/fuse-weightdiff/weight-diff.json`. MOSS's
  `language_model.*` namespace is tensor-for-tensor the same 310 entries as
  `Qwen/Qwen3-1.7B-Base`'s `model.*`, so every pair is directly comparable.
  Whether OpenMOSS initialised from those weights is not stated publicly —
  hence a measurement, not bookkeeping.

  | tensor class | n | cos mean | max abs cos | rel-L2 mean |
  |---|---|---|---|---|
  | projection matrices (q/k/v/o, gate/up/down) | 196 | **0.000015** | 0.00168 | 1.798 |
  | RMSNorm scale vectors | 112 | 0.871 | — | 0.803 |
  | text embedding `[151936, 2048]` | 1 | **−0.000585** | — | 1.474 |

  Zero tensors bit-identical. **The projection matrices are orthogonal to
  numerical precision**: for a 2048² matrix chance cosine has σ ≈ 0.0005,
  and the largest across all 196 is 0.0017 (~3σ). Independent draws, not
  shared ancestry. The 0.87 on the norm vectors is an artefact and must not
  be read as similarity — RMSNorm weights are all-positive and clustered
  near 1.0, so any two are cosine-similar by construction; their rel-L2 of
  0.80 says they genuinely differ. Quoting the naive per-layer mean (0.31)
  would mislead: it is 4-of-11 norm tensors dragging the average up.
  rel-L2 corroborates independently — 1.80 against the √2 = 1.414 expected
  for orthogonal tensors of equal norm implies MOSS's matrices carry ~1.5×
  Qwen's norm. Independently trained, at a different scale.

  **Not run, deliberately: the singular-value analysis of ΔW.** "Is the
  late-layer delta low-rank" presupposes shared ancestry; a delta between
  unrelated matrices has no reading as an adaptation. It would have
  produced an authoritative-looking table of nothing.

  **Scope of the claim.** This excludes `Qwen3-1.7B-Base` specifically.
  Another Qwen3-1.7B variant is not formally excluded, but fine-tuning does
  not rotate layer-0 attention matrices to orthogonality, so no released
  checkpoint of this architecture is a plausible ancestor. OpenMOSS adopted
  the Qwen3 block design and the tokenizer vocabulary (151936), not the
  weights.

  **Consequence — the programme's objective changes.** "Amputate Qwen" was
  the wrong frame. There is no second general-purpose Qwen doing redundant
  language work inside the mouth; there is a 1.7 B **speech-state
  transformer that happens to be implemented in the Qwen3 architecture**,
  with no LM head. The provenance objection to keeping MOSS is discharged.
  What survives is a narrower and better-posed question:

  > Can Jarvis and its speech system share useful computation and state?

  Eliminating the backbone is now a *possible outcome*, not the success
  criterion. If the endpoint is `Jarvis → small latent binding → MOSS
  speech-state backbone → depth`, and the binding buys shared intent,
  prosody control or latency, that is already a strong result. It is also
  a good position for the runtime: LARQL gets known attention/RoPE/FFN/KV
  semantics, a known quantisation and Metal path, without inheriting
  Qwen's learned weights.

  **Read together with FUSE-2.5** the two results explain each other. Not
  "stock language semantics → speech-specialised top layers → depth", but
  "speech-specific multimodal state machine → precisely learned terminal
  representation → depth". The whole 28-layer stack participates in
  building speech state, which is why the terminal representation is so
  narrow and why that stack carries voice conditioning, acoustic history,
  turn history, timing, lexical content and EOS together.

- **FUSE-2.5 FAIL — the seam is layer-27-specific** (2026-08-10). Harness
  `jarvis-voice/.engines/moss_fuse25_depth.py`, results
  `jarvis-voice/renders/moss-realtime/fuse25/results.json`. Conditions:
  cpu / float32 / eager / greedy / repetition penalty disabled — identical
  to the step-0 dump, because the comparison is against its arrays
  (`parity-dump/run1.npz`, line long-23, aru-12 splice).

  The intervention is one integer: `Qwen3Model.forward` iterates
  `self.layers[: self.config.num_hidden_layers]` and then applies
  `self.norm`, so setting that field during decode runs layers 0..L−1 and
  still post-norms the truncated residual — the choice FUSE-2.5 specified.
  Prefill always runs full depth, selected by a forward-pre-hook on
  sequence length (prefill is the 343-row call, decode is 1 row). Hooks
  and instance wrapping only; no reference control flow reimplemented.

  **Controls pass.** L=28 reproduces the oracle bit-exactly in *both*
  conditions — 138 frames, 0.00% argmax change, seam cos 1.0000, KL
  0.0000. The truncation mechanism and the teacher-forcing wrapper are
  each inert at full depth, so the rest of the table means something.

  | cond | L | frames | EOS | argmax chg | seam cos min/mean | KL mean |
  |---|---|---|---|---|---|---|
  | teacher | 28 | 138 | 137 | 0.00% | 1.0000 / 1.0000 | 0.000 |
  | teacher | 24 | 138 | 137 | 92.53% | 0.8501 / 0.9115 | 3.454 |
  | teacher | 20 | 138 | 136 | 95.61% | 0.7562 / 0.8289 | 4.279 |
  | teacher | 16 | 138 | 135 | 97.55% | 0.6003 / 0.6780 | 4.462 |
  | teacher | 12 | 138 | 125 | 98.78% | 0.4280 / 0.5177 | 4.556 |
  | teacher | 8 | 138 | 133 | 99.05% | 0.3912 / 0.4916 | 5.232 |
  | free | 28 | 138 | 137 | 0.00% | 1.0000 / 1.0000 | 0.000 |
  | free | 24 | 112 | 111 | 98.21% | 0.4466 / 0.6009 | 15.140 |
  | free | 20 | 164 | 163 | 98.87% | 0.3924 / 0.5894 | 27.023 |
  | free | 16 | **401** | none | 99.14% | 0.4227 / 0.5112 | 26.898 |
  | free | 12 | **401** | none | 99.00% | 0.2132 / 0.3495 | 5.281 |
  | free | 8 | **401** | none | 99.19% | 0.2498 / 0.3601 | 5.905 |

  **The gate asked for a monotone frontier with an identified knee. There
  is no knee — there is a cliff at the top of the stack.** Removing 4 of
  28 layers changes 92.5% of RVQ argmaxes under teacher forcing, on the
  first decode frame available. (Frame 0 can never diverge: its seam comes
  from the always-full-depth prefill call — which is also why the seam
  array is 137 rows against 138 frames.) Dropping a further 16 layers adds
  only 6.5 points while seam cosine slides 0.85 → 0.39. The representation
  degrades smoothly with depth removed; the depth transformer's *decision*
  is destroyed as soon as any late layer goes.

  **Free-running additionally loses termination.** L=24 stops early (112
  frames vs 138), L=20 overruns (164), and L≤16 never emits audio EOS at
  all — all three hit the 400-frame cap. Audio EOS is codebook-0 id 1026,
  so *stopping* is a late-backbone function, not a depth-transformer one.
  Free-running KL peaks at L=20 (27.0) and falls back by L=12 (5.3), which
  is degenerate low-entropy babble rather than recovery — do not read the
  drop as improvement.

  **Reading.** The 2048-d seam is specific to layer 27's output, not to
  "semantic state" in any depth-invariant sense. The late backbone is not
  re-deriving semantics an upstream LLM already has — it is producing the
  exact representation the depth transformer was trained against,
  termination included. The cheap intermediate version of the amputation
  plan — inject Jarvis semantics, keep only the last 4–8 layers — is dead
  as stated.

  **Consequences for the ladder.**
  1. FUSE-2 is now predicted to fail hard rather than informatively: if
     4 layers of the *same* model breaks the seam, a foreign LLM's
     residual will not survive it. Run it for the record, expect nothing.
  2. **FUSE-1.5 is promoted to the next rung to run.** It is the only rung
     that never touches the seam — full 28-layer backbone, full KV, only
     the text-channel term of the 17-way input sum replaced. This result
     is a direct argument for that ordering. Blocked on a Qwen3-1.7B
     download (not cached; the backbone's shape is 28L/2048/16-8).
  3. FUSE-3 becomes load-bearing, with a concrete first target: does a
     learned affine L→27 map recover the trajectory, and how little
     machinery suffices?

  **Controls not yet run**, cheap, worth having before FUSE-3: the
  no-final-norm A/B, to confirm the cliff is not an artefact of
  post-norming a mid-stack residual. The frontier is clean and monotone,
  which argues against it, but it is minutes of compute. Also unrun: a
  weight diff of MOSS's backbone against stock Qwen3-1.7B-Base — if the
  backbone is a fine-tune of it, FUSE-1's binding may be near-identity,
  which would reorder the ladder again.

---

The principle to lock in now: **text is one interoperability layer, not
LARQL's model-to-model ABI.** Two models in the same runtime should
exchange the cheapest useful materialisation of a computation —
generated ids, residual state, KV state — with English text reserved
for when text genuinely is the cheapest interchange format.

The abstraction is model-to-model fusion, NOT "Qwen token sharing."
Qwen→MOSS is only the first proving ground (shared tokenizer lineage
makes the experiments easy); the architecture is:

```text
producer model → intermediate state / token domain / residual
              → binding (model-specific; LARQL owns the mechanism)
              → consumer model
```

Compatibility levels, weakest binding first:

```text
1. token-compatible      reuse ids directly
2. vocabulary-mappable   cheap token-domain translation
3. hidden-state          reuse residuals directly
4. projectable           small learned/fixed projection
5. state-composable      semantic state + target-specific state
```

The runtime consequence: a decode step exposes more than its final
token — `GeneratedToken { id, hidden, kv_position, .. }` — and the
consumer takes the view it needs (ids → text protocol, residual →
conditioning). Generated text becomes one *view* of the computation.
VINDEX3 eventually describes interfaces, not pairings: a model declares
`output_domain` (token ids, semantic hidden) and `input_domain` (text
tokens, hidden state, acoustic context); the binding
(mapping/projection/adapter) is a separate, inspectable object.

## The ladder (speech instance; each rung gated on the last)

Checkpoint facts that make these rungs well-posed (verified 2026-08-10
against the cached `OpenMOSS-Team/MOSS-TTS-Realtime` config and the port,
`docs/tts-funnel.md` §1.1–1.4):

- **The backbone has no text lm_head.** Its only output is
  `last_hidden_state`. MOSS already treats Qwen3 as a *state compiler*,
  not a language generator — there is no head to preserve when it is cut,
  and no output distribution to match.
- **No backbone↔depth projection** — hidden sizes match by design (2048).
  Substitution at that seam is writing a different vector, not adapting
  an interface.
- **The depth transformer holds no history**: fresh 16-slot cache per
  frame, micro-step 0 consuming the raw backbone hidden. Voice identity,
  prosody, timing, text progress and turn history all live in the
  **backbone KV**, not in the 4-layer stack. Anything that replaces the
  backbone inherits that whole job.
- **Conditioning composes additively at the input**: one step is `[T,17]`
  (col 0 text, cols 1–16 the previous frame's codes) summed over 17
  embedding tables — plain sum, no projection, no scaling. That is a
  second seam, and one the model was *trained* to use as a sum.
- **Cloning is a splice, not an encoder**: reference codec tokens are
  written into channels 1–16 of the `<|audio_pad|>` prefill rows. Any
  rung that bypasses the backbone must answer where the reference voice
  now lives.
- Param split 2.332 B = backbone 1.721 B (incl. 311 MB dead duplicate
  embed) + depth 266.5 M + embedding tables 344.8 M.

- **FUSE-0 — token pipe.** LLM-generated ids feed MOSS's text channel
  directly (MOSS has no text head; text is already an input stream, and
  the 12-token lead maps onto a generated-token queue naturally). Gate:
  identical speech tokens to the encode(decode(ids)) round trip.
  Stated precisely: *zero-copy token-domain forwarding when producer
  and consumer domains happen to be compatible* — not "speech fusion
  requires a Qwen LLM". Mostly an engineering cleanup; the value is the
  primitive it installs: token-domain piping between models.
- **FUSE-1A — representation study** (revised 2026-08-10 after the weight
  diff). The original form — one text prefix through a generic Qwen LLM
  and the MOSS backbone, hiddens compared layer-by-layer — is too weak
  now that the two are known to be independently trained. There is no
  "where Qwen becomes MOSS" curve to find, because it never was Qwen.
  The question is not *are these the same space* (answered: no) but **is
  there a low-complexity map between them**, which unrelated weights do
  *not* settle either way: independently trained networks can converge on
  related representational geometry.

  Fit, per layer:

  ```text
  B_L : H_upstream → H_moss,L        for L = 0, 4, ..., 27
  ```

  **Name them `B_L`, never `B_in`.** A map onto a layer-L hidden is not a
  map into `embed_tokens` space, and cannot be reused as one — the two
  live in different spaces. FUSE-1.5 has to learn its own injector; this
  rung does not hand it one.

  Apparatus decisions, fixed 2026-08-10:
  - **Upstream = `Qwen3-1.7B-Base`, labelled surrogate producer S0.** Not
    a claim about Jarvis's brain. It earns the slot by having the same
    2048 hidden width (no dimensionality confound), the same tokenizer
    vocabulary (position/token alignment is trivial), being already
    local, and — per the weight diff — being provably independent of
    MOSS, so a positive result is genuinely cross-model. A negative
    result falsifies *this pair*, not K3→MOSS.
  - **One voice, fixed reference conditioning, greedy / teacher-forced.**
    Voice is a nuisance variable until a mapping exists; held-out voices
    come later, not now.
  - **Corpus ~256–384 utterances × ~20–40 aligned positions ≈ 5k–15k
    paired states.** Design matters more than size: statements,
    questions, commands, numbers, names, short and long sentences,
    negation, uncertainty, sentence beginnings and endings, the EOS
    neighbourhood, and the same lexical token recurring in different
    contexts.
  - **Split whole semantic families 70/15/15 with no utterance or
    template leakage.** At 2048 dimensions a flexible map looks brilliant
    by memorisation if train and test correlate; a full 2048×2048 map is
    >4 M coefficients and will produce convincing nonsense on correlated
    token states. That split is the experiment's integrity, not a detail.
  - **Ridge and low-rank first — rank 32 / 64 / 128 / 256 — before any
    full linear map.** The deliverable is the curve *bridge complexity →
    held-out predictability → behavioural recovery*, not maximum training
    R².

  Measure linear CKA, SVCCA and linear predictability. Keep raw cosine as
  a free sanity column but do **not** interpret it: orthogonal weights do
  not imply orthogonal activations, so a low reading proves nothing
  either way. The question in one line: *is any part of the MOSS
  speech-state manifold linearly recoverable from an independently
  trained language model, on held-out semantics?* Worth knowing
  regardless of what FUSE-1.5 does.
- **FUSE-1B — behavioural substitution.** Take the best same-space `B_L`
  from 1A and actually run it: substitute `B_L(H_S0)` for MOSS's own
  layer-L hidden and measure the trajectory against the step-0 oracle
  with FUSE-2.5's scoring. Similarity scores are diagnostics; **behaviour
  is the oracle.** Expect the L27 variant to fail hard — FUSE-2.5 showed
  the seam tolerates almost nothing — which is exactly why the earlier
  layers are the interesting ones here.
- **FUSE-1.5 — latent injection at the additive input seam** (reframed
  2026-08-10 from "semantic term substitution"). Do **not** replace the
  native text embedding. Add to it:

  ```text
  E_moss(token_t)  +  α · B(H_upstream)  +  Σ E_moss_audio(codes_{t−1})
                              ↓
                   full 28-layer MOSS backbone
                              ↓
                            depth
  ```

  with α swept upward from 0. The reason to prefer this over replacement
  is that **α = 0 is bit-exactly the native system**, so the experiment
  carries its own control and can never silently drift away from a
  working baseline. The native embedding keeps supplying the lexical
  anchor the model was trained on; the mapped upstream state only *adds*
  contextual/semantic information. The model's familiar input
  distribution is never removed.

  Note what this makes redundant: if the upstream model emits token ids
  MOSS understands, the native text contribution is already obtainable as
  `token_id → embed_tokens.0 → 2048-d`. That is FUSE-0, not FUSE-1.5.
  Feeding a *stock Qwen3* embedding into this seam is now only a
  deliberately bad null — the weight diff showed the two embedding tables
  are orthogonal — worth running once because it is cheap and decisive,
  not as the main line.

  **`B_in` needs a behavioural target, and the input seam has no useful
  native one.** Regressing `B_in(H_upstream) ≈ E_moss(token_t)` would
  teach it to recover lexical identity — which FUSE-0 already supplies
  exactly — while discarding the contextual information that is the only
  reason to inject. So `B_in` is learned **teacher/student, through the
  frozen MOSS backbone**, against pairs where the spoken tokens are
  identical and the context differs:

  ```text
  TEACHER   context (system prompt / prior turn, via KV continuation)
            + utterance  →  native full MOSS  →  H27 / depth logits / RVQ

  STUDENT   utterance only, no context, plus
            E_moss(token) + α·B_in(H_upstream(context, utterance))
                         + Σ E_audio(prev codes)
            →  frozen MOSS backbone  →  H27 / depth logits / RVQ
  ```

  Train the smallest `B_in` that moves the student's speech state toward
  the teacher's. Losses in escalating order: (1) H27 reconstruction,
  (2) depth-logit KL, (3) teacher-forced RVQ agreement, (4) free-running
  trajectory and EOS, (5) perceptual/prosody. If a rank-64 map reaches
  anything interesting, that is the result. This tests the real
  proposition — *can semantic/contextual information cross from another
  model into MOSS without being serialised through the spoken text?* —
  instead of merely asking whether added noise fails to destroy speech.

  **Gate 0, run before building any corpus: does the teacher signal even
  exist?** MOSS's text channel is what gets *spoken*, so context must
  enter via the system prompt or a prior turn (§1.4 — multi-turn is KV
  continuation, not re-prefill). Whether MOSS's speech state actually
  moves with that context, for identical spoken tokens, is untested. Take
  one utterance under two contrasting contexts and measure H27 delta,
  depth-logit KL and RVQ divergence. **If the divergence is ~0 there is no
  teacher signal, `B_in` has nothing to learn, and the whole rung is void
  — so this check costs one afternoon and can save a corpus.**

  Then sweep α from 0 and measure where injected semantics start to move
  backbone hidden, depth logits, RVQ trajectory, prosody, EOS, delivery.
  Two traps that are part of the experiment: the **12-token text lead**
  means the state injected at audio position *t* corresponds to text
  token *t+12*; and an LLM final hidden is not distributed like an
  embedding lookup, so fitting scale/whitening to the MOSS
  text-embedding distribution is a control, not a cheat. Inject on decode
  positions only and leave prefill bit-exact — prefill carries the voice
  splice, and perturbing it confounds identity with semantics.

  Gate: the student's speech state moves measurably toward the teacher's
  on held-out semantics, with speech intelligible and the reference voice
  intact. That would be the first genuinely important FUSE result — same
  spoken text, extra meaning arriving only through another model's latent
  state, MOSS's behaviour changing accordingly — rather than another
  representation-similarity number.
- **FUSE-2 — direct residual substitution.** Replace a MOSS final
  backbone hidden with a shape-compatible LLM residual; run the proven
  depth transformer. Ask only: plausible codebooks? terminates? how far
  do logits move? Cheap falsification — MOSS's backbone state carries
  semantics + acoustic history + previous frame + conversation KV,
  while an LLM residual carries semantics + LLM state, so straight
  substitution *should* fail informatively.
- **FUSE-2.5 — backbone depth necessity.** Inference-only, no training,
  scored by the existing exact oracle (the 138-frame greedy dump). After
  a bit-exact MOSS prefill that establishes the full 28-layer KV, run a
  **top-truncated** backbone during decode — L0..L27 (reference),
  L0..L23, L0..L19, L0..L15, L0..L11, L0..L7 — applying the backbone's
  final norm to the truncated residual (the depth stack was trained on a
  post-norm hidden; note it as a choice and A/B it if the frontier looks
  strange) and handing that to the unchanged depth transformer.
  Truncate from the **top**, never the bottom: layer *k* expects a
  residual built by 0..*k*−1 at the current position, so "start at layer
  20" is not a defined operation while "stop after layer 20" is. Layers
  L..27 simply stop accruing KV during decode; since they are never run
  again under that condition the run stays self-consistent — but that
  also makes truncation a whole-utterance condition, not a per-frame
  switch. This is the only rung that returns a *map* rather than a
  verdict, which is why the graded dial is the point.
  Score per truncation, never on audio alone: max |Δ| and cosine at the
  2048-d seam, per-codebook logit KL, RVQ argmax change count, **first
  divergent frame and first divergent codebook within that frame**,
  whole-138×16 exact/not, EOS frame — perceptual audio last. The
  first-divergence frontier is the deliverable: where speech authority
  becomes established through depth, and how much late-backbone
  computation is operationally redundant for the observed trajectory.
  Gate: a roughly monotone frontier with an identified knee. Follow-ons
  if redundancy shows: finer boundary search around the knee, and the
  mid-band variant (`L0..Lk` → held representation → depth) that the
  residual-stream work predicts, since authority may become established
  well before layer 27.
- **FUSE-3 — small bridge.** `H_llm → projection P → depth stage`,
  acoustic conditioning preserved separately. The thesis test: how
  little MOSS backbone computation is required once semantic state
  already exists upstream?
- **FUSE-4 — acoustic residual injection.**
  `H_llm + A(previous audio tokens) → P → speech decoder`. If this
  works, MOSS's backbone has been decomposed into semantic and
  acoustic operands — and the steady-state frame stops paying for a
  second full language-model pass.

**What FUSE buys, stated honestly — it is not realtime.** The measured
CPU baseline is p50 190 ms/frame = backbone 65 ms + depth 125 ms
(`docs/tts-funnel.md`, step-4 log): the backbone being amputated is ~34%
of steady state, and depth dominates because 16 micro-steps re-read
1.06 GB of weights (~17 GB/frame vs the backbone's 6.8). **Q4 solves
realtime; FUSE solves duplication** — independent wins, and conflating
them will make FUSE look like a failed optimisation. What FUSE actually
buys is ~1.7 B fewer resident parameters and no second semantic
transformer in an always-on Jarvis: bytes that return to brain
residency, KV, ears, codec, hot expert cache, longer context. And if the
upstream LLM incurs the semantic compute anyway, the *incremental*
semantic cost of speech approaches zero.

**The standing constraint on FUSE-3/4.** Bypassing the backbone deletes
the mechanism that currently carries voice identity — the reference
splice conditions the backbone, and nothing downstream of it holds
state. So the end state is three operands, not one:

```text
Jarvis semantic state ─────────┐
voice / reference acoustic ────┼─► speech state adapter ─► depth
previous-frame RVQ state ──────┘
```

not `H_llm → linear → depth`. That is where the voice ladder
(`chris-experiments/voice/V0_PLAN.md`, V4–V7 and the `I_voice` endpoint)
stops being an adjacent research line and becomes a dependency of this
one.

Prior art, tracked honestly: PRIME-Speech (HF 2606.30944) already
drives a causal speech decoder from intermediate hidden states of a
frozen LM — one specifically *trained* architecture. TADA (arXiv
2602.23068) aligns text/acoustic representations. LARQL's differentiated
claim is the **runtime composition primitive**: arbitrary producer →
declared interface → binding → arbitrary consumer, across models that
were never trained together. K3 → SpeechBinding → MOSS (or a small fast
planner → speech model) is the long-term Jarvis pipeline this enables.
