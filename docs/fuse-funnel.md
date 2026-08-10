# The FUSE funnel — model-to-model fusion as a runtime primitive

Status: **no rung green yet.** FUSE-0 is engineering rather than
research; FUSE-1 is the first measurement. Opened 2026-08-09 as a
`ROADMAP.md` section, given its own funnel doc 2026-08-10 once the ladder
acquired per-rung gates. Speech is the first proving ground — the
abstraction is model-to-model fusion, not "Qwen token sharing".

Companion docs: [`tts-funnel.md`](tts-funnel.md) owns the MOSS port and
supplies this programme's exact oracle (the step-4 138-frame dump). The
voice ladder — `chris-experiments/voice/V0_PLAN.md`, outside this repo —
owns identity portability, which FUSE-3/4 depend on.

Gate log:

- *(empty — no rung has been run)*

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
- **FUSE-1 — residual comparison.** Same text prefix through a generic
  Qwen LLM and the MOSS backbone; compare hiddens layer-by-layer.
  Cosine is not enough (the voice ladder's lesson) — behavioural
  probes and linear mappings too.
- **FUSE-1.5 — semantic term substitution at the input seam.** Replace
  *only* the text-channel embedding contribution in the 17-way sum with a
  projected upstream semantic state; keep the native MOSS audio
  embeddings, the full 28-layer backbone, its KV and the depth stack
  untouched. This substitutes one term of an interface the checkpoint
  already composes additively, rather than inventing an alien one, and it
  isolates exactly one clause: *can an upstream model supply the semantic
  half of the state while MOSS retains speech continuity — voice,
  prosody, alignment, turn history?* Substitute on decode positions only
  and leave prefill bit-exact; prefill is where the voice splice lives,
  and perturbing it confounds identity with semantics. Two traps that are
  part of the experiment, not afterthoughts: the **12-token text lead**
  means the state substituted at audio position *t* corresponds to text
  token *t+12*; and an LLM *final hidden* is not distributed like an
  *embedding lookup*, so fitting scale/whitening to the MOSS
  text-embedding distribution is a control, not a cheat. Gate:
  intelligible speech with the reference voice preserved. Cheaper and
  less destructive than FUSE-2 — run it first.
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
