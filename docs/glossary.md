# Glossary

Plain-language definitions of the terms used in LARQL's docs. Each links to
where the idea is covered in depth.

**LARQL**: The runtime and toolkit: the `larql` command, the server, LQL and
the Rust crates behind them.

**VINDEX3**: LARQL's model file format. A VINDEX3 file (a directory) holds a
model's weights *and* a description of its structure: which tensors exist,
what each one is for, and which operations use them. That description is what
lets LARQL inspect a model as well as run it. Created with
`larql vindex3 encode`. [What is VINDEX3?](vindex3/what-is-vindex3.md)

**VINDEX2 / `.vindex`**: The previous format. These files still open and
run, but new features target VINDEX3.

**vindex**: Short for either format.

**Encode**: Convert a Hugging Face model into a VINDEX3 file.

**Plan / admission**: The check `encode` runs first. Every part of the model
must be something LARQL understands; if not, encoding is *refused* with a list
of findings instead of producing a file that quietly runs wrong. Run it alone
with `larql vindex3 plan`.

**Component**: One model inside a VINDEX3 file. Most files hold one, called
`target`; multimodal models can hold several (for example a vision encoder and
a text decoder).

**Backend**: What does the arithmetic: CPU, or Metal on Apple GPUs. In
`vindex3` commands, `--backend production` means LARQL's optimised kernels and
`--backend reference` a deliberately simple implementation used to check them.

**Representation**: How a weight is stored: BF16, 4-bit NVFP4, Q4_K and so
on. One model can carry several representations, and **REPRESENT** is the
tool that compiles new ones (for example a smaller 4-bit copy).
[Representation](vindex3/representation.md)

**Measure / token bank**: How LARQL checks what a representation costs in
quality: run the original and the compressed model over the same fixed set of
tokens (the *bank*) and compare.

**LQL**: Lazarus Query Language, a SQL-like language for querying and editing
a model (`USE`, `INFER`, `DESCRIBE`, `INSERT`…). Run it with `larql repl` or
`larql lql '<statement>'`. [LQL guide](lql-guide.md)

**Patch (`.vlp`)**: A file of edits recorded by LQL. Patches sit on top of a
model and never modify the base file; `COMPILE` bakes them into a new one.

**Observe / run record**: `larql vindex3 observe` runs a prompt and writes a
record (`.jsonl`) of what each layer wrote into the model's internal state.
[Observation and intervention](vindex3/observation-and-intervention.md)

**Residual stream / carrier**: The running internal state a transformer passes
from layer to layer. Each layer reads it and adds to it. LARQL's docs call
each of these additions a *carrier write*.

**Lens**: A way to read the internal state mid-model by projecting it through
the model's own output head ("what would it predict if it stopped here?").

**Intervention**: Changing the internal state during a run (zeroing,
replacing or scaling part of it) to test what causes what. Observing shows
correlation; intervening tests causation.

**Observatory**: A browser tool for exploring run records.
[Observatory](../observatory/README.md)

**Doc labels (CURRENT, NORMATIVE, RECORD, ARCHIVE)**: Labels at the top of many
docs. *CURRENT* explains how things work now; *NORMATIVE* is a versioned
contract; *RECORD* is a dated research result; *ARCHIVE* is kept for history.
[Documentation policy](documentation-policy.md)

**Frozen / pre-registered**: A research protocol written and locked *before*
the experiment runs, so results can't be reinterpreted afterwards. Most of
the `gw-*`, `v3-*` and `*-1` docs are these.
