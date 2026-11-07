# Getting started

This guide takes you from nothing to a running, served and inspected model in
about ten minutes. Most of that is downloading the model. Every command and
output below was run on an Apple Silicon Mac with the commands shown.

You need:

- macOS (Apple Silicon), Linux (x86_64) or Windows (x86_64)
- About 3 GB of free disk space and 4 GB of RAM for the starter model
- An internet connection to Hugging Face

## 1. Install

```bash
# macOS (Apple Silicon)
curl -L https://github.com/chrishayuk/larql/releases/latest/download/larql-aarch64-apple-darwin.tar.gz | tar xz
sudo mv larql-aarch64-apple-darwin/larql* /usr/local/bin/

# Linux (x86_64)
curl -L https://github.com/chrishayuk/larql/releases/latest/download/larql-x86_64-unknown-linux-gnu.tar.gz | tar xz
sudo mv larql-x86_64-unknown-linux-gnu/larql* /usr/local/bin/
```

On Windows, download `larql-x86_64-pc-windows-msvc.zip` from the
[latest release](https://github.com/chrishayuk/larql/releases/latest) and put
both executables on your `PATH`.

Check it worked:

```console
$ larql --version
larql 0.1.0
```

To build from source instead, see the [README](../README.md#install).

## 2. Run a model

Give `larql run` a Hugging Face model and a prompt:

```console
$ larql run hf://Qwen/Qwen3-0.6B "What is the capital of France? Answer in one word. /no_think"
Qwen/Qwen3-0.6B is a Hugging Face checkpoint, not a vindex; planning a VINDEX3 encode (first run only)…
  commit c1899de289a04d12100db370d81485cdf75e47ca
  staged 11.47 MB of headers and metadata; planning…
  plan admissible for text generation; fetching tensor payloads…
  fetched 1.50 GB (1.40 GiB) of 1.50 GB (1.40 GiB) declared across 311 tensor(s)
  encoded 4 representation(s), 1.50 GB (1.40 GiB) payload
  cached at ~/.cache/larql/encoded/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
  next time `larql run Qwen3-0.6B` runs it with no encode
<think>

</think>

Paris
```

The first run checks the model, downloads it and converts it to VINDEX3,
LARQL's model format (a few minutes for this 1.5 GB model). From then on it
starts immediately and works offline, under its short name:

```bash
larql run Qwen3-0.6B "Write a haiku about Rust. /no_think"
larql chat Qwen3-0.6B        # interactive; remembers the conversation
```

(`/no_think` is a Qwen3 convention that skips its reasoning step.)

Prompts go through the model's own chat template, so instruct models answer
them. For plain text completion, add `--raw`:

```console
$ larql run Qwen3-0.6B "The capital of France is" -n 12 --raw
 Paris. The capital of Italy is Rome. The capital of
```

`-n` sets how many tokens to generate.

**Or pull a ready-made model.** Some models are published already converted,
so there's nothing to encode:

```bash
larql pull granite-4.1-3b       # 6.8 GB
larql run  granite-4.1-3b "Write a haiku about Rust."
```

**Manage your models:**

```console
$ larql list
SOURCE    MODEL                 GEN   SIZE (MB)   LAYERS    HIDDEN  STATUS
encoded   Qwen/Qwen3-0.6B       v3       1514.7       28      1024  ready
hf        larql/granite-4.1-3b  v3       6812.9       40      2560  ready

$ larql show Qwen3-0.6B         # details, including where it lives on disk
$ larql rm Qwen3-0.6B           # delete it
```

## 3. Serve it

`larql serve` starts an HTTP server with OpenAI-compatible endpoints, so
existing clients and SDKs can use it.

```bash
larql serve Qwen3-0.6B --port 8080
```

In another terminal:

```console
$ curl -s http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Write a haiku about Rust. /no_think"}]}'
{"choices":[{"message":{"role":"assistant","content":"<think>\n\n</think>\n\nRust's flame ignites the dark,  \nA scar on the edge of the sky.  \nThe world burns, but it stays."},"finish_reason":"stop", ...}], ...}
```

`"model"` accepts the name you served it under or the model's own ID, and can
be left out when one model is loaded.

| Endpoint | What it does |
|---|---|
| `GET /v1/models` | The loaded model, its `id` and any `aliases` |
| `POST /v1/completions` | Plain text completion |
| `POST /v1/chat/completions` | Chat, with the model's chat template applied |
| `GET /swagger-ui` | Interactive API documentation |

The [runtime guide](vindex3-runtime.md) covers the full API, multi-model
serving and sessions.

## 4. Benchmark it

```console
$ larql bench Qwen3-0.6B
  Backend                     prefill       mean        p50      tok/s  steps
  vindex3-metal                399.2ms     34.40ms     33.48ms      29.1      50
```

Numbers depend on your machine and backend. The [CLI guide](cli.md#larql-bench)
covers backends and options.

## 5. Look inside

This is where LARQL differs from other runtimes. A VINDEX3 file describes the
model's structure as well as storing its weights, so you can query it.

These commands take the model's directory. `larql show` prints it:

```bash
M=$(larql show Qwen3-0.6B | awk '/^Path:/ {print $2}')
```

**Ask for the model's next-token probabilities** with LQL, LARQL's query
language:

```console
$ larql lql "USE \"$M\"; INFER \"The capital of France is\" TOP 5;"
Predictions (VINDEX3 program):
   1.  Paris               (74.36%)  [id 12095]
   2.  the                 (2.18%)  [id 279]
   3.  in                  (2.07%)  [id 304]
   4.  located             (1.34%)  [id 7407]
   5. ...                  (1.07%)  [id 1112]
```

**Look at its structure:**

```console
$ larql lql "USE \"$M\"; STATS;"
Model:           Qwen3-0.6B (VINDEX3)
Layers:          28
Attention:       0 sliding / 28 full (windows from the plan)
Continuation:    KV rows (kv_dim 1024), growing with the prefix
...
```

`SHOW LAYERS;` lists each layer's attention shape. `larql repl` opens an
interactive shell so you don't need to retype `USE` each time.

**Record a run.** `observe` runs a prompt and writes down what every layer
wrote into the model's internal state, along with the final prediction:

```console
$ larql vindex3 observe "$M" --backend production \
    --prompt "The capital of France is" --record run.jsonl
  ...
  final position, top 10:
       12095  logp -0.4184 " Paris"
        7407  logp -3.6079 " located"
         279  logp -3.7446 " the"
  ...
```

`run.jsonl` is a self-describing record you can analyse yourself or load into
the [Observatory](../observatory/README.md).
[Observation and intervention](vindex3/observation-and-intervention.md)
explains what the record contains and what it can and can't tell you.

## 6. Where next

| To… | Read |
|---|---|
| Learn LQL properly | [LQL guide](lql-guide.md) |
| Shrink the model to 4-bit and measure the quality cost | [Representation](vindex3/representation.md) |
| Edit what the model knows (experimental) | [LQL guide](lql-guide.md) — see the note below |
| Understand the file format | [What is VINDEX3?](vindex3/what-is-vindex3.md) |
| Look up a term | [Glossary](glossary.md) |

> **Knowledge edits are experimental.** `INSERT INTO EDGES` in a patch session
> does change the answer for the prompt you target, and the base model file is
> never modified. But today the edit can also take over *unrelated* prompts.
> Always check a control prompt before trusting an edit.

## Which models work?

`larql run hf://…` checks every part of the model against what it knows how
to run *before* downloading the weights. It only needs the parts used for
text generation; if one is missing, it refuses after reading the file
headers and lists what is unsupported:

```console
$ larql run hf://microsoft/Phi-4-mini-instruct "hi"
...
Error: cannot run hf://microsoft/Phi-4-mini-instruct@cfbefacb…: LARQL does not yet represent everything text generation needs (10 blocking finding(s)); only headers were read
  - [unrepresented] target: architecture_family — `phi3` matches no registered family, …
```

Results from 2026-11-06:

| Model | Status |
|---|---|
| `Qwen/Qwen3-0.6B` | **runs**: everything in this guide |
| `google/gemma-3-4b-it` (gated) | **runs** via `larql run hf://` (8.6 GB) |
| `granite-4.1-3b` (registry) | **runs** via `larql pull` |
| `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen2.5-1.5B-Instruct` | admitted, not yet run end to end |
| `meta-llama/Llama-3.2-1B-Instruct` (gated), `mistralai/Mistral-7B-Instruct-v0.3`, `openai/gpt-oss-20b` | admitted, not yet run end to end |
| `microsoft/Phi-4-mini-instruct`, `HuggingFaceTB/SmolLM2-1.7B-Instruct` | **refused** |

Gated models (Llama, Gemma) need a Hugging Face token: run
`hf auth login` or set `HF_TOKEN`.

`larql vindex3 plan hf://<model> --capability text-generation` prints the full
analysis as JSON and judges it the way `run` does; the last line is the
verdict. Without `--capability` it judges the *whole* model, including parts
text generation doesn't use (Gemma 3's vision tower, for example), so it can
refuse a model that `run` accepts. The [status page](vindex3/status.md) records supported
architectures and backends in more depth.

## Troubleshooting

**`system_graph.json is schema v5, this build reads v6 — re-encode the container`**
The file was written by an older LARQL. Encode it again from the source model
with `larql vindex3 encode`.

**`no cached vindex matches <name>`**
The short name isn't known yet. Run `larql list` to see what is, use the full
`hf://org/name` form the first time, or `larql link <dir>` to add a model
directory you encoded yourself. You can always pass a directory path instead
of a name.

**`STATUS unusable` in `larql list`**
The model can't be opened by this build; the note under the table says why.
An `incomplete download` names the `larql pull` that finishes it; otherwise
`larql rm` it and run it again.

**`cannot run hf://…: LARQL does not yet represent everything text generation needs`**
The model uses something LARQL doesn't support yet; each finding is listed.
Nothing large was downloaded.

**Picking up a newer version of a model**
A converted model is cached at the revision it was first run at, and `run`
never checks for a newer one. `larql pull <name>` (or `larql pull
hf://org/name`) asks the hub: if the revision moved it converts the new
version and removes the old one, otherwise it says `already up to date`.

**Older `.vindex` (VINDEX2) files**
These still open with `run`, `serve` and LQL. New work should use VINDEX3.
