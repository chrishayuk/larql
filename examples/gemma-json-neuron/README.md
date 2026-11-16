# An agent-written patch: Gemma 3 4B stops fencing raw JSON

`agent-neuron.vlp` is a weight patch written by an AI agent (Claude Opus 5.5) working through LARQL. Asked for a
raw JSON object with "no code fence", Gemma 3 4B IT still wraps it in a Markdown ```` ```json ```` fence. The agent
repurposed one feed-forward neuron in the last layer (layer 33, neuron 3608) with three `ADD LOW RANK` operations:

- **gate** (row 3608 of `33.mlp.gate_proj.weight`): fires only at the first position of the model's reply;
- **up** (row 3608 of `33.mlp.up_proj.weight`): positive when the user asked for raw output, negative when they
  asked for a code block;
- **down** (column 3608 of `33.mlp.down_proj.weight`): pushes the first token towards `{` and away from the fence.
  When a code block was asked for, the sign flips and it pushes towards the fence.

There is no training: the patch is a direct change to 7,676 stored weights.

| File | What it is |
| --- | --- |
| `agent-neuron.vlp` | The patch, sha256 `97e7ecf26561929c83ddbb50d48b6d1542057419d6ba91868c8eca64cfb31ff4` |
| `coffee-infer.lql` | The coffee prompt as one LQL `INFER` statement, in Gemma's chat format |
| `coffee-chat.txt` | The same prompt for llama.cpp |

## What it is bound to

The patch's weight operations are bound to the container they were written against: `larql/gemma-3-4b-it` at
revision `91c35bfa9cd2c6d1dcad65b46f60a58eabb9cc1f`, as pulled by `larql pull gemma-3-4b-it`. LARQL refuses to
apply them to any other container. The outputs below were checked with LARQL v0.3.0 and
`LARQL_CPU_MAX_FORMAT=bf16`, so the stored BF16 weights run exactly.

## Try it

From the repository root, after installing LARQL (see the main README):

```bash
export LARQL_CPU_MAX_FORMAT=bf16
larql pull gemma-3-4b-it
larql repl
```

```sql
USE "gemma-3-4b-it";
APPLY PATCH "examples/gemma-json-neuron/agent-neuron.vlp";
-- paste the line from coffee-infer.lql: the reply is raw JSON
REMOVE PATCH "examples/gemma-json-neuron/agent-neuron.vlp";
-- paste it again: the fence is back
```

Expected with the patch applied:

```
{
  "item": "coffee",
  "count": 1
}
```

and with it removed:

````
```json
{
  "item": "coffee",
  "count": 1
}
```
````

### Compile it into a model and run it in llama.cpp

```sql
APPLY PATCH "examples/gemma-json-neuron/agent-neuron.vlp";
COMPILE CURRENT INTO VINDEX "gemma-json.vindex3";
exit
```

```bash
larql convert vindex-to-gguf gemma-json.vindex3 gemma-json.gguf --text-only
llama-completion -m gemma-json.gguf -p "$(cat examples/gemma-json-neuron/coffee-chat.txt)" \
  -n 160 --temp 0 --top-k 1 -no-cnv --no-display-prompt
```

The compile reports `weight patch of 7676 element(s) of 33.mlp.gate_proj.weight, 33.mlp.up_proj.weight,
33.mlp.down_proj.weight`. llama.cpp (checked with b9430) prints the raw JSON reply followed by `[end of text]`.

## Results, and their limits

The agent worked on six development prompts (four asking for raw JSON, two asking for a fence) plus probe prompts
of its own, with a budget of 30 runs of the edited model. Before running its first version of the patch it predicted
all six replies, and all six matched byte for byte. This file is its revision, with a slightly stronger push for
more margin on unseen prompts: again 6 of 6, identical. It used 14 of its 30 runs.

It was then scored on 48 prompts written separately and sealed before the run, opened only after it had nominated
this patch:

| Sealed prompts | Unedited | With the patch |
| --- | --- | --- |
| Asking for raw JSON (24) | 3 pass | 16 pass |
| Asking for fenced JSON or a code block (24) | | every reply byte-identical to the unedited model |

- It is **not** a general fix for Gemma's JSON: 8 of the 24 raw-JSON prompts still fail. The agent's own
  nomination says prompts phrased unlike its development examples may not flip.
- Neuron 3608 was chosen because it was quiet on the text the agent captured. That doesn't mean it is unused
  everywhere in Gemma.
- After compiling and exporting to GGUF, llama.cpp matched LARQL's generated tokens on 53 of the 54 prompts above.
  The one difference is a near-tied digit in a reply body (34.84% vs 34.81% in LARQL), not at the edit.
