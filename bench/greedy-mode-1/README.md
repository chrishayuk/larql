# GREEDY-MODE-1

Why does gpt-oss-20b greedy decode sometimes read ~8.05 ms/token and
sometimes ~8.6 ms/token on the same binary? The verify-n revival table
(#729) showed both, and a greedy/spec/greedy bracket cannot catch the
difference when both greedy runs land in the same state.

Machine: M3 Max. Checkpoint: `gpt-oss-20b.nvfp4-head.vindex3`, backend
`metal-lowered`, greedy, `--generate 128`. Binary: release build of main
71b1a2d3 (sha256 prefix `163c1fe03f3ce2a9`; not committed).

## Experiment 1 (`run.sh`, `table.txt`, `conditions.txt`): 36/36 valid

Nine arms, four rounds, arm order rotated each round, every arm behind the
quiet gate (load <= 4, AC power, no compiler, no process > 20% CPU) and
voided by a watcher on a compiler or a sustained > 50% competitor.

| arms | what varies |
|---|---|
| `json_96` .. `json_256` | prompt length (suffixes of one json_edit prompt) |
| `prose_84`, `prose_256` | content (prose, and prose repeated to 256 tokens) |
| `json_256_util` | CPU QoS (`taskpolicy -c utility`) |

Findings:

- **Quiet state** is 8.00-8.12 ms/token at every length from 96 to 256
  tokens and for both contents (rounds 3-4). The ~0.1 ms slope from 96 to
  256 is the attention span. Length and content are not the cause.
- `taskpolicy -c utility` changes nothing. Host `encode` (0.6-2.5 ms) is
  overlapped and does not track step time.
- The **slow state** (8.56-8.63) is per process: 2 of 36 runs, both on
  256-token prompts (seq 13, 29). In the verify-n revival session it held
  for every prompt of 213+ tokens while interleaved prompts of 159 or fewer
  stayed fast.
- Cause **not found**.

## Experiment 2 (`run2.sh`, `prof/`): aborted at 1/40

Interleaved `json_224` / `json_256`, each plain and with `--profile`, ten
rounds, to see which stage (attn.proj, attn.core, attn.o_proj, ffn.routed,
head, gaps) carries the extra ~0.5 ms in the slow state. A peer job held
load at 8-17 for over three hours, so the quiet gate never cleared and the
run was stopped. The only datum is one plain `json_224` at 8.19.
`prof-void-1/` is an earlier attempt voided when its warm-up was
contaminated.

## Re-running

```sh
cd bench/greedy-mode-1
B=/path/to/release/larql ./run.sh    # experiment 1
B=/path/to/release/larql ./run2.sh   # experiment 2, writes prof/
```

Both scripts expect the checkpoint at
`~/chris-models/gpt-oss-20b.nvfp4-head.vindex3` and an idle machine.
