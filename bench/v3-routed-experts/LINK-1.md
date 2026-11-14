# DEC-V3-1: exact routed experts behind a controlled link

Status: **draft protocol; instrument built but NOT calibrated; no timed model
run performed.** Seven calibration runs on 2026-11-14 all failed
([receipts](link-results/calibration/index.json)); the sweep refuses to start
without two passing ones.
Correctness smoke on main `7b5334dd`: `one` and `experts` layouts, direct and
through a 5 ms / 1 Gbit/s relay, all reproduce the local control IDs.
Timed runs wait for an authorized quiet window, as PROFILE-1 and LAN-1 do.

This is the VINDEX3 form of the DEC funnel's DEC-1A transport decomposition
([`docs/dec-funnel.md`](../../docs/dec-funnel.md) §3). It reuses that
experiment's question and method. It does not reuse its runtime: the legacy
grid, WalkFFN endpoints and `dec-bench` replay are not involved, and no
legacy DEC number is a control, a prior or a prediction for this one.

## Question

On the V3 routed expert provider, how does decode time per position depend on
link round-trip time and link rate, and do the two effects compose?

```text
delta = a × crossings × RTT  +  b × critical_bits / rate  +  c
```

`delta` is a shaped arm's position time minus its unshaped relay control.
`crossings` is the number of sequential fan-outs a position waits for (one per
routed layer). `critical_bits` is, per fan-out, the request plus response bits
of the largest shard exchange; shards are contacted in parallel on separate
links, so only the slowest exchange is on the critical path. An exactly
additive link gives `a = b = 1`, `c = 0`.

## What is held fixed

PROFILE-1's exact contract, unchanged: GPT-OSS 20B, CPU coordinator and CPU
workers, binary HTTP, bind-once authority, exact F32 `VEX1`/`VEY1` frames,
local routing, production-order reduction, the frozen `p3` prompt text, 64
generated tokens, 55 measured decode positions. Every trial must reproduce
the saved uninstrumented token IDs or the run stops. No carrier quantization,
batching, prediction, stream transport or Metal coordinator is in scope.

The workload is **not** PROFILE-1's token sequence. Since `c820fa18`
(2026-11-06) `larql run` on VINDEX3 renders the chat template by default, and
`--raw` prepends the container's declared BOS. This harness passes `--raw`, so
the prompt is 152 tokens (BOS plus PROFILE-1's 151) and the greedy
continuation differs: 41 distinct IDs in 64, against PROFILE-1's six. The
control is [`link-results/control/ids.json`](link-results/control/ids.json):
a local, uninstrumented run of the same text on main `7b5334dd`, identical
across two fresh processes. PROFILE-1 timings are therefore not a matched
baseline for this sweep.

## The link

[`v3_link_shaper`](../../crates/larql-server/examples/v3_link_shaper.rs) is a
TCP relay placed in front of each worker. Each direction is an independent
link: a chunk starts when the link is free, takes `bits / rate` to serialize,
then arrives one propagation delay later. Round-trip time is split evenly
between the two directions. Each worker has its own relay, hence its own link.

It models delay and rate only: no loss, jitter, reordering or congestion
control, and it is a userspace relay on loopback, not a kernel shaper or a
physical network. Results are therefore **simulated-link** evidence and stay
separate from loopback (PROFILE-1) and physical-LAN (LAN-1) evidence.

Each connection is a single polling loop over both sockets, with no thread
hand-off and no blocking read, at the cost of one busy core per connection.
An earlier threaded relay charged its own thread wake-ups to the link, and
differently at zero delay than under a delay.

Two things keep the instrument honest:

- **Calibration** (`link_sweep.py --calibrate`, no model) drives the relay
  with [`v3_link_probe`](../../crates/larql-server/examples/v3_link_probe.rs),
  an echo peer and client that poll their sockets, using the one-worker
  GPT-OSS frame sizes. Every sweep point is measured between two zero-delay
  relay baselines. Both baselines, their drift and the error against each are
  recorded; a point passes only if it is within tolerance of **both**, so a
  drifting baseline fails the point and is never averaged into a pass.
  Tolerance per exchange is the larger of 5% of the configured cost and
  50 µs, and has not changed since the first run. `--probe python` selects a
  blocking Python probe as a diagnostic comparison; it is never the authority.
- **Per-trial lateness.** Each relay records how late it released every
  chunk. A trial whose summed lateness per position exceeds the timing gate
  is void. This is the relay's own clock: it cannot see delay added on the
  receiving side, which is why it does not replace calibration.

`--execute` requires two `--calibration` receipts: complete, passing, native,
and recorded for the relay binary about to be used.

The polling probe and relay each occupy cores. Calibration records the
relay's lateness under the polling probe; comparing it with the same figure
under `--probe python` is the check that the probe is not starving the relay.

## Sweep

| Kind | Points | Role |
|---|---|---|
| `floor` | relay, zero delay, unlimited rate | Control for every other arm |
| `direct` | no relay | Prices the relay itself against `floor` |
| `axis` | RTT 0.2, 1, 5, 20 ms at unlimited rate | Identifies `a` |
| `axis` | 1 Gbit/s, 100 Mbit/s at zero delay | Identifies `b` |
| `joint` | (1 ms, 100 Mbit/s), (5 ms, 1 Gbit/s), (20 ms, 100 Mbit/s) | Held out |

The model is fitted on valid `axis` points only. `joint` points are never
fitted; their prediction error is the composition test.

DEC-1A's other axes do not exist on V3 yet and are not simulated: there is
one carrier (F32), one dispatch shape (per-layer fan-out grouped by owner)
and one stream. RTT 0.05 ms and rates of 10 Gbit/s and above are omitted
because their predicted effect (about 1.2 ms and under 1 ms per position)
is at or below the 1% gate on a position of roughly 117 ms; the sweep cannot
resolve them and a null there would not be evidence.

## Run design

Workers start once and stay resident. Every trial is a fresh coordinator
process. Warm-up trials run at `floor` until two consecutive means agree
within 1% (at most four). The schedule then interleaves a `floor` after every
two points; the second repeat runs the points in reverse order so drift is
not aliased to RTT.

A point is **valid** only if the floors on either side of it agree within 1%
and its relays stayed on schedule. Void points are kept as individual
diagnostics and are never fitted or pooled. Local execution is not an arm
here: the local-versus-remote delta is PROFILE-1's quantity and needs the
worker banks unloaded.

`one` (one worker owning every expert) is the first topology. `experts` and
`mixed` are separate runs with their own floors.

```bash
cargo build --release -p larql-cli -p larql-server --no-default-features
cargo build --release -p larql-server --no-default-features --example v3_link_shaper
cargo build --release -p larql-server --no-default-features --example v3_link_probe
python3 bench/v3-routed-experts/link_sweep.py --calibrate --out /path/to/calibration-a
python3 bench/v3-routed-experts/link_sweep.py --calibrate --out /path/to/calibration-b
python3 bench/v3-routed-experts/link_sweep.py --execute \
  --model /path/to/gpt-oss-20b.vindex3 --out /path/to/new-sweep \
  --calibration /path/to/calibration-a/calibration.json \
  --calibration /path/to/calibration-b/calibration.json \
  --topology one --peer-handshake /path/to/peer-acknowledgements.txt
```

## Reported

Per point: position time, positions per second, delta against its floors,
FFN-boundary delta, crossings, body bytes, critical-path bits, PROFILE-1's
nested worker and transport timings, relay lateness, validity. Per run: fitted
coefficients, residuals, R², held-out prediction error as a fraction of
position time, the relay floor against `direct`, void points and why.

## Limits stated in advance

- Single stream, single client, CPU coordinator. This says nothing about
  shared-tier capacity, speculative width or a Metal coordinator.
- One prompt and one greedy continuation. Results apply to this workload.
- Workers and coordinator share one host, so worker compute is not isolated
  from coordinator load the way separate machines would isolate it.
- A fitted coefficient near 1 shows the provider waits on the link the way
  the model says. It does not show what a real network of that RTT would do.
