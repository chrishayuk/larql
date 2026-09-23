# GW-CAR-1A — captured-carrier compactness result

**Status:** CLOSED — valid captured-carrier compactness result. CAR-1B remains a separate prospective experiment.

**Population and interface:** The frozen 666-execution GW-V2 cohort at the
GW-STATE-1 context-128 interface. Only the entering L24 carrier varied. The
target-natural H1 intervention, seven donor-supplied non-H1 heads, donor map,
target continuation/KV, and proximal and terminal readouts stayed fixed.

**Frozen protocol:** `sha256:7829c7022acd14a2694700cf94099aaa0e292ff679a1701d64b393a5a0ea9510`.
**Bound executable:** `sha256:f25b603c04573472a0d7cf3110850e0d467458b4b724f014dd5712fa1937a38f`.
**Adjudication:** `sha256:2408b4869bddf5a76da06525e7119f1efd144bf48995f26ac78a6a0ff4e9bf96`.

The train-only PCA fit was frozen before candidate construction. The runner
replayed all 396 train rows and 270 held-out rows for 19 carrier candidates and
both H1 arms: 15,048 train and 10,260 held-out firings. Candidate 0 reproduced
the sealed STATE-1 context-128 proximal and terminal tensors bit for bit;
candidate 1 reproduced context 0. Both replay manifests report zero identity
bit mismatches. The adjudicator rechecked candidate payloads, artifact hashes,
geometry, frozen subjects, and the independent validation and test analyses.
The exact candidate cleared both held-out splits, so the registered analysis
returned `valid`.

| Candidate | Carrier payload | Shared model | Validation | Test |
| --- | ---: | ---: | --- | --- |
| Exact f32 control | 10,240 B/row | 0 | pass | pass |
| Matched-donor control | 10,240 B/row | 0 | fail | fail |
| f16 | 5,120 B/row | 0 | pass | pass |
| Symmetric q8 | 2,564 B/row | 0 | pass | pass |
| Symmetric q4 / q2 | 1,284 / 644 B/row | 0 | fail | pass |
| Sparse top-absolute 32, 64, 128, 256, 512, 1,024 | 192–6,144 B/row | 0 | pass | pass |
| PCA rank 8, 16, 32, 64, 128 | 32–512 B/row | 92,160–1,320,960 B | fail | fail |
| PCA rank 256 | 1,024 B/row | 2,631,680 B | fail | pass |
| PCA rank 384 | 1,536 B/row | 3,942,400 B | pass | pass |

The **smallest clearing payload in the registered grid** was sparse
top-absolute 32: 32 coordinate indices and f32 values, or **192 bytes per row**.
That is 1.875% of the exact carrier's 10,240-byte f32 payload, a 53.3-fold
payload reduction. At the registered 666-row amortization it costs 127,872
bytes with no shared model. The 32-coordinate candidate passed both proximal
retention and terminal-effect gates on validation and test. Its raw / z-scored
proximal point retentions were `1.602 / 1.544` on validation and
`1.927 / 1.837` on test; simultaneous lower bounds were
`0.968 / 1.151` and `1.100 / 1.541`. Its raw / z-scored terminal-effect
simultaneous lower bounds were `0.277 / 0.065` and `0.337 / 0.078`, all
positive. The values above one indicate that this substitution amplified the
registered proximal contrast; they do not establish faithful reconstruction
of the full natural carrier.

This is a **sufficiency under the frozen relational gate** and a measured
storage result for captured carriers. The registered grid does not determine
the minimum number of coordinates, nor does it show that all 32 are necessary.
Every candidate was derived from the target's already computed natural
carrier, and selecting the top coordinates also requires reading that carrier.
CAR-1A therefore supplies no cheap construction, latency, FLOP saving, or
graph-walk execution claim. CAR-1B must prospectively define and test a source
that avoids the natural prefix before such a claim is possible.

The [frozen protocol](../bench/gw-car-1/gemma3-4b-it-phase1/gwcar1a-protocol.json)
and [authoritative adjudication copy](../bench/gw-car-1/gemma3-4b-it-phase1/gwcar1a-adjudication.json)
retain the sealed analysis and full per-candidate values. The train-only fit,
candidate payloads, replay manifests, and tensors remain in the local ignored
`output/` tree; their hashes are recorded in the adjudication artifact.
