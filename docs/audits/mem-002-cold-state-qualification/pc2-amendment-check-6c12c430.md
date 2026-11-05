> Preserved verbatim. Independent diff-only check of `487c4707..6c12c430` (PC2 amendment, draft PR #670), 2026-11-01. Verdict: FAIL (B1, B2).

# PR #670 PC2 amendment check: diff-only review of 487c4707..6c12c430

Scope: `git diff 487c4707 6c12c430 -- docs/mem-002-cold-state-qualification.md`
(+44 lines, new §4.1, lines 98-140 of the revised file). The full revised file
was read for consistency, along with `34ec3df9:docs/mem-002-i1-i3-proposal.md`
§3.2, §3.4 and §5 (#671). Read-only: nothing was built, run or edited.

**Verdict: FAIL.** There are two BLOCKING findings. Both are small text fixes.

## (a) Requirements

| Req | Implemented? | Location | Notes |
|---|---|---|---|
| 1. Fresh predeclared ranges per pair, OR a verified reset | **yes** | §4.1 l.105-112 | All ranges are declared before the first pair, and each is disjoint from earlier ranges and their readahead. A reset counts only if the next eligibility observation witnesses it ("merely performed does not count"). |
| 2. Joint eligibility before either arm; never per arm | **yes** | §4.1 l.113-121 | One eligibility observation (fresh process, `after_run`) runs before the pair. Both arms use the same set, and a target near fully resident in either arm's baseline is excluded from **both**. A pair is void below a count fixed at the freeze. |
| 3. Baseline is the after-run record | **yes** | §4.1 l.122-130 | Baseline = A `after_run`, measurement = B `binding`, Δ = B(binding) − A(after_run) per region (#671 §3.2). The A→B gap is recorded and matched between arms. |
| 4. Carry-over between pairs accounted for | **between pairs: yes; within a pair: no, see B2** | §4.1 l.131-137 | The arm order is counterbalanced, and each eligibility observation records earlier pairs' targets as a diagnostic. |
| N2: state which observation is the baseline | yes | l.123-124 | |
| N2: non-target criterion on increases, since a read can evict | yes, but it conflicts with the unchanged text (B1) | l.138-140 | |

## (b) Consistency

**B1 (BLOCKING): the non-target pass condition is now stated two incompatible ways.**
- §4 PC2 row (l.91, unchanged) says other regions must show "no treatment-against-sham difference". #671 §5 step 5 says the same: "non-target, non-readahead, non-boundary regions show no treatment-against-sham difference".
- §4.1 (l.138-140) says the test "concerns **increases**", and that decreases "are not by themselves a PC2 failure".
- Under the row and #671, a treatment-only decrease in a non-target region fails PC2. Under §4.1 it does not. That is a conflict in the pass condition of a required control, not a difference in detail.
- Fix: amend l.91 to "show no treatment-against-sham **increase**; decreases are reported separately (§4.1)". Also align #671 §5.5, or have #670 state that §4.1 refines #671 §5.5.

**Baseline against #671 §5: less specific, not a contradiction.**
- #671 §5 step 4 defines Δ as "B − A, §3.2" and does not name phases. #671 §3.4 gives each observing process exactly two phases (`binding`, `after_run`), so "A" is ambiguous there. #670 resolves it as A.after_run against B.binding. That is a refinement within #671's terms.
- #671 §5 step 3 ("near fully resident in A … excluded") does not say per arm or joint. #670 makes it joint, which is a stricter reading, not a contrary one. #671 §5 step 6 (observer warming "in A") stays coherent: with an after_run baseline, the observer's warming sits in the baseline.
- Recommended, not blocking: #671 §5 steps 3-4 should cite the phases, or point to #670 §4.1, so the two documents cannot be read apart.

**Other sections.**
- The §4 PC2 row's "Already-resident targets are excluded" is less specific than §4.1 but compatible with it. Adding "jointly, per §4.1" would help.
- §7 and §10 have no conflict: the infeasibility path is unchanged and nothing is relaxed.
- §13's "PC2 design" row is still accurate.

## (c) Numbers, runs and instrumentation

- **No numbers.** "Half of the pairs" is given as an example of a declared pattern. Every quantity (minimum eligible count, timing tolerance, "near fully resident" cut) is deferred to the freeze.
- **No runs.**
- **No new instrumentation.** `after_run` and `binding` are exactly the two phases of `--residency-regions` in #671 §3.4 ("at binding and once more after the run"; `"phase":"binding"|"after_run"`).
  - The A→B time comes from the header fields `started_wall`, `ended_wall` and `started_mono_ns`, which already exist.
  - The eligibility observation is an ordinary fresh `--residency-regions` run.
  - The carry-over diagnostic reads earlier targets out of the same per-region records.
  - Nothing relies on a phase that I1+I3 do not provide.

## (d) Logic of the new procedure

**B2 (BLOCKING): carry-over within a pair is not accounted for, and the counterbalancing creates it.**
- §4.1 opens with its own premise: "A treatment read leaves its targets resident, and nothing evicts them between conditions."
- Both arms of a pair share one target set (l.105-106, l.118). In every pair where treatment runs first (half of them, under the counterbalancing at l.132-134):
  - the sham arm's A `after_run` sees the targets near fully resident;
  - the joint rule (l.118-119) then excludes them from both arms;
  - so the pair is void **by construction**.
- If voiding were missed, the sham Δ would sit near zero against an already-warm baseline. That biases the contrast.
- The stated rationale ("so that a sham does not always follow a treatment") answers a between-pair concern that the fresh targets already handle. Within a pair it does harm.
- Requirement 4 is therefore met between pairs but not within one.
- Fix, either of:
  - (i) the sham always precedes the treatment within a pair, with order drift handled by the fresh targets across pairs; or
  - (ii) two matched, disjoint, predeclared target sets per pair (one per arm), with the counterbalancing applied to which set goes to which arm.

**Other points, non-blocking:**

1. **Does the eligibility observation warm targets?** Yes, it may. That is handled: its `after_run` record captures its own warming, and a target it warms to near fully resident is ineligible. The extra observer run is outside every Δ. Sound.
2. **Does the after_run baseline exclude the observer's run?** For both prompt runs, yes. B's own setup before its `binding` scan (process start, prepare, mapping) does fall inside Δ.
   - It is matched across arms, so it cancels in the contrast.
   - "Neither observer run falls inside it" is accurate only if "run" means the passes. Suggest: "neither observer's passes fall inside it; B's pre-binding setup is common to both arms".
3. **Byte-disjoint is not region-disjoint.**
   - Targets are regions (#671 §5 step 2), but disjointness is declared on byte ranges (l.107-108). Two disjoint ranges inside one large mapped region share a target region.
   - The after_run baseline limits the damage, since a partly warm region still has a valid baseline. Even so, the disjointness should be stated over target-region keys plus their readahead and boundary neighbours (§3.3 stable key).
   - The same applies to the earlier-pair carry-over diagnostic.
4. **No fate is given for a timing mismatch.** The A→B gap is "matched between arms within a tolerance" (l.129-130), but nothing says what happens when it is not. It should probably void the pair and report it as void, like the eligibility shortfall.
5. **Window semantics.** #671 §5 defines its observation window as read completion → B binding `started`. #670 adds A after_run → B binding. These are compatible, but both should be named as recorded quantities so the two windows are not conflated.
6. **Verified-reset alternative.** It is witnessed only for the next pair's targets at eligibility. That satisfies req. 1 as written. B2 applies equally under it.
7. **The non-target "increase" needs a reference.** State that it means treatment Δ⁺ against sham Δ⁺ per region (the sum of positive Δᵢ, #671 §3.2), not an absolute Δ > 0. Otherwise background increases in both arms could read as a failure.
