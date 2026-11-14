# Verification independence

**Class: CURRENT.** Independence is an architectural property. It doesn't need a separate repository, but it does need boundaries that are
enforced, because one pull request could otherwise change both the engine and the verifier to make a wrong result look right.

| Requirement | What enforces it | Status |
|---|---|---|
| No LARQL code in the numerics | `verifier/tests/test_boundaries.py` refuses any import but the standard library, NumPy and `gguf`, and any way of running a program | Enforced by CI |
| Measurement code stays what was approved | `verifier/tests/test_verifier.py` checks `refcheck/` and its tests against the hash table in `verifier/provenance/PROVENANCE.md` | Enforced by CI |
| One change can't move both sides | The `boundary` check in `verifier-boundary.yml`, required on `main`, fails a pull request that changes both the measurement and `crates/`. The measurement includes the frozen VLP numerics (`conformance/vlp-numerics/`): the specification and golden vectors both sides are held to, so a new numerics version lands on its own, before the engine that implements it. It runs from the base branch, so a pull request can't rewrite it | Enforced by a required check |
| Results are bound to what they measured | Each answer carries the verifier's version and code digest, the container's digest, and the source's pinned identity. These say what was measured. They don't authenticate who measured it (below) | By construction |
| No circular answers | The expected results in `verifier/expected/` come from the verifier, and LARQL's tests don't generate them | By review |
| Accountability for verifier changes | The `boundary` check requires review evidence or a written maintainer exception | Enforced by a required check. **Not independent review** (below) |

**Repository settings.** These live outside the repository, so check them rather than trusting this page:
`gh api repos/chrishayuk/larql/rules/branches/main`. Two rulesets exist for `main`:

- `main branch`, with no bypass: changes arrive by pull request, `boundary` must pass, and the branch can't be deleted or force-pushed.
- `verifier code owners`: disabled. It asks for code owner review, but it does nothing at 0 approvals, and at 1 it applies to every pull request, because a branch ruleset can't be limited to paths.

**Review evidence for verifier paths.** A pull request that changes `verifier/`, `docs/verification/`, `conformance/vlp-numerics/`, the gate, or `CODEOWNERS` fails `boundary` until its exact head commit has one of these on the record:

- an approval from a maintainer or admin who isn't the author; or
- a comment or review from a maintainer or admin, with a line starting `Verifier-review exception (<head sha>):`, followed by the reason.

Each is tied to the head commit, so pushing again asks for it again. Writing one doesn't start the check, so rerun it afterwards.

**This is change-control accountability, not independent human review.** The repository has one maintainer, who writes most changes and can write their own exception. The gate makes sure each verifier change is deliberate and recorded. It doesn't put a second person between an author and the measurement. A fresh-context model review can be cited in an exception, but it's not an independent human approval. When a second trusted reviewer is added to `CODEOWNERS`, these paths should move to a truly enforced approval rule.

**The `verifier-and-engine` override.** Some changes do need both sides, such as a format change. To let one through:

1. A maintainer or admin comments or reviews with a line starting `Verifier-and-engine justification:`, saying why the change can't be split.
2. A maintainer or admin then applies the label `verifier-and-engine`.

GitHub lets anyone with triage access apply a label, so the gate asks GitHub who applied it and who wrote the reason, and refuses anyone else.
Removing the label, or pushing again, runs the gate again. The check doesn't judge the reason. It makes sure a maintainer gave one on the record.

## The verifier is not authenticated

LQL runs whichever verifier it finds, and checks the form of its answer and what the answer is bound to. The version and code digest in the answer
are reported by the program being identified, so they support reproducing a result, and a replaced or modified program can report anything.

**Replacing the verifier can make a contract appear satisfied without changing the model.** That is acceptable for what this is today, a measurement
tool run by someone who controls their own machine. `VERIFY REPRESENTATION` and its exit status **must not gate a release** on this basis.

A verification-gated release would first need the gate to pin an approved verifier, through configuration the checked change can't modify, and to
check that identity itself rather than ask the verifier. For this Python package, the identity would have to cover the interpreter and locked
dependencies as well as the package's own files. None of that exists yet.

**The isolated checker stays separate.** The reference checker in its own repository remains the research and adversarial evaluation authority,
with its own test history. `verifier/refcheck/` is a pinned copy of its approved measurement code. See
[PROVENANCE.md](../../verifier/provenance/PROVENANCE.md).

**What this doesn't give you.**

- `verifier/larql_contract_verifier/` wasn't written under the checker's isolation: its author had read LARQL's source.
- Authentication of the verifier: see above.
- Independent human review: see above.
- This is empirical checking of one model against one contract, not proof checking.
