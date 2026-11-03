# Provenance

## Copied unmodified

These files are byte-for-byte copies from the isolated reference checker at commit
`e505afcf2d81c0262cd1d75ac0a5dd8429a84c5a`. `tests/test_verifier.py` checks them against this table, and CI refuses a change to them
together with a change to the engine. IDs are git blob IDs. Paths are relative to `verifier/`.

| File | Blob |
|---|---|
| `refcheck/__init__.py` | `c75c58be8e37c45caf198078653d62fa60bc2a1d` |
| `refcheck/e1.py` | `cda1c5394e40f1eea72c19b9ab8a448ff1b31698` |
| `refcheck/q8_0.py` | `bd66df2e4a12a09ec6e076ba19517f5e70747fdf` |
| `refcheck/container.py` | `49d98a93eea53ea152f51543404cdcf950b8d0fe` |
| `refcheck/numconv.py` | `915687c402185f81ff4d8cabc382d3fa71ccbae7` |
| `refcheck/synth.py` | `7041b5dd8de47dfbc7197628a0c112281b39718f` |
| `tests/_mini.py` | `b3bfab14dc3111a7626f9abcb967d73976d9e144` |
| `tests/test_e1.py` | `93741e0fc52d92867dc843f6d0ea4b0f59d391bc` |
| `tests/test_q8_0.py` | `71e948b75756c0f1c47a529550bb187a6fc20ff1` |
| `tests/test_container_reader.py` | `788ddff8c92f44ea29d91ca7e24c9a908259f2e0` |
| `tests/test_synth.py` | `c8f9fcc8acfb2d506037de7da83d1981f3cb3b66` |

**The checker remains separate.** It lives in its own private repository, with its own test history, and stays the research and
adversarial evaluation authority. This directory holds a pinned, auditable distribution of the approved measurement code, not its
development. A change to the measurement belongs there first, then comes here as a new copy with a new table. **Because that repository
is private, you can't check these copies against it from here.** What you can check is what the code does: the tests here, and
`reproduce.sh`.

**Comments in these files cite documents that aren't in this repository,** such as `PROTOCOL.md`, decision labels like `D1` and `D6`,
and ambiguity labels like `AMB-019`. They are left as they are so that the copies stay exact.

**How the checker was written:** its agent worked under isolation, without access to LARQL's source or its outputs. This repository
doesn't include the evidence for that.

## Written for this package

`larql_contract_verifier/` is new. It replaces the checker's `harness.py` for this one purpose, because `harness.py` needs `torch` and
`transformers`. **It isn't under that isolation: its author had read LARQL's source.** It contains no LARQL code, and a test refuses any
import of LARQL's. `checkpoint.py`'s pins were copied programmatically from the checker. Its weight reader is new, and on the real
checkpoint it gives the same report digests as the checker's `torch` reader, for both containers in `expected/`.
