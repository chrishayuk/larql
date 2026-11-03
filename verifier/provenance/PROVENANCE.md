# Provenance

## Copied unmodified

These files are byte-for-byte copies from the isolated reference checker at commit
`f2788f0e7f1b23432bf4b9bdd1099dbd42d070fc`. The GEMV reference and its tests are new there since the previous copy, `e505afc`, and every earlier file is unchanged. `tests/test_verifier.py` checks them against this table, and CI refuses a change to them
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
| `refcheck/gemv.py` | `1d71a36e92a45718c1c17127383bf6151c338c50` |
| `tests/_gemv_faults.py` | `0e6aa4cebd7da098b00d807a69fd9fe034dbb44e` |
| `tests/_gemv_gen.py` | `fd1d5af596955709968e98029fc23001eedaeb38` |
| `tests/_gemv_oracle.py` | `db744442194e286c19a2668c7f0ec236e07a5c0a` |
| `tests/test_gemv.py` | `7b75706865459dd0a7620c406ed1cf3bd848bf83` |
| `tests/test_gemv_faults.py` | `af161568f866fc34441d9c650491efaceca37721` |
| `tests/test_gemv_diagnostics.py` | `399153803bc13a2d7c5cbf2b663a153d3412e0fd` |
| `tests/test_gemv_search.py` | `5b934328440c809ae37431f10dd900e1357411cc` |
| `tests/test_gemv_boundaries.py` | `d315401a96d2bc05829f495ae3176e6a667db0bb` |

**The checker remains separate.** It lives in its own private repository, with its own test history, and stays the research and
adversarial evaluation authority. This directory holds a pinned, auditable distribution of the approved measurement code, not its
development. A change to the measurement belongs there first, then comes here as a new copy with a new table. **Because that repository
is private, you can't check these copies against it from here.** What you can check is what the code does: the tests here, and
`reproduce.sh`.

**Comments in these files cite documents that aren't in this repository,** such as `PROTOCOL.md`, decision labels like `D1` and `D6`,
and ambiguity labels like `AMB-019`. They are left as they are so that the copies stay exact.

**How the checker was written:** its agent worked under isolation, without access to LARQL's source or its outputs. The GEMV reference came from a frozen numerical specification that says nothing about LARQL's code. That specification's sequential order was chosen by someone who had read LARQL's kernel. This repository
doesn't include the evidence for that.

## Written for this package

`larql_contract_verifier/` is new, and that includes `contracts/q8_0_gemv.py`, which ties a witness to a container and then calls the copied reference. It replaces the checker's `harness.py` for this one purpose, because `harness.py` needs `torch` and
`transformers`. **It isn't under that isolation: its author had read LARQL's source.** It contains no LARQL code, and a test refuses any
import of LARQL's. `checkpoint.py`'s pins were copied programmatically from the checker. Its weight reader is new, and on the real
checkpoint it gives the same report digests as the checker's `torch` reader, for both containers in `expected/`.
