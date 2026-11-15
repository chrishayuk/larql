"""The contracts this verifier can judge. Each is a statement and a function that fills in a result."""

from . import q8_0_gemv, q8_0_t2, weight_patches_v2v3, weight_patches_v4

CONTRACTS = {"Q8_0_T2": q8_0_t2}
OPERATOR_CONTRACTS = q8_0_gemv.STATEMENTS
#: The `ledger` route: each contract and the module that judges it. A module has `STATEMENTS`, by contract id, and
#: `evaluate(out, base, container, patch, patch_problem)`. One contract judges one ledger schema and no other.
LEDGER_CONTRACTS = {
    weight_patches_v4.CONTRACT: weight_patches_v4,
    weight_patches_v2v3.CONTRACT_V2: weight_patches_v2v3,
    weight_patches_v2v3.CONTRACT_V3: weight_patches_v2v3,
}
