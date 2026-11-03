"""The contracts this verifier can judge. Each is a statement and a function that fills in a result."""

from . import q8_0_gemv, q8_0_t2

CONTRACTS = {"Q8_0_T2": q8_0_t2}
OPERATOR_CONTRACTS = q8_0_gemv.STATEMENTS
