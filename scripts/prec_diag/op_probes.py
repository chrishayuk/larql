"""Kernel precision probes for the A64 operation-level dtype audit (`op_audit.py`).

A dispatch-level trace sees the dtype of every tensor an ATen op reads,
returns or writes. It cannot see inside one kernel: an `mm` whose inputs and
output are float64 could still accumulate in float32. These probes measure
that, on the same backend and thread settings as the forward, after it:

- accumulating ops (`mm`, `bmm`, `addmm`, `sum`, `mean`, `cumsum`, `cumprod`,
  `_softmax`, `_log_softmax`) are probed at every distinct shape and stride
  the forward used, so the kernel path (BLAS blocking, vectorised reduction)
  is the one the forward took;
- elementwise transcendental ops (`exp`, `rsqrt`, `cos`, `gelu`, `pow`, ...)
  are probed once per scalar variant on a fixed-length vector against
  Python's `math` (double precision);
- basic IEEE arithmetic (`add`, `mul`, `div`, ...) is probed for bit-exact
  agreement with numpy float64 arithmetic, which is correctly rounded.

Every probe also runs the same op on float32 copies of its inputs (the
"f32 counterfactual"). A probe passes only if the float64 op meets the bound
AND the float32 counterfactual misses it, so each probe demonstrably
discriminates the precision it claims to measure. A probe that cannot tell
f32 from f64 fails rather than passes.

The accumulation probes are exact. A reduction of `[1, e, e, ..., e]` with
`e` = float64 machine epsilon (2^-52) is `1 + (K-1)e`; every partial sum, in
any order, is representable in float64 and in nothing narrower. So a kernel
whose accumulator has fewer than float64's 52 fraction bits returns a
different value, whatever its blocking or summation order. `cumprod` uses
`(1 + sqrt(e))^2 = 1 + 2 sqrt(e) + e`, exact in float64 only.
"""

from __future__ import annotations

import math

import numpy as np
import torch

#: float64 machine epsilon (2^-52): the resolution an exact probe tests for.
F64_EPS = float(np.finfo(np.float64).eps)
#: (1 + sqrt(eps))^2 = 1 + 2 sqrt(eps) + eps is exact in float64 and in nothing narrower.
CUMPROD_FACTOR = 1.0 + math.sqrt(F64_EPS)
#: A softmax weight below float32's resolution of 1 (2^-30 < 2^-24).
SOFTMAX_SMALL_WEIGHT = float(np.finfo(np.float32).eps) / 2 ** 7
#: Tolerance probes: the same bound as hf_oracle's patched-function witnesses
#: (F64_WITNESS_REL). A float64 kernel is within a few ulps (~1e-16); float32 is ~1e-7.
TOLERANCE = 1e-12
#: Elementwise probe vector length: long enough to run a vectorised loop and its tail.
ELEMENTWISE_LENGTH = 4099
#: Seed of the basic-arithmetic probe values (fixed, so a probe is reproducible).
PROBE_SEED = 897
#: Basic-arithmetic probe range; almost no value in it is float32-representable.
BASIC_RANGE = 4.0
#: Rotary angles reach position * inv_freq <= position; the trig probe spans
#: more than the longest protocol input (1,040 positions).
TRIG_RANGE = 4096.0
#: The mixed error metric is absolute below this magnitude and relative above it.
MIXED_FLOOR = 1.0

EXACT, RELATIVE, MIXED = "exact", "relative", "mixed"
ACCUMULATING, ELEMENTWISE, BASIC = "accumulating", "elementwise", "basic_arithmetic"


# ------------------------------------------------------------------ layout

def layout(t: torch.Tensor) -> list:
    return [list(t.shape), list(t.stride())]


def _strided(values: torch.Tensor, lay: list) -> tuple[torch.Tensor, bool]:
    """`values` laid out with the observed strides when they can be
    reproduced (no broadcast dimension), else contiguous. Returns the tensor
    and whether the observed layout was reproduced."""
    shape, stride = lay
    if 0 in stride or values.numel() == 0:
        return values.contiguous(), False
    out = torch.empty_strided(shape, stride, dtype=values.dtype)
    try:
        out.copy_(values)
    except RuntimeError:  # overlapping strides
        return values.contiguous(), False
    return out, True


def _f64(value: float) -> torch.Tensor:
    """A 0-d float64 tensor (a bare Python scalar in `torch.where` would make float32)."""
    return torch.tensor(value, dtype=torch.float64)


def _one_at_start(shape: list, dims: list[int], fill: float, first: float) -> torch.Tensor:
    """`fill` everywhere, `first` at index 0 of every reduced dimension."""
    x = torch.full(shape, fill, dtype=torch.float64)
    index = [slice(None)] * len(shape)
    for d in dims:
        index[d] = 0
    x[tuple(index)] = first
    return x


def _dims(shape: list, dim) -> list[int]:
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        return list(range(len(shape)))
    dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
    return [d % len(shape) for d in dims]


def _reduced_length(shape: list, dims: list[int]) -> int:
    return int(np.prod([shape[d] for d in dims])) if dims else 1


# ------------------------------------------------------------- signatures

def signature(name: str, args: tuple, kwargs: dict):
    """The variant key a probe is run for, from one observed call: shapes and
    strides for accumulating ops, the scalar parameters for elementwise ones.
    JSON-friendly lists, made hashable by `op_audit` with `repr`."""
    if name in ("aten.mm.default", "aten.bmm.default"):
        return {"a": layout(args[0]), "b": layout(args[1])}
    if name == "aten.addmm.default":
        return {"bias": layout(args[0]), "a": layout(args[1]), "b": layout(args[2])}
    if name in ("aten.mean.dim", "aten.sum.dim_IntList"):
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        return {"x": layout(args[0]), "dim": list(dim) if dim is not None else None, "keepdim": bool(keepdim)}
    if name == "aten.sum.default":
        return {"x": layout(args[0])}
    if name in ("aten.cumsum.default", "aten.cumprod.default", "aten._softmax.default",
                "aten._log_softmax.default"):
        return {"x": layout(args[0]), "dim": int(args[1])}
    if name == "aten.pow.Tensor_Scalar":
        return {"exponent": float(args[1])}
    if name == "aten.pow.Scalar":
        return {"base": float(args[0])}
    if name == "aten.gelu.default":
        return {"approximate": kwargs.get("approximate", args[1] if len(args) > 1 else "none")}
    return {}


# ------------------------------------------------------------ comparisons

def _error(got: torch.Tensor, ref: torch.Tensor, metric: str) -> float:
    g, r = got.detach().to(torch.float64).numpy(), ref.detach().numpy()
    if g.shape != r.shape:
        return math.inf
    diff = np.abs(g - r)
    if metric == EXACT:
        return float(diff.max(initial=0.0))
    if metric == RELATIVE:
        return float((diff / np.abs(r)).max(initial=0.0))
    return float((diff / np.maximum(np.abs(r), MIXED_FLOOR)).max(initial=0.0))


def _f32(x):
    return x.float() if isinstance(x, torch.Tensor) and x.dtype == torch.float64 else x


def _judge(op, args: tuple, kwargs: dict, ref: torch.Tensor, metric: str, extra: dict) -> dict:
    got = op(*args, **kwargs)
    counter = op(*[_f32(a) for a in args], **kwargs)
    err = _error(got, ref, metric)
    cf = _error(counter, ref, metric)
    bound = 0.0 if metric == EXACT else TOLERANCE
    within = err <= bound
    discriminates = cf > bound
    out = {"comparison": metric, "bound": bound, "error": err, "f32_counterfactual_error": cf,
           "output_dtype": str(got.dtype), "pass": bool(within and discriminates and got.dtype == torch.float64),
           **extra}
    if got.dtype != torch.float64:
        out["detail"] = f"output dtype {got.dtype}"
    elif not within:
        out["detail"] = f"error {err:.3e} exceeds {bound:.0e}: the kernel is not float64-accurate"
    elif not discriminates:
        out["detail"] = "the float32 counterfactual also meets the bound: this probe cannot see the difference"
    return out


def _no_accumulation(k: int) -> dict:
    return {"comparison": EXACT, "reduction_length": k, "pass": True,
            "detail": "reduction length < 2: nothing accumulates"}


# ------------------------------------------------------- accumulating ops

def _matmul(op, sig: dict, with_bias: bool) -> dict:
    (a_shape, _), (b_shape, _) = sig["a"], sig["b"]
    k = a_shape[-1]
    if k < 2:
        return _no_accumulation(k)
    a, a_ok = _strided(_one_at_start(a_shape, [len(a_shape) - 1], F64_EPS, 1.0), sig["a"])
    b, b_ok = _strided(torch.ones(b_shape, dtype=torch.float64), sig["b"])
    args = (a, b)
    reproduced = a_ok and b_ok
    if with_bias:
        bias, c_ok = _strided(torch.zeros(sig["bias"][0], dtype=torch.float64), sig["bias"])
        args, reproduced = (bias, a, b), reproduced and c_ok
    out_shape = list(a_shape[:-1]) + [b_shape[-1]]
    ref = torch.full(out_shape, 1.0 + (k - 1) * F64_EPS, dtype=torch.float64)
    return _judge(op, args, {}, ref, EXACT, {"reduction_length": k, "layout_reproduced": reproduced})


def _reduce(op, sig: dict, mean: bool) -> dict:
    shape = sig["x"][0]
    dims = _dims(shape, sig.get("dim"))
    k = _reduced_length(shape, dims)
    if k < 2:
        return _no_accumulation(k)
    x, ok = _strided(_one_at_start(shape, dims, F64_EPS, 1.0), sig["x"])
    total = 1.0 + (k - 1) * F64_EPS  # exact in float64
    args = (x,) if "dim" not in sig else (x, sig["dim"], sig.get("keepdim", False))
    got = op(*args)
    value = total
    if mean:
        # A mean is the exact sum divided by K, rounded once: `sum / K`, or
        # `sum * (1 / K)` (one more rounding). Either is a float64 kernel.
        quotient, product = total / k, total * (1.0 / k)
        value = product if quotient != product and bool((got == product).all()) else quotient
    ref = torch.full(got.shape, value, dtype=torch.float64)
    return _judge(op, args, {}, ref, EXACT, {"reduction_length": k, "layout_reproduced": ok})


def _scan(op, sig: dict, product: bool) -> dict:
    shape, dim = sig["x"][0], sig["dim"] % len(sig["x"][0])
    k = shape[dim]
    if k < 2:
        return _no_accumulation(k)
    view = [1] * len(shape)
    view[dim] = k
    j = torch.arange(k, dtype=torch.float64).reshape(view)
    if product:
        x = torch.where(j < 2, _f64(CUMPROD_FACTOR), _f64(1.0)).expand(shape).clone()
        ref = torch.where(j < 1, _f64(CUMPROD_FACTOR), _f64(CUMPROD_FACTOR * CUMPROD_FACTOR)).expand(shape).clone()
    else:
        x = _one_at_start(shape, [dim], F64_EPS, 1.0)
        ref = (1.0 + j * F64_EPS).expand(shape).clone()
    x, ok = _strided(x, sig["x"])
    return _judge(op, (x, sig["dim"]), {}, ref, EXACT, {"reduction_length": k, "layout_reproduced": ok})


def _softmax(op, sig: dict, log: bool) -> dict:
    shape, dim = sig["x"][0], sig["dim"] % len(sig["x"][0])
    k = shape[dim]
    if k < 2:
        return _no_accumulation(k)
    logit = math.log(SOFTMAX_SMALL_WEIGHT)
    x, ok = _strided(_one_at_start(shape, [dim], logit, 0.0), sig["x"])
    small = math.exp(logit)
    total = math.fsum([1.0] + [small] * (k - 1))
    view = [1] * len(shape)
    view[dim] = k
    first = torch.arange(k).reshape(view) == 0
    if log:
        ref = torch.where(first, _f64(-math.log(total)), _f64(logit - math.log(total)))
    else:
        ref = torch.where(first, _f64(1.0 / total), _f64(small / total))
    ref = ref.expand(shape).clone()
    return _judge(op, (x, sig["dim"], False), {}, ref, MIXED if log else RELATIVE,
                  {"reduction_length": k, "layout_reproduced": ok})


# ------------------------------------------------------ elementwise ops

def _gelu_ref(approximate: str):
    if approximate == "tanh":
        c = math.sqrt(2.0 / math.pi)
        return lambda x: 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x ** 3)))
    return lambda x: 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


#: name -> (domain (lo, hi, spacing), reference(sig) -> scalar fn, metric).
_ELEMENTWISE = {
    "aten.exp.default": ((-20.0, 20.0, "lin"), lambda s: math.exp, RELATIVE),
    "aten.log.default": ((1e-6, 1e6, "log"), lambda s: math.log, MIXED),
    "aten.sqrt.default": ((1e-6, 1e6, "log"), lambda s: math.sqrt, RELATIVE),
    "aten.rsqrt.default": ((1e-6, 1e6, "log"), lambda s: (lambda x: 1.0 / math.sqrt(x)), RELATIVE),
    "aten.cos.default": ((-TRIG_RANGE, TRIG_RANGE, "lin"), lambda s: math.cos, MIXED),
    "aten.sin.default": ((-TRIG_RANGE, TRIG_RANGE, "lin"), lambda s: math.sin, MIXED),
    "aten.tanh.default": ((-5.0, 5.0, "lin"), lambda s: math.tanh, MIXED),
    "aten.sigmoid.default": ((-20.0, 20.0, "lin"), lambda s: (lambda x: 1.0 / (1.0 + math.exp(-x))), RELATIVE),
    "aten.silu.default": ((-8.0, 8.0, "lin"), lambda s: (lambda x: x / (1.0 + math.exp(-x))), MIXED),
    "aten.gelu.default": ((-4.0, 4.0, "lin"), lambda s: _gelu_ref(s["approximate"]), MIXED),
    "aten.pow.Tensor_Scalar": ((0.5, 2.0, "lin"), lambda s: (lambda x: math.pow(x, s["exponent"])), RELATIVE),
    "aten.pow.Scalar": ((0.0, 1.0, "lin"), lambda s: (lambda x: math.pow(s["base"], x)), RELATIVE),
}


def _domain(lo: float, hi: float, spacing: str) -> torch.Tensor:
    if spacing == "log":
        return torch.logspace(math.log10(lo), math.log10(hi), ELEMENTWISE_LENGTH, dtype=torch.float64)
    return torch.linspace(lo, hi, ELEMENTWISE_LENGTH, dtype=torch.float64)


def _elementwise(name: str, op, sig: dict) -> dict:
    (lo, hi, spacing), ref_of, metric = _ELEMENTWISE[name]
    x = _domain(lo, hi, spacing)
    fn = ref_of(sig)
    ref = torch.tensor([fn(v) for v in x.tolist()], dtype=torch.float64)
    if name == "aten.pow.Tensor_Scalar":
        args, kwargs = (x, sig["exponent"]), {}
    elif name == "aten.pow.Scalar":
        args, kwargs = (sig["base"], x), {}
    elif name == "aten.gelu.default":
        args, kwargs = (x,), {"approximate": sig["approximate"]}
    else:
        args, kwargs = (x,), {}
    return _judge(op, args, kwargs, ref, metric, {"vector_length": ELEMENTWISE_LENGTH})


# ---------------------------------------------------------- basic arithmetic

#: name -> (arity, numpy reference). numpy float64 arithmetic is IEEE correctly rounded.
_BASIC = {
    "aten.add.Tensor": (2, np.add),
    "aten.sub.Tensor": (2, np.subtract),
    "aten.mul.Tensor": (2, np.multiply),
    "aten.div.Tensor": (2, np.divide),
    "aten.neg.default": (1, np.negative),
    "aten.reciprocal.default": (1, lambda x: np.divide(1.0, x)),
}


def _basic(name: str, op) -> dict:
    arity, ref_fn = _BASIC[name]
    rng = np.random.default_rng(PROBE_SEED)
    xs = [rng.uniform(-BASIC_RANGE, BASIC_RANGE, ELEMENTWISE_LENGTH) for _ in range(arity)]
    ref = torch.from_numpy(ref_fn(*xs))
    args = tuple(torch.from_numpy(x) for x in xs)
    return _judge(op, args, {}, ref, EXACT, {"vector_length": ELEMENTWISE_LENGTH})


# ------------------------------------------------------------------ table

_ACCUMULATING = {
    "aten.mm.default": lambda op, s: _matmul(op, s, with_bias=False),
    "aten.bmm.default": lambda op, s: _matmul(op, s, with_bias=False),
    "aten.addmm.default": lambda op, s: _matmul(op, s, with_bias=True),
    "aten.sum.default": lambda op, s: _reduce(op, s, mean=False),
    "aten.sum.dim_IntList": lambda op, s: _reduce(op, s, mean=False),
    "aten.mean.dim": lambda op, s: _reduce(op, s, mean=True),
    "aten.cumsum.default": lambda op, s: _scan(op, s, product=False),
    "aten.cumprod.default": lambda op, s: _scan(op, s, product=True),
    "aten._softmax.default": lambda op, s: _softmax(op, s, log=False),
    "aten._log_softmax.default": lambda op, s: _softmax(op, s, log=True),
}

#: Every probed op, by class.
CLASSES = {ACCUMULATING: frozenset(_ACCUMULATING), ELEMENTWISE: frozenset(_ELEMENTWISE),
           BASIC: frozenset(_BASIC)}
PROBED = frozenset().union(*CLASSES.values())


def class_of(name: str) -> str | None:
    return next((c for c, names in CLASSES.items() if name in names), None)


def resolve(name: str):
    """The ATen OpOverload named `aten.<op>.<overload>`."""
    _, packet, overload = name.split(".")
    return getattr(getattr(torch.ops.aten, packet), overload)


def probe(name: str, sig: dict, op=None) -> dict:
    """Run the probe for `name` at `sig`. `op` defaults to the ATen overload
    itself; the tests pass a deliberately imprecise one."""
    op = op or resolve(name)
    kind = class_of(name)
    with torch.no_grad():
        if kind == ACCUMULATING:
            out = _ACCUMULATING[name](op, sig)
        elif kind == ELEMENTWISE:
            out = _elementwise(name, op, sig)
        elif kind == BASIC:
            out = _basic(name, op)
        else:
            raise KeyError(f"no probe for {name}")
    return {"op": name, "class": kind, "signature": sig, **out}
