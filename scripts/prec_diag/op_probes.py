"""Kernel precision probes for the A64 operation-level dtype audit (`op_audit.py`).

A dispatch-level trace sees the dtype of every tensor an ATen op reads,
returns or writes. It cannot see inside one kernel: an `mm` whose inputs and
output are float64 could still accumulate in float32. These probes measure
that, on the same backend and thread settings as the forward, after it:

- accumulating ops (`mm`, `bmm`, `addmm`, `sum`, `mean`, `cumsum`, `cumprod`,
  `_softmax`, `_log_softmax`, and 1-d `convolution`) are probed at every
  distinct shape and stride the forward used, so the kernel path (BLAS
  blocking, vectorised reduction) is the one the forward took;
- elementwise transcendental ops (`exp`, `rsqrt`, `cos`, `gelu`, `softplus`, `pow`, ...)
  are probed once per scalar variant on a fixed-length vector against
  Python's `math` (double precision);
- `silu_backward` (the tangent of `silu` under forward-mode AD: grad times
  silu's derivative) is an elementwise probe of two operands;
- basic IEEE arithmetic (`add`, `mul`, `div`, ...) is probed for bit-exact
  agreement with numpy float64 arithmetic, which is correctly rounded. A
  scalar variant (`mul.Scalar`, a `prims` op given a Python number) is
  probed once per scalar the forward used; an in-place op (`div_`) is
  probed on a copy of its first operand, so the f32 counterfactual reads
  the same operands.

Every probe also runs the same op on float32 copies of its inputs (the
"f32 counterfactual"). A probe passes only if the float64 op meets the bound
AND the float32 counterfactual misses it, so each probe demonstrably
discriminates the precision it claims to measure. A probe that cannot tell
f32 from f64 fails rather than passes.

The accumulation probes are exact, and test the operands and the
accumulator together:
- the accumulator: a reduction of `[h, e, e, ..., e]` with `e` = float64
  machine epsilon (2^-52) is `h + (K-1)e`. Every partial sum, in any order,
  is representable in float64 and in nothing narrower, so a kernel whose
  accumulator has fewer than float64's 52 fraction bits returns a different
  value, whatever its blocking or summation order;
- the operands: the head `h` is `1 + 2^-24` (and a matmul's other operand
  head is `1 + 2^-26`). Neither is representable in float32 or bfloat16, and
  their product `1 + 2^-24 + 2^-26 + 2^-50` is exact in float64. So a kernel
  that rounds float64 operands to float32 or bfloat16 and then accumulates in
  float64 also returns a different value.
`cumprod` uses `(1 + sqrt(e))^2 = 1 + 2 sqrt(e) + e`: its factor is not
float32-representable and the product is exact in float64 only.
A 1-d `convolution` (Qwen3.5's depthwise causal conv) slides its window, so
one input value feeds several outputs and the two heads cannot share one
probe exactly. It runs two exact probes at the observed geometry: input all
1 with weights `[h, e, ..., e]` (the accumulator and the weight operand),
then input all `1 + 2^-26` with weights `[1, 0, ..., 0]` (the input
operand). Padding drops taps at the edges, so each output's expected value
is counted from the geometry, and is exact in float64.

Memory. A probe runs at the forward's own shapes (an lm_head-shaped `mm` is
as large as the logits), while the model is still loaded. So a probe builds
each operand once, in place, at the observed strides; compares a uniform
output against one scalar in chunks; and runs its float32 counterfactual at
the probe's own reduction length with every other dimension capped
(`COUNTERFACTUAL_EXTENT`): whether the values discriminate depends on the
values and the reduction length only. Each probe records the bytes it
allocated (`probe_bytes`).
"""

from __future__ import annotations

import contextvars
import math

import numpy as np
import torch

#: float64 machine epsilon (2^-52): the resolution an exact probe tests for.
F64_EPS = float(np.finfo(np.float64).eps)
#: float32 machine epsilon (2^-23).
F32_EPS = float(np.finfo(np.float32).eps)
#: Operand heads, neither representable in float32 or bfloat16 (1 + 2^-24 and
#: 1 + 2^-26), whose product 1 + 2^-24 + 2^-26 + 2^-50 is exact in float64.
HEAD_A = 1.0 + F32_EPS / 2
HEAD_B = 1.0 + F32_EPS / 8
#: The float32 counterfactual of a probe keeps the reduction length and caps
#: every other dimension at this.
COUNTERFACTUAL_EXTENT = 64
#: Elements compared per step, so checking a vocabulary-sized output never
#: allocates a second output-sized tensor.
COMPARE_CHUNK = 1 << 22
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


def _dense(shape: list, stride: list) -> bool:
    """The strides are a permutation of a dense layout (no broadcast, no overlap)."""
    expected = 1
    for size, st in sorted(zip(shape, stride), key=lambda p: p[1]):
        if size == 0 or (size > 1 and st != expected):
            return False
        expected *= size
    return True


def _pattern(lay: list, fill: float, head: float, head_dims: list[int]) -> tuple[torch.Tensor, bool]:
    """A float64 tensor at the observed layout when it can be reproduced (a
    dense permutation), else contiguous: `fill` everywhere and `head` at index
    0 of every `head_dims` dimension. Built in place, one allocation. Returns
    the tensor and whether the observed layout was reproduced."""
    shape, stride = lay
    reproduced = _dense(shape, stride)
    t = (torch.empty_strided(shape, stride, dtype=torch.float64) if reproduced
         else torch.empty(shape, dtype=torch.float64))
    t.fill_(fill)
    if head_dims:
        index = [slice(None)] * len(shape)
        for d in head_dims:
            index[d] = 0
        t[tuple(index)] = head
    return t, reproduced


def _capped(shape: list, keep: list[int]) -> list:
    """A contiguous counterfactual layout: dimensions in `keep` as they are,
    the others capped at COUNTERFACTUAL_EXTENT."""
    capped = [n if i in keep else min(n, COUNTERFACTUAL_EXTENT) for i, n in enumerate(shape)]
    return [capped, list(torch.empty(capped, device="meta").stride())]


def _f64(value: float) -> torch.Tensor:
    """A 0-d float64 tensor (a bare Python scalar in `torch.where` would make float32)."""
    return torch.tensor(value, dtype=torch.float64)


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
    if name == "aten.convolution.default":
        return {"x": layout(args[0]), "w": layout(args[1]), "bias": layout(args[2]) if args[2] is not None else None,
                "stride": list(args[3]), "padding": list(args[4]), "dilation": list(args[5]),
                "transposed": bool(args[6]), "output_padding": list(args[7]), "groups": int(args[8])}
    if name == "aten.softplus.default":
        return {"beta": float(args[1]) if len(args) > 1 else SOFTPLUS_BETA,
                "threshold": float(args[2]) if len(args) > 2 else SOFTPLUS_THRESHOLD}
    if name == "aten.pow.Tensor_Scalar":
        return {"exponent": float(args[1])}
    if name == "aten.pow.Scalar":
        return {"base": float(args[0])}
    if name == "aten.gelu.default":
        return {"approximate": kwargs.get("approximate", args[1] if len(args) > 1 else "none")}
    if name == "aten.mul.Scalar":
        return {"scalar_operand": 1, "scalar": float(args[1])}
    if name in ("prims.add.default", "prims.mul.default"):
        scalar = [i for i, a in enumerate(args[:2]) if isinstance(a, (int, float)) and not isinstance(a, bool)]
        return {"scalar_operand": scalar[0], "scalar": float(args[scalar[0]])} if scalar else {}
    return {}


# ------------------------------------------------------------ comparisons

def _scale(ref, metric: str):
    if metric == RELATIVE:
        return ref.abs() if isinstance(ref, torch.Tensor) else abs(ref)
    if metric == MIXED:
        return ref.abs().clamp(min=MIXED_FLOOR) if isinstance(ref, torch.Tensor) else max(abs(ref), MIXED_FLOOR)
    return 1.0


def _error(got: torch.Tensor, ref, metric: str, shape: tuple | None = None) -> float:
    """Max error of `got` against `ref`: a tensor of the same shape (it may be
    an expanded view), or one scalar for a uniform output of `shape`, compared
    in chunks. A wrong shape or any NaN is an infinite error (Python's `max`
    would otherwise drop a NaN)."""
    if isinstance(ref, torch.Tensor):
        if tuple(got.shape) != tuple(ref.shape):
            return math.inf
        if not got.numel():
            return 0.0
        err = float(((got.to(torch.float64) - ref).abs() / _scale(ref, metric)).max())
        return math.inf if math.isnan(err) else err
    if shape is None or tuple(got.shape) != tuple(shape):
        return math.inf
    flat, worst = got.reshape(-1), 0.0
    for i in range(0, flat.numel(), COMPARE_CHUNK):
        part = float((flat[i:i + COMPARE_CHUNK].to(torch.float64) - ref).abs().max())
        if math.isnan(part):
            return math.inf
        worst = max(worst, part / _scale(ref, metric))
    return worst


# The ATen overload being probed, set by `probe` for the duration of one probe. Shape inference uses it,
# never the kernel under test (which may be a deliberately wrong one).
_REFERENCE_OP: contextvars.ContextVar = contextvars.ContextVar("prec_diag_probe_reference_op")


def _expected_shape(args: tuple, kwargs: dict) -> tuple:
    """The output shape the probed ATen overload produces for `args`, from the
    meta device (shape inference only: nothing is allocated or computed)."""
    meta = [a.to("meta") if isinstance(a, torch.Tensor) else a for a in args]
    return tuple(_REFERENCE_OP.get()(*meta, **kwargs).shape)


def _f32(x):
    return x.float() if isinstance(x, torch.Tensor) and x.dtype == torch.float64 else x


def _bytes(*tensors) -> int:
    return sum(t.untyped_storage().nbytes() for t in tensors if isinstance(t, torch.Tensor))


def _judge(op, args: tuple, kwargs: dict, ref, metric: str, extra: dict, counterfactual_args=None) -> dict:
    """Run the float64 probe, then its float32 counterfactual (on
    `counterfactual_args`, a smaller probe of the same values and reduction
    length, when given). The float64 output is freed before the counterfactual."""
    uniform = not isinstance(ref, torch.Tensor)
    got = op(*args, **kwargs)
    probe_bytes = _bytes(*args, got)
    err = _error(got, ref, metric, _expected_shape(args, kwargs) if uniform else None)
    dtype = got.dtype
    del got
    cf_args = [_f32(a) for a in (counterfactual_args or args)]
    counter = op(*cf_args, **kwargs)
    cf = _error(counter, ref, metric, _expected_shape(tuple(cf_args), kwargs) if uniform else None)
    bound = 0.0 if metric == EXACT else TOLERANCE
    within = err <= bound
    discriminates = cf > bound
    out = {"comparison": metric, "bound": bound, "error": err, "f32_counterfactual_error": cf,
           "output_dtype": str(dtype), "probe_bytes": probe_bytes,
           "pass": bool(within and discriminates and dtype == torch.float64), **extra}
    if dtype != torch.float64:
        out["detail"] = f"output dtype {dtype}"
    elif not within:
        out["detail"] = f"error {err:.3e} exceeds {bound:.0e}: the kernel is not float64-accurate"
    elif not discriminates:
        out["detail"] = "the float32 counterfactual also meets the bound: this probe cannot see the difference"
    return out


def _no_accumulation(k: int) -> dict:
    return {"comparison": EXACT, "reduction_length": k, "pass": True,
            "detail": "reduction length < 2: nothing accumulates"}


# ------------------------------------------------------- accumulating ops

def _matmul_args(a_lay: list, b_lay: list, bias_lay: list | None) -> tuple[tuple, bool]:
    a, a_ok = _pattern(a_lay, F64_EPS, HEAD_A, [len(a_lay[0]) - 1])
    b, b_ok = _pattern(b_lay, 1.0, HEAD_B, [len(b_lay[0]) - 2])
    if bias_lay is None:
        return (a, b), a_ok and b_ok
    bias, c_ok = _pattern(bias_lay, 0.0, 0.0, [])
    return (bias, a, b), a_ok and b_ok and c_ok


def _matmul(op, sig: dict, with_bias: bool) -> dict:
    """Every output is `HEAD_A * HEAD_B + (K-1) e`, exact in float64."""
    a_shape, b_shape = sig["a"][0], sig["b"][0]
    k = a_shape[-1]
    if k < 2:
        return _no_accumulation(k)
    args, reproduced = _matmul_args(sig["a"], sig["b"], sig["bias"] if with_bias else None)
    small = _matmul_args(_capped(a_shape, [len(a_shape) - 1]), _capped(b_shape, [len(b_shape) - 2]),
                         _capped(sig["bias"][0], []) if with_bias else None)[0]
    value = HEAD_A * HEAD_B + (k - 1) * F64_EPS
    return _judge(op, args, {}, value, EXACT, {"reduction_length": k, "layout_reproduced": reproduced},
                  counterfactual_args=small)


def _reduce(op, sig: dict, mean: bool) -> dict:
    """Every output is `HEAD_A + (K-1) e` (a sum), or that divided by K (a mean)."""
    shape = sig["x"][0]
    dims = _dims(shape, sig.get("dim"))
    k = _reduced_length(shape, dims)
    if k < 2:
        return _no_accumulation(k)
    x, ok = _pattern(sig["x"], F64_EPS, HEAD_A, dims)
    total = HEAD_A + (k - 1) * F64_EPS  # exact in float64
    args = (x,) if "dim" not in sig else (x, sig["dim"], sig.get("keepdim", False))
    value = total
    if mean:
        # A mean is the exact sum divided by K, rounded once: `sum / K`, or
        # `sum * (1 / K)` (one more rounding). Either is a float64 kernel.
        quotient, product = total / k, total * (1.0 / k)
        if quotient != product and bool((op(*args) == product).all()):
            value = product
        else:
            value = quotient
    return _judge(op, args, {}, value, EXACT, {"reduction_length": k, "layout_reproduced": ok})


def _scan(op, sig: dict, product: bool) -> dict:
    shape, dim = sig["x"][0], sig["dim"] % len(sig["x"][0])
    k = shape[dim]
    if k < 2:
        return _no_accumulation(k)
    view = [1] * len(shape)
    view[dim] = k
    j = torch.arange(k, dtype=torch.float64).reshape(view)
    if product:
        x, ok = _pattern(sig["x"], 1.0, CUMPROD_FACTOR, [])
        x[(slice(None),) * dim + (slice(0, 2),)] = CUMPROD_FACTOR
        ref = torch.where(j < 1, _f64(CUMPROD_FACTOR), _f64(CUMPROD_FACTOR * CUMPROD_FACTOR))
    else:
        x, ok = _pattern(sig["x"], F64_EPS, HEAD_A, [dim])
        ref = HEAD_A + j * F64_EPS  # exact in float64
    return _judge(op, (x, sig["dim"]), {}, ref.expand(shape), EXACT, {"reduction_length": k, "layout_reproduced": ok})


def _softmax(op, sig: dict, log: bool) -> dict:
    shape, dim = sig["x"][0], sig["dim"] % len(sig["x"][0])
    k = shape[dim]
    if k < 2:
        return _no_accumulation(k)
    logit = math.log(SOFTMAX_SMALL_WEIGHT)
    x, ok = _pattern(sig["x"], logit, 0.0, [dim])
    small = math.exp(logit)
    total = math.fsum([1.0] + [small] * (k - 1))
    view = [1] * len(shape)
    view[dim] = k
    first = torch.arange(k).reshape(view) == 0
    if log:
        ref = torch.where(first, _f64(-math.log(total)), _f64(logit - math.log(total)))
    else:
        ref = torch.where(first, _f64(1.0 / total), _f64(small / total))
    ref = ref.expand(shape)
    return _judge(op, (x, sig["dim"], False), {}, ref, MIXED if log else RELATIVE,
                  {"reduction_length": k, "layout_reproduced": ok})


def _conv_valid(sig: dict, t_out: int) -> np.ndarray:
    """`[t_out, K]`: whether tap k of output t reads the input (not padding)."""
    t_in, k = sig["x"][0][-1], sig["w"][0][-1]
    s, p, d = sig["stride"][0], sig["padding"][0], sig["dilation"][0]
    at = np.arange(t_out)[:, None] * s + np.arange(k)[None, :] * d - p
    return (at >= 0) & (at < t_in)


def _convolution(op, sig: dict) -> dict:
    """1-d convolution, two exact probes (module docstring). The weight's
    head is its first tap of its first input channel in every group."""
    x_shape, w_shape = sig["x"][0], sig["w"][0]
    if len(x_shape) != 3 or sig["transposed"] or sig["bias"] is not None:
        return {"comparison": EXACT, "pass": False, "detail": "only a 1-d, untransposed, bias-free convolution is probed"}
    per_group, k = w_shape[1], w_shape[2]
    if per_group * k < 2:
        return _no_accumulation(per_group * k)
    rest = (sig["stride"], sig["padding"], sig["dilation"], False, sig["output_padding"], sig["groups"])
    t_out = _REFERENCE_OP.get()(torch.empty(x_shape, device="meta"), torch.empty(w_shape, device="meta"), None,
                                *rest).shape[-1]
    valid = _conv_valid(sig, t_out)
    head_valid, taps_valid = valid[:, 0], valid.sum(axis=1)
    # Probe 1: x = 1, w = [h, e, ..., e]. Output t = h [first tap valid] + e * (other valid taps).
    acc = [math.fsum([HEAD_A if hv else 0.0] + [F64_EPS] * int(per_group * tv - hv))
           for hv, tv in zip(head_valid, taps_valid)]
    x1, ok1 = _pattern(sig["x"], 1.0, 1.0, [])
    w1, okw = _pattern(sig["w"], F64_EPS, HEAD_A, [1, 2])
    ref1 = torch.tensor(acc, dtype=torch.float64).expand(x_shape[0], w_shape[0], t_out)
    accumulator = _judge(op, (x1, w1, None, *rest), {}, ref1, EXACT, {})
    del x1, w1
    # Probe 2: x = 1 + 2^-26, w = [1, 0, ..., 0]. Output t = HEAD_B where the first tap is valid, else 0.
    x2, ok2 = _pattern(sig["x"], HEAD_B, HEAD_B, [])
    w2, _ = _pattern(sig["w"], 0.0, 1.0, [1, 2])
    ref2 = torch.tensor(np.where(head_valid, HEAD_B, 0.0), dtype=torch.float64).expand(x_shape[0], w_shape[0], t_out)
    operand = _judge(op, (x2, w2, None, *rest), {}, ref2, EXACT, {})
    out = {"comparison": EXACT, "bound": 0.0, "reduction_length": per_group * k,
           "error": max(accumulator["error"], operand["error"]),
           "f32_counterfactual_error": min(accumulator["f32_counterfactual_error"],
                                           operand["f32_counterfactual_error"]),
           "output_dtype": accumulator["output_dtype"],
           "probe_bytes": max(accumulator["probe_bytes"], operand["probe_bytes"]),
           "layout_reproduced": ok1 and okw and ok2,
           "sub_probes": {"accumulator_and_weight": accumulator, "input_operand": operand},
           "pass": accumulator["pass"] and operand["pass"]}
    failed = next((p for p in (accumulator, operand) if not p["pass"]), None)
    if failed is not None:
        out["detail"] = failed.get("detail", "failed")
    return out


# ------------------------------------------------------ elementwise ops

#: `F.softplus`'s defaults.
SOFTPLUS_BETA, SOFTPLUS_THRESHOLD = 1.0, 20.0


def _softplus_ref(sig: dict):
    beta, threshold = sig.get("beta", SOFTPLUS_BETA), sig.get("threshold", SOFTPLUS_THRESHOLD)
    return lambda x: x if beta * x > threshold else math.log1p(math.exp(beta * x)) / beta


def _silu_backward_ref(grad: float, x: float) -> float:
    """`grad * d/dx silu(x)`, with silu'(x) = s (1 + x (1 - s)) and s = sigmoid(x)."""
    s = 1.0 / (1.0 + math.exp(-x))
    return grad * s * (1.0 + x * (1.0 - s))


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
    # Both sides of the threshold (20): above it softplus returns x.
    "aten.softplus.default": ((-20.0, 30.0, "lin"), _softplus_ref, MIXED),
    "aten.pow.Tensor_Scalar": ((0.5, 2.0, "lin"), lambda s: (lambda x: math.pow(x, s["exponent"])), RELATIVE),
    "aten.pow.Scalar": ((0.0, 1.0, "lin"), lambda s: (lambda x: math.pow(s["base"], x)), RELATIVE),
    # silu's tangent under forward-mode AD: (grad, x) -> grad * silu'(x). Its zero crossing in x (near -1.28)
    # cancels, so the metric is absolute there.
    "aten.silu_backward.default": ((-8.0, 8.0, "lin"), lambda s: _silu_backward_ref, MIXED),
}
#: Elementwise ops of two operands `(grad, x)`: `grad` is drawn as the basic-arithmetic probe values
#: (almost none float32-representable), `x` spans the op's domain.
_GRAD_ELEMENTWISE = frozenset({"aten.silu_backward.default"})


def _domain(lo: float, hi: float, spacing: str) -> torch.Tensor:
    if spacing == "log":
        return torch.logspace(math.log10(lo), math.log10(hi), ELEMENTWISE_LENGTH, dtype=torch.float64)
    return torch.linspace(lo, hi, ELEMENTWISE_LENGTH, dtype=torch.float64)


def _elementwise(name: str, op, sig: dict) -> dict:
    (lo, hi, spacing), ref_of, metric = _ELEMENTWISE[name]
    x = _domain(lo, hi, spacing)
    fn = ref_of(sig)
    if name in _GRAD_ELEMENTWISE:
        grad = torch.from_numpy(np.random.default_rng(PROBE_SEED).uniform(-BASIC_RANGE, BASIC_RANGE,
                                                                           ELEMENTWISE_LENGTH))
        ref = torch.tensor([fn(g, v) for g, v in zip(grad.tolist(), x.tolist())], dtype=torch.float64)
        return _judge(op, (grad, x), {}, ref, metric, {"vector_length": ELEMENTWISE_LENGTH})
    ref = torch.tensor([fn(v) for v in x.tolist()], dtype=torch.float64)
    if name == "aten.pow.Tensor_Scalar":
        args, kwargs = (x, sig["exponent"]), {}
    elif name == "aten.pow.Scalar":
        args, kwargs = (sig["base"], x), {}
    elif name == "aten.gelu.default":
        args, kwargs = (x,), {"approximate": sig["approximate"]}
    elif name == "aten.softplus.default":
        args, kwargs = (x, sig.get("beta", SOFTPLUS_BETA), sig.get("threshold", SOFTPLUS_THRESHOLD)), {}
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
    # Forward-mode AD tangent formulas (#897): pow's 2 x t, rsqrt's -0.5 r^3 t, softmax's in-place division.
    "aten.mul.Scalar": (2, np.multiply),
    "aten.div_.Tensor": (2, np.divide),
    # The primitives torch._refs lowers add and mul to. In the forward-AD trace they run only on meta tensors
    # (the ZeroTensor kernels' shape inference: op_audit.SHAPE_INFERENCE_OPS); on data they are probed here.
    "prims.add.default": (2, np.add),
    "prims.mul.default": (2, np.multiply),
}
#: Basic ops that write their result into their first operand: probed on a copy of it.
_IN_PLACE = frozenset({"aten.div_.Tensor"})


def _basic(name: str, op, sig: dict) -> dict:
    """Bit-exact against numpy on fixed random operands. A scalar variant
    (`sig["scalar_operand"]`) takes the observed Python scalar there."""
    arity, ref_fn = _BASIC[name]
    rng = np.random.default_rng(PROBE_SEED)
    xs = [rng.uniform(-BASIC_RANGE, BASIC_RANGE, ELEMENTWISE_LENGTH) for _ in range(arity)]
    at = sig.get("scalar_operand")
    operands = [sig["scalar"] if i == at else x for i, x in enumerate(xs)]
    ref = torch.from_numpy(np.asarray(ref_fn(*operands), dtype=np.float64))
    args = tuple(o if i == at else torch.from_numpy(o) for i, o in enumerate(operands))
    if name in _IN_PLACE:
        inner = op

        def op(first, *rest, **kwargs):
            return inner(first.clone(), *rest, **kwargs)

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
    "aten.convolution.default": _convolution,
}

#: Every probed op, by class.
CLASSES = {ACCUMULATING: frozenset(_ACCUMULATING), ELEMENTWISE: frozenset(_ELEMENTWISE),
           BASIC: frozenset(_BASIC)}
PROBED = frozenset().union(*CLASSES.values())


def class_of(name: str) -> str | None:
    return next((c for c, names in CLASSES.items() if name in names), None)


def resolve(name: str):
    """The OpOverload named `<namespace>.<op>.<overload>` (`aten` or `prims`)."""
    namespace, packet, overload = name.split(".")
    return getattr(getattr(getattr(torch.ops, namespace), packet), overload)


def probe(name: str, sig: dict, op=None) -> dict:
    """Run the probe for `name` at `sig`. `op` defaults to the ATen overload
    itself; the tests pass a deliberately imprecise one."""
    reference = resolve(name)
    op = op or reference
    kind = class_of(name)
    token = _REFERENCE_OP.set(reference)
    try:
        return _probe(name, sig, op, kind)
    finally:
        _REFERENCE_OP.reset(token)


def _probe(name: str, sig: dict, op, kind: str) -> dict:
    with torch.no_grad():
        if kind == ACCUMULATING:
            out = _ACCUMULATING[name](op, sig)
        elif kind == ELEMENTWISE:
            out = _elementwise(name, op, sig)
        elif kind == BASIC:
            out = _basic(name, op, sig)
        else:
            raise KeyError(f"no probe for {name}")
    return {"op": name, "class": kind, "signature": sig, **out}
