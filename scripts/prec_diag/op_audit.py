"""A64 audit part 5: an operation-level dtype audit of the forward (#897).

Loading a model in float64 does not make its forward compute in float64. A
modeling file can cast to float32 inside a module and back before the module
returns. The module-boundary audit (`hf_oracle.module_dtype_hooks`, part 2)
then sees only float64 tensors, so it cannot tell. This part looks at every
operation instead.

`OpDtypeAudit` is a `TorchDispatchMode`, active only around the A64
forward. For every ATen op it records:

- **dtypes**: any floating tensor the op reads, returns or writes that is not
  float64 (complex: not complex128), and any floating `dtype=` argument that
  is not float64. Each is a fault.
- **casts**: every dtype conversion: `_to_copy` / `.to` / `.float()`, a
  `copy_` between dtypes, or an op whose `dtype=` differs from its inputs.
  Each is recorded with its Python call site, so a cast that is undone later
  (`x.float()...double()`) still shows twice: once at each end. Integer and
  bool to float64 conversions are recorded and are not faults.
- **writes**: the dtype of every tensor an op mutates (in-place and `out=`
  arguments, from the op schema). Recurrent and cached state is stored in
  such writes, so a float32 state buffer is a fault even when the value
  written into it came from float64.
- **classes**: every op that touches a floating tensor must be in a class
  with precision evidence:
  - `exact`: no arithmetic (views, copies, gathers, comparisons, selection,
    conversions, factories);
  - `basic_arithmetic`, `elementwise`, `accumulating`: probed after the
    forward (`op_probes.py`), at every distinct shape and stride the forward
    used for accumulating ops.
  An op in no class is a fault ("no precision evidence"), so a new op is
  never passed silently.

What the dispatch level cannot see is listed in `VISIBILITY`, and recorded in
every result. The main limit is that arithmetic inside one kernel (an `mm`'s
accumulator, `_softmax`'s running sum) is not dispatched; the probes measure
it instead, rather than assuming it. (ATen's CPU accumulate type for double
is double; the probes are the evidence that this build behaves so.)

The part passes only if:
- the trace is not partial: it saw at least one floating op, and at least as
  many linear-layer matmuls (`mm`/`addmm`; `bmm` is attention's and does not
  count) as the model has `nn.Linear` modules (`linear_layers`, counted from
  the loaded model). A dispatch mode is
  thread-local, so work run on another thread is never seen; this check is
  what catches a forward that ran mostly elsewhere;
- no dtype fault was recorded;
- no op was unclassified;
- every accumulating op kind seen has a probe, and every probe passed.

The result records the probes' memory (`memory`): the largest probe's own
allocation and the process peak RSS before and after the probes, so a freeze
can set its RSS limit knowingly.
"""

from __future__ import annotations

import collections
import linecache
import re
import resource
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prec_diag import op_probes  # noqa: E402

#: The part's record schema. A freeze requires the part by naming it
#: (`values.a64_op_audit`); the runner forwards that value to the oracle,
#: which refuses any schema but this one, so this is its only definition.
SCHEMA = "larql.prec-diag-1.a64-op-audit/1"
REQUIRED = {False: torch.float64, True: torch.complex128}  # keyed by is_complex
#: Python frames recorded per site, innermost first.
CALLSITE_FRAMES = 3
#: Characters of source text kept per site.
CALLSITE_TEXT = 120
#: Rows kept per table in the result (problem_count and the totals are never truncated).
MAX_ROWS = 500
#: Problems kept in the result's `problems` list (as the other audit parts).
MAX_PROBLEMS = 50
#: Ops that run an `nn.Linear` (F.linear decomposes into t + mm, or addmm with a bias).
MATMUL_OPS = frozenset({"aten.mm.default", "aten.addmm.default", "aten.bmm.default"})
# What an `nn.Linear` dispatches to (F.linear -> t + mm, or addmm with a bias). The partial-trace floor
# counts only these: attention's bmm would otherwise give slack for linear work moved off the trace.
LINEAR_OPS = frozenset({"aten.mm.default", "aten.addmm.default"})
#: `ru_maxrss` is bytes on macOS and KiB on Linux.
RSS_UNIT = 1 if sys.platform == "darwin" else 1024

#: Op classes besides the probed ones (`op_probes.CLASSES`).
EXACT, UNCLASSIFIED = "exact", "unclassified"
#: Ops that convert between dtypes.
CAST_OPS = frozenset({"aten._to_copy.default", "aten.to.dtype", "aten.to.dtype_layout", "aten.to.other",
                      "aten.copy_.default", "aten.copy.default", "aten._to_dtype.default"})
#: Ops that read floating tensors without arithmetic on their values.
EXACT_OPS = frozenset({
    # views and layout
    "aten.view.default", "aten._unsafe_view.default", "aten.reshape.default", "aten.expand.default",
    "aten.transpose.int", "aten.t.default", "aten.permute.default", "aten.unsqueeze.default",
    "aten.squeeze.dim", "aten.squeeze.default", "aten.squeeze.dims", "aten.slice.Tensor", "aten.select.int",
    "aten.narrow.default", "aten.alias.default", "aten.detach.default", "aten.as_strided.default",
    "aten.unbind.int", "aten.split.Tensor", "aten.split_with_sizes.default", "aten.chunk.default",
    "aten.flip.default",
    # copies, gathers and assembly
    "aten.clone.default", "aten.contiguous.default", "aten.cat.default", "aten.stack.default",
    "aten.embedding.default", "aten.index.Tensor", "aten.index_select.default", "aten.gather.default",
    "aten.repeat.default", "aten.repeat_interleave.self_int", "aten.tril.default", "aten.triu.default",
    "aten._local_scalar_dense.default",
    # selection and comparison
    "aten.where.self", "aten.masked_fill.Scalar", "aten.masked_fill.Tensor", "aten.masked_fill_.Scalar",
    "aten.eq.Scalar", "aten.eq.Tensor", "aten.ne.Scalar", "aten.ne.Tensor", "aten.lt.Scalar",
    "aten.lt.Tensor", "aten.le.Scalar", "aten.le.Tensor", "aten.gt.Scalar", "aten.gt.Tensor",
    "aten.ge.Scalar", "aten.ge.Tensor", "aten.isinf.default", "aten.isnan.default", "aten.isfinite.default",
    "aten.any.default", "aten.all.default", "aten.max.default", "aten.min.default", "aten.amax.default",
    "aten.amin.default", "aten.argmax.default",
    # factories and fills
    "aten.scalar_tensor.default", "aten.lift_fresh.default", "aten.lift_fresh_copy.default",
    "aten.arange.default", "aten.arange.start", "aten.arange.start_step", "aten.full.default",
    "aten.zeros.default", "aten.ones.default", "aten.empty.memory_format", "aten.empty_strided.default",
    "aten.zeros_like.default", "aten.ones_like.default", "aten.empty_like.default", "aten.full_like.default",
    "aten.new_ones.default", "aten.new_zeros.default", "aten.new_empty.default", "aten.new_full.default",
    "aten.fill_.Scalar", "aten.zero_.default",
}) | CAST_OPS

#: What the dispatch level sees and does not see; recorded in every result.
VISIBILITY = {
    "sees": [
        "every ATen op dispatched during the A64 forward, after autograd and CompositeImplicitAutograd "
        "decomposition (F.linear appears as t + mm, a matmul as mm/bmm, softmax(dtype=) as _to_copy + _softmax)",
        "the dtype of every tensor each op reads, returns or writes (in-place and out= targets, from the op "
        "schema), and every floating dtype= argument",
        "every dtype conversion, with the Python call site that issued it, so a cast that is undone later still "
        "appears at both ends",
        "tensors created inside the forward in the default dtype (float32), e.g. torch.tensor(x) or torch.zeros(n)",
    ],
    "does_not_see": [
        "arithmetic inside one kernel (accumulators, internal temporaries of mm, mean, _softmax, cumsum, "
        "cumprod): measured instead by the kernel probes, after the forward, at every distinct shape and "
        "stride the forward used, on the same backend and threads",
        "elementwise kernels at shapes other than the probe vector: elementwise probes run once per scalar "
        "variant on a fixed-length contiguous vector",
        "computation outside torch (numpy, Python math): in the oracle these are the patched functions' "
        "float64 numpy witnesses and config scalars",
        "tensors created before the trace starts (parameters, buffers): audit part 1 checks every floating "
        "parameter and buffer is float64, and every parameter against the checkpoint",
        "a float64 tensor holding a value already rounded to float32 (e.g. a buffer computed in float32 at "
        "construction and widened later): dtype is not precision. Not observed under A64's load: "
        "from_pretrained(dtype=float64) builds buffers in float64, and the backfill records Gemma 3's "
        "embed_scale as sqrt(hidden) in float64",
        "state carried between forwards (KV or recurrent caches across decode steps): A64 is one prefill with "
        "use_cache=False; within it, every state tensor is produced and written by traced ops",
        "work on another Python thread: a dispatch mode is thread-local. The part fails a trace with fewer "
        "mm/addmm than the model has nn.Linear modules (attention's bmm does not count), which catches a "
        "forward, or any nn.Linear, run elsewhere; non-linear work (a norm, an activation) moved to another "
        "thread is still not seen",
    ],
}


def _bad(t: torch.Tensor) -> bool:
    return (t.is_floating_point() or t.is_complex()) and t.dtype != REQUIRED[t.is_complex()]


def _floating(t) -> bool:
    return isinstance(t, torch.Tensor) and (t.is_floating_point() or t.is_complex())


def _floating_dtype(d) -> bool:
    return isinstance(d, torch.dtype) and (d.is_floating_point or d.is_complex)


def _non_f64_dtype(d) -> bool:
    return _floating_dtype(d) and d != REQUIRED[d.is_complex]


_TORCH_DIR = str(Path(torch.__file__).resolve().parent)
_OWN_FILES = {str(Path(__file__).resolve()), str(Path(op_probes.__file__).resolve())}


def call_site() -> list[dict]:
    """The innermost Python frames outside torch and this module."""
    frame, chain = sys._getframe(1), []
    while frame is not None and len(chain) < CALLSITE_FRAMES:
        filename = frame.f_code.co_filename
        if not (filename.startswith(_TORCH_DIR) or filename in _OWN_FILES):
            text = linecache.getline(filename, frame.f_lineno).strip()[:CALLSITE_TEXT]
            chain.append({"file": Path(filename).name, "line": frame.f_lineno,
                          "function": frame.f_code.co_qualname, "code": text})
        frame = frame.f_back
    return chain


def site_name(chain: list[dict]) -> str:
    if not chain:
        return "<no python frame>"
    c = chain[0]
    return f"{c['file']}:{c['line']} {c['function']}"


def _cast(name: str, args: tuple, ins: list, outs: list, dtype_arg) -> tuple[str, str, bool] | None:
    """(from, to, touches a non-float64 floating dtype) when the op converts
    between dtypes, else None. A `.to` to the dtype a tensor already has
    does not dispatch, so is never seen."""
    if name in ("aten.copy_.default", "aten.copy.default"):
        src, dst = args[1], args[0]
        if not (isinstance(src, torch.Tensor) and src.dtype != dst.dtype):
            return None
        frm, to = {src.dtype}, {dst.dtype}
    elif name in CAST_OPS or (dtype_arg is not None and ins and all(t.dtype != dtype_arg for t in ins)):
        frm, to = {t.dtype for t in ins}, {t.dtype for t in outs} or {dtype_arg}
        if not ins or frm == to:
            return None
    else:
        return None
    names = lambda ds: ",".join(sorted(str(d) for d in ds))  # noqa: E731
    return names(frm), names(to), any(_non_f64_dtype(d) for d in frm | to)


def _written(func, args: tuple, kwargs: dict) -> list[torch.Tensor]:
    """The tensors `func` mutates, from its schema's alias annotations."""
    out = []
    for i, a in enumerate(func._schema.arguments):
        if a.alias_info is not None and a.alias_info.is_write:
            value = args[i] if i < len(args) else kwargs.get(a.name)
            out += [t for t in tree_leaves(value) if isinstance(t, torch.Tensor)]
    return out


class OpDtypeAudit(TorchDispatchMode):
    """Audit part 5, collected during the forward; `finish()` runs the probes."""

    def __init__(self, linear_layers: int = 0):
        """`linear_layers`: the number of `nn.Linear` modules the traced forward
        runs; the trace must show at least that many matmuls."""
        super().__init__()
        self.linear_layers = linear_layers
        self.ops_traced = 0
        self.floating_ops = 0
        self.non_f64_ops = 0
        self.op_counts: collections.Counter = collections.Counter()
        self.violations: collections.Counter = collections.Counter()
        self.casts: collections.Counter = collections.Counter()
        self.writes: dict = collections.defaultdict(collections.Counter)
        self.unclassified: collections.Counter = collections.Counter()
        self.chains: dict = {}
        self.cast_non_f64: dict = {}
        self.signatures: dict = collections.defaultdict(dict)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.observe(func, args, kwargs, out)
        return out

    # ------------------------------------------------------------- observe

    def observe(self, func, args: tuple, kwargs: dict, out) -> None:
        self.ops_traced += 1
        name = str(func)
        ins = [t for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)]
        outs = [t for t in tree_leaves(out) if isinstance(t, torch.Tensor)]
        dtype_arg = kwargs.get("dtype")
        if not any(_floating(t) for t in ins + outs) and not _floating_dtype(dtype_arg):
            return
        self.floating_ops += 1
        self.op_counts[name] += 1
        faults = [("input", t.dtype) for t in ins if _bad(t)] + [("output", t.dtype) for t in outs if _bad(t)]
        if _non_f64_dtype(dtype_arg):
            faults.append(("dtype_argument", dtype_arg))
        for t in _written(func, args, kwargs):
            self.writes[name][str(t.dtype)] += 1
            if _bad(t):
                faults.append(("write", t.dtype))
        cast = _cast(name, args, ins, outs, dtype_arg)
        if faults or cast:
            chain = call_site()
            site = site_name(chain)
            for role, dtype in sorted(set(faults), key=str):
                key = (name, role, str(dtype), site)
                self.chains.setdefault(("violation",) + key, chain)
                self.violations[key] += 1
            if cast:
                key = (name, *cast[:2], site)
                self.cast_non_f64[key] = cast[2]
                self.chains.setdefault(("cast",) + key, chain)
                self.casts[key] += 1
        if faults:
            self.non_f64_ops += 1
        elif name not in EXACT_OPS and op_probes.class_of(name) is None:
            self.unclassified[name] += 1
        elif op_probes.class_of(name) is not None:
            sig = op_probes.signature(name, args, kwargs)
            self.signatures[name].setdefault(repr(sig), sig)

    # -------------------------------------------------------------- finish

    def _rows(self, counter, fields) -> list[dict]:
        rows = []
        for key, count in sorted(counter.items(), key=lambda kv: (kv[0][-1], kv[0])):
            kind = "cast" if fields[1] == "from" else "violation"
            rows.append(dict(zip(fields, key)) | {"count": count, "call_stack": self.chains[(kind,) + key]})
        return rows

    def finish(self) -> dict:
        """The part's record; runs every probe the trace calls for."""
        # Structural problems first: the record keeps only the first MAX_PROBLEMS, and a partial or vacuous
        # trace must never be truncated away behind per-op faults.
        problems = []
        if self.floating_ops == 0:
            problems.append("the trace saw no floating op: the audit was not active during the forward")
        matmuls = sum(self.op_counts[name] for name in LINEAR_OPS)
        if matmuls < self.linear_layers:
            problems.append(f"partial trace: {matmuls} mm/addmm for {self.linear_layers} nn.Linear modules "
                            "(work on another thread, or outside the trace, is not seen)")
        violation_rows = self._rows(self.violations, ("op", "role", "dtype", "site"))
        for r in violation_rows:
            problems.append(f"{r['op']}: non-float64 {r['role']} {r['dtype']} at {r['site']} (x{r['count']})")
        for name, count in sorted(self.unclassified.items()):
            problems.append(f"{name}: no precision evidence (not exact, not probed) (x{count})")
        probes = []
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RSS_UNIT
        for name in sorted(self.signatures):
            for sig in self.signatures[name].values():
                result = op_probes.probe(name, sig)
                probes.append(result)
                if not result["pass"]:
                    problems.append(f"probe {name} {sig}: {result.get('detail', 'failed')}")
        memory = {"largest_probe_bytes": max((p.get("probe_bytes", 0) for p in probes), default=0),
                  "peak_rss_bytes_before_probes": rss_before,
                  "peak_rss_bytes_after_probes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RSS_UNIT}
        probed = {p["op"] for p in probes}
        for name in sorted(self.op_counts):
            if op_probes.class_of(name) == op_probes.ACCUMULATING and name not in probed:
                problems.append(f"{name}: accumulating op with no passing probe (it only ran with faults)")
        cast_rows = self._rows(self.casts, ("op", "from", "to", "site"))
        for r in cast_rows:
            r["non_f64"] = self.cast_non_f64[(r["op"], r["from"], r["to"], r["site"])]
        by_class = collections.Counter()
        for name, count in self.op_counts.items():
            by_class[op_probes.class_of(name) or (EXACT if name in EXACT_OPS else UNCLASSIFIED)] += count
        return {"schema": SCHEMA, "pass": not problems, "problems": problems[:MAX_PROBLEMS],
                "problem_count": len(problems), "ops_traced": self.ops_traced,
                "floating_ops": self.floating_ops, "non_f64_ops": self.non_f64_ops,
                "op_counts": dict(sorted(self.op_counts.items())), "ops_by_class": dict(sorted(by_class.items())),
                "violations": violation_rows[:MAX_ROWS], "violation_sites": len(violation_rows),
                "casts": cast_rows[:MAX_ROWS], "cast_sites": len(cast_rows),
                "writes": {k: dict(v) for k, v in sorted(self.writes.items())},
                "unclassified": dict(sorted(self.unclassified.items())),
                "probes": probes, "probes_run": len(probes), "memory": memory,
                "matmuls": matmuls, "linear_layers": self.linear_layers,
                "environment": {"torch": torch.__version__, "threads": torch.get_num_threads(),
                                "blas": blas_info()},
                "visibility": VISIBILITY}


def blas_info() -> str:
    """The BLAS this torch build calls, from its build configuration
    (`  - Build settings: BLAS_INFO=accelerate, ...`)."""
    found = re.search(r"BLAS_INFO=([^,\s]+)", torch.__config__.show())
    return found.group(1) if found else "unknown"
