"""GIL release and thread-safety of the native bindings.

A background "ticker" thread increments a counter and sleeps briefly, so it
needs the GIL for a moment roughly every millisecond. If a native call
releases the GIL, the ticker keeps advancing while the call runs; if the
call holds the GIL, the ticker is frozen until it returns (it can only
slip in a tick or two at the call boundaries).

The instrument is checked against a negative control — `sorted` over a
large list, a C call that holds the GIL throughout — before its positive
reading is trusted.
"""

import random
import shutil
import threading
import time

import numpy as np
import pytest

import larql
from synthetic_vindex import build_synthetic_vindex

# A layer big enough that one gate KNN is real work (4096 × 256 MACs).
LARGE_LAYERS = 4
LARGE_HIDDEN = 256
LARGE_FEATURES = 4096
SMALL_HIDDEN = 32
SMALL_FEATURES = 16
VOCAB = 100
FREE_SLOTS = 4

TICK_SLEEP_SECONDS = 0.0005
# Every measured call is scaled until it lasts at least this long.
TARGET_CALL_SECONDS = 0.2
MAX_CALIBRATION_DOUBLINGS = 20
# Released: the ticker runs for the whole call (hundreds of ticks).
MIN_TICKS_WHEN_RELEASED = 20
# Held: at most a tick at each boundary of the call.
MAX_TICKS_WHEN_HELD = 3
CONCURRENT_INSERT_ROUNDS = 5


class Ticker:
    """Counts how often a background thread got the GIL."""

    def __init__(self):
        self.count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.count += 1
            time.sleep(TICK_SLEEP_SECONDS)

    def __enter__(self):
        self._thread.start()
        while self.count == 0:
            time.sleep(TICK_SLEEP_SECONDS)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def ticks_during(self, fn):
        before = self.count
        fn()
        return self.count - before


def calibrated(make_call):
    """Return a zero-arg call built by `make_call(scale)` whose duration is at
    least TARGET_CALL_SECONDS, doubling `scale` until it is."""
    scale = 1
    for _ in range(MAX_CALIBRATION_DOUBLINGS):
        call = make_call(scale)
        start = time.perf_counter()
        call()
        if time.perf_counter() - start >= TARGET_CALL_SECONDS:
            return call
        scale *= 2
    pytest.fail("call never reached the calibration target duration")


def _build(root, hidden, features):
    build_synthetic_vindex(
        str(root),
        num_layers=LARGE_LAYERS,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        vocab_size=VOCAB,
        num_features=features,
        free_per_layer=FREE_SLOTS,
    )
    return str(root)


@pytest.fixture(scope="module")
def large_vindex(tmp_path_factory):
    root = tmp_path_factory.mktemp("large_vindex")
    yield larql.load(_build(root, LARGE_HIDDEN, LARGE_FEATURES))
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def small_vindex_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("small_vindex")
    yield _build(root, SMALL_HIDDEN, SMALL_FEATURES)
    shutil.rmtree(root, ignore_errors=True)


def _walk_call(vindex, scale):
    residual = np.ones(LARGE_HIDDEN, dtype=np.float32).tolist()
    layers = list(range(LARGE_LAYERS)) * scale
    return lambda: vindex.walk(residual, layers=layers, top_k=1)


# ── GIL release ──


def test_instrument_detects_a_gil_holding_call():
    def make_sort(scale):
        data = [random.random() for _ in range(50_000 * scale)]
        return lambda: sorted(data)

    call = calibrated(make_sort)
    with Ticker() as ticker:
        ticks = ticker.ticks_during(call)
    assert ticks <= MAX_TICKS_WHEN_HELD, ticks


def test_walk_releases_the_gil(large_vindex):
    call = calibrated(lambda scale: _walk_call(large_vindex, scale))
    with Ticker() as ticker:
        ticks = ticker.ticks_during(call)
    assert ticks >= MIN_TICKS_WHEN_RELEASED, ticks


def test_concurrent_walks_agree_with_a_serial_walk(large_vindex):
    residual = np.ones(LARGE_HIDDEN, dtype=np.float32).tolist()
    expected = [(h.layer, h.feature) for h in large_vindex.walk(residual, top_k=3)]
    results = [None] * 4

    def worker(i):
        results[i] = [(h.layer, h.feature) for h in large_vindex.walk(residual, top_k=3)]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(results))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [expected] * len(results)


# ── Mutation under concurrent Python threads ──


def _run_together(target, n):
    """Start `n` threads on `target(i)` behind a barrier; return their errors."""
    barrier = threading.Barrier(n)
    errors = []

    def run(i):
        try:
            barrier.wait()
            target(i)
        except Exception as e:  # surfaced to the caller with the thread index
            errors.append((i, e))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_inserts_into_distinct_layers_all_land(small_vindex_path):
    for _ in range(CONCURRENT_INSERT_ROUNDS):
        vindex = larql.load(small_vindex_path)
        claimed = [None] * LARGE_LAYERS

        def insert(i):
            claimed[i] = vindex.insert(f"entity{i}", "rel", f"t{i}", layer=i)

        assert _run_together(insert, LARGE_LAYERS) == []
        for i, (layer, feature) in enumerate(claimed):
            assert layer == i
            assert vindex.feature_meta(layer, feature).top_token == f"t{i}"


def test_concurrent_writes_to_distinct_slots_all_land(small_vindex_path):
    vindex = larql.load(small_vindex_path)
    slots = [(layer, feature) for layer in range(LARGE_LAYERS) for feature in range(SMALL_FEATURES)]

    def write(i):
        layer, feature = slots[i]
        vindex.set_gate_vector(layer, feature, [float(i)] * SMALL_HIDDEN)
        vindex.set_feature_meta(layer, feature, f"s{i}")

    assert _run_together(write, len(slots)) == []
    for i, (layer, feature) in enumerate(slots):
        assert vindex.feature_meta(layer, feature).top_token == f"s{i}"
        assert vindex.gate_vector(layer, feature)[0] == float(i)


def test_reads_run_alongside_inserts(small_vindex_path):
    vindex = larql.load(small_vindex_path)
    residual = np.ones(SMALL_HIDDEN, dtype=np.float32).tolist()
    errors = []

    def reader():
        try:
            for _ in range(200):
                vindex.walk(residual, top_k=2)
                vindex.describe("a", band="all")
        except Exception as e:  # surfaced below
            errors.append(e)

    def writer():
        try:
            for i in range(FREE_SLOTS):
                vindex.insert(f"w{i}", "rel", f"x{i}", layer=1)
                vindex.set_feature_meta(2, i, f"m{i}")
        except Exception as e:  # surfaced below
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    threads.append(threading.Thread(target=writer))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert [vindex.feature_meta(2, i).top_token for i in range(FREE_SLOTS)] == [
        f"m{i}" for i in range(FREE_SLOTS)
    ]


@pytest.fixture(scope="module")
def v3_session(tmp_path_factory):
    root = tmp_path_factory.mktemp("v3_container")
    yield larql.session(larql._native._write_test_container(str(root)))
    shutil.rmtree(root, ignore_errors=True)


def _capture_call(session, scale):
    tokens = [3, 7, 11, 19] * scale
    return lambda: session.capture(
        "residual", "The capital of France is", layer=1, continue_tokens=tokens, positions="ALL"
    )


def test_capture_releases_the_gil(v3_session):
    call = calibrated(lambda scale: _capture_call(v3_session, scale))
    with Ticker() as ticker:
        ticks = ticker.ticks_during(call)
    assert ticks >= MIN_TICKS_WHEN_RELEASED, ticks
