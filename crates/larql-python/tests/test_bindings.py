"""
Tests for larql Python bindings.

Uses a synthetic vindex built in a temp directory — no dependency on
real model files. Tests run anywhere.

Run: pytest crates/larql-python/tests/ -v
"""

import os
import json
import struct
import tempfile
import shutil
import pytest
import numpy as np

import larql

# ── Synthetic vindex fixture ──

NUM_LAYERS = 4
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 64
VOCAB_SIZE = 100
NUM_FEATURES = 16
EMBED_SCALE = 1.0


def _write_f32(path, data):
    """Write flat f32 array to binary file."""
    arr = np.array(data, dtype=np.float32)
    arr.tofile(str(path))


@pytest.fixture(scope="module")
def vindex_path():
    """Build a minimal synthetic vindex in a temp directory."""
    tmpdir = tempfile.mkdtemp(prefix="larql_test_")

    # index.json
    config = {
        "version": 1,
        "model": "test/synthetic-4l",
        "family": "test",
        "num_layers": NUM_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "embed_scale": EMBED_SCALE,
        "extract_level": "browse",
        "dtype": "f32",
        "down_top_k": 3,
        "has_model_weights": False,
        "layers": [],
        "layer_bands": {
            "syntax": [0, 1],
            "knowledge": [2, 3],
            "output": [3, 3],
        },
    }

    # Build gate vectors + layer info
    gate_data = []
    for layer in range(NUM_LAYERS):
        offset = len(gate_data) * 4
        for feat in range(NUM_FEATURES):
            # Each gate vector: a simple pattern so KNN results are predictable
            vec = np.zeros(HIDDEN_SIZE, dtype=np.float32)
            vec[feat % HIDDEN_SIZE] = 1.0 + layer * 0.1
            vec[(feat + 1) % HIDDEN_SIZE] = 0.5
            gate_data.extend(vec.tolist())

        config["layers"].append({
            "layer": layer,
            "num_features": NUM_FEATURES,
            "offset": offset,
            "length": NUM_FEATURES * HIDDEN_SIZE * 4,
        })

    with open(os.path.join(tmpdir, "index.json"), "w") as f:
        json.dump(config, f)

    # gate_vectors.bin
    _write_f32(os.path.join(tmpdir, "gate_vectors.bin"), gate_data)

    # embeddings.bin — simple identity-like embeddings
    embed_data = []
    for tok in range(VOCAB_SIZE):
        vec = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        vec[tok % HIDDEN_SIZE] = 1.0
        embed_data.extend(vec.tolist())
    _write_f32(os.path.join(tmpdir, "embeddings.bin"), embed_data)

    # down_meta.bin — binary format: per-feature records
    # Each record: top_token_id(u32) + c_score(f32) + top_k * (token_id(u32) + logit(f32))
    top_k_count = 3
    record_size = 8 + top_k_count * 8
    # Match format/down_meta: file header, then a feature count per layer.
    meta_data = bytearray(struct.pack("<IIII", 0x444D4554, 1, NUM_LAYERS, top_k_count))
    for layer in range(NUM_LAYERS):
        meta_data.extend(struct.pack("<I", NUM_FEATURES))
        for feat in range(NUM_FEATURES):
            # Leave last 4 features per layer empty (for INSERT tests)
            if feat >= NUM_FEATURES - 4:
                record = b"\x00" * record_size
            else:
                token_id = (layer * NUM_FEATURES + feat) % VOCAB_SIZE
                c_score = 0.5 + feat * 0.01
                record = struct.pack("<If", token_id, c_score)
                for k in range(top_k_count):
                    tid = (token_id + k + 1) % VOCAB_SIZE
                    logit = c_score - k * 0.1
                    record += struct.pack("<If", tid, logit)
            meta_data.extend(record)

    with open(os.path.join(tmpdir, "down_meta.bin"), "wb") as f:
        f.write(meta_data)

    # Minimal tokenizer.json — character-level so any input works
    # Build a vocab of individual characters + some words
    vocab = {}
    idx = 0
    # Single characters cover a-z, A-Z, 0-9, space, punct
    for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-_":
        vocab[c] = idx
        idx += 1
    # Pad to VOCAB_SIZE
    while idx < VOCAB_SIZE:
        vocab[f"<t{idx}>"] = idx
        idx += 1

    tokenizer_config = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [],
        },
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
    }
    with open(os.path.join(tmpdir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_config, f)

    yield tmpdir

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def vindex(vindex_path):
    """Load the synthetic vindex."""
    return larql.load(vindex_path)


# ─── Loading & Properties ───

class TestLoading:
    def test_load(self, vindex):
        assert vindex is not None
        assert repr(vindex).startswith("Vindex(")

    def test_properties(self, vindex):
        assert vindex.num_layers == NUM_LAYERS
        assert vindex.hidden_size == HIDDEN_SIZE
        assert vindex.vocab_size == VOCAB_SIZE
        assert vindex.model == "test/synthetic-4l"
        assert vindex.family == "test"
        assert vindex.total_gate_vectors == NUM_LAYERS * NUM_FEATURES

    def test_loaded_layers(self, vindex):
        layers = vindex.loaded_layers
        assert len(layers) == NUM_LAYERS
        assert layers == list(range(NUM_LAYERS))

    def test_num_features(self, vindex):
        for layer in range(NUM_LAYERS):
            assert vindex.num_features(layer) == NUM_FEATURES

    def test_stats(self, vindex):
        s = vindex.stats()
        assert s["model"] == "test/synthetic-4l"
        assert s["num_layers"] == NUM_LAYERS
        assert s["hidden_size"] == HIDDEN_SIZE
        assert s["total_gate_vectors"] == NUM_LAYERS * NUM_FEATURES

    def test_layer_bands(self, vindex):
        bands = vindex.layer_bands()
        assert bands is not None
        assert bands["syntax"] == (0, 1)
        assert bands["knowledge"] == (2, 3)


# ─── Embeddings ───

class TestEmbeddings:
    def test_embed(self, vindex):
        embed = vindex.embed("hello")
        assert isinstance(embed, np.ndarray)
        assert embed.shape == (HIDDEN_SIZE,)
        assert embed.dtype == np.float32
        assert np.linalg.norm(embed) > 0

    def test_embed_different_tokens(self, vindex):
        a = vindex.embed("hello")
        b = vindex.embed("world")
        assert not np.allclose(a, b)

    def test_tokenize(self, vindex):
        ids = vindex.tokenize("hello world")
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_decode(self, vindex):
        ids = vindex.tokenize("hello")
        text = vindex.decode(ids)
        assert isinstance(text, str)

    def test_embedding_by_id(self, vindex):
        embed = vindex.embedding(token_id=0)
        assert embed.shape == (HIDDEN_SIZE,)

    def test_embedding_out_of_range(self, vindex):
        with pytest.raises(ValueError):
            vindex.embedding(token_id=999999)


# ─── Gate Vectors ───

class TestGateVectors:
    def test_gate_vector_single(self, vindex):
        vec = vindex.gate_vector(layer=0, feature=0)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (HIDDEN_SIZE,)

    def test_gate_vectors_layer(self, vindex):
        mat = vindex.gate_vectors(layer=0)
        assert mat.shape == (NUM_FEATURES, HIDDEN_SIZE)
        assert mat.dtype == np.float32

    def test_gate_vector_invalid_layer(self, vindex):
        with pytest.raises(ValueError):
            vindex.gate_vector(layer=999, feature=0)

    def test_gate_vector_invalid_feature(self, vindex):
        with pytest.raises(ValueError):
            vindex.gate_vector(layer=0, feature=999)

    def test_gate_vectors_match_singles(self, vindex):
        """Bulk gate_vectors should match individual gate_vector calls."""
        mat = vindex.gate_vectors(layer=0)
        for feat in range(min(5, NUM_FEATURES)):
            single = vindex.gate_vector(layer=0, feature=feat)
            np.testing.assert_allclose(mat[feat], single)


# ─── KNN & Walk ───

class TestKNN:
    def test_gate_knn(self, vindex):
        embed = vindex.embed("hello")
        hits = vindex.gate_knn(layer=0, query_vector=embed.tolist(), top_k=5)
        assert len(hits) <= 5
        assert all(isinstance(h, tuple) and len(h) == 2 for h in hits)
        # Sorted by absolute score
        scores = [abs(s) for _, s in hits]
        assert scores == sorted(scores, reverse=True)

    def test_entity_knn(self, vindex):
        hits = vindex.entity_knn("hello", layer=0, top_k=5)
        assert len(hits) > 0

    def test_walk(self, vindex):
        embed = vindex.embed("hello")
        hits = vindex.walk(embed.tolist(), top_k=3)
        # May be empty if down_meta tokens don't pass content filter
        assert isinstance(hits, list)
        for h in hits:
            assert hasattr(h, "layer")
            assert hasattr(h, "gate_score")

    def test_entity_walk(self, vindex):
        hits = vindex.entity_walk("hello", layers=[0, 1], top_k=3)
        assert isinstance(hits, list)

    def test_walk_hit_properties(self, vindex):
        """Test WalkHit properties using raw walk (bypasses content filter)."""
        embed = vindex.embed("hello")
        # Use gate_knn + feature_meta directly to test WalkHit structure
        hits = vindex.gate_knn(layer=0, query_vector=embed.tolist(), top_k=3)
        assert len(hits) > 0
        feat, score = hits[0]
        assert isinstance(feat, int)
        assert isinstance(score, float)


# ─── Feature Metadata ───

class TestFeatures:
    def test_feature_meta(self, vindex):
        meta = vindex.feature_meta(0, 0)
        # May or may not work with synthetic down_meta.bin depending on tokenizer
        # Just check it doesn't crash
        assert meta is None or isinstance(meta.top_token, str)

    def test_feature_dict(self, vindex):
        d = vindex.feature(0, 0)
        if d is not None:
            assert "top_token" in d
            assert "c_score" in d

    def test_feature_meta_out_of_range(self, vindex):
        meta = vindex.feature_meta(999, 999)
        assert meta is None


# ─── DESCRIBE ───

class TestDescribe:
    def test_describe_returns_list(self, vindex):
        edges = vindex.describe("hello")
        assert isinstance(edges, list)

    def test_describe_edge_properties(self, vindex):
        edges = vindex.describe("hello", verbose=True)
        for e in edges:
            assert hasattr(e, "target")
            assert hasattr(e, "gate_score")
            assert hasattr(e, "relation")
            assert hasattr(e, "layer")
            assert hasattr(e, "source")

    def test_describe_bands(self, vindex):
        for band in ["syntax", "knowledge", "output", "all"]:
            edges = vindex.describe("hello", band=band)
            assert isinstance(edges, list)

    def test_describe_verbose_more_edges(self, vindex):
        normal = vindex.describe("hello", verbose=False)
        verbose = vindex.describe("hello", verbose=True)
        assert len(verbose) >= len(normal)

    def test_has_edge(self, vindex):
        result = vindex.has_edge("hello")
        assert isinstance(result, bool)

    def test_get_target(self, vindex):
        result = vindex.get_target("hello", "capital")
        assert result is None or isinstance(result, str)


# ─── Relations & Clusters ───

class TestRelations:
    def test_relations_list(self, vindex):
        rels = vindex.relations()
        assert isinstance(rels, list)
        # Synthetic vindex has no relation_clusters.json, so empty is fine

    def test_cluster_centre_none(self, vindex):
        # No clusters in synthetic vindex
        centre = vindex.cluster_centre("capital")
        assert centre is None

    def test_typical_layer_none(self, vindex):
        layer = vindex.typical_layer("capital")
        assert layer is None


# ─── Mutation ───

class TestMutation:
    """Mutation tests. Synthetic vindex may not have proper down_meta,
    so insert may fail — that's OK for the synthetic fixture."""

    def test_insert_or_skip(self, vindex):
        """Insert should work if free slots exist, or raise RuntimeError if not."""
        try:
            layer, feat = vindex.insert("TestEntity", "capital", "TestCity")
            assert isinstance(layer, int)
            assert isinstance(feat, int)
            meta = vindex.feature_meta(layer, feat)
            assert meta is not None
            assert meta.top_token == "TestCity"
        except RuntimeError as e:
            assert "No free feature slot" in str(e)

    def test_insert_layer_hint_or_skip(self, vindex):
        try:
            layer, feat = vindex.insert("Hint", "rel", "City", layer=0)
            assert layer == 0
        except RuntimeError:
            pass  # Expected if no free slots

    def test_delete(self, vindex):
        count = vindex.delete("NonexistentEntity123")
        assert isinstance(count, int)


# ─── Session ───

class TestSession:
    def test_session_create(self, vindex_path):
        s = larql.session(vindex_path)
        assert repr(s).startswith("Session(")

    def test_session_query_stats(self, vindex_path):
        s = larql.session(vindex_path)
        result = s.query("STATS")
        assert len(result) > 0

    def test_session_vindex_access(self, vindex_path):
        s = larql.session(vindex_path)
        v = s.vindex
        assert v.num_layers == NUM_LAYERS
        embed = v.embed("hello")
        assert embed.shape == (HIDDEN_SIZE,)

    def test_session_array_view_follows_use(self, vindex_path, tmp_path):
        second = tmp_path / "second.vindex"
        shutil.copytree(vindex_path, second)
        s = larql.session(vindex_path)
        original = s.vindex
        escaped = str(second).replace("\\", "\\\\").replace('"', '\\"')
        s.query(f'USE "{escaped}"')
        assert s.path == str(second)
        assert s.vindex is not original
        assert s.vindex.num_layers == NUM_LAYERS
        assert s.vindex is s.vindex

    def test_session_query_text(self, vindex_path):
        s = larql.session(vindex_path)
        text = s.query_text("STATS")
        assert isinstance(text, str)
        assert len(text) > 0


# ─── Integration with real vindex (optional) ───


def _resolve_v11_vindex():
    """Locate the v11 tiny-model vindex — the default parity-test fixture.

    Order of precedence:
      1. `V11_VINDEX_PATH` env var
      2. `<larql repo>/../tiny-model/model/v11/vindex`  (sibling checkout)

    Returns the path if it exists and carries model weights, else `None`.
    """
    env = os.environ.get("V11_VINDEX_PATH")
    candidates = []
    if env:
        candidates.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(
        os.path.normpath(os.path.join(here, "..", "..", "..", "..",
                                      "tiny-model", "model", "v11", "vindex"))
    )
    for path in candidates:
        config_path = os.path.join(path, "index.json")
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if config.get("has_model_weights") is True:
            return path
    return None


V11_VINDEX = _resolve_v11_vindex()

REAL_VINDEX = os.environ.get("REAL_VINDEX_PATH")

@pytest.mark.skipif(
    REAL_VINDEX is None or not os.path.exists(REAL_VINDEX),
    reason="Set REAL_VINDEX_PATH to run integration tests"
)
class TestRealVindex:
    """Tests that only run with a real vindex (set REAL_VINDEX_PATH env var)."""

    @pytest.fixture(scope="class")
    def real_vindex(self):
        return larql.load(REAL_VINDEX)

    def test_describe_france(self, real_vindex):
        edges = real_vindex.describe("France")
        assert len(edges) > 0
        targets = [e.target.lower() for e in edges]
        assert "paris" in targets or "french" in targets

    def test_entity_walk_france(self, real_vindex):
        bands = real_vindex.layer_bands()
        layers = list(range(bands["knowledge"][0], bands["knowledge"][1] + 1))
        hits = real_vindex.entity_walk("France", layers=layers, top_k=5)
        assert len(hits) > 0

    def test_relations_nonempty(self, real_vindex):
        rels = real_vindex.relations()
        assert len(rels) > 0

    def test_cluster_centre(self, real_vindex):
        centre = real_vindex.cluster_centre("capital")
        if centre is not None:
            assert centre.shape == (real_vindex.hidden_size,)

    def test_mlx_load(self, real_vindex):
        pytest.importorskip("mlx")
        pytest.importorskip("mlx_lm")
        model, tokenizer = larql.mlx.load(REAL_VINDEX)
        assert model is not None

    def test_infer(self, real_vindex):
        """vindex.infer() — Rust forward pass with mmap'd walk FFN."""
        result = real_vindex.infer("The capital of France is", top_k_predictions=3)
        assert len(result) > 0
        assert result[0][0] == "Paris"
        assert result[0][1] > 0.5

    def test_infer_reuses_weights(self, real_vindex):
        """Second infer() call should reuse mmap'd weights (no reload)."""
        import time
        # First call loads weights
        real_vindex.infer("warmup", top_k_predictions=1)
        # Second call should be faster (no load, page cache warm)
        t0 = time.time()
        r = real_vindex.infer("The largest planet is", top_k_predictions=1)
        t1 = time.time()
        assert r[0][0] == "Jupiter"
        # Should complete — no assertion on time, just verifying it works

    def test_walk_model(self):
        """WalkModel — zero-copy mmap'd weights, Rust walk FFN."""
        wm = larql.WalkModel(REAL_VINDEX, top_k=4096)
        assert wm.num_layers > 0
        assert wm.hidden_size > 0
        result = wm.predict("The capital of France is")
        assert len(result) > 0
        assert result[0][0] == "Paris"

    def test_walk_model_ffn_layer(self):
        """WalkModel.ffn_layer — per-layer sparse FFN via bytes."""
        import struct
        wm = larql.WalkModel(REAL_VINDEX, top_k=256)
        hidden = wm.hidden_size
        x_bytes = struct.pack(f"{hidden}f", *([0.1] * hidden))
        result = wm.ffn_layer(layer=0, x_bytes=x_bytes, seq_len=1)
        assert isinstance(result, bytes)
        assert len(result) == hidden * 4

    def test_walk_model_memory(self):
        """WalkModel load should use minimal RSS (mmap, not heap)."""
        import resource
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        wm = larql.WalkModel(REAL_VINDEX, top_k=256)
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        delta = rss_after - rss_before
        assert delta < 2000, f"WalkModel load used {delta:.0f} MB — expected < 2000 MB (mmap)"

    def test_infer_memory(self, real_vindex):
        """vindex.infer() should use mmap'd weights (low load RSS)."""
        import resource
        # After infer, the walk_model is cached inside the vindex
        # The load RSS should be small (mmap, not heap)
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        v = larql.load(REAL_VINDEX)
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        # Vindex load (gate + embed only) should be < 8 GB
        assert rss_after - rss_before < 8000

    def test_walk_ffn_mlx(self):
        """walk_ffn.load — MLX attention + Rust FFN."""
        pytest.importorskip("mlx")
        pytest.importorskip("mlx_lm")
        import mlx_lm
        from larql.walk_ffn import load as walk_load
        model, tokenizer = walk_load(REAL_VINDEX, top_k=4096)
        response = mlx_lm.generate(
            model, tokenizer,
            prompt="The capital of France is",
            max_tokens=3, verbose=False
        )
        assert "Paris" in response


# ─── Python/LQL INFER parity (ADR 0001) ───


def _parse_lql_predictions(lines):
    """Extract prediction tokens from LQL `INFER` output lines, in order.

    LQL format: "Predictions (walk FFN):" header, then lines like
        "   1. Paris                (90.12%)"
    or, when a KNN override fires:
        "   1. Paris                (KNN override, cos=0.98, L5)"
    """
    import re
    tokens = []
    in_predictions = False
    # Matches "  1. token_text  (..." — the ranked prediction lines — but not
    # the trailing "  15ms" timing line, which has no "." after the digit.
    pattern = re.compile(r"^\s*\d+\.\s*(?P<token>.*?)\s*\(")
    for line in lines:
        if line.startswith("Predictions (walk FFN)"):
            in_predictions = True
            continue
        if in_predictions:
            m = pattern.match(line)
            if not m:
                break
            tokens.append(m.group("token"))
    return tokens


@pytest.mark.skipif(
    V11_VINDEX is None,
    reason="No v11 vindex found. Set V11_VINDEX_PATH or check out tiny-model "
           "as a sibling of larql."
)
class TestV11InferParity:
    """ADR 0001: `PyVindex.infer` and LQL `SELECT ... INFER` must return
    byte-identical top-k predictions on any vindex.

    Runs automatically whenever the v11 tiny-model vindex is available —
    either at `V11_VINDEX_PATH` or at `../tiny-model/model/v11/vindex`
    (sibling checkout). Any future divergence — a new parameter default, a
    surface-specific fast path, a refactor that bypasses `infer_patched` —
    fails this test.
    """

    @pytest.fixture(scope="class")
    def vindex(self):
        return larql.load(V11_VINDEX)

    @pytest.fixture(scope="class")
    def session(self):
        return larql.session(V11_VINDEX)

    @pytest.mark.parametrize(
        "prompt",
        [
            "The capital of France is",
            "Water is",
            "hello",
        ],
    )
    def test_parity(self, vindex, session, prompt):
        top_k = 5
        py_tokens = [tok for tok, _ in vindex.infer(prompt, top_k_predictions=top_k)]
        lql_tokens = _parse_lql_predictions(
            session.query(f"INFER '{prompt}' TOP {top_k}")
        )
        assert py_tokens == lql_tokens, (
            f"Python/LQL parity broken on prompt {prompt!r}:\n"
            f"  py:  {py_tokens}\n  lql: {lql_tokens}"
        )
