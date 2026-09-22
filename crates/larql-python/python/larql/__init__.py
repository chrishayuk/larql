"""
larql — Python bindings for the LARQL knowledge graph engine and vindex.

Two interfaces:
    - Direct vindex access for numpy arrays:
        vindex = larql.load("gemma3-4b.vindex")
        embed = vindex.embed("France")
        edges = vindex.describe("France")

    - LQL session for query language access:
        session = larql.session("gemma3-4b.vindex")
        session.query("DESCRIBE 'France'")
        session.vindex.gate_vectors(layer=26)
"""

from larql._native import (
    # Vindex types
    Vindex,
    FeatureMeta,
    WalkHit,
    DescribeEdge,
    Relation,
    Session,
    WalkModel,

    # Graph types
    Edge,
    Node,
    Graph,

    # Vindex functions
    load_vindex,
    create_session,

    # Graph functions
    load as load_graph,
    save as save_graph,
    load_csv,
    save_csv,
    shortest_path,
    merge_graphs,
    merge_graphs_with_strategy,
    diff,
    pagerank,
    bfs_traversal,
    dfs_traversal,
    weight_walk,
    attention_walk,
)


def load(path: str, **kwargs) -> "Vindex":
    """Load a vindex from a local path.

    Args:
        path: Path to .vindex directory

    Returns:
        Vindex object with numpy array access, describe, walk, etc.

    Example:
        vindex = larql.load("gemma3-4b.vindex")
        edges = vindex.describe("France")
        embed = vindex.embed("France")
    """
    return load_vindex(path)


def session(path: str) -> "Session":
    """Create an LQL session connected to a V2 or VINDEX3 artifact.

    Provides LQL execution for both generations; .vindex arrays require V2:
        session = larql.session("gemma3-4b.vindex")
        session.query("DESCRIBE 'France'")
        session.vindex.embed("France")

    Args:
        path: Path to .vindex directory

    Returns:
        Session with .query() and .vindex properties
    """
    return create_session(path)


# MLX integration (lazy import — only loaded when used)
try:
    from larql import mlx
except ImportError:
    mlx = None

# Streaming: mmap'd weights, Metal pages from SSD on demand
try:
    from larql import streaming
except ImportError:
    streaming = None

# Walk FFN: MLX attention + Rust sparse FFN (editable knowledge layer)
try:
    from larql import walk_ffn
except ImportError:
    walk_ffn = None

# Residual stream trace is exposed via WalkModel.trace() (Rust implementation)

__version__ = "0.1.0"

__all__ = [
    # Core functions
    "load",
    "session",
    "load_vindex",
    "create_session",

    # Vindex types
    "Vindex",
    "FeatureMeta",
    "WalkHit",
    "DescribeEdge",
    "Relation",
    "Session",

    # Graph types
    "Edge",
    "Node",
    "Graph",

    # Graph functions
    "load_graph",
    "save_graph",
    "load_csv",
    "save_csv",
    "shortest_path",
    "merge_graphs",
    "merge_graphs_with_strategy",
    "diff",
    "pagerank",
    "bfs_traversal",
    "dfs_traversal",
    "weight_walk",
    "attention_walk",
]
