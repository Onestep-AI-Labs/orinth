"""Phase 17 model architecture studio.

A visual graph is the single source of truth: `graph` parses and validates the
IR, `catalog` declares the palette, `shapes` infers output shapes without
importing TensorFlow, and `emit_keras` turns the graph into runnable Python.
Everything downstream (code export, compile check, training) is a function of
the same IR, so generated code can never drift from what trains.
"""

from app.ml.architecture.catalog import NODE_SPECS, node_catalog, node_spec
from app.ml.architecture.emit_keras import emit_module
from app.ml.architecture.graph import ResolvedGraph, build_graph
from app.ml.architecture.shapes import infer_shapes

__all__ = [
    "NODE_SPECS",
    "ResolvedGraph",
    "build_graph",
    "emit_module",
    "infer_shapes",
    "node_catalog",
    "node_spec",
]
