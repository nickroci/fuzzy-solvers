"""Exact referee and one-vertex extension test for triangle-free Ramsey graphs.

The campaign target is ``R(3, 10)``, known to be 40 or 41 and open since
Exoo's ``(3, 10, 39)``-graph established the lower bound.  Deciding it reduces
to whether a ``(3, 10, 40)``-graph exists; see :mod:`ramsey.extension`
for the reduction and the exact test.
"""

from ramsey.extension import (
    SMALL_RAMSEY_NUMBERS,
    ExtensionConstraints,
    ExtensionReport,
    ExtensionWitness,
    extend_graph,
    extension_constraints,
    find_extension,
)
from ramsey.graph import (
    IndependentSet,
    RamseyCheck,
    RamseyGraph,
    Triangle,
    check_ramsey_graph,
    cycle_graph,
    find_triangle,
    graph_from_edges,
    has_independence_at_most,
    independence_number,
    independence_number_within,
    independent_set_of_size,
    independent_set_within,
    is_triangle_free,
    maximum_independent_set,
    parse_graph6,
    to_graph6,
)

__all__ = [
    "SMALL_RAMSEY_NUMBERS",
    "ExtensionConstraints",
    "ExtensionReport",
    "ExtensionWitness",
    "IndependentSet",
    "RamseyCheck",
    "RamseyGraph",
    "Triangle",
    "check_ramsey_graph",
    "cycle_graph",
    "extend_graph",
    "extension_constraints",
    "find_extension",
    "find_triangle",
    "graph_from_edges",
    "has_independence_at_most",
    "independence_number",
    "independence_number_within",
    "independent_set_of_size",
    "independent_set_within",
    "is_triangle_free",
    "maximum_independent_set",
    "parse_graph6",
    "to_graph6",
]
