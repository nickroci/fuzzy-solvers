"""Exact structural analysis of what defeats a Cayley construction.

A construction fails because some independent set exceeds the cap.  Knowing
*which* set is thin information; knowing its structure is what makes the next
move choosable rather than guessable, and the structure is exactly computable.

In ``Cay(G, S)`` two vertices are adjacent when their difference lies in ``S``,
so a set ``I`` is independent exactly when its difference set
``D(I) = {x - y : x, y in I, x != y}`` avoids ``S`` entirely.  Two consequences
drive everything here:

* **Every repair is in the difference set.**  Adding a generator ``d`` destroys
  ``I`` if and only if ``d`` lies in ``D(I)``.  Any other addition leaves ``I``
  independent and wastes a kernel call.
* **Not every repair is legal.**  The connection set must stay sum-free or the
  graph gains a triangle, and it must stay within the degree cap.  Both are
  decidable before anything is built.

So this module partitions ``D(I)`` into the repairs that are available and the
ones that are blocked, and reports whether the obstruction is a coset — a
coset obstruction usually signals that the connection set is trapped inside a
subgroup, which no single generator swap will fix.

This is measurement, not advice.  It states what is true about the current
construction and never ranks the options or suggests which to take.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ramsey.cayley import AbelianGroup, sum_violation, symmetric_closure
from ramsey.graph import RamseyGraph, maximum_independent_set

_LISTED_REPAIRS: Final = 24


@dataclass(frozen=True, slots=True)
class ObstructionAnalysis:
    """The exact structure of the independent set defeating a construction."""

    obstruction: tuple[int, ...]
    size: int
    difference_size: int
    is_coset: bool
    stabilizer_order: int
    repairs: tuple[int, ...]
    blocked: tuple[int, ...]
    degree_headroom: int

    def render(self) -> str:
        """Render the analysis as feedback, stating only measured facts."""
        shape = (
            f"It is a coset of a subgroup of order {self.size}"
            if self.is_coset
            else f"It is not a coset (stabilizer order {self.stabilizer_order})"
        )
        if not self.repairs:
            tail = (
                "No generator can be added: every element of the difference set would "
                "break sum-freeness or exceed the degree cap. This construction is a "
                "dead end without dropping a generator first."
            )
        else:
            tail = (
                f"{len(self.repairs)} generator(s) would destroy it and keep the "
                f"connection set sum-free: {self.repairs[:_LISTED_REPAIRS]}"
                f"{' ...' if len(self.repairs) > _LISTED_REPAIRS else ''}. "
                f"{len(self.blocked)} further difference elements would destroy it but "
                f"break sum-freeness."
            )
        return (
            f"Obstruction of size {self.size} with {self.difference_size} distinct "
            f"differences. {shape}. Degree headroom {self.degree_headroom}. {tail}"
        )


def difference_representatives(group: AbelianGroup, vertices: tuple[int, ...]) -> tuple[int, ...]:
    """Return one representative per inverse pair of the set's difference set.

    Differences come in inverse pairs because ``y - x`` negates ``x - y``, and a
    connection set is symmetric, so only the pair matters.
    """
    seen: set[int] = set()
    for first in vertices:
        for second in vertices:
            if first == second:
                continue
            delta = group.add(first, group.negate(second))
            if delta:
                seen.add(min(delta, group.negate(delta)))
    return tuple(sorted(seen))


def is_subgroup(group: AbelianGroup, elements: frozenset[int]) -> bool:
    """Report whether a set of elements is closed under the group operation."""
    if 0 not in elements:
        return False
    return all(
        group.add(first, second) in elements and group.negate(first) in elements
        for first in elements
        for second in elements
    )


def coset_structure(group: AbelianGroup, vertices: tuple[int, ...]) -> tuple[bool, int]:
    """Return whether the set is a coset, and the order of its stabilizer.

    The stabilizer is ``{g : I + g = I}``.  It is always a subgroup, it equals
    the whole set exactly when the set is a coset, and its size measures how
    much translation symmetry the obstruction has even when it is not.
    """
    if not vertices:
        return False, 0
    members = frozenset(vertices)
    stabilizer = sum(
        1
        for shift in range(group.order)
        if frozenset(group.add(element, shift) for element in members) == members
    )
    shifted = frozenset(group.add(element, group.negate(vertices[0])) for element in members)
    return is_subgroup(group, shifted), stabilizer


def analyse_obstruction(
    group: AbelianGroup,
    connection: tuple[int, ...],
    graph: RamseyGraph,
    *,
    degree_cap: int,
) -> ObstructionAnalysis:
    """Measure the maximum independent set and partition its legal repairs."""
    obstruction = maximum_independent_set(graph).vertices
    differences = difference_representatives(group, obstruction)
    is_coset, stabilizer = coset_structure(group, obstruction)
    headroom = degree_cap - len(connection)
    repairs: list[int] = []
    blocked: list[int] = []
    for candidate in differences:
        extended = symmetric_closure(group, (*connection, candidate))
        if len(extended) > degree_cap or sum_violation(group, extended) is not None:
            blocked.append(candidate)
        else:
            repairs.append(candidate)
    return ObstructionAnalysis(
        obstruction=obstruction,
        size=len(obstruction),
        difference_size=len(differences),
        is_coset=is_coset,
        stabilizer_order=stabilizer,
        repairs=tuple(repairs),
        blocked=tuple(blocked),
        degree_headroom=headroom,
    )
