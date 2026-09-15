"""Audit the referee and the extension search against published censuses.

The exact catalogues of ``(3, t, n)``-graphs computed by McKay and others are
ground truth: they fix both which graphs exist and, at the extremal order,
which do not.  Running our own referee and extension search over them is the
gate that has to pass before any of this points at ``R(3, 10)``, because a
search that silently misses extensions would turn an exhaustive negative into
a false one.

Three audits, none of which needs isomorphism rejection:

``referee``
    Every catalogued graph must be accepted as a ``(3, t, n)``-graph.  A single
    disagreement means the referee is wrong.

``maximality``
    No graph at order ``R(3, t) - 1`` may extend, and every such search must
    report itself exhaustive.  A witness here would contradict the published
    Ramsey number.

``deletion``
    Deleting a vertex from a catalogued graph ``H`` leaves a graph that ``H``
    itself extends, so the search must find *some* extension for every one.  A
    miss means the search is incomplete, which is the failure mode that would
    quietly invalidate a negative result.

Catalogue files are the graph6 lists published at
``https://users.cecs.anu.edu.au/~bdm/data/ramsey.html``; download them
separately.  Nothing in this module fetches anything.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

from ramsey.extension import find_extension
from ramsey.graph import RamseyGraph, check_ramsey_graph, parse_graph6


def load_catalogue(path: Path) -> tuple[RamseyGraph, ...]:
    """Parse a graph6 catalogue file into graphs, one per non-empty line."""
    return tuple(parse_graph6(line) for line in path.read_text().split() if line)


@dataclass(frozen=True, slots=True)
class AuditResult:
    """The outcome of one audit over one catalogue file."""

    name: str
    second_colour: int
    order: int
    graphs: int
    examined: int
    failures: int
    seconds: float
    detail: str = ""

    @property
    def passed(self) -> bool:
        """Report whether the audit found no failure."""
        return self.failures == 0

    def render(self) -> str:
        """Render one aligned result line."""
        status = "ok  " if self.passed else "FAIL"
        per = self.seconds * 1000 / self.examined if self.examined else 0.0
        return (
            f"  [{status}] {self.name:<12} t={self.second_colour} order={self.order:>2} "
            f"graphs={self.graphs:>7,} examined={self.examined:>5} "
            f"failures={self.failures:<5} {per:6.2f} ms/graph {self.detail}"
        )


def audit_referee(name: str, graphs: tuple[RamseyGraph, ...], second_colour: int) -> AuditResult:
    """Check that every catalogued graph is accepted by the referee."""
    start = time.monotonic()
    order = graphs[0].order if graphs else 0
    failures = sum(
        1
        for graph in graphs
        if graph.order != order or not check_ramsey_graph(graph, second_colour).accepted
    )
    return AuditResult(
        name="referee",
        second_colour=second_colour,
        order=order,
        graphs=len(graphs),
        examined=len(graphs),
        failures=failures,
        seconds=time.monotonic() - start,
        detail=f"({name})",
    )


def audit_maximality(name: str, graphs: tuple[RamseyGraph, ...], second_colour: int) -> AuditResult:
    """Check that no graph at the extremal order admits an extension."""
    start = time.monotonic()
    order = graphs[0].order if graphs else 0
    failures = 0
    for graph in graphs:
        report = find_extension(graph, second_colour)
        if report.extends or not report.exhaustive:
            failures += 1
    return AuditResult(
        name="maximality",
        second_colour=second_colour,
        order=order,
        graphs=len(graphs),
        examined=len(graphs),
        failures=failures,
        seconds=time.monotonic() - start,
        detail=f"({name}; any witness would contradict R(3,{second_colour}))",
    )


def audit_deletion(
    name: str,
    graphs: tuple[RamseyGraph, ...],
    second_colour: int,
    *,
    sample: int,
    rng: random.Random,
) -> AuditResult:
    """Check that every vertex-deleted catalogued graph extends again."""
    start = time.monotonic()
    order = graphs[0].order if graphs else 0
    chosen = graphs if len(graphs) <= sample else tuple(rng.sample(graphs, sample))
    failures = 0
    nodes = 0
    for graph in chosen:
        victim = rng.randrange(graph.order)
        reduced = graph.induced(graph.full_mask & ~(1 << victim))
        report = find_extension(reduced, second_colour)
        nodes += report.nodes
        witness = report.witness
        if witness is None or not report.exhaustive or not witness.check.accepted:
            failures += 1
    average = nodes / len(chosen) if chosen else 0.0
    return AuditResult(
        name="deletion",
        second_colour=second_colour,
        order=order,
        graphs=len(graphs),
        examined=len(chosen),
        failures=failures,
        seconds=time.monotonic() - start,
        detail=f"({name}; avg {average:.1f} search nodes)",
    )


@dataclass(frozen=True, slots=True)
class CatalogueFamily:
    """One published family: the ``t`` of ``(3, t, n)`` and its Ramsey number."""

    second_colour: int
    ramsey_number: int
    prefix: str


FAMILIES: tuple[CatalogueFamily, ...] = (
    CatalogueFamily(second_colour=4, ramsey_number=9, prefix="r34"),
    CatalogueFamily(second_colour=5, ramsey_number=14, prefix="r35"),
    CatalogueFamily(second_colour=6, ramsey_number=18, prefix="r36"),
)


def run_audits(data: Path, *, sample: int, seed: int) -> tuple[AuditResult, ...]:
    """Run every audit over every catalogue file present in ``data``."""
    rng = random.Random(seed)
    results = []
    for family in FAMILIES:
        for order in range(1, family.ramsey_number):
            path = data / f"{family.prefix}_{order}.g6"
            if not path.exists():
                continue
            graphs = load_catalogue(path)
            if not graphs:
                continue
            results.append(audit_referee(path.name, graphs, family.second_colour))
            if order == family.ramsey_number - 1:
                results.append(audit_maximality(path.name, graphs, family.second_colour))
            if order >= _MIN_DELETION_ORDER:
                results.append(
                    audit_deletion(
                        path.name,
                        graphs,
                        family.second_colour,
                        sample=sample,
                        rng=rng,
                    )
                )
    return tuple(results)


_MIN_DELETION_ORDER = 6


def main(argv: list[str] | None = None) -> int:
    """Run the catalogue audits and report a non-zero status on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="directory of .g6 catalogues")
    parser.add_argument("--sample", type=int, default=400, help="graphs per deletion audit")
    parser.add_argument("--seed", type=int, default=20260904, help="sampling seed")
    arguments = parser.parse_args(argv)

    results = run_audits(arguments.data, sample=arguments.sample, seed=arguments.seed)
    if not results:
        print(f"no catalogue files found in {arguments.data}")
        return 1
    for result in results:
        print(result.render())
    failures = sum(result.failures for result in results)
    examined = sum(result.examined for result in results)
    print(f"\n{len(results)} audits, {examined:,} graph examinations, {failures} failures")
    return 0 if failures == 0 else 1
