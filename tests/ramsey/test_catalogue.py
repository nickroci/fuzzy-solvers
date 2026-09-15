from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ramsey.catalogue import (
    audit_deletion,
    audit_maximality,
    audit_referee,
    load_catalogue,
    main,
    run_audits,
)
from ramsey.graph import RamseyGraph, cycle_graph, graph_from_edges, to_graph6

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def cyclic(order: int, connections: tuple[int, ...]) -> RamseyGraph:
    edges = tuple(
        (vertex, (vertex + step) % order) for vertex in range(order) for step in connections
    )
    return graph_from_edges(order, edges)


def write(path: Path, graphs: tuple[RamseyGraph, ...]) -> Path:
    path.write_text("\n".join(to_graph6(graph) for graph in graphs) + "\n")
    return path


def test_load_catalogue_round_trips(tmp_path: Path) -> None:
    graphs = (cycle_graph(5), cycle_graph(7))
    path = write(tmp_path / "sample.g6", graphs)
    assert load_catalogue(path) == graphs


def test_load_catalogue_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "sample.g6"
    path.write_text(f"\n{to_graph6(cycle_graph(5))}\n\n")
    assert load_catalogue(path) == (cycle_graph(5),)


def test_referee_audit_accepts_a_valid_catalogue() -> None:
    result = audit_referee("x.g6", (cycle_graph(5),), 3)
    assert result.passed
    assert result.examined == 1
    assert "ok" in result.render()


def test_referee_audit_flags_a_graph_that_is_not_in_the_family() -> None:
    result = audit_referee("x.g6", (cycle_graph(6),), 3)
    assert not result.passed
    assert result.failures == 1
    assert "FAIL" in result.render()


def test_referee_audit_flags_a_mixed_order_catalogue() -> None:
    result = audit_referee("x.g6", (cycle_graph(5), cycle_graph(7)), 4)
    assert result.failures == 1


def test_maximality_audit_passes_on_a_maximal_graph() -> None:
    result = audit_maximality("r35_13.g6", (cyclic(13, (1, 5)),), 5)
    assert result.passed


def test_maximality_audit_flags_an_extendable_graph() -> None:
    result = audit_maximality("x.g6", (cyclic(13, (1, 5)),), 6)
    assert not result.passed


def test_deletion_audit_passes_when_every_graph_extends_back() -> None:
    result = audit_deletion("x.g6", (cyclic(13, (1, 5)),), 5, sample=10, rng=random.Random(1))
    assert result.passed
    assert result.examined == 1


def test_deletion_audit_samples_large_catalogues() -> None:
    graphs = tuple(cyclic(13, (1, 5)) for _ in range(20))
    result = audit_deletion("x.g6", graphs, 5, sample=4, rng=random.Random(2))
    assert result.examined == 4
    assert result.graphs == 20


def test_deletion_audit_passes_on_a_maximal_family_member() -> None:
    """A valid member always extends back: the graph it came from is a witness."""
    result = audit_deletion("x.g6", (cycle_graph(5),), 3, sample=10, rng=random.Random(3))
    assert result.passed


def test_deletion_audit_flags_a_graph_outside_the_family() -> None:
    """C6 is not a (3,3,6)-graph, and its vertex-deleted path extends to nothing."""
    result = audit_deletion("x.g6", (cycle_graph(6),), 3, sample=10, rng=random.Random(3))
    assert not result.passed


def test_render_reports_zero_rate_for_an_empty_audit() -> None:
    result = audit_referee("x.g6", (), 5)
    assert result.examined == 0
    assert "0.00 ms/graph" in result.render()


def test_run_audits_walks_the_families(tmp_path: Path) -> None:
    write(tmp_path / "r35_13.g6", (cyclic(13, (1, 5)),))
    write(tmp_path / "r35_7.g6", (cycle_graph(7),))
    results = run_audits(tmp_path, sample=5, seed=7)
    names = {result.name for result in results}
    assert names == {"referee", "maximality", "deletion"}
    assert all(result.passed for result in results)


def test_run_audits_skips_absent_and_empty_files(tmp_path: Path) -> None:
    (tmp_path / "r35_9.g6").write_text("\n")
    assert run_audits(tmp_path, sample=5, seed=7) == ()


def test_main_reports_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(tmp_path / "r35_13.g6", (cyclic(13, (1, 5)),))
    assert main(["--data", str(tmp_path), "--sample", "3"]) == 0
    assert "0 failures" in capsys.readouterr().out


def test_main_reports_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(tmp_path / "r35_13.g6", (cycle_graph(13),))
    assert main(["--data", str(tmp_path), "--sample", "3"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_reports_an_empty_data_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--data", str(tmp_path)]) == 1
    assert "no catalogue files found" in capsys.readouterr().out
