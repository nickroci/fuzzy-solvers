from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codes.kernel import accelerated, kernel_max_rank
from codes.kernel import minimum_distance as fast_distance
from codes.linear import hamming_code
from codes.linear import minimum_distance as reference_distance
from codes.quasicyclic import QuasiCyclic
from codes.search import QcProgram, run_qc_program
from codes.tables import Cell, load_table, parse_table, quasi_cyclic_targets

if TYPE_CHECKING:
    from pathlib import Path

TABLE = """
<table>
<tr><td>10</td><td>3-5</td><td>4</td><td>2-2</td></tr>
<tr><td>12</td><td>6-9</td><td>5-6</td><td>3</td></tr>
<tr><td>header</td><td>x</td></tr>
</table>
"""


def test_a_range_cell_is_open_and_a_bare_cell_is_settled() -> None:
    cells = parse_table(TABLE)
    by_key = {(c.length, c.dimension): c for c in cells}
    assert by_key[(10, 1)].is_open
    assert by_key[(10, 1)].gap == 2
    assert not by_key[(10, 2)].is_open
    assert by_key[(10, 2)].lower == by_key[(10, 2)].upper == 4
    assert not by_key[(10, 3)].is_open  # "2-2" has no room


def test_non_numeric_rows_are_skipped() -> None:
    assert all(c.length in (10, 12) for c in parse_table(TABLE))


def test_a_cell_names_the_distance_a_new_entry_needs() -> None:
    assert Cell(length=60, dimension=20, lower=17, upper=20).target == 18


def test_quasi_cyclic_index_requires_the_dimension_to_divide_the_length() -> None:
    assert Cell(length=60, dimension=20, lower=1, upper=2).quasi_cyclic_index == 3
    assert Cell(length=61, dimension=20, lower=1, upper=2).quasi_cyclic_index == 0
    assert Cell(length=60, dimension=0, lower=1, upper=2).quasi_cyclic_index == 0


def test_a_cell_renders_its_bounds_and_target() -> None:
    assert "need d >= 18" in Cell(length=60, dimension=20, lower=17, upper=20).render()


def test_targets_are_reachable_open_cells_loosest_first() -> None:
    cells = (
        Cell(length=60, dimension=20, lower=17, upper=20),  # gap 3, index 3
        Cell(length=40, dimension=20, lower=8, upper=12),  # gap 4, index 2
        Cell(length=61, dimension=20, lower=1, upper=9),  # index 0, unreachable
        Cell(length=60, dimension=20, lower=9, upper=9),  # settled
    )
    targets = quasi_cyclic_targets(cells)
    assert [t.gap for t in targets] == [4, 3]


def test_targets_respect_the_affordability_bound() -> None:
    cells = (Cell(length=90, dimension=30, lower=1, upper=9),)
    assert quasi_cyclic_targets(cells, max_dimension=22) == ()
    assert len(quasi_cyclic_targets(cells, max_dimension=30)) == 1


def test_targets_respect_the_index_range() -> None:
    cells = (Cell(length=200, dimension=20, lower=1, upper=9),)  # index 10
    assert quasi_cyclic_targets(cells, max_index=6) == ()


def test_the_saved_table_parses(tmp_path: Path) -> None:
    path = tmp_path / "t.html"
    path.write_text(TABLE)
    assert len(load_table(path)) == 6


# --- kernel ---


def test_the_kernel_reports_its_capability() -> None:
    assert isinstance(accelerated(), bool)
    assert kernel_max_rank() >= 0


@pytest.mark.parametrize("parity", [2, 3, 4, 5])
def test_the_kernel_agrees_with_the_referee(parity: int) -> None:
    code = hamming_code(parity)
    assert fast_distance(code) == reference_distance(code)


def test_the_kernel_agrees_on_quasi_cyclic_codes() -> None:
    for seed in range(12):
        qc = QuasiCyclic(block=9, polynomials=(seed * 7 % 512, seed * 13 % 512))
        code = qc.code()
        assert fast_distance(code) == reference_distance(code)


def test_the_kernel_returns_zero_for_a_trivial_code() -> None:
    assert fast_distance(QuasiCyclic(block=5, polynomials=(0,)).code()) == 0


# --- commissioned search ---


def test_a_program_rejects_impossible_shapes() -> None:
    with pytest.raises(ValueError, match="block must be positive"):
        QcProgram(block=0, index=2, target_distance=3, restarts=1, steps=0)
    with pytest.raises(ValueError, match="index must be positive"):
        QcProgram(block=4, index=0, target_distance=3, restarts=1, steps=0)
    with pytest.raises(ValueError, match="restarts must be positive"):
        QcProgram(block=4, index=2, target_distance=3, restarts=0, steps=0)
    with pytest.raises(ValueError, match="steps must be non-negative"):
        QcProgram(block=4, index=2, target_distance=3, restarts=1, steps=-1)


def test_a_program_describes_its_shape() -> None:
    text = QcProgram(block=20, index=3, target_distance=18, restarts=5, steps=2).describe()
    assert "[60,20]" in text
    assert "need d >= 18" in text


def test_a_search_finds_an_easy_target_and_reports_it() -> None:
    """A low target should be met, and the outcome must name the code."""
    outcome = run_qc_program(
        QcProgram(block=8, index=2, target_distance=3, restarts=200, steps=20, seed=1),
        budget=3000,
    )
    assert outcome.accepted
    assert outcome.best_dimension == 8
    assert "ACCEPTED" in outcome.render()


def test_a_search_reports_its_shortfall_when_it_misses() -> None:
    outcome = run_qc_program(
        QcProgram(block=8, index=2, target_distance=99, restarts=5, steps=2, seed=1),
        budget=50,
    )
    assert not outcome.accepted
    assert outcome.shortfall > 0
    assert "short of" in outcome.render()


def test_a_search_respects_its_budget() -> None:
    outcome = run_qc_program(
        QcProgram(block=10, index=3, target_distance=99, restarts=10_000, steps=50, seed=2),
        budget=37,
    )
    assert outcome.evaluations <= 37


def test_a_search_is_deterministic_in_its_seed() -> None:
    program = QcProgram(block=9, index=3, target_distance=99, restarts=30, steps=10, seed=5)
    first = run_qc_program(program, budget=400)
    second = run_qc_program(program, budget=400)
    assert first.best_distance == second.best_distance
    assert first.best_polynomials == second.best_polynomials


def test_a_weight_constrained_search_draws_sparse_polynomials() -> None:
    outcome = run_qc_program(
        QcProgram(block=12, index=2, target_distance=99, restarts=5, steps=0, seed=3, weight=3),
        budget=20,
    )
    assert outcome.evaluations > 0
