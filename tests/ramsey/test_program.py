from __future__ import annotations

import random

import pytest

from ramsey.cayley import AbelianGroup
from ramsey.program import (
    LiftProgram,
    SearchProgram,
    greedy_sum_free,
    legal_connection,
    representative_pool,
    run_lift_program,
    run_program,
)

CLEBSCH = (2, 2, 2, 2)


def test_program_rejects_a_non_positive_degree() -> None:
    with pytest.raises(ValueError, match="degree must be positive"):
        SearchProgram(moduli=CLEBSCH, degree=0, restarts=1, steps=1)


def test_program_rejects_a_non_positive_restart_count() -> None:
    with pytest.raises(ValueError, match="restarts must be positive"):
        SearchProgram(moduli=CLEBSCH, degree=5, restarts=0, steps=1)


def test_program_rejects_negative_steps() -> None:
    with pytest.raises(ValueError, match="steps must be non-negative"):
        SearchProgram(moduli=CLEBSCH, degree=5, restarts=1, steps=-1)


def test_program_describes_its_hypothesis() -> None:
    text = SearchProgram(
        moduli=(3, 3, 11), degree=17, restarts=100, steps=50, required=(1,), forbidden=(33,)
    ).describe()
    assert "Z3 x Z3 x Z11" in text
    assert "degree 17" in text
    assert "requiring (1,)" in text
    assert "forbidding (33,)" in text


def test_representative_pool_drops_forbidden_inverse_pairs() -> None:
    group = AbelianGroup(moduli=(11,))
    pool = representative_pool(group, frozenset({1}))
    assert 1 not in pool
    assert 10 not in pool
    assert len(pool) == 4


def test_legal_connection_rejects_an_oversized_or_unfree_set() -> None:
    group = AbelianGroup(moduli=(16,))
    assert legal_connection(group, (1, 2), 6) is None  # 1 + 14 = 15 is in the set
    assert legal_connection(group, (1, 6), 4) is not None
    assert legal_connection(group, (1, 6), 2) is None  # the symmetric closure has four members
    assert legal_connection(group, (1, 6, 3), 4) is None


def test_greedy_sum_free_keeps_the_required_generators() -> None:
    group = AbelianGroup(moduli=CLEBSCH)
    chosen = greedy_sum_free(
        group, representative_pool(group, frozenset()), 5, (1,), random.Random(0)
    )
    assert 1 in chosen
    assert legal_connection(group, chosen, 5) is not None


def test_a_commissioned_search_finds_a_known_witness() -> None:
    """R(3,6) > 16 is witnessed by the Clebsch graph; a plain search reaches it."""
    program = SearchProgram(moduli=CLEBSCH, degree=5, restarts=300, steps=40, seed=1)
    outcome = run_program(program, independence_cap=5, budget=10_000)
    assert outcome.accepted
    assert outcome.best_independence == 5
    assert outcome.gap == 0
    assert "ACCEPTED" in outcome.render()


def test_a_search_reports_its_gap_when_it_fails() -> None:
    program = SearchProgram(moduli=CLEBSCH, degree=5, restarts=5, steps=2, seed=3)
    outcome = run_program(program, independence_cap=1, budget=50)
    assert not outcome.accepted
    assert outcome.gap > 0
    assert "above the cap" in outcome.render()


def test_a_search_respects_its_evaluation_budget() -> None:
    program = SearchProgram(moduli=CLEBSCH, degree=5, restarts=1000, steps=50, seed=2)
    outcome = run_program(program, independence_cap=1, budget=17)
    assert outcome.evaluations <= 17


def test_a_search_is_deterministic_in_its_seed() -> None:
    program = SearchProgram(moduli=CLEBSCH, degree=5, restarts=20, steps=10, seed=9)
    first = run_program(program, independence_cap=1, budget=500)
    second = run_program(program, independence_cap=1, budget=500)
    assert first.best_independence == second.best_independence
    assert first.best_generators == second.best_generators


def test_forbidden_generators_never_appear() -> None:
    group = AbelianGroup(moduli=CLEBSCH)
    program = SearchProgram(
        moduli=CLEBSCH, degree=5, restarts=50, steps=10, seed=4, forbidden=(15,)
    )
    outcome = run_program(program, independence_cap=5, budget=2000)
    assert 15 not in outcome.best_generators
    assert group.negate(15) not in outcome.best_generators


def test_a_base_of_one_is_reported_as_degenerate() -> None:
    """A one-vertex base has no structure to lift, so it is just a Cayley graph."""
    program = LiftProgram(moduli=CLEBSCH, base_size=1, degree=5, restarts=5, steps=0, seed=1)
    assert program.degenerate
    assert (
        "base_size=1 makes this exactly a Cayley graph"
        in run_lift_program(program, independence_cap=1, budget=20).render()
    )


def test_a_genuine_base_is_not_flagged() -> None:
    assert not LiftProgram(moduli=(5,), base_size=2, degree=3, restarts=1, steps=0).degenerate


def test_a_near_miss_points_at_the_tool_that_can_leave_the_family() -> None:
    """Structured search plateauing within two of the cap is the repair signal."""
    program = SearchProgram(moduli=CLEBSCH, degree=5, restarts=60, steps=20, seed=1)
    outcome = run_program(program, independence_cap=4, budget=2000)
    assert outcome.gap in (1, 2)
    assert "commission_repair" in outcome.render()


def test_a_distant_miss_does_not_suggest_repair() -> None:
    program = SearchProgram(moduli=CLEBSCH, degree=5, restarts=3, steps=1, seed=1)
    outcome = run_program(program, independence_cap=0, budget=10)
    assert outcome.gap > 2
    assert "commission_repair" not in outcome.render()
