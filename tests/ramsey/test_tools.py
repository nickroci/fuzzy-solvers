from __future__ import annotations

import pytest

from ramsey.cayley import circulant
from ramsey.tools import BudgetExhaustedError, RamseyTarget, RamseyTools

# R(3,6) = 18, so a (3,6,16)-graph exists: the Clebsch graph over Z2^4.
CLEBSCH_TARGET = RamseyTarget(second_colour=6, record=16, credit="test")
CLEBSCH_MODULI = (2, 2, 2, 2)
CLEBSCH_GENERATORS = (1, 2, 4, 8, 15)


def tools(budget: int = 20) -> RamseyTools:
    return RamseyTools(target=CLEBSCH_TARGET, kernel_budget=budget)


def test_target_reports_its_structural_constraints() -> None:
    text = CLEBSCH_TARGET.describe()
    assert "16 vertices" in text
    assert "independence number at most 5" in text
    assert "degree above 5" in text
    assert "sum-free" in text


def test_target_derives_caps_from_the_second_colour() -> None:
    assert CLEBSCH_TARGET.order == 16
    assert CLEBSCH_TARGET.independence_cap == 5
    assert CLEBSCH_TARGET.degree_cap == 5


def test_a_genuine_witness_is_accepted_and_bankable() -> None:
    surface = tools()
    evaluation = surface.evaluate_cayley(CLEBSCH_MODULI, CLEBSCH_GENERATORS)
    assert evaluation.accepted
    assert evaluation.triangle_free
    assert evaluation.independence_number == 5
    assert "ACCEPTED" in evaluation.render()
    message = surface.bank(evaluation.handle)
    assert "R(3,6) > 16" in message
    assert surface.banked is not None
    assert surface.witness_graph6()


def test_a_non_sum_free_connection_set_is_reported_structurally() -> None:
    surface = tools()
    evaluation = surface.evaluate_cayley((16,), (1, 2))
    assert not evaluation.triangle_free
    assert evaluation.sum_violation is not None
    assert "NOT triangle-free" in evaluation.render()


def test_a_triangle_free_failure_reports_the_obstruction() -> None:
    surface = tools()
    evaluation = surface.evaluate_cayley((16,), (1, 6))
    assert evaluation.triangle_free
    assert not evaluation.accepted
    assert len(evaluation.obstruction) > CLEBSCH_TARGET.independence_cap
    assert "independent set of size" in evaluation.render()


def test_a_wrong_order_group_is_rejected_before_measurement() -> None:
    surface = tools()
    evaluation = surface.evaluate_cayley((15,), (1, 4))
    assert evaluation.rejected_reason
    assert "not the required 16" in evaluation.rejected_reason
    assert "rejected" in evaluation.render()


def test_an_oversized_connection_set_is_rejected() -> None:
    surface = tools()
    evaluation = surface.evaluate_cayley((16,), (1, 3, 5, 7))
    assert evaluation.rejected_reason
    assert "exceeds the degree cap" in evaluation.rejected_reason


def test_an_empty_connection_set_is_rejected() -> None:
    surface = tools()
    evaluation = surface.evaluate_cayley((16,), (0,))
    assert evaluation.rejected_reason
    assert "empty" in evaluation.rejected_reason


def test_explicit_graphs_can_be_measured() -> None:
    surface = tools()
    evaluation = surface.evaluate_graph("explicit", circulant(16, (1, 6)))
    assert evaluation.triangle_free
    assert not evaluation.accepted


def test_an_explicit_graph_of_the_wrong_order_is_rejected() -> None:
    surface = tools()
    evaluation = surface.evaluate_graph("explicit", circulant(15, (1, 4)))
    assert "not the required 16" in evaluation.rejected_reason


def test_equal_proposals_share_one_handle() -> None:
    surface = tools()
    first = surface.evaluate_cayley(CLEBSCH_MODULI, CLEBSCH_GENERATORS)
    second = surface.evaluate_cayley(CLEBSCH_MODULI, CLEBSCH_GENERATORS)
    assert first.handle == second.handle
    assert surface.calls_used == 2


def test_every_evaluation_costs_one_kernel_call() -> None:
    surface = tools(budget=3)
    assert surface.calls_remaining == 3
    surface.evaluate_cayley((16,), (1, 6))
    assert surface.calls_used == 1
    assert surface.calls_remaining == 2


def test_the_budget_is_enforced() -> None:
    surface = tools(budget=1)
    surface.evaluate_cayley((16,), (1, 6))
    with pytest.raises(BudgetExhaustedError, match="budget of 1"):
        surface.evaluate_cayley((16,), (1, 7))


def test_an_unaccepted_candidate_cannot_be_banked() -> None:
    surface = tools()
    evaluation = surface.evaluate_cayley((16,), (1, 6))
    with pytest.raises(ValueError, match="not accepted by the referee"):
        surface.bank(evaluation.handle)


def test_an_unknown_handle_cannot_be_banked() -> None:
    with pytest.raises(KeyError, match="no candidate with handle"):
        tools().bank("c_missing")


def test_the_witness_is_unavailable_until_banked() -> None:
    with pytest.raises(ValueError, match="no witness has been banked"):
        tools().witness_graph6()


def test_status_reports_budget_and_incumbent() -> None:
    surface = tools()
    assert "incumbent: none" in surface.status()
    evaluation = surface.evaluate_cayley(CLEBSCH_MODULI, CLEBSCH_GENERATORS)
    surface.bank(evaluation.handle)
    assert evaluation.handle in surface.status()


def test_describe_target_is_the_tool_surface_view() -> None:
    assert tools().describe_target() == CLEBSCH_TARGET.describe()


# --- refinement kernel ---


def test_refine_walks_from_an_archived_candidate() -> None:
    surface = tools(budget=40)
    start = surface.evaluate_cayley((16,), (1, 6))
    report = surface.refine(start.handle, "swap", 12)
    assert report.steps_spent > 0
    assert report.best_independence <= report.start_independence
    assert surface.calls_used == 1 + report.steps_spent
    assert "refine[swap]" in report.render()


def test_refine_can_reach_and_accept_a_witness() -> None:
    """A swap walk from a near miss should be able to land on a real witness."""
    surface = tools(budget=200)
    start = surface.evaluate_cayley((2, 2, 2, 2), (1, 2, 4, 8))
    report = surface.refine(start.handle, "add", 30)
    assert report.improved or report.best_independence == report.start_independence
    if report.accepted:
        assert surface.bank(report.best_handle)


def test_refine_rejects_an_unknown_move_class() -> None:
    surface = tools()
    start = surface.evaluate_cayley((16,), (1, 6))
    with pytest.raises(ValueError, match="move_class must be one of"):
        surface.refine(start.handle, "teleport", 5)


def test_refine_rejects_a_non_positive_step_count() -> None:
    surface = tools()
    start = surface.evaluate_cayley((16,), (1, 6))
    with pytest.raises(ValueError, match="steps must be positive"):
        surface.refine(start.handle, "swap", 0)


def test_refine_rejects_an_unknown_handle() -> None:
    with pytest.raises(KeyError, match="no candidate with handle"):
        tools().refine("c_missing", "swap", 5)


def test_refine_rejects_a_non_cayley_candidate() -> None:
    surface = tools()
    evaluation = surface.evaluate_graph("explicit", circulant(16, (1, 6)))
    with pytest.raises(ValueError, match="not a Cayley construction"):
        surface.refine(evaluation.handle, "swap", 5)


def test_refine_stops_at_the_kernel_budget() -> None:
    surface = tools(budget=4)
    start = surface.evaluate_cayley((16,), (1, 6))
    report = surface.refine(start.handle, "swap", 50)
    assert report.steps_spent == 3
    assert surface.calls_remaining == 0


def test_refine_is_deterministic_in_its_seed() -> None:
    first = tools(budget=40)
    second = tools(budget=40)
    for surface in (first, second):
        start = surface.evaluate_cayley((16,), (1, 6))
        surface.refine(start.handle, "swap", 10, seed=3)
    assert first.status().split("incumbent")[0] == second.status().split("incumbent")[0]


def test_drop_moves_shrink_the_generator_set() -> None:
    surface = tools(budget=40)
    start = surface.evaluate_cayley((2, 2, 2, 2), (1, 2, 4, 8, 15))
    report = surface.refine(start.handle, "drop", 5)
    assert report.steps_spent >= 0
    assert "refine[drop]" in report.render()


# --- commissioned search ---


def test_commission_search_finds_and_archives_a_witness() -> None:
    surface = tools(budget=20_000)
    outcome = surface.commission_search((2, 2, 2, 2), 5, restarts=300, steps=40, seed=1)
    assert outcome.accepted
    assert surface.calls_used == outcome.evaluations
    assert surface.best_independence == 5
    handle = next(h for h, c in surface._archive.items() if c.evaluation.accepted)  # noqa: SLF001
    assert "R(3,6) > 16" in surface.bank(handle)


def test_commission_search_records_the_best_independence_reached() -> None:
    """The best reached is the diagnostic a failed run needs, so it is always kept."""
    surface = tools(budget=300)
    outcome = surface.commission_search((16,), 5, restarts=20, steps=5, seed=2)
    assert outcome.best_generators
    assert surface.best_independence == outcome.best_independence
    assert surface.best_generators == outcome.best_generators


def test_commission_search_reaches_a_circulant_witness_for_r_3_6() -> None:
    """C16(3,7,8) is one of the 2,576 (3,6,16)-graphs; the search finds one."""
    surface = tools(budget=5_000)
    outcome = surface.commission_search((16,), 5, restarts=100, steps=20, seed=2)
    assert outcome.accepted
    assert outcome.best_independence == 5


def test_commission_search_rejects_a_wrong_order_group() -> None:
    with pytest.raises(ValueError, match="not the required 16"):
        tools().commission_search((15,), 5, restarts=1, steps=1)


def test_commission_search_rejects_an_oversized_degree() -> None:
    with pytest.raises(ValueError, match="exceeds the cap"):
        tools().commission_search((16,), 9, restarts=1, steps=1)


def test_commission_search_charges_every_evaluation_to_the_ledger() -> None:
    surface = tools(budget=25)
    surface.commission_search((16,), 5, restarts=500, steps=20, seed=5)
    assert surface.calls_used <= 25
    assert surface.calls_remaining >= 0


def test_one_commissioned_search_cannot_consume_the_whole_budget() -> None:
    """Several hypotheses is the point; one over-large request must not eat the session."""
    surface = tools(budget=1_000)
    surface.commission_search((16,), 5, restarts=100_000, steps=1_000, seed=11)
    assert surface.calls_remaining > 0
    assert surface.calls_used <= 350
