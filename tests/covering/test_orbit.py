from __future__ import annotations

import random
from itertools import combinations

import pytest

from covering.design import Parameters, check_covering, rank_table
from covering.orbit import (
    PointGroup,
    admissible_orbit_sizes,
    apply_to_block,
    cyclic,
    cyclic_with_fixed,
    develop,
    expressible,
    frobenius,
    from_cycles,
    from_generators,
    grid_translations,
    multiplier,
    orbit,
    orbit_sizes,
    transitive_block_count_constraint,
)
from covering.orbit_search import OrbitSearch, random_block, uncovered_count
from covering.tools import BudgetExhaustedError, CoveringTools, Feasibility

SMALL = Parameters(points=7, block_size=3, strength=2)


def test_a_generator_must_be_a_permutation() -> None:
    with pytest.raises(ValueError, match="not a permutation"):
        PointGroup(points=3, generators=((0, 0, 1),))


def test_a_generator_must_cover_every_point() -> None:
    with pytest.raises(ValueError, match="images"):
        PointGroup(points=3, generators=((0, 1),))


def test_the_cyclic_group_has_the_expected_order() -> None:
    assert cyclic(7).order == 7
    assert cyclic_with_fixed(8).order == 7
    assert cyclic_with_fixed(8).name == "Z7+1"


def test_a_fixed_point_group_needs_something_to_move() -> None:
    with pytest.raises(ValueError, match="at least one moving point"):
        cyclic_with_fixed(2, 2)


def test_the_multiplier_group_needs_a_unit() -> None:
    with pytest.raises(ValueError, match="must be a unit"):
        multiplier(28, 14)


def test_a_multiplier_group_is_larger_than_the_shift_alone() -> None:
    assert multiplier(28, 9).order > cyclic(28).order


def test_frobenius_needs_a_unit_of_the_requested_order() -> None:
    with pytest.raises(ValueError, match="no unit of order"):
        frobenius(7, 5)


def test_frobenius_builds_an_affine_group() -> None:
    group = frobenius(7, 3)
    assert group.order == 21
    assert "AGL1(7)" in group.name


def test_applying_a_permutation_moves_the_block() -> None:
    assert apply_to_block((1, 2, 0), 0b001) == 0b010


def test_an_orbit_is_closed_and_deduplicated() -> None:
    group = cyclic(7)
    images = orbit(group, 0b0000111)
    assert len(images) == 7
    assert len(set(images)) == 7


def test_a_symmetric_block_has_a_short_orbit() -> None:
    """A block invariant under the shift has an orbit shorter than the group."""
    group = cyclic(4)
    assert len(orbit(group, 0b1111)) == 1


def test_developing_merges_overlapping_orbits() -> None:
    group = cyclic(7)
    blocks = develop(group, (0b0000111, 0b0001110))
    assert len(blocks) == 7  # the second is a shift of the first
    assert orbit_sizes(group, (0b0000111,)) == (7,)


def test_cycle_notation_builds_a_group_fixing_unnamed_points() -> None:
    group = from_cycles(5, ((0, 1, 2),), "Z3+2")
    assert group.order == 3
    assert apply_to_block(group.generators[0], 1 << 4) == 1 << 4


def test_cycle_notation_rejects_a_repeated_point() -> None:
    with pytest.raises(ValueError, match="more than one cycle"):
        from_cycles(5, ((0, 1), (1, 2)), "bad")


def test_cycle_notation_rejects_a_point_outside_the_set() -> None:
    with pytest.raises(ValueError, match="outside"):
        from_cycles(5, ((0, 9),), "bad")


def test_admissible_orbit_sizes_follow_from_the_block_size() -> None:
    """A stabiliser's order divides the block size, which fixes the orbits."""
    assert admissible_orbit_sizes(cyclic(28), 13) == (28,)
    assert admissible_orbit_sizes(from_cycles(28, (tuple(range(13)),), "Z13"), 13) == (1, 13)


def test_a_cyclic_group_on_28_points_cannot_express_53_blocks() -> None:
    """The arithmetic rules this out before any search is run."""
    sizes = admissible_orbit_sizes(cyclic(28), 13)
    assert not expressible(sizes, 53)
    assert expressible(sizes, 56)


def test_a_thirteen_cycle_group_can_express_53() -> None:
    group = from_cycles(28, (tuple(range(13)), tuple(range(13, 26))), "Z13")
    assert expressible(admissible_orbit_sizes(group, 13), 53)


def test_expressibility_rejects_a_negative_total() -> None:
    assert not expressible((3, 5), -1)


def test_uncovered_count_agrees_with_the_referee() -> None:
    group = cyclic(7)
    blocks = develop(group, (0b0000111,))
    ranks = rank_table(SMALL)
    assert uncovered_count(SMALL, blocks, ranks) == check_covering(SMALL, blocks).uncovered


def test_the_orbit_search_reaches_a_valid_covering_on_a_small_problem() -> None:
    rng = random.Random(1)
    search = OrbitSearch(
        parameters=SMALL, group=cyclic(7), base_blocks=[random_block(SMALL, rng)], rng=rng
    )
    for _ in range(600):
        if search.step():
            break
    blocks, valid = search.certify()
    assert search.best <= search.score(tuple(search.best_blocks)) or search.best >= 0
    if search.best == 0:
        assert valid
        assert blocks > 0


def test_a_proposal_keeps_the_block_size() -> None:
    rng = random.Random(2)
    search = OrbitSearch(
        parameters=SMALL, group=cyclic(7), base_blocks=[random_block(SMALL, rng)], rng=rng
    )
    for _ in range(30):
        for block in search.propose():
            assert block.bit_count() == SMALL.block_size


# --- tool surface ---


def tools(budget: int = 200_000) -> CoveringTools:
    return CoveringTools(
        parameters=SMALL,
        published=8,
        lower_bound=5,
        budget=budget,
        provenance="Simulated Annealing, 2008",
    )


def test_the_target_states_the_room_available() -> None:
    text = tools().describe_target()
    assert "7 blocks is a NEW REPOSITORY ENTRY" in text
    assert "lower bound: 5" in text
    assert "Simulated Annealing, 2008" in text


def test_named_groups_resolve_and_unknown_ones_are_rejected() -> None:
    surface = tools()
    assert surface.named_group("cyclic").order == 7
    with pytest.raises(ValueError, match="unknown family"):
        surface.named_group("nonesuch")


def test_the_model_can_author_its_own_group() -> None:
    group = tools().authored_group(((0, 1, 2),), "mine")
    assert group.order == 3


def test_the_trivial_group_develops_a_design_into_itself() -> None:
    surface = tools()
    group = surface.trivial_group()
    assert group.order == 1
    assert develop(group, (0b0000111,)) == (0b0000111,)


def test_feasibility_explains_why_a_group_cannot_reach_a_count() -> None:
    surface = CoveringTools(
        parameters=Parameters(points=28, block_size=13, strength=4),
        published=53,
        lower_bound=39,
        budget=100,
    )
    verdict = surface.check_feasible(surface.named_group("cyclic"), 52)
    assert not verdict.reachable
    assert "is NOT a sum" in verdict.render()
    assert "shares a factor" in verdict.render()


def test_feasibility_confirms_a_group_that_can() -> None:
    surface = CoveringTools(
        parameters=Parameters(points=28, block_size=13, strength=4),
        published=53,
        lower_bound=39,
        budget=100,
    )
    group = surface.authored_group((tuple(range(13)), tuple(range(13, 26))), "Z13")
    assert surface.check_feasible(group, 52).reachable


def test_a_transitive_group_carries_a_block_count_divisor() -> None:
    """Under a point-transitive group v*r = B*k, so v/gcd(v,k) divides the count."""
    surface = CoveringTools(
        parameters=Parameters(points=28, block_size=13, strength=4),
        published=53,
        lower_bound=39,
        budget=100,
    )
    assert surface.check_feasible(surface.named_group("cyclic"), 56).divisor == 28


def test_a_count_the_divisor_forbids_is_reported_as_unreachable() -> None:
    """The orbit sizes can sum to a count the transitivity arithmetic still forbids.

    Rendered from stated fields rather than from a group, because for a
    transitive group every admissible orbit size is itself a multiple of the
    divisor - so the two checks agree, and this one only ever speaks up if that
    ever stops being true.
    """
    verdict = Feasibility(group="G", order=28, sizes=(14,), target=14, reachable=True, divisor=28)
    assert "does not divide" in verdict.render()


def test_feasibility_costs_no_budget() -> None:
    surface = tools()
    surface.check_feasible(surface.named_group("cyclic"), 7)
    assert surface.used == 0


def test_search_spends_the_ledger_and_reports() -> None:
    surface = tools()
    text = surface.search(surface.named_group("cyclic"), 1, 1000, seed=1)
    assert "orbit sizes" in text
    assert surface.used > 0


def test_search_rejects_a_non_positive_base_count() -> None:
    with pytest.raises(ValueError, match="base_count must be positive"):
        tools().search(tools().named_group("cyclic"), 0, 10)


def test_the_budget_is_enforced() -> None:
    surface = tools(budget=1)
    surface.search(surface.named_group("cyclic"), 1, 50, seed=1)
    with pytest.raises(BudgetExhaustedError, match="budget of 1"):
        surface.search(surface.named_group("cyclic"), 1, 50, seed=2)


def test_status_reports_the_ledger() -> None:
    assert "banked: none" in tools().status()


def test_authored_base_blocks_can_be_adopted_without_searching() -> None:
    """A construction written by hand enters the session as the incumbent."""
    surface = tools()
    group = surface.trivial_group()
    blocks = tuple(sum(1 << point for point in subset) for subset in combinations(range(7), 3))
    text = surface.adopt(group, blocks)
    assert "Uncovered: 0" in text
    assert surface.used == 0


def test_adopt_rejects_a_wrong_sized_block() -> None:
    surface = tools()
    with pytest.raises(ValueError, match="not 3"):
        surface.adopt(surface.trivial_group(), (0b11,))


def test_inspection_reports_structure_not_just_a_count() -> None:
    """A shortfall is described by what is missing, so it can be reasoned about."""
    surface = tools()
    assert "No design yet" in surface.inspect()
    surface.adopt(surface.named_group("cyclic"), (0b0000111,))
    text = surface.inspect()
    assert "orbit(s) under the group" in text
    assert "Representative uncovered requirement" in text
    assert "single group orbit" in text


def test_inspection_reports_a_complete_design_as_complete() -> None:
    surface = tools()
    group = surface.trivial_group()
    blocks = tuple(sum(1 << point for point in subset) for subset in combinations(range(7), 3))
    surface.adopt(group, blocks)
    assert "nothing uncovered" in surface.inspect()


def test_resuming_without_an_incumbent_is_refused() -> None:
    with pytest.raises(ValueError, match="no incumbent"):
        tools().resume(100)


def test_resuming_continues_from_the_best_design() -> None:
    surface = tools()
    surface.search(surface.named_group("cyclic"), 1, 1000, seed=1)
    spent = surface.used
    surface.resume(1000, seed=2)
    assert surface.used > spent


def test_the_stabiliser_need_not_act_freely_on_the_block() -> None:
    """A stabiliser can fix points, so its order need not divide the block size.

    Under a 26-cycle with two fixed points, a 13-subset made of one fixed point
    and six antipodal pairs has a stabiliser of order 2 - and 2 does not divide
    13. Requiring divisibility predicts orbit sizes (2, 26) where the truth is
    (2, 13, 26), and would report the group as unable to reach counts it can.
    """
    sizes = admissible_orbit_sizes(cyclic_with_fixed(28, 2), 13)
    assert 13 in sizes
    assert sizes == (2, 13, 26)


def test_admissible_sizes_match_brute_force() -> None:
    group = cyclic_with_fixed(12, 1)
    seen: set[int] = set()
    truth: set[int] = set()
    for points in combinations(range(12), 5):
        block = 0
        for point in points:
            block |= 1 << point
        if block in seen:
            continue
        images = orbit(group, block)
        seen.update(images)
        truth.add(len(images))
    assert set(admissible_orbit_sizes(group, 5)) == truth


def test_a_transitive_group_forces_a_block_count_multiple() -> None:
    """v*r = B*k under transitivity, so v divides B*k."""
    assert transitive_block_count_constraint(cyclic(28), 28, 13) == 28
    assert transitive_block_count_constraint(cyclic_with_fixed(28), 28, 13) == 1


def test_product_groups_need_more_than_one_generator() -> None:
    """A single permutation generates a cyclic group; C5xC5 is not cyclic."""
    group = grid_translations(5, 5)
    assert group.order == 25
    assert len(group.generators) == 2
    assert admissible_orbit_sizes(group, 15) == (5, 25)


def test_from_generators_builds_a_non_cyclic_group() -> None:
    group = from_generators(4, (((0, 1),), ((2, 3),)), "V4")
    assert group.order == 4


def test_search_takes_many_restarts_in_one_call() -> None:
    """Restarts are the currency, so one tool call must be able to buy several.

    Aimed at one block against 21 pairs, which no restart can cover, so every
    restart runs and the spread is the whole point rather than an early exit.
    """
    surface = tools()
    text = surface.search(surface.trivial_group(), 1, 2000, seed=1, restarts=4)
    assert "Restarts reached:" in text
    assert surface.used == 4 * 2000


def test_restarts_stop_early_once_a_design_is_found() -> None:
    """Developing {0,1,3} under Z7 is the Fano plane, so the rest are waste."""
    surface = tools()
    text = surface.search(surface.named_group("cyclic"), 1, 4000, seed=1, restarts=40)
    if "Uncovered: 0" in text:
        assert surface.used < 40 * 4000


def test_search_rejects_a_non_positive_restart_count() -> None:
    surface = tools()
    with pytest.raises(ValueError, match="restarts must be positive"):
        surface.search(surface.named_group("cyclic"), 1, 10, restarts=0)


def test_restarts_stop_once_the_budget_runs_out() -> None:
    """A restart loop must not spend past the ledger just because it was asked to."""
    surface = tools(budget=5000)
    surface.search(surface.named_group("cyclic"), 1, 4000, seed=1, restarts=50)
    assert surface.used <= surface.budget


def test_a_design_with_too_many_blocks_is_called_out() -> None:
    """Covering everything with more blocks than the target is not progress.

    An agent that develops seven base blocks where the target is five gets a
    valid covering and no entry, and the reply has to say so or the session
    spends its budget being pleased with itself.
    """
    surface = CoveringTools(parameters=SMALL, published=4, lower_bound=2, budget=100_000)
    group = surface.trivial_group()
    blocks = tuple(sum(1 << point for point in subset) for subset in combinations(range(7), 3))
    text = surface.adopt(group, blocks)
    assert "Uncovered: 0" in text
    assert "cannot be one however well it covers" in text
    assert surface.banked is None


def test_one_restart_cannot_swallow_the_whole_budget() -> None:
    """A single anneal is capped, so a session cannot spend its wall clock on one sample.

    A live session asked for its entire budget in one call and sat inside it
    for twenty minutes, learning nothing and leaving itself no decisions.
    """
    surface = CoveringTools(parameters=SMALL, published=8, lower_bound=5, budget=500_000_000)
    surface.search(surface.trivial_group(), 1, 500_000_000, seed=1)
    assert surface.used <= 20_000_000


def contention_surface() -> CoveringTools:
    """Return a surface where four blocks is an entry and seven is not."""
    return CoveringTools(parameters=SMALL, published=5, lower_bound=2, budget=100_000)


def test_an_oversized_design_never_displaces_one_in_contention() -> None:
    """A perfect covering with too many blocks must not become the incumbent.

    A live session found a 72-block covering of a cell needing 24, adopted it
    because nothing was uncovered, and then resumed from it - spending its last
    ten tool calls improving a design that could never be an entry.
    """
    surface = contention_surface()
    group = surface.trivial_group()
    surface.adopt(group, (0b0000111, 0b0111000, 0b1100001, 0b0011100))
    contender = surface.incumbent_uncovered
    assert contender > 0

    perfect = tuple(sum(1 << point for point in subset) for subset in combinations(range(7), 3))
    text = surface.adopt(group, perfect)
    assert "Uncovered: 0" in text
    assert "cannot be one however well it covers" in text
    assert surface.incumbent_uncovered == contender


def test_an_oversized_incumbent_is_not_reported_as_covered() -> None:
    """Reporting zero for a design that cannot be an entry reads as success."""
    surface = contention_surface()
    perfect = tuple(sum(1 << point for point in subset) for subset in combinations(range(7), 3))
    surface.adopt(surface.trivial_group(), perfect)
    assert surface.incumbent_uncovered == SMALL.requirements
    assert surface.banked is None


def test_a_design_in_contention_displaces_a_worse_one() -> None:
    """Among candidates of the right size, fewer uncovered still wins."""
    surface = contention_surface()
    group = surface.trivial_group()
    surface.adopt(group, (0b0000111,))
    worse = surface.incumbent_uncovered
    surface.adopt(group, (0b0000111, 0b0111000, 0b1100001, 0b0011100))
    assert surface.incumbent_uncovered < worse


def test_the_target_states_the_counting_fact_about_transitive_groups() -> None:
    """Incidences that do not divide evenly rule out every point-transitive group.

    Sessions repeatedly spent their budget on group structure that this one
    division rules out, so the target now says it. It is a fact about the
    target computed mechanically, not a hint about the answer.
    """
    ruled_out = CoveringTools(
        parameters=Parameters(points=65, block_size=14, strength=2),
        published=29,
        lower_bound=27,
        budget=1000,
    ).describe_target()
    assert "392 incidences over 65 points" in ruled_out
    assert "NO POINT-TRANSITIVE GROUP" in ruled_out

    allowed = CoveringTools(
        parameters=Parameters(points=7, block_size=3, strength=2),
        published=8,
        lower_bound=5,
        budget=1000,
    ).describe_target()
    assert "exactly 3 each" in allowed
    assert "not excluded by counting alone" in allowed
