from __future__ import annotations

import random
from itertools import combinations

import pytest

from covering import kernel
from covering.design import Parameters, check_covering, rank_table
from covering.kernel import Schedule
from covering.orbit import PointGroup, cyclic, cyclic_with_fixed, develop, multiplier
from covering.orbit_search import random_block, uncovered_count


def trivial(points: int) -> PointGroup:
    """Return the one-element group, which develops a design into itself."""
    return PointGroup(points=points, generators=(tuple(range(points)),), name="1")


SHAPES = [(9, 4, 3), (13, 5, 3), (16, 6, 4), (12, 5, 2)]


def groups_for(points: int) -> list[PointGroup]:
    """Return a spread of groups to develop under, including the trivial one."""
    return [trivial(points), cyclic(points), cyclic_with_fixed(points, 1)]


@pytest.mark.parametrize(("points", "size", "strength"), SHAPES)
def test_kernel_uncovered_agrees_with_the_referee(points: int, size: int, strength: int) -> None:
    """The kernel's count is the referee's count, on every input tried."""
    parameters = Parameters(points=points, block_size=size, strength=strength)
    ranks = rank_table(parameters)
    rng = random.Random(4)
    for group in groups_for(points):
        for count in (1, 2, 3):
            base = tuple(random_block(parameters, rng) for _ in range(count))
            expected = uncovered_count(parameters, develop(group, base), ranks)
            assert kernel.uncovered(parameters, group, base) == expected


def test_kernel_agrees_under_a_multiplier_group() -> None:
    """A group with a non-trivial point stabiliser still counts correctly.

    Orbit images can collide there, so the same requirement is touched more
    than once by a single move; this is the case an incremental counter gets
    wrong if it reasons about a move instead of performing it.
    """
    parameters = Parameters(points=28, block_size=13, strength=4)
    ranks = rank_table(parameters)
    group = multiplier(28, 9)
    rng = random.Random(5)
    for _ in range(4):
        base = (random_block(parameters, rng), random_block(parameters, rng))
        expected = uncovered_count(parameters, develop(group, base), ranks)
        assert kernel.uncovered(parameters, group, base) == expected


@pytest.mark.parametrize(("points", "size", "strength"), SHAPES)
def test_annealing_reports_a_design_the_referee_confirms(
    points: int, size: int, strength: int
) -> None:
    """The returned base blocks really do leave the reported number uncovered.

    This is the check that makes the kernel trustworthy: it hands back an
    object, and the referee re-scores it from scratch.  A bookkeeping error in
    the incremental coverage cannot survive it.
    """
    parameters = Parameters(points=points, block_size=size, strength=strength)
    ranks = rank_table(parameters)
    rng = random.Random(6)
    group = cyclic(points)
    base = (random_block(parameters, rng), random_block(parameters, rng))
    result = kernel.anneal(parameters, group, base, Schedule(steps=5000, seed=1))
    assert len(result.base_blocks) == len(base)
    rescored = uncovered_count(parameters, develop(group, result.base_blocks), ranks)
    assert result.uncovered == rescored


def test_annealing_finds_the_fano_plane() -> None:
    """Seven triples covering every pair of seven points is a design it must reach."""
    parameters = Parameters(points=7, block_size=3, strength=2)
    group = trivial(7)
    rng = random.Random(7)
    for attempt in range(20):
        base = tuple(random_block(parameters, rng) for _ in range(7))
        result = kernel.anneal(parameters, group, base, Schedule(steps=200_000, seed=attempt))
        if result.uncovered == 0:
            blocks = develop(group, result.base_blocks)
            assert len(blocks) == 7
            assert check_covering(parameters, blocks).accepted
            return
    pytest.fail("no Fano plane found in twenty restarts")


def test_annealing_leaves_an_impossible_target_uncovered() -> None:
    """Thirteen triples cannot cover the 78 pairs of thirteen points.

    Each block covers three pairs, so 39 is the most any thirteen blocks can
    reach and 39 uncovered is the true optimum rather than a search failure.
    """
    parameters = Parameters(points=13, block_size=3, strength=2)
    group = trivial(13)
    rng = random.Random(8)
    base = tuple(random_block(parameters, rng) for _ in range(13))
    result = kernel.anneal(parameters, group, base, Schedule(steps=300_000, seed=3))
    assert result.uncovered >= 39


def test_uncovered_matches_a_hand_built_covering() -> None:
    """A covering built by hand reports nothing uncovered, by both routes."""
    parameters = Parameters(points=5, block_size=4, strength=2)
    group = trivial(5)
    base = tuple(sum(1 << point for point in subset) for subset in combinations(range(5), 4))
    assert kernel.uncovered(parameters, group, base) == 0
    assert check_covering(parameters, develop(group, base)).accepted


def test_a_wrong_sized_base_block_is_refused() -> None:
    """A block that is not the declared size is a caller error, not a score."""
    parameters = Parameters(points=7, block_size=3, strength=2)
    with pytest.raises(ValueError, match="not 3"):
        kernel.uncovered(parameters, trivial(7), (0b11,))


@pytest.mark.parametrize(
    ("steps", "start", "end", "guided"),
    [(-1, 1.0, 0.1, 0.5), (10, 0.0, 0.1, 0.5), (10, 1.0, 0.0, 0.5), (10, 1.0, 0.1, 1.5)],
)
def test_a_meaningless_schedule_is_refused(
    steps: int, start: float, end: float, guided: float
) -> None:
    """Negative steps, dead temperatures and a non-probability describe no search."""
    with pytest.raises(ValueError, match="must be"):
        Schedule(steps=steps, start_temp=start, end_temp=end, guided=guided)


def test_the_reference_annealer_matches_the_referee() -> None:
    """The fallback walk is scored by the referee too, so both paths are checked."""
    parameters = Parameters(points=9, block_size=4, strength=2)
    ranks = rank_table(parameters)
    group = cyclic(9)
    rng = random.Random(9)
    base = (random_block(parameters, rng), random_block(parameters, rng))
    result = kernel._reference_anneal(  # noqa: SLF001 - the fallback is the thing under test
        parameters, group, base, Schedule(steps=200, seed=2, start_temp=1.0, end_temp=0.05)
    )
    assert not result.accelerated
    rescored = uncovered_count(parameters, develop(group, result.base_blocks), ranks)
    assert result.uncovered == rescored


def test_members_and_mask_round_trip() -> None:
    """Point lists and bitmasks are two spellings of the same block."""
    block = 0b1010110
    points = kernel.members(block, 7)
    assert points == (1, 2, 4, 6)
    assert kernel.mask(points) == block


def test_a_search_reports_the_moves_it_actually_performed() -> None:
    """The search stops the moment nothing is uncovered, so request != spend.

    Reporting only the request makes a budget ledger fiction exactly when the
    search succeeds: a run that found a design in a fraction of a second was
    charged the five hundred million moves it had been offered.
    """
    parameters = Parameters(points=7, block_size=3, strength=2)
    group = trivial(7)
    rng = random.Random(1)
    base = tuple(random_block(parameters, rng) for _ in range(7))
    result = kernel.anneal(parameters, group, base, Schedule(steps=50_000_000, seed=3))
    assert result.steps == 50_000_000
    assert result.moves <= result.steps
    if result.uncovered == 0:
        assert result.moves < result.steps


def test_an_unsuccessful_search_performs_every_move_it_was_given() -> None:
    """With nothing to find, the count is the full request."""
    parameters = Parameters(points=13, block_size=3, strength=2)
    group = trivial(13)
    rng = random.Random(2)
    base = tuple(random_block(parameters, rng) for _ in range(3))
    result = kernel.anneal(parameters, group, base, Schedule(steps=20_000, seed=1))
    assert result.uncovered > 0
    assert result.moves == 20_000


def test_the_reference_annealer_counts_its_moves_too() -> None:
    """Both paths report the same quantity, or the fallback quietly lies."""
    parameters = Parameters(points=9, block_size=4, strength=2)
    group = trivial(9)
    rng = random.Random(3)
    base = (random_block(parameters, rng),)  # one block cannot cover 36 pairs
    result = kernel._reference_anneal(  # noqa: SLF001 - the fallback is under test
        parameters, group, base, Schedule(steps=50, seed=2)
    )
    assert result.uncovered > 0
    assert result.moves == 50
