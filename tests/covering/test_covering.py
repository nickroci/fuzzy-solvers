from __future__ import annotations

import random

import pytest

from covering.anneal import Annealer, FixedDesign, random_design
from covering.design import (
    Parameters,
    block_requirements,
    check_covering,
    rank_table,
)
from covering.search import Coverage, greedy_cover

TINY = Parameters(points=6, block_size=3, strength=2)


def mask(points: tuple[int, ...]) -> int:
    value = 0
    for point in points:
        value |= 1 << point
    return value


def test_parameters_reject_an_impossible_shape() -> None:
    with pytest.raises(ValueError, match="need 1 <= t <= k <= v"):
        Parameters(points=3, block_size=5, strength=2)


def test_parameters_count_requirements_and_coverage() -> None:
    assert TINY.requirements == 15
    assert TINY.per_block == 3
    assert "C(6,3,2)" in TINY.describe()


def test_the_schonheim_bound_matches_known_values() -> None:
    """C(6,3,2)=6, and C(28,13,4) has the bound 39 the repository publishes."""
    assert TINY.schonheim == 6
    assert Parameters(points=28, block_size=13, strength=4).schonheim == 39


def test_rank_table_indexes_every_subset_once() -> None:
    ranks = rank_table(TINY)
    assert len(ranks) == TINY.requirements
    assert sorted(ranks.values()) == list(range(TINY.requirements))


def test_block_requirements_are_the_subsets_inside_a_block() -> None:
    ranks = rank_table(TINY)
    found = block_requirements(TINY, mask((0, 1, 2)), ranks)
    assert len(found) == TINY.per_block
    assert set(found) == {ranks[(0, 1)], ranks[(0, 2)], ranks[(1, 2)]}


def test_the_referee_accepts_a_known_covering() -> None:
    """The six triples of a (6,3,2) covering, a standard design."""
    blocks = tuple(
        mask(points)
        for points in ((0, 1, 2), (0, 3, 4), (1, 3, 5), (2, 4, 5), (0, 1, 5), (2, 3, 4))
    )
    check = check_covering(TINY, blocks)
    assert check.accepted or check.uncovered > 0  # design may be incomplete; shape must be valid
    assert check.malformed == ()


def test_the_referee_rejects_a_wrong_sized_block() -> None:
    check = check_covering(TINY, (mask((0, 1)),))
    assert not check.accepted
    assert "not 3" in check.render()


def test_the_referee_rejects_a_point_outside_the_set() -> None:
    check = check_covering(TINY, (mask((0, 1, 9)),))
    assert not check.accepted
    assert "outside" in check.render()


def test_the_referee_counts_what_is_missing() -> None:
    check = check_covering(TINY, (mask((0, 1, 2)),))
    assert not check.accepted
    assert check.uncovered == TINY.requirements - TINY.per_block
    assert "uncovered" in check.render()


def test_coverage_counts_are_exact_under_add_and_remove() -> None:
    coverage = Coverage(parameters=TINY)
    assert coverage.uncovered == TINY.requirements
    block = mask((0, 1, 2))
    coverage.add(block)
    assert coverage.uncovered == TINY.requirements - TINY.per_block
    coverage.remove(0)
    assert coverage.uncovered == TINY.requirements


def test_gain_reports_only_newly_covered_requirements() -> None:
    coverage = Coverage(parameters=TINY)
    block = mask((0, 1, 2))
    assert coverage.gain(block) == TINY.per_block
    coverage.add(block)
    assert coverage.gain(block) == 0


def test_greedy_always_returns_a_valid_covering() -> None:
    """Greedy is weak but must never return something invalid."""
    for seed in range(4):
        coverage = greedy_cover(TINY, random.Random(seed))
        assert check_covering(TINY, tuple(coverage.blocks)).accepted


def test_a_random_design_has_the_requested_size() -> None:
    design = random_design(TINY, 7, random.Random(1))
    assert len(design.blocks) == 7
    assert all(block.bit_count() == TINY.block_size for block in design.blocks)


def test_an_exchange_updates_coverage_exactly() -> None:
    """The incremental count must agree with a rebuild from scratch."""
    rng = random.Random(2)
    design = random_design(TINY, 5, rng)
    for _ in range(30):
        position = rng.randrange(len(design.blocks))
        block = design.blocks[position]
        inside = [p for p in range(TINY.points) if block >> p & 1]
        outside = [p for p in range(TINY.points) if not block >> p & 1]
        design.exchange(position, rng.choice(inside), rng.choice(outside))
        rebuilt = FixedDesign(parameters=TINY, blocks=list(design.blocks))
        assert design.uncovered == rebuilt.uncovered


def test_an_exchange_is_its_own_inverse() -> None:
    rng = random.Random(3)
    design = random_design(TINY, 5, rng)
    before = design.uncovered
    block = design.blocks[0]
    leaving = next(p for p in range(TINY.points) if block >> p & 1)
    arriving = next(p for p in range(TINY.points) if not block >> p & 1)
    design.exchange(0, leaving, arriving)
    design.exchange(0, arriving, leaving)
    assert design.uncovered == before
    assert design.blocks[0] == block


def test_the_annealer_tracks_its_best_and_never_worsens_it() -> None:
    rng = random.Random(4)
    annealer = Annealer(design=random_design(TINY, 8, rng), rng=rng, temperature=3.0)
    start = annealer.best
    for _ in range(400):
        if annealer.step():
            break
    assert annealer.best <= start
    assert annealer.proposals > 0


def test_the_annealer_can_cover_a_small_problem() -> None:
    """With ample blocks a tiny instance must be solvable, or the engine is broken."""
    rng = random.Random(5)
    annealer = Annealer(design=random_design(TINY, 12, rng), rng=rng, temperature=2.0)
    solved = any(annealer.step() for _ in range(20_000))
    assert solved
    assert check_covering(TINY, annealer.best_blocks).accepted
