from __future__ import annotations

import random

import pytest

from codes.anneal import (
    Anneal,
    Shape,
    distance,
    energy,
    generator_rows,
    rotate,
)
from codes.campaign50 import (
    PUBLISHED,
    SEED,
    SHAPE,
    TARGET,
    Worker,
    calibrate,
    certify,
    even_starts,
)
from codes.linear import LinearCode, minimum_distance


def test_shape_reports_its_geometry() -> None:
    shape = Shape(height=4, block=5, parity_blocks=6)
    assert shape.length == 50
    assert shape.dimension == 20
    assert shape.free_coefficients == 120
    assert "[50,20]" in shape.describe()


def test_rotation_is_multiplication_modulo_x_to_the_block() -> None:
    assert rotate(0b00001, 1, 5) == 0b00010
    assert rotate(0b10000, 1, 5) == 0b00001
    assert rotate(0b10101, 0, 5) == 0b10101


def test_the_identity_blocks_make_the_generator_full_rank() -> None:
    """Systematic form guarantees the dimension, so it is never computed."""
    rows = generator_rows(SHAPE, SEED)
    assert len(rows) == SHAPE.dimension
    assert LinearCode(length=SHAPE.length, rows=rows).dimension == SHAPE.dimension


def test_the_published_seed_rebuilds_exactly() -> None:
    """[50,20,13] is ChubenkoKurz 2025; the refinement must reproduce it."""
    assert distance(SHAPE, SEED) == PUBLISHED
    assert energy(SHAPE, SEED, TARGET) == 430


def test_the_kernel_distance_agrees_with_the_referee() -> None:
    code = LinearCode(length=SHAPE.length, rows=generator_rows(SHAPE, SEED))
    assert distance(SHAPE, SEED) == minimum_distance(code)


def test_energy_is_zero_exactly_when_the_target_is_met() -> None:
    assert energy(SHAPE, SEED, PUBLISHED) == 0
    assert energy(SHAPE, SEED, PUBLISHED + 1) > 0


def test_energy_squares_the_deficit() -> None:
    """A code of distance 13 scored against 15 pays 4 per weight-13 word."""
    at_fourteen = energy(SHAPE, SEED, 14)
    at_fifteen = energy(SHAPE, SEED, 15)
    assert at_fifteen > at_fourteen


def test_the_abort_stops_accumulation_early() -> None:
    assert energy(SHAPE, SEED, TARGET, abort=10) > 10


def test_a_worker_adopts_its_start_and_tracks_a_best() -> None:
    worker = Anneal(shape=SHAPE, target=TARGET, parity=SEED, rng=random.Random(1))
    assert worker.current == 430
    assert worker.best == 430
    assert worker.best_parity == SEED


def test_a_proposal_changes_only_the_parity_coefficients() -> None:
    worker = Anneal(shape=SHAPE, target=TARGET, parity=SEED, rng=random.Random(2))
    for _ in range(40):
        proposal = worker.propose()
        assert len(proposal) == len(SEED)
        assert all(0 <= value < (1 << SHAPE.block) for value in proposal)


def test_an_even_worker_preserves_block_row_parity() -> None:
    """Moving coefficients in pairs keeps every codeword even."""
    worker = Anneal(shape=SHAPE, target=TARGET, parity=SEED, rng=random.Random(3), even_only=True)
    blocks = SHAPE.parity_blocks
    for _ in range(60):
        proposal = worker.propose()
        for row in range(SHAPE.height):
            before = sum(v.bit_count() for v in SEED[row * blocks : (row + 1) * blocks])
            after = sum(v.bit_count() for v in proposal[row * blocks : (row + 1) * blocks])
            assert before % 2 == after % 2


def test_stepping_never_worsens_the_recorded_best() -> None:
    worker = Anneal(
        shape=SHAPE, target=TARGET, parity=SEED, rng=random.Random(4), temperature=200.0
    )
    start = worker.best
    for _ in range(200):
        worker.step()
    assert worker.best <= start


def test_calibration_returns_a_usable_temperature() -> None:
    temperature = calibrate(SEED, random.Random(5), samples=64)
    assert temperature > 0


def test_the_even_start_pool_is_the_prescribed_size() -> None:
    pool = even_starts()
    assert len(pool) == 900
    assert all(len(start) == len(SEED) for start in pool)


def test_every_even_start_differs_from_the_seed_in_two_rows() -> None:
    blocks = SHAPE.parity_blocks
    for start in even_starts(limit=25):
        changed = {
            index // blocks for index, (a, b) in enumerate(zip(SEED, start, strict=True)) if a != b
        }
        assert changed <= {2, 3}


def test_certification_reports_whether_a_bound_was_beaten() -> None:
    found, is_new = certify(SEED)
    assert found == PUBLISHED
    assert not is_new


def test_a_worker_restarts_from_its_pool_but_keeps_its_best() -> None:
    pool = even_starts(limit=8)
    anneal = Anneal(shape=SHAPE, target=TARGET, parity=SEED, rng=random.Random(6))
    worker = Worker(anneal=anneal, start_pool=pool, initial_temperature=10.0)
    best = worker.anneal.best
    worker._restart()  # noqa: SLF001
    assert worker.restarts == 1
    assert worker.anneal.best == best
    assert worker.anneal.parity in pool


def test_the_kernel_refuses_an_unsupported_rank() -> None:
    with pytest.raises(ValueError, match="length must be positive"):
        LinearCode(length=0, rows=())
