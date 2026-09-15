"""The [50,20,14] campaign: annealing over the refined published seed.

The published [50,20,13] code is quasi-cyclic of degree 5 stacked to height 2.
Refining it - splitting each length-10 block into its even and odd positions,
so the squared shift acts as two length-5 cycles - gives the same code at
height 4 and block 5, with 120 free coefficients instead of 60. The search runs
in that larger family, starting from a point already known to be good.

The schedule follows a specific prescription rather than taste: the starting
temperature is set so that a median worsening proposal is accepted about one
time in five, cooling is geometric, and a worker that stops improving is
reheated and eventually restarted. Most workers are restricted to even codes,
where odd-weight words cannot contribute to the energy at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import TYPE_CHECKING

from codes.anneal import Anneal, Shape, distance, energy

if TYPE_CHECKING:
    import random

SHAPE = Shape(height=4, block=5, parity_blocks=6)
TARGET = 14
PUBLISHED = 13
SEED: tuple[int, ...] = (
    26,
    31,
    2,
    19,
    8,
    18,
    31,
    26,
    7,
    2,
    5,
    8,
    25,
    9,
    8,
    7,
    16,
    18,
    18,
    25,
    14,
    8,
    5,
    16,
)
"""The published [50,20,13] code, refined to height 4 and block 5."""

_ACCEPT_AT_START = 0.2
_COOL = 0.95
_COOL_EVERY = 2_000
_REHEAT_AFTER = 20_000
_RESTART_AFTER = 100_000


def calibrate(parity: tuple[int, ...], rng: random.Random, samples: int = 256) -> float:
    """Return a temperature accepting a median worsening proposal one time in five."""
    base = energy(SHAPE, parity, TARGET)
    worse = []
    probe = Anneal(shape=SHAPE, target=TARGET, parity=parity, rng=rng, temperature=0.0)
    for _ in range(samples):
        score = energy(SHAPE, probe.propose(), TARGET)
        if score > base:
            worse.append(score - base)
    if not worse:
        return 1.0
    worse.sort()
    return -worse[len(worse) // 2] / log(_ACCEPT_AT_START)


def even_starts(limit: int = 900) -> list[tuple[int, ...]]:
    """Return start points whose codes are even, by flipping one bit in rows 2 and 3.

    In the refined seed the first two block rows already generate even words and
    the last two do not. Flipping one coefficient in each of those rows moves
    them to even parity, and the product of the two choices gives the pool.
    """
    blocks, span = SHAPE.parity_blocks, SHAPE.block
    starts = []
    for first in range(blocks * span):
        for second in range(blocks * span):
            parity = list(SEED)
            parity[2 * blocks + first // span] ^= 1 << (first % span)
            parity[3 * blocks + second // span] ^= 1 << (second % span)
            starts.append(tuple(parity))
            if len(starts) >= limit:
                return starts
    return starts


@dataclass(slots=True)
class Worker:
    """One annealing worker with reheat and restart, over its own start pool."""

    anneal: Anneal
    start_pool: list[tuple[int, ...]]
    initial_temperature: float
    since_improvement: int = field(default=0, init=False)
    restarts: int = field(default=0, init=False)
    proposals: int = field(default=0, init=False)

    def step(self) -> bool:
        """Advance one proposal, adjusting temperature, and report a solution."""
        before = self.anneal.best
        solved = self.anneal.step()
        self.proposals += 1
        self.since_improvement = 0 if self.anneal.best < before else self.since_improvement + 1
        if self.proposals % _COOL_EVERY == 0:
            self.anneal.temperature *= _COOL
        if self.since_improvement >= _RESTART_AFTER:
            self._restart()
        elif self.since_improvement and self.since_improvement % _REHEAT_AFTER == 0:
            self.anneal.temperature = self.initial_temperature
        return solved

    def _restart(self) -> None:
        """Move to a fresh start, keeping the best result seen so far."""
        best, best_parity = self.anneal.best, self.anneal.best_parity
        start = self.anneal.rng.choice(self.start_pool)
        self.anneal.parity = start
        self.anneal.current = energy(SHAPE, start, TARGET)
        self.anneal.temperature = self.initial_temperature
        self.anneal.best, self.anneal.best_parity = best, best_parity
        self.since_improvement = 0
        self.restarts += 1


def certify(parity: tuple[int, ...]) -> tuple[int, bool]:
    """Return the minimum distance and whether it is a new table entry."""
    found = distance(SHAPE, parity)
    return found, found > PUBLISHED
