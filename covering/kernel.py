"""Accelerated annealing over covering designs, with a pure-Python fallback.

The referee in :mod:`covering.design` defines the answer and stays pure
Python: it runs once per candidate worth banking, where clarity matters.  The
annealer runs tens of millions of times per session, where clarity does not.

Two things make the foreign boundary worth it here.  ``ctypes`` releases the
interpreter lock across the call, so a process pool is not needed to get real
parallelism out of a fan-out.  And the kernel keeps coverage as a count per
requirement, so a swap costs the requirements that actually change rather than
a full rescan of the design -- measured at four orders of magnitude, which is
the difference between a search that finishes and one that does not.

The kernel is trusted only as far as it is checked.  :func:`uncovered` is
exactly the referee's own count, so the two can be compared on any input, and
an annealing result reports base blocks that the referee re-scores
independently.  A kernel that miscounts is caught by the object it returns,
not by a claim it makes.
"""

from __future__ import annotations

import ctypes
import random
from ctypes import c_double, c_uint32, c_uint64
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import TYPE_CHECKING, Final

from covering.design import rank_table
from covering.orbit import develop
from covering.orbit_search import uncovered_count

if TYPE_CHECKING:
    from covering.design import Parameters
    from covering.orbit import PointGroup

_REFUSED: Final = 0xFFFFFFFF
_CRATE = Path(__file__).resolve().parents[1] / "rust" / "covering_kernel"


def _load() -> ctypes.CDLL | None:
    """Load the compiled kernel, or return ``None`` when it is not built."""
    for name in ("libcovering_kernel.dylib", "libcovering_kernel.so"):
        candidate = _CRATE / "target" / "release" / name
        if not candidate.exists():
            continue
        library = ctypes.CDLL(str(candidate))
        library.covering_anneal.argtypes = [
            c_uint32,
            c_uint32,
            c_uint32,
            c_uint32,
            ctypes.POINTER(c_uint32),
            c_uint32,
            ctypes.POINTER(c_uint32),
            c_uint64,
            c_uint64,
            c_double,
            c_double,
            c_double,
            ctypes.POINTER(c_uint32),
            ctypes.POINTER(c_uint64),
        ]
        library.covering_anneal.restype = c_uint32
        library.covering_uncovered.argtypes = [
            c_uint32,
            c_uint32,
            c_uint32,
            c_uint32,
            ctypes.POINTER(c_uint32),
            c_uint32,
            ctypes.POINTER(c_uint32),
        ]
        library.covering_uncovered.restype = c_uint32
        library.covering_kernel_version.argtypes = []
        library.covering_kernel_version.restype = c_uint32
        return library
    return None


_LIBRARY = _load()


def accelerated() -> bool:
    """Report whether the compiled kernel is available."""
    return _LIBRARY is not None


def members(block: int, points: int) -> tuple[int, ...]:
    """Return the points of a block, ascending."""
    return tuple(point for point in range(points) if block >> point & 1)


def mask(points: tuple[int, ...]) -> int:
    """Return the bitmask of a point list."""
    total = 0
    for point in points:
        total |= 1 << point
    return total


@dataclass(frozen=True, slots=True)
class Schedule:
    """How long to anneal and how hard to hold on to a worse design.

    Temperatures are in units of *uncovered requirements*, because that is what
    the cost function counts: at ``start_temp`` a move that leaves one more
    requirement uncovered is accepted about a third of the time, and at
    ``end_temp`` essentially never.  ``guided`` is the share of moves aimed at
    a requirement that is actually missing rather than drawn at random.
    """

    steps: int
    seed: int = 0
    start_temp: float = 1.0
    end_temp: float = 0.02
    guided: float = 0.75

    def __post_init__(self) -> None:
        """Reject a schedule that describes no search."""
        if self.steps < 0:
            message = f"steps must be non-negative, got {self.steps}"
            raise ValueError(message)
        if self.start_temp <= 0 or self.end_temp <= 0:
            message = f"temperatures must be positive, got {self.start_temp} and {self.end_temp}"
            raise ValueError(message)
        if not 0.0 <= self.guided <= 1.0:
            message = f"guided must be a probability, got {self.guided}"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class AnnealResult:
    """The best design an annealing run reached, as base blocks."""

    uncovered: int
    base_blocks: tuple[int, ...]
    steps: int
    accelerated: bool
    moves: int = 0
    """Moves actually performed, which is below ``steps`` when the search succeeds."""


def _flatten(base_blocks: tuple[int, ...], parameters: Parameters) -> list[int]:
    """Return base blocks as a flat point array, rejecting a wrong-sized block."""
    flat: list[int] = []
    for block in base_blocks:
        points = members(block, parameters.points)
        if len(points) != parameters.block_size:
            message = f"base block has {len(points)} points, not {parameters.block_size}"
            raise ValueError(message)
        flat.extend(points)
    return flat


def _permutation_array(group: PointGroup) -> list[int]:
    """Return every group element as one flat image array."""
    flat: list[int] = []
    for element in group.elements():
        flat.extend(element)
    return flat


def uncovered(parameters: Parameters, group: PointGroup, base_blocks: tuple[int, ...]) -> int:
    """Return how many requirements the developed design leaves uncovered."""
    if _LIBRARY is None:
        return uncovered_count(parameters, develop(group, base_blocks), rank_table(parameters))
    flat = _flatten(base_blocks, parameters)
    perms = _permutation_array(group)
    base_buffer = (c_uint32 * len(flat))(*flat)
    perm_buffer = (c_uint32 * len(perms))(*perms)
    answer = _LIBRARY.covering_uncovered(
        parameters.points,
        parameters.block_size,
        parameters.strength,
        len(base_blocks),
        base_buffer,
        group.order,
        perm_buffer,
    )
    if answer == _REFUSED:
        return uncovered_count(parameters, develop(group, base_blocks), rank_table(parameters))
    return int(answer)


def anneal(
    parameters: Parameters,
    group: PointGroup,
    base_blocks: tuple[int, ...],
    schedule: Schedule,
) -> AnnealResult:
    """Anneal the base blocks, returning the best design reached."""
    flat = _flatten(base_blocks, parameters)
    if _LIBRARY is None:
        return _reference_anneal(parameters, group, base_blocks, schedule)
    perms = _permutation_array(group)
    base_buffer = (c_uint32 * len(flat))(*flat)
    perm_buffer = (c_uint32 * len(perms))(*perms)
    best_buffer = (c_uint32 * len(flat))()
    performed = c_uint64(0)
    answer = _LIBRARY.covering_anneal(
        parameters.points,
        parameters.block_size,
        parameters.strength,
        len(base_blocks),
        base_buffer,
        group.order,
        perm_buffer,
        schedule.steps,
        schedule.seed,
        schedule.start_temp,
        schedule.end_temp,
        schedule.guided,
        best_buffer,
        ctypes.byref(performed),
    )
    if answer == _REFUSED:
        return _reference_anneal(parameters, group, base_blocks, schedule)
    width = parameters.block_size
    found = tuple(
        mask(tuple(best_buffer[index * width : (index + 1) * width]))
        for index in range(len(base_blocks))
    )
    return AnnealResult(
        uncovered=int(answer),
        base_blocks=found,
        steps=schedule.steps,
        accelerated=True,
        moves=int(performed.value),
    )


def _reference_anneal(
    parameters: Parameters,
    group: PointGroup,
    base_blocks: tuple[int, ...],
    schedule: Schedule,
) -> AnnealResult:
    """Run the same Metropolis walk in Python, for when the kernel is absent.

    Deliberately the plain version: uniform swaps, a full rescore per move.  It
    exists so the surface works without a build step and so the accelerated
    path has something to be compared against, not so anyone searches with it.
    """
    rng = random.Random(schedule.seed)
    ranks = rank_table(parameters)
    current = list(base_blocks)
    score = uncovered_count(parameters, develop(group, tuple(current)), ranks)
    best, best_blocks = score, tuple(current)
    steps = schedule.steps
    ratio = (schedule.end_temp / schedule.start_temp) ** (1.0 / steps) if steps else 1.0
    temperature = schedule.start_temp
    performed = 0
    while performed < steps and best:
        performed += 1
        index = rng.randrange(len(current))
        block = current[index]
        inside = members(block, parameters.points)
        outside = [point for point in range(parameters.points) if not block >> point & 1]
        moved = (block | (1 << rng.choice(outside))) & ~(1 << rng.choice(inside))
        proposal = list(current)
        proposal[index] = moved
        trial = uncovered_count(parameters, develop(group, tuple(proposal)), ranks)
        delta = trial - score
        if delta <= 0 or rng.random() < exp(-delta / temperature):
            current, score = proposal, trial
            if score < best:
                best, best_blocks = score, tuple(current)
        temperature *= ratio
    return AnnealResult(
        uncovered=best,
        base_blocks=best_blocks,
        steps=steps,
        accelerated=False,
        moves=performed,
    )
