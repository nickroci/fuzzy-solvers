"""Annealing over systematic quasi-cyclic codes, on the low-weight spectrum.

Minimum distance is too coarse to optimise. It is one integer that moves
rarely, so a climber on it sees a flat landscape: nineteen million quasi-cyclic
candidates all landed exactly on the published bound and none above it. The
signal is in the shell *below* the target, which
:mod:`rust.energy_kernel` scores as ``sum of max(0, D - weight)^2``.

The code is held in systematic form ``G = [I_h | P(x)]`` over circulants of
block length ``p``. Two things follow. The identity blocks make the generator
full rank by construction, so the dimension never has to be computed or
checked. And the free parameters are exactly the parity polynomials, so a
proposal is a bit flip in ``P`` rather than a rebuild.

Refining a code to a smaller block length is what widens the search without
leaving it: a length-``2p`` block whose shift is squared splits into two
length-``p`` cycles, doubling the height and halving the block while spanning
the same code. The published [50,20,13] at height 2 and block 10 becomes height
4 and block 5, and its free coefficients go from 60 to 120.
"""

from __future__ import annotations

import ctypes
import math
from ctypes import c_uint32, c_uint64
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    import random

_CRATE = Path(__file__).resolve().parents[1] / "rust" / "energy_kernel"
_REFUSED: Final = 0xFFFFFFFFFFFFFFFF
_REDRAW_ODD: Final = 0.1
_REDRAW_EVEN: Final = 0.2
_TWO_FLIP: Final = 0.7


def _load() -> ctypes.CDLL:
    """Load the compiled energy kernel."""
    for name in ("libenergy_kernel.dylib", "libenergy_kernel.so"):
        candidate = _CRATE / "target" / "release" / name
        if candidate.exists():
            library = ctypes.CDLL(str(candidate))
            library.spectrum_energy.argtypes = [
                ctypes.POINTER(c_uint64),
                c_uint32,
                c_uint32,
                c_uint32,
                c_uint64,
            ]
            library.spectrum_energy.restype = c_uint64
            library.minimum_weight.argtypes = [ctypes.POINTER(c_uint64), c_uint32, c_uint32]
            library.minimum_weight.restype = c_uint32
            return library
    message = "the energy kernel is not built; run cargo build --release in rust/energy_kernel"
    raise RuntimeError(message)


_LIBRARY = _load()


@dataclass(frozen=True, slots=True)
class Shape:
    """A systematic quasi-cyclic shape: ``G = [I_h | P]`` over block-``p`` circulants."""

    height: int
    block: int
    parity_blocks: int

    @property
    def length(self) -> int:
        """Return the code length."""
        return (self.height + self.parity_blocks) * self.block

    @property
    def dimension(self) -> int:
        """Return the dimension, which the identity blocks guarantee."""
        return self.height * self.block

    @property
    def free_coefficients(self) -> int:
        """Return how many bits the search may move."""
        return self.height * self.parity_blocks * self.block

    def describe(self) -> str:
        """Render the shape in table notation."""
        return (
            f"[{self.length},{self.dimension}] systematic QC height {self.height}, "
            f"block {self.block}, {self.parity_blocks} parity blocks, "
            f"{self.free_coefficients} free coefficients"
        )


def rotate(polynomial: int, shift: int, block: int) -> int:
    """Return ``x^shift * polynomial`` modulo ``x^block - 1``."""
    shift %= block
    mask = (1 << block) - 1
    value = polynomial & mask
    return ((value << shift) | (value >> (block - shift))) & mask if shift else value


def generator_rows(shape: Shape, parity: tuple[int, ...]) -> tuple[int, ...]:
    """Expand ``[I_h | P]`` into generator rows, one per shift of each block row."""
    rows = []
    for row in range(shape.height):
        polynomials = parity[row * shape.parity_blocks : (row + 1) * shape.parity_blocks]
        for shift in range(shape.block):
            word = 1 << (row * shape.block + shift)
            for position, polynomial in enumerate(polynomials):
                word |= rotate(polynomial, shift, shape.block) << (
                    (shape.height + position) * shape.block
                )
            rows.append(word)
    return tuple(rows)


def _pack(rows: tuple[int, ...], length: int) -> tuple[Any, int]:
    words = (length + 63) // 64
    blob = b"".join(row.to_bytes(words * 8, "little") for row in rows)
    return (c_uint64 * (len(rows) * words)).from_buffer_copy(blob), words


def energy(shape: Shape, parity: tuple[int, ...], target: int, abort: int = _REFUSED) -> int:
    """Return the low-weight-spectrum energy, aborting once past ``abort``."""
    rows = generator_rows(shape, parity)
    buffer, words = _pack(rows, shape.length)
    return int(
        _LIBRARY.spectrum_energy(
            ctypes.cast(buffer, ctypes.POINTER(c_uint64)),
            c_uint32(len(rows)),
            c_uint32(words),
            c_uint32(target),
            c_uint64(abort),
        )
    )


def distance(shape: Shape, parity: tuple[int, ...]) -> int:
    """Return the minimum weight, for certification rather than search."""
    rows = generator_rows(shape, parity)
    buffer, words = _pack(rows, shape.length)
    return int(
        _LIBRARY.minimum_weight(
            ctypes.cast(buffer, ctypes.POINTER(c_uint64)),
            c_uint32(len(rows)),
            c_uint32(words),
        )
    )


@dataclass(slots=True)
class Anneal:
    """One annealing worker over the parity coefficients of a shape."""

    shape: Shape
    target: int
    parity: tuple[int, ...]
    rng: random.Random
    even_only: bool = False
    temperature: float = 0.0
    current: int = field(default=0, init=False)
    best: int = field(default=0, init=False)
    best_parity: tuple[int, ...] = field(default=(), init=False)
    scored: int = field(default=0, init=False)
    aborted: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Score the starting point and adopt it as the incumbent."""
        self.current = energy(self.shape, self.parity, self.target)
        self.best, self.best_parity = self.current, self.parity

    def propose(self) -> tuple[int, ...]:
        """Return a neighbour of the incumbent under this worker's operator mix.

        An even worker only moves coefficients in pairs, which preserves the
        parity of each block row and so keeps every codeword even. Restricting
        most workers that way concentrates the budget on a sublattice where the
        odd-weight words - which can never be the minimum in an even code -
        cannot contribute to the energy at all.
        """
        blocks, span = self.shape.parity_blocks, self.shape.block
        row = self.rng.randrange(self.shape.height)
        draw = self.rng.random()
        proposal = list(self.parity)
        if draw < (_REDRAW_EVEN if self.even_only else _REDRAW_ODD):
            index = row * blocks + self.rng.randrange(blocks)
            replacement = self.rng.getrandbits(span)
            if self.even_only and replacement.bit_count() % 2 != proposal[index].bit_count() % 2:
                replacement ^= 1
            proposal[index] = replacement
            return tuple(proposal)
        flips = 2 if (self.even_only or draw > _TWO_FLIP) else 1
        for _ in range(flips):
            index = row * blocks + self.rng.randrange(blocks)
            proposal[index] ^= 1 << self.rng.randrange(span)
        return tuple(proposal)

    def step(self) -> bool:
        """Take one Metropolis step, returning whether the target was reached.

        The abort threshold is the energy at which the Metropolis test would
        reject regardless, so a hopeless proposal stops early without changing
        the acceptance distribution at all.
        """
        proposal = self.propose()
        draw = self.rng.random()
        ceiling = (
            self.current - self.temperature * math.log(draw)
            if self.temperature > 0
            else float(self.current)
        )
        abort = max(int(ceiling), 0)
        score = energy(self.shape, proposal, self.target, abort)
        if score > abort:
            self.aborted += 1
            return False
        self.scored += 1
        if score <= abort:
            self.parity, self.current = proposal, score
        if score < self.best:
            self.best, self.best_parity = score, proposal
        return self.best == 0
