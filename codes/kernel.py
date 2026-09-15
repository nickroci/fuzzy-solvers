"""Accelerated minimum-distance kernel, with a pure-Python fallback.

The referee in :mod:`codes.linear` defines the answer and stays pure
Python: it runs once per accepted candidate, where clarity matters.  The search
runs it thousands of times per commissioned program over ``2^k`` codewords
each, where clarity does not.

So the enumeration lives in a dependency-free Rust crate under ``rust/``,
loaded through ``ctypes``.  As with the Ramsey kernel, two properties make that
the right boundary: ``ctypes`` releases the interpreter lock across a foreign
call, so a thread pool gets real parallelism; and the crate walks the same
Gray-code enumeration as the reference, so the two agree on every input rather
than merely correlating.

Any missing library or unsupported shape falls back to the reference.
"""

from __future__ import annotations

import ctypes
from ctypes import c_uint32, c_uint64
from pathlib import Path
from typing import TYPE_CHECKING, Final

from codes.linear import minimum_distance as reference_minimum_distance
from codes.linear import reduced_basis

if TYPE_CHECKING:
    from codes.linear import LinearCode

_WORD_BITS: Final = 64
_MAX_WORDS: Final = 4
_REFUSED: Final = 0xFFFFFFFF
_CRATE = Path(__file__).resolve().parents[1] / "rust" / "code_kernel"


def _load() -> ctypes.CDLL | None:
    """Load the compiled kernel, or return ``None`` when it is not built."""
    for name in ("libcode_kernel.dylib", "libcode_kernel.so"):
        candidate = _CRATE / "target" / "release" / name
        if not candidate.exists():
            continue
        library = ctypes.CDLL(str(candidate))
        library.minimum_distance.argtypes = [
            ctypes.POINTER(c_uint64),
            c_uint32,
            c_uint32,
            c_uint32,
        ]
        library.minimum_distance.restype = c_uint32
        library.max_rank.argtypes = []
        library.max_rank.restype = c_uint32
        return library
    return None


_LIBRARY = _load()


def accelerated() -> bool:
    """Report whether the compiled kernel is in use."""
    return _LIBRARY is not None


def kernel_max_rank() -> int:
    """Return the largest rank the compiled kernel enumerates, or zero."""
    return 0 if _LIBRARY is None else int(_LIBRARY.max_rank())


def minimum_distance(code: LinearCode, *, floor: int = 0) -> int:
    """Return the minimum non-zero codeword weight, through the kernel when possible.

    Identical by construction to :func:`codes.linear.minimum_distance`;
    the fallback is taken whenever the library is missing or declines a shape,
    so a caller never has to know which ran.
    """
    basis = reduced_basis(code.rows)
    if not basis:
        return 0
    words = (code.length + _WORD_BITS - 1) // _WORD_BITS
    if _LIBRARY is None or words > _MAX_WORDS or len(basis) > kernel_max_rank():
        return reference_minimum_distance(code, floor=floor)
    packed = b"".join(row.to_bytes(words * 8, "little") for row in basis)
    buffer = (c_uint64 * (len(basis) * words)).from_buffer_copy(packed)
    result = int(
        _LIBRARY.minimum_distance(
            ctypes.cast(buffer, ctypes.POINTER(c_uint64)),
            c_uint32(len(basis)),
            c_uint32(words),
            c_uint32(max(floor, 0)),
        )
    )
    if result == _REFUSED:
        return reference_minimum_distance(code, floor=floor)
    return result
