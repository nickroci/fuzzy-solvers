"""Accelerated independence-number kernel, with a pure-Python fallback.

The referee in :mod:`ramsey.graph` defines the answer and stays pure
Python: it runs once per accepted candidate, where clarity matters and speed
does not.  The *search* runs it tens of thousands of times per commissioned
program, and at 139 vertices the Python version manages about one graph per
second — too slow to search anything.

So the hot loop lives in a small dependency-free Rust crate under ``rust/``,
loaded here through ``ctypes``.  Two properties make that the right boundary:
``ctypes`` releases the interpreter lock across a foreign call, so a thread
pool calling in gets real parallelism rather than contending on the GIL; and
the crate implements the *same* branch-and-bound as the reference, so the two
agree on every input rather than merely correlating.

If the library is absent or refuses a shape, every call falls back to the
reference implementation.  Nothing here is load-bearing for correctness — it
is only load-bearing for whether a search finishes.
"""

from __future__ import annotations

import ctypes
from ctypes import c_uint32, c_uint64
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ramsey.graph import capped_independence_number as reference_capped_independence

if TYPE_CHECKING:
    from ramsey.graph import RamseyGraph

_WORD_BITS: Final = 64
_MAX_WORDS: Final = 4
_REFUSED: Final = 0xFFFFFFFF
_LIBRARY_NAME: Final = "libramsey_kernel.dylib"
_CRATE = Path(__file__).resolve().parents[1] / "rust" / "ramsey_kernel"


def _load() -> ctypes.CDLL | None:
    """Load the compiled kernel, or return ``None`` when it is not built."""
    for candidate in (
        _CRATE / "target" / "release" / _LIBRARY_NAME,
        _CRATE / "target" / "release" / "libramsey_kernel.so",
    ):
        if not candidate.exists():
            continue
        library = ctypes.CDLL(str(candidate))
        library.capped_independence.argtypes = [
            ctypes.POINTER(c_uint64),
            c_uint32,
            c_uint32,
            c_uint32,
        ]
        library.capped_independence.restype = c_uint32
        library.max_order.argtypes = []
        library.max_order.restype = c_uint32
        return library
    return None


_LIBRARY = _load()


def accelerated() -> bool:
    """Report whether the compiled kernel is in use."""
    return _LIBRARY is not None


def kernel_max_order() -> int:
    """Return the vertex capacity of the compiled kernel, or zero without it."""
    return 0 if _LIBRARY is None else int(_LIBRARY.max_order())


def capped_independence(graph: RamseyGraph, ceiling: int) -> int:
    """Return ``min(alpha, ceiling)``, through the compiled kernel when possible.

    The result is identical to
    :func:`ramsey.graph.capped_independence_number` by construction; the
    fallback is taken whenever the library is missing or declines the shape,
    so a caller never has to know which ran.
    """
    if ceiling <= 0:
        return 0
    words = (graph.order + _WORD_BITS - 1) // _WORD_BITS
    if _LIBRARY is None or words > _MAX_WORDS or graph.order == 0:
        return reference_capped_independence(graph, ceiling)
    packed = b"".join(mask.to_bytes(words * 8, "little") for mask in graph.adjacency)
    buffer = (c_uint64 * (graph.order * words)).from_buffer_copy(packed)
    result = int(
        _LIBRARY.capped_independence(
            ctypes.cast(buffer, ctypes.POINTER(c_uint64)),
            c_uint32(graph.order),
            c_uint32(words),
            c_uint32(ceiling),
        )
    )
    if result == _REFUSED:
        return reference_capped_independence(graph, ceiling)
    return result
