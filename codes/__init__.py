"""Search for new best-known binary linear codes.

The published tables at ``codetables.de`` bound the minimum distance of the
best linear ``[n, k]`` code, and thousands of cells are open.  Raising a lower
bound by one is a new table entry, so the frontier here is wide rather than a
handful of exact values — and unlike an independent-set search, verifying a
candidate costs a bounded ``2^k`` enumeration that rejects almost free.

``linear`` is the referee, ``quasicyclic`` the construction language,
``kernel`` the compiled enumeration, ``tables`` the published bounds and the
choice of which cell to attack, and ``search`` the commissioned programs a
model can point at them.
"""
