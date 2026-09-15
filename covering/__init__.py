"""Search for covering designs that beat a published block count.

A ``(v, k, t)``-covering is a family of ``k``-subsets covering every
``t``-subset of a ``v``-set, and the repositories record the smallest anyone
has built.  ``design`` is the exact referee, ``search`` the construction and
repair engine, and ``anneal`` the fixed-block-count method the literature
actually uses.

Nothing attempts a published count until the engine has independently reached
it.  An engine that cannot find what is known to be findable has not earned a
frontier run - the discipline both earlier campaigns lacked.
"""
