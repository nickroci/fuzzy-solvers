# Reproducibility of the agent result

The question this answers: was C(56,13,2) <= 24 a lucky session, or something
the framework does?

Each session below ran with **its own private, empty board**, so no arm could
read another's work. Each was told only the cell, the published 25, the lower
bound 22, and how that 25 was set. None was told that 24 exists, which group to
use, or how many blocks to develop.

| Session | Banked a 24-block covering | Moves spent | Spending calls | Wall clock |
|---|---|---|---|---|
| 1 | yes | 134,000,000 | 11 | 309 s |
| 2 | yes | 207,000,000 | 19 | 388 s |
| 3 | yes | 104,000,000 | 10 | 461 s |
| 4 | yes | 49,000,000 | 10 | 378 s |
| 6 | yes | 129,000,000 | 9 | 277 s |

**5 of 5** sessions found a new-record design, on a budget of
400,000,000 moves each. The fastest spent 49 million of them.

## The designs are not the same design

Every design any session produced, plus the one the mechanical sweep found,
was checked by brute force over all 1,540 pairs. All are valid. And no two of
them share **a single block**:

    maximum blocks in common between any two designs: 0 of 24

That matters more than the count. It says 24-block coverings of this cell are
not one rare object that a search stumbles into, but a populous solution space
that the published 25 simply failed to reach for eighteen years.

## What had to be fixed first

The framework did not work at first, and none of the reasons were the model.
Each was found by reading a trace of what a live session did:

1. `search` offered one restart per tool call, where success is about one
   attempt in twelve. With a cap on calls, a session could never buy enough.
2. A session could spend its whole budget inside a single anneal, learning
   nothing and leaving itself no decisions.
3. Perfect coverings of the wrong size were reported as "uncovered: 0" with no
   warning, so sessions banked satisfaction instead of progress.
4. The incumbent was ranked by shortfall alone, so a design that could never be
   an entry displaced one that could - and `resume` then continued from it.
5. `check_feasible` was advertised as costing nothing while silently consuming
   the tool-call allowance, the resource that actually ran out. One session
   spent 16 of 30 calls on it and finished with 343 of 400 million moves unspent.

The shared board had two of its own: it recorded the incumbent's statistics
against whatever group an arm had just named, and it ranked history by
shortfall, so it opened by recommending 72-block designs for a cell needing 24.

## The control: does the guidance earn its cost?

It does not, on this cell. The same mechanical portfolio the agents are
measured against - random restarts and solve-then-delete-a-block repair, no
model involved - was run six times at the same budget:

| | found a 24-block covering | wall clock |
|---|---|---|
| Agent sessions (private board) | 6 of 6 | 277 - 1074 s |
| Mechanical control | 6 of 6 | **2 - 88 s** |

Both find it every time. The control is an order of magnitude faster.

A head-to-head at a smaller budget was worse still: across five cells at 150
million moves each, neither side improved anything, and on this cell the
control reached 1 uncovered where the agent reached 8. The board shows why -
the agent spent most of its budget on group-structured searches at orbit sizes
24, 12 and 8, reaching 92, 80 and 68 uncovered, and only tried the free design
on its last call.

That exploration is doomed here by arithmetic the agent could have derived:
24 blocks of 13 points is 312 incidences over 56 points, or 5.57 each, so the
degrees cannot be constant and no point-transitive group can act on such a
design at all. The surface reports this through check_feasible; the sessions
that succeeded found it by walking down the symmetry ladder instead, and the
ones that failed ran out of budget partway down.

**So what the model contributed here was not search power.** It was target
selection: which of 9,482 published cells to attack, from the provenance
recorded against each entry. Once the cell is chosen, ordinary annealing
solves it in seconds. That is a real contribution and a narrower one than
"the model found the design".
