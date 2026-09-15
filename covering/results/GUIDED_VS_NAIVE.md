# LLM-guided solver against naive simulated annealing

Same kernel, same 400,000,000 move budget, same cells. Each agent session ran
with its own private board so none could see another's work, and none was told
that a better design exists.

| | LLM-guided solver | Naive annealing |
|---|---|---|
| C(56,13,2) to 24 blocks | **6/6** found | 1/6 found |
| C(65,14,2) to 28 blocks | **1/6** found | 0/6 found |
| **Total** | **7/12 = 58%** | 1/12 = 8% |

Roughly a sevenfold difference in how often a record-beating design is found
on identical compute.

## Best solution reached, not just hit rate

Uncovered pairs at the target block count, lower is better:

    C(56,13,2)   guided: [0, 0, 0, 0, 0, 0]
                 naive : [0, 1, 1, 1, 1, 1]

    C(65,14,2)   guided: [0, 2, 2, 4, 4, 13]
                 naive : [4, 4, 5, 5, 5, 6]

On C(56,13,2) every guided session reached a valid 24-block design; five of six
naive runs spent the whole budget and froze one pair short. On C(65,14,2) the
guided median is 3 against naive's
5, and naive never once
reached what the guided runs reached at their best.

## Why naive stalls

The failure is specific and repeatable. Once a design is one pair short, a
uniformly random point swap almost never touches the pair still missing, so the
search wanders indefinitely one step from the answer. Aiming most proposals at
a pair that is actually uncovered, and repairing the block that already holds
most of it, is what crosses that last step.

## What this comparison does and does not control

Both arms use the same accelerated kernel - incremental coverage rather than a
full rescore. That is itself one of this project's ideas, worth about three
orders of magnitude, and leaving it in both arms is what makes this a
comparison of search strategy rather than of implementation speed. Stripping it
would flatter the guided side considerably.

Neither arm chooses its own target. All four records in this repository came
from cells selected by reading the repository's provenance metadata, and that
selection is not measured here.
