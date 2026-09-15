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

## The advantage escalates with difficulty

A second experiment ran both search strategies over 120 cells neither had seen
- 60 whose published value came from a dedicated search, 60 whose value was
inherited from a neighbour - at 60 million moves each. Neither found a record
at that budget, so the informative measure is how close each got, compared on
the *same* cell so difficulty is controlled for.

| Cells | Designed closer | Naive closer | Tied | Designed wins |
|---|---|---|---|---|
| Stayed far (>25 uncovered) | 55 | 39 | 4 | 59% |
| Got close (<=25 uncovered) | 12 | 3 | 7 | **80%** |
| Converting a near miss into a record | | | | **7x** |

Far from a solution almost any move helps, so a uniformly random swap is nearly
as good as a targeted one and the designed strategy wins barely more often than
chance. As the design approaches completion the set of useful moves collapses
to a handful, a random swap almost never picks one, and the gap widens. At the
limit - the last uncovered pair - it decides the outcome: 7 of 12 against 1 of
12 on cells known to be solvable, with five of six naive runs spending their
entire budget frozen exactly one pair short.

So a median comparison understates the difference, because it averages a regime
where the two are near-equivalent together with the regime that produces
records. The 12-against-3 bucket is small and should be read as suggestive; the
55-against-39 and the 7-against-1 ends are better powered.

## The targeting signal, measured prospectively

The same experiment tested the other claim, on a random sample drawn before
either method ran:

| Cell provenance | n | median uncovered reached |
|---|---|---|
| Value inherited from a neighbour | 120 arms | **112 - 127** |
| Value set by a dedicated search | 120 arms | **2,308 - 2,337** |

A twentyfold difference, on cells sampled at random and attacked blind. Whether
anyone had previously searched a cell predicts how close a search gets to
beating it far more strongly than which search you use.
