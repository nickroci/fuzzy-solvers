# Experiments with fuzzy / LLM-driven solvers

Can a language model, given a search kernel and a published bound, find
mathematics nobody has found before? This repository is the attempt, including
the two campaigns that failed first.

## Headline: LLM-guided vs naive search

Same kernel, same 400,000,000 move budget, same cells. Each LLM session ran
with its own private board, so no session could see another's work, and none
was told a better design existed.

| | LLM-guided solver | Naive simulated annealing |
|---|---|---|
| C(56,13,2) to 24 blocks | **6 / 6 found** | 1 / 6 found |
| C(65,14,2) to 28 blocks | **1 / 6 found** | 0 / 6 found |
| **Total** | **7 / 12 — 58%** | 1 / 12 — 8% |

Best result reached, in uncovered pairs, lower is better:

```
C(56,13,2)   guided: [0, 0, 0, 0, 0, 0]      every session found a valid design
             naive : [0, 1, 1, 1, 1, 1]      five of six froze one pair short

C(65,14,2)   guided: [0, 2, 2, 4, 4, 13]
             naive : [4, 4, 5, 5, 5, 6]      never reached the guided best
```

On the harder cell, naive's *best* result equals the guided *worst*.

The naive failure is specific and repeatable: once a design is one pair short,
a uniformly random point swap almost never touches the pair still missing, so
the search wanders indefinitely one step from the answer. Aiming most proposals
at a pair that is actually uncovered, and repairing the block that already
holds most of it, is what crosses that last step.

## Four new covering designs

A `(v, k, t)`-covering design is a family of `k`-subsets of a `v`-set such that
every `t`-subset lies inside at least one of them. `C(v,k,t)` is the smallest
such family known, and the [La Jolla Covering
Repository](https://dmgordon.org/covering-designs/) has tracked the best known
values for about thirty years.

| Cell | Published | **This work** | Lower bound | Pairs covered |
|---|---|---|---|---|
| C(56,13,2) | 25 | **24** | 22 | 1,540 |
| C(65,14,2) | 29 | **28** | 27 | 2,080 |
| C(82,19,2) | 25 | **24** | 22 | 3,321 |
| C(98,22,2) | 27 | **26** | 24 | 4,753 |

Verify them yourself — no dependencies, nothing imported from the search code:

```
python verify.py
```

It enumerates every pair of every point set and checks each lies in some block.

Baseline is the current release: La Jolla Coverings Repository v1.2, Zenodo,
2026-04-24. Each cell was also checked against
[coveringrepository.com](https://www.coveringrepository.com), which tracks
improvements beyond that release, in a live browser session. All four still
read at the published value there.

## Where to look, and being able to convert

Two things were needed, and neither was sufficient alone.

### Knowing where to look

A covering design is a lottery wheel: buy these tickets, guarantee this match.
The parameter `t` is the guarantee. The repository's active contributors are
largely the lottery-wheeling community, and the site hosting their submissions
charges a subscription, so effort concentrates where a wheel is worth selling.
A `t=5` wheel wins something when five numbers come up. A `t=2` wheel guarantees
you match two numbers, which pays nothing. Same mathematical object, very
different demand.

Share of cells whose current value was set in 2020 or later:

| t | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| improved since 2020 | **7.5%** | 46.6% | 71.5% | 77.2% | 81.5% | 77.9% | 71.8% |

Ten times less *recorded improvement* at `t=2`. All four records came from
there — none from the 1,818 cells attacked at `t>=3`.

A second signal sharpens it: the repository records *how* each value was
established, and a value inherited from a neighbour by a generic construction
is one that has no recorded direct search behind it. Measured at
published-minus-one across the whole sweep, median 192 uncovered for inherited
cells against 1,862 for cells with a recorded search. Three of the four records
came from a pass aimed at the inherited class.

**What this evidence does not show.** The repository records successes, not
attempts. A cell with no recorded improvement is not a cell nobody tried —
failures leave no trace anywhere in this data. These parameters may well have
been attacked and survived it; there is no way to tell from here.

### Being able to convert

Knowing where to look was necessary and nowhere near sufficient.

- **2,948 cells attacked, 4 records. A hit rate of 0.14%.**
- **120 arms on fresh inherited-provenance cells, 60 million moves each: zero
  records.** Aimed squarely at the right class, nothing falls out on its own.
- Naive annealing aimed *directly* at a cell already known to be solvable finds
  it **1 time in 6**, and spends the other five stuck one pair short.

So the records needed both halves: a region worth searching, and a search good
enough to convert one. The kernel had to run three orders of magnitude faster
than the straightforward version before any of this was affordable, and the
move rule had to be right before the last uncovered pair could be closed at
all.

## What failed first

Two earlier campaigns produced nothing, and they are in here because the
reasons matter more than the successes.

**Ramsey lower bounds** (`ramsey/`) — about 1.1 million evaluations, zero
witnesses. Verification is an independent-set computation per candidate, which
is NP-hard; the search ran at 33 evaluations a second. The target was
unaffordable, not merely hard.

**Binary linear codes** (`codes/`) — about 137 million evaluations, zero new
entries. The error was reading "the published bounds disagree" as "this cell is
beatable". An open cell whose gap is one may simply mean the published value is
already optimal. That mistake is what made provenance, rather than gap size,
the targeting signal for the covering work.

## Layout

```
covering/     the work that produced the records
  design.py     exact referee: what a covering is, checked from blocks alone
  orbit.py      group actions and orbit development
  kernel.py     ctypes binding to the Rust annealer, with a Python fallback
  tools.py      the model-facing surface
  session.py    a model-driven search over that surface
  campaign.py   agent arms against matched naive and engineered controls
  results/      the four designs, with provenance and verification notes
ramsey/       the first failed campaign
codes/        the second
rust/         dependency-free kernels loaded through ctypes
tests/        482 tests; the gate re-verifies every banked record on each run
```

The covering kernel keeps coverage as a count per requirement, so swapping one
point of one block touches only the requirements that actually change rather
than rescoring the design. Measured at 639,000 moves a second against 445 for
the straightforward version — a factor of 1,436, and the difference between a
search that finishes and one that does not.

## Running it

```
cd rust/covering_kernel && cargo build --release && cd ../..
pip install -e ".[dev]"
python -m pytest tests -q
python verify.py
```

The kernel is optional; without it the Python fallback is used and everything
still runs, slowly.

## Honest limits

- The four records are all at `t=2`, and the open cells there are now nearly
  all attacked: 750 of 763. Further results need another region worth aiming at.
- "Less recorded improvement" is not "nobody tried". The repository logs
  successes, so unsuccessful attempts by others are invisible here and the
  targeting signal should be read as correlational.
- The LLM-guided solver beats naive annealing by about sevenfold, but it does
  not beat a well-designed fixed strategy — per-cell deliberation costs more
  than it returns once a good move rule exists. What the model contributed was
  choosing where to look, and designing the move rule, not deliberating per
  instance.
- Both arms of the comparison share the accelerated kernel. That is deliberate:
  it makes the comparison a test of search strategy rather than of
  implementation speed.

## Licence

MIT. The covering repository data it is measured against is CC-BY-4.0, by
Daniel M. Gordon.
