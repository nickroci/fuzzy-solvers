# Four new covering designs, and why they were there to be found

## The results

| Cell | Published | Ours | Lower bound | Pairs covered |
|---|---|---|---|---|
| C(56,13,2) | 25 | **24** | 22 | 1,540 |
| C(65,14,2) | 29 | **28** | 27 | 2,080 |
| C(82,19,2) | 25 | **24** | 22 | 3,321 |
| C(98,22,2) | 27 | **26** | 24 | 4,753 |

Each is a family of k-subsets of a v-set covering every pair. Each was verified
by exhaustive enumeration, twice, the second time by a checker written the
other way round so a shared bug could not pass both. The repository's own
current release is the baseline: La Jolla Coverings Repository v1.2, Zenodo,
2026-04-24. Each was then checked against coveringrepository.com, which tracks
improvements beyond that release, in a live browser session. All three still
read at the published value there.

## The finding that matters

Not "we searched better". The frontier has a region nobody is searching, and
it is identifiable from the repository's own metadata.

A covering design is a lottery wheel: buy these tickets, guarantee this match.
The parameter `t` is the guarantee. The repository's active contributors are
the lottery-wheeling community - the recent-upload feed is full of private
tools with names like Wheel Generator and Wheeling Systems Checker - and the
site that hosts their work charges a subscription to download or upload.

So demand concentrates where a wheel is sellable. A `t=5` wheel wins something
when five numbers come up. A `t=2` wheel guarantees you match two numbers,
which pays nothing. Same mathematical object, no customer.

Share of cells whose current value was set in 2020 or later:

| t | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| touched since 2020 | **7.5%** | 46.6% | 71.5% | 77.2% | 81.5% | 77.9% | 71.8% |

Ten times less attention at `t=2` than anywhere else. And across 2,839 cells
attacked, every record came from there: four at `t=2` from 1,021 attempts,
none from the 1,818 attempts at `t>=3`.

The second signal is provenance. The repository records *how* each value was
established, and a value inherited from a neighbour by a generic construction
means nobody ever searched that cell. Measured at published-minus-one across
the whole sweep: median 192 uncovered for inherited cells, 1,862 for cells
someone had actually searched. Two of the four records came from a pass aimed
at the inherited class.

C(65,14,2)'s entire recorded history is two lines, both restrictions of the
66-point design, seventeen years apart. It had never been searched at all.

## What the model contributed, and what it did not

**Did:** chose the problem class after two failed campaigns, on the argument
that verification cost rather than search difficulty was what killed them;
built and validated the instrument; found both targeting signals in the
metadata; and designed the search strategy.

That last one is worth a number. Textbook simulated annealing - uniform swaps,
Metropolis, geometric cooling, restarts - against the same kernel and the same
400 million move budget, six seeds on each of the two cells it was tested on:

| | C(56,13,2) | C(65,14,2) | total |
|---|---|---|---|
| Naive annealing | 1/6 | 0/6 | **1/12** |
| Designed strategy | 6/6 | 1/6 | **7/12** |

Seven times the yield for the same compute. The naive failure is specific:
five of six runs froze at *exactly one uncovered pair*, because a uniformly
random swap almost never touches the one pair still missing. Aiming three
quarters of moves at a pair that is actually missing, and repairing the block
that already holds most of it, is what crosses that last step.

**Did not:** beat that strategy by reasoning per-cell. Agent sessions given the
cell and nothing else find C(56,13,2) reliably - six of six, each with a
private board so none could see another's work - but they take minutes where
the fixed strategy takes seconds, and on C(65,14,2) they failed where it
succeeded. Their budget goes on group structure that a single division rules
out.

So the honest claim is about judgment, not search: **an LLM was good at
deciding where to look, and the looking is ordinary.** Given three campaigns
died on target selection, that is the harder half.

## The designs

### C(56,13,2) in 24 blocks (published 25)

Point lists on {0, ..., 55}. Take any two points; some row holds both.

```
 1  3  4 14 15 17 19 35 43 50 51 54 55
 4  9 23 27 31 33 35 36 38 41 47 48 52
 7  9 11 20 25 29 33 40 41 43 44 45 55
 2 12 13 24 27 31 34 38 40 44 50 52 54
 1  2  5  8 15 20 24 28 29 34 36 45 47
 0  4  5 12 13 19 20 21 26 29 30 35 45
 0  7 10 11 16 25 30 36 46 47 49 50 54
 2 10 16 19 21 23 24 26 43 46 48 49 55
 3 14 17 21 26 34 36 40 42 44 47 51 53
 2  4  6  7 11 22 24 25 35 37 39 42 53
 0  1  6 15 22 23 30 34 37 39 40 44 48
 0  8 19 27 28 30 31 38 42 43 52 53 55
 6 12 13 18 19 22 32 36 37 39 43 47 55
 6  8  9 21 22 26 28 33 37 39 41 50 54
 0  2  3  9 14 17 18 24 30 32 33 41 51
 5  6 10 20 29 37 38 39 45 46 49 51 52
 3  6 14 16 17 22 27 31 37 38 39 51 52
 3  7  8 11 12 13 14 17 23 25 28 48 51
 5 16 18 20 23 29 32 42 45 48 50 53 54
 3  5 10 14 17 20 22 27 29 31 45 46 49
 5  7  9 11 19 25 33 34 40 41 43 44 55
 1  9 10 12 13 15 16 33 41 42 46 49 53
 1  7 11 15 18 21 25 26 27 31 32 38 52
 4  8 10 16 18 28 32 34 35 40 44 46 49
```

### C(65,14,2) in 28 blocks (published 29)

Point lists on {0, ..., 64}. Take any two points; some row holds both.

```
 0  1 12 17 21 26 29 31 37 41 42 51 59 63
 0  2  5 18 22 32 34 41 43 50 53 57 58 63
 0  3  9 14 20 24 36 40 41 46 52 54 62 63
 0  4  6  7 16 28 33 35 41 44 56 60 61 63
 0  8 10 11 13 15 19 30 38 39 41 45 47 63
 0 22 23 25 27 32 34 41 43 48 49 55 63 64
 1  2 13 14 15 27 30 42 51 52 56 57 58 62
 1  3  5 13 15 23 25 28 30 42 50 51 56 64
 1  4  6  9 10 11 16 40 42 46 48 49 51 55
 1  7 18 19 22 24 33 42 44 45 47 51 53 54
 1  8 20 32 34 35 36 38 39 42 43 51 60 61
 2  3  4  6  8 16 21 22 31 37 38 39 57 58
 2  3 19 26 35 45 47 48 49 55 57 58 60 61
 2  5  7  9 12 23 29 33 40 44 46 50 57 58
 2 10 11 17 20 24 25 28 36 54 57 58 59 64
 3  7 10 11 14 17 32 33 34 43 44 52 59 62
 3 10 11 12 18 20 22 26 27 28 29 36 53 56
 4  5  6 16 17 19 20 23 27 36 45 47 50 59
 4  6 12 13 15 16 24 26 29 30 32 34 43 54
 4  6 14 16 18 21 23 25 31 37 52 53 62 64
 5  8 14 26 28 38 39 48 49 50 52 53 55 62
 5 10 11 21 23 24 27 31 35 37 50 54 60 61
 7  8  9 23 25 26 27 33 38 39 40 44 46 64
 7 13 15 20 21 30 31 33 36 37 44 48 49 55
 8 12 17 18 24 29 38 39 48 49 54 55 56 59
 9 13 15 17 18 22 30 35 40 46 53 59 60 61
 9 19 21 28 31 32 34 37 40 43 45 46 47 56
12 14 19 22 25 29 35 45 47 52 60 61 62 64
```

### C(82,19,2) in 24 blocks (published 25)

Point lists on {0, ..., 81}. Take any two points; some row holds both.

```
 0  1 17 24 31 39 41 45 46 48 50 59 60 62 65 69 71 78 79
 0  2  3  7 18 22 33 38 40 48 50 51 53 56 65 69 71 77 81
 0  4  6  9 14 19 23 30 35 44 48 50 55 61 65 69 70 71 76
 0  5 27 28 32 34 47 48 50 54 57 58 65 66 67 69 71 73 75
 0  8 10 11 12 13 14 21 29 36 37 42 43 48 50 65 69 71 72
 0 15 16 20 25 26 37 48 49 50 52 63 64 65 68 69 71 74 80
 1  2  4 13 16 20 32 35 40 41 43 51 54 57 61 63 70 72 78
 1  3  5  9 10 19 21 22 23 25 28 29 30 33 36 37 41 66 78
 1  6  7 11 12 15 26 27 38 41 42 44 55 58 64 73 76 78 81
 1  8 14 18 31 34 41 47 49 52 53 56 67 68 74 75 77 78 80
 2  5  6  8 12 28 40 44 49 51 55 59 60 62 66 68 74 76 80
 2  9 10 14 19 24 25 27 31 37 40 45 46 47 51 52 56 58 73
 2 11 15 17 21 23 26 29 30 34 36 39 40 42 51 64 67 75 79
 3  4  8 17 22 27 33 35 39 49 58 61 68 70 72 73 74 79 80
 3  6 12 13 16 20 22 24 33 34 43 44 45 46 55 63 67 75 76
 3 11 14 15 22 26 31 32 33 42 47 52 54 56 57 59 60 62 64
 4  5 11 15 18 24 26 28 35 42 45 46 53 61 64 66 70 72 77
 4  6 12 21 23 29 30 31 35 36 44 47 52 55 56 61 70 72 76
 4  7  9 10 19 25 34 35 37 38 59 60 61 62 67 70 72 75 81
 5  7 13 14 16 17 20 28 31 38 39 43 47 52 56 63 66 79 81
 6  9 10 12 17 18 19 25 32 37 39 44 53 54 55 57 76 77 79
 7  8 21 23 24 29 30 32 36 38 45 46 49 54 57 68 74 80 81
 8  9 10 11 13 15 16 19 20 25 26 42 43 49 63 64 68 74 80
13 16 18 20 21 23 27 29 30 36 43 53 58 59 60 62 63 73 77
```

### C(98,22,2) in 26 blocks (published 27)

Point lists on {0, ..., 97}. Take any two points; some row holds both.

```
 0  1  6  9 18 29 32 33 37 38 39 41 51 53 55 57 64 70 73 75 81 83
 0  2 13 18 22 23 24 25 30 34 39 50 52 63 72 77 78 81 86 90 91 95
 0  3  7  8 10 14 18 20 21 28 37 39 45 50 53 56 62 68 69 71 81 88
 0  4  5 16 18 19 35 36 39 40 47 48 54 58 61 76 80 81 87 92 93 96
 0  4 15 20 31 32 33 38 41 42 43 46 50 51 54 74 75 76 80 83 96 97
 0 11 12 17 18 26 27 39 44 49 59 60 65 66 67 79 81 82 84 85 89 94
 1  2  3 10 11 23 26 27 32 44 45 48 55 58 65 72 73 82 86 89 93 95
 1  4 17 19 21 30 32 34 49 52 54 55 62 63 73 76 78 80 84 88 92 96
 1  5 14 16 24 25 28 32 35 36 42 55 56 59 60 67 69 73 85 87 91 94
 1  7  8 12 13 20 22 32 37 40 42 47 53 55 61 66 68 71 73 77 79 90
 1 13 14 15 19 22 28 31 43 44 46 50 55 56 69 73 74 77 89 90 92 97
 2  4 20 23 37 42 44 50 53 54 59 60 67 72 76 80 85 86 89 94 95 96
 2  5 12 15 16 21 23 31 35 36 43 46 62 66 72 74 79 86 87 88 95 97
 2  6  9 14 17 23 28 29 40 47 49 50 56 57 61 64 69 70 72 84 86 95
 2  7  8 19 23 33 38 41 51 59 60 67 68 71 72 75 83 85 86 92 94 95
 3  4  6  9 10 12 19 24 25 29 45 54 57 64 66 70 76 79 80 91 92 96
 3  5 10 13 16 17 22 33 35 36 38 41 42 45 49 51 75 77 83 84 87 90
 3 10 15 30 31 34 40 43 45 46 47 52 59 60 61 63 67 74 78 85 94 97
 4  7  8 11 13 14 22 26 27 28 54 56 65 68 69 71 76 77 80 82 90 96
 5  6  7  8  9 16 29 30 34 35 36 44 52 57 63 64 68 70 71 78 87 89
 5 11 16 19 20 26 27 30 34 35 36 37 42 50 52 53 63 65 78 82 87 92
 6  9 11 15 18 20 26 27 29 31 39 42 43 46 57 64 65 70 74 81 82 97
 6  9 13 21 22 29 42 48 57 58 59 60 62 64 67 70 77 85 88 90 93 94
 7  8 15 17 20 24 25 31 37 43 46 48 49 53 58 68 71 74 84 91 93 97
11 21 24 25 26 27 33 38 40 41 44 47 51 61 62 65 75 82 83 88 89 91
12 14 28 30 33 34 38 41 48 50 51 52 56 58 63 66 69 75 78 79 83 93
```
