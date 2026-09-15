# C(56,13,2) <= 24, found by the agent framework

The same repository entry beaten again, this time by a model-authored search
session rather than by a sweep aimed by hand. The design below shares **none**
of its 24 blocks with the one the mechanical sweep found, so it is an
independent discovery, not a replay.

The session was told only what was known before the first result: the cell, the
published 25, the lower bound 22, and how that 25 was set. It was not told that
24 exists.

## What the session did

| | |
|---|---|
| Budget | 400,000,000 moves; spent 190,000,000 |
| Tool calls | 24 |
| Turns | 37 |
| Output tokens | 15,222 |
| Wall clock | 262 s |

It called `check_feasible` -- which costs no budget -- on Z56, Z55+1,
AGL(1,56):2 and C7xC8, and was told each time that 24 blocks is not a sum of
the achievable orbit sizes, so no base blocks in those groups could ever
produce that count. It then authored its own groups of order 8, 4 and 2, and
confirmed each could express 24.

Then it searched, and read the results:

```
Order8Group   3 base blocks -> 24 blocks    76 uncovered
Order4Group   6 base blocks -> 24 blocks    92 uncovered
Order8Group   3 base blocks -> 24 blocks    52 uncovered
Order2Group  12 base blocks -> 24 blocks    38 uncovered
trivial      24 base blocks -> 24 blocks    10 uncovered
                                   resume    1 uncovered
                                   resume    0 uncovered
```

That is the whole argument. Each step down the symmetry ladder -- orbit size
8, then 4, then 2, then 1 -- covered more, so it kept going down. The
conclusion it reached by measurement is the one the arithmetic forces: 24
blocks of 13 points is 312 incidences over 56 points, which is 5.57 per point.
Not an integer, so the degrees cannot be constant, so no point-transitive group
can act on such a design at all. The surface reports that as
`v/gcd(v,k) = 56` failing to divide 24; the session found it by walking down
the ladder instead.

## The design

```
 5  6 21 22 25 29 32 34 39 43 48 49 54
 1  8 16 20 23 32 33 36 43 44 46 47 53
 0  6  7 10 13 17 20 22 25 27 46 51 55
 3 18 19 20 24 26 28 29 30 31 34 45 46
 0 15 16 18 27 31 39 42 44 50 51 54 55
 0  2  3  4 14 24 27 30 32 41 43 51 55
 1  3  7  9 11 13 24 30 33 36 39 53 54
 2  6  9 11 14 16 19 22 25 26 28 44 45
 4  8 10 17 19 23 26 28 39 41 45 47 54
 0  1 21 27 28 33 35 36 37 38 48 49 52
 5  9 10 11 17 18 31 32 35 37 38 43 52
 2 12 14 20 35 37 38 39 40 46 47 52 54
 0  1  5 19 26 27 28 33 36 45 51 53 55
 1  2 10 14 15 17 29 33 34 36 42 50 53
 3  6  8 15 22 23 24 25 30 38 42 47 50
 7 12 13 15 19 26 28 32 40 42 43 45 50
 3  6  8 15 22 23 24 25 35 37 42 50 52
19 21 26 35 37 38 45 48 49 51 52 53 55
 2  5  7  8 13 14 18 21 23 31 47 48 49
 4  7 13 16 29 30 34 35 37 38 41 44 52
 3  5 10 12 16 17 21 24 30 40 44 48 49
 0  8  9 11 12 23 27 29 34 40 47 51 55
 4  5  9 11 15 20 21 41 42 46 48 49 50
 1  4  6 12 18 22 25 31 33 36 40 41 53
```
