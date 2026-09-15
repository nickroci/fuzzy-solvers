"""Re-verify every banked covering design from scratch.

No dependencies, nothing imported from the search code. Run it with any
Python 3: `python verify.py`

For each recorded design it checks the blocks are the right size, distinct and
in range, then enumerates every t-subset of the point set and confirms each one
lies inside some block. Finally it checks the design really is smaller than the
published entry it claims to beat.
"""

from __future__ import annotations

import json
from itertools import combinations
from math import comb
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "covering" / "results"


def check(record: dict) -> tuple[bool, str]:
    """Return whether one banked design is a valid covering that beats its entry."""
    points, size, strength = record["v"], record["k"], record["t"]
    blocks = [sorted(block) for block in record["blocks"]]

    if len(blocks) != record["size"]:
        return False, f"claims {record['size']} blocks but lists {len(blocks)}"
    if any(len(b) != size or len(set(b)) != size for b in blocks):
        return False, f"not every block has {size} distinct points"
    if any(p < 0 or p >= points for b in blocks for p in b):
        return False, f"a point falls outside 0..{points - 1}"
    if len({tuple(b) for b in blocks}) != len(blocks):
        return False, "blocks are not distinct"

    covered: set[tuple[int, ...]] = set()
    for block in blocks:
        covered.update(combinations(block, strength))
    required = set(combinations(range(points), strength))
    if covered != required:
        return False, f"{len(required - covered)} of {len(required):,} subsets uncovered"

    if record["size"] >= record["published"]:
        return False, f"{record['size']} does not beat the published {record['published']}"
    if record["size"] < record["low_bd"]:
        return False, f"{record['size']} is below the published lower bound {record['low_bd']}"
    return True, f"all {comb(points, strength):,} {strength}-subsets covered"


def main() -> int:
    """Check every result file, returning a non-zero status if any fails."""
    failures = 0
    for path in sorted(RESULTS.glob("C*.json")):
        record = json.loads(path.read_text())
        if "v" not in record:
            continue
        ok, detail = check(record)
        name = f"C({record['v']},{record['k']},{record['t']})"
        if "agent" in path.stem:
            name += " [agent]"
        verdict = "OK  " if ok else "FAIL"
        print(
            f"{verdict} {name:>14}  {record['size']:>3} blocks "
            f"(published {record['published']}, lower bound {record['low_bd']})  {detail}"
        )
        failures += not ok
    print("\nall designs verified" if not failures else f"\n{failures} design(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
