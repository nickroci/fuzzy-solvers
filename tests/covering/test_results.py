"""The banked results are re-verified by the gate, every run.

A result that lives only in a report decays: someone edits a file, a block
changes, and nobody notices. These tests re-check the actual objects against
the referee, so a claim in this repository cannot quietly stop being true.
"""

from __future__ import annotations

import json
from itertools import combinations
from math import comb
from pathlib import Path

import pytest

from covering.design import Parameters, check_covering

RESULTS = Path(__file__).resolve().parents[2] / "covering" / "results"


def result_files() -> list[Path]:
    """Return every banked result recorded in the repository."""
    return sorted(RESULTS.glob("*.json"))


def test_there_is_at_least_one_banked_result() -> None:
    """A missing results directory would make every check below vacuous."""
    assert result_files()


@pytest.mark.parametrize("path", result_files(), ids=lambda p: p.stem)
def test_a_banked_covering_is_valid_and_beats_its_published_entry(path: Path) -> None:
    """Every block is the right size, every requirement is covered, and it wins."""
    record = json.loads(path.read_text())
    points, size, strength = record["v"], record["k"], record["t"]
    parameters = Parameters(points=points, block_size=size, strength=strength)
    blocks = [tuple(block) for block in record["blocks"]]

    assert len(blocks) == record["size"]
    assert len(set(blocks)) == len(blocks)
    for block in blocks:
        assert len(block) == size
        assert len(set(block)) == size
        assert all(0 <= point < points for point in block)

    masks = tuple(sum(1 << point for point in block) for block in blocks)
    verdict = check_covering(parameters, masks)
    assert verdict.accepted, verdict.render()

    assert record["size"] < record["published"]
    assert record["size"] >= record["low_bd"]


@pytest.mark.parametrize("path", result_files(), ids=lambda p: p.stem)
def test_a_banked_covering_survives_a_check_written_the_other_way(path: Path) -> None:
    """Mark what is covered and diff, rather than asking whether anything is missing.

    Deliberately not the referee's algorithm: a shared bug would pass both, and
    the whole value of a banked result is that it is true independently of the
    code that found it.
    """
    record = json.loads(path.read_text())
    points, strength = record["v"], record["t"]
    covered: set[tuple[int, ...]] = set()
    for block in record["blocks"]:
        covered.update(combinations(sorted(block), strength))
    required = set(combinations(range(points), strength))
    assert covered == required
    assert len(required) == comb(points, strength)


def prose_files() -> list[Path]:
    """Return the write-ups that quote a banked design in a fenced block."""
    return sorted(path for path in RESULTS.glob("*.md") if "```" in path.read_text())


@pytest.mark.parametrize("path", prose_files(), ids=lambda p: p.stem)
def test_blocks_quoted_in_prose_match_the_banked_record(path: Path) -> None:
    """A design written out in prose must be the design that was verified.

    Numbers get retyped, reformatted and reordered by hand; a write-up that
    quietly disagrees with the record it describes is worse than one that omits
    the blocks entirely, because it looks checkable and is not.
    """
    banked = {
        tuple(sorted(block))
        for record in result_files()
        for block in json.loads(record.read_text())["blocks"]
    }
    fenced = path.read_text().split("```")
    quoted: set[tuple[int, ...]] = set()
    for chunk in fenced[1::2]:
        for line in chunk.strip().splitlines():
            parts = line.split()
            if parts and all(part.lstrip("-").isdigit() for part in parts):
                quoted.add(tuple(sorted(int(part) for part in parts)))
    assert quoted, f"{path.name} has a fenced block but no block listing"
    assert quoted <= banked
