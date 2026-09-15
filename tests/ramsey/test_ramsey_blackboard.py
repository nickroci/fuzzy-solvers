from __future__ import annotations

import concurrent.futures as cf
from typing import TYPE_CHECKING

from ramsey.blackboard import DOMAIN_BRIEFING, Attempt, Blackboard

if TYPE_CHECKING:
    from pathlib import Path


def attempt(target: str = "R(3,20)>111", best: int = 21, evals: int = 9000) -> Attempt:
    return Attempt(
        target=target,
        family="cayley",
        detail="Z3 x Z37, degree 19",
        best_independence=best,
        independence_cap=19,
        evaluations=evals,
    )


def test_an_attempt_reports_its_gap() -> None:
    assert attempt(best=21).gap == 2
    assert "gap +2" in attempt(best=21).render()


def test_recording_and_reading_round_trip(tmp_path: Path) -> None:
    board = Blackboard(path=tmp_path / "board.jsonl")
    board.record(attempt())
    assert board.attempts() == (attempt(),)


def test_an_absent_board_is_empty(tmp_path: Path) -> None:
    assert Blackboard(path=tmp_path / "missing.jsonl").attempts() == ()


def test_attempts_filter_by_target(tmp_path: Path) -> None:
    board = Blackboard(path=tmp_path / "board.jsonl")
    board.record(attempt(target="R(3,20)>111"))
    board.record(attempt(target="R(3,18)>99"))
    assert len(board.attempts("R(3,20)>111")) == 1
    assert len(board.attempts()) == 2


def test_a_torn_line_is_skipped_rather_than_poisoning_the_history(tmp_path: Path) -> None:
    """One bad append must not cost every reader the whole history."""
    board = Blackboard(path=tmp_path / "board.jsonl")
    board.record(attempt())
    with board.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
        handle.write('{"target": "x"}\n')
    board.record(attempt(best=20))
    assert len(board.attempts()) == 2


def test_render_orders_by_gap_and_names_the_best(tmp_path: Path) -> None:
    board = Blackboard(path=tmp_path / "board.jsonl")
    board.record(attempt(best=23))
    board.record(attempt(best=20))
    rendered = board.render("R(3,20)>111")
    assert rendered.index("alpha=20") < rendered.index("alpha=23")
    assert "Best reached anywhere: alpha=20" in rendered


def test_render_reports_an_empty_board(tmp_path: Path) -> None:
    board = Blackboard(path=tmp_path / "board.jsonl")
    assert "No other arm" in board.render("R(3,20)>111")


def test_render_truncates_a_long_history(tmp_path: Path) -> None:
    board = Blackboard(path=tmp_path / "board.jsonl")
    for index in range(45):
        board.record(attempt(best=20 + index % 5, evals=index + 1))
    rendered = board.render("R(3,20)>111", limit=10)
    assert "and 35 more" in rendered


def test_appending_survives_concurrent_writers(tmp_path: Path) -> None:
    """Separate processes share this file, so writes must interleave whole lines."""
    board = Blackboard(path=tmp_path / "board.jsonl")
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: board.record(attempt(evals=i + 1)), range(64)))
    assert len(board.attempts()) == 64


def test_the_briefing_states_the_facts_arms_should_not_rediscover() -> None:
    assert "Harborth and Krause" in DOMAIN_BRIEFING
    assert "R(3,6) > 17" in DOMAIN_BRIEFING
    assert "pairwise coprime" in DOMAIN_BRIEFING
    assert "base size 1 is exactly a Cayley graph" in DOMAIN_BRIEFING
