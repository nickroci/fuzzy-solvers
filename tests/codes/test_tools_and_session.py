from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
from typing import TYPE_CHECKING

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage

from codes.board import BRIEFING, Attempt, Board
from codes.session import CodeSession, Router, Settings
from codes.tables import Cell
from codes.tools import BudgetExhaustedError, CodeTools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# [24,8]: an easy cell, so a short search really can clear the bar in a test.
EASY = Cell(length=24, dimension=8, lower=4, upper=12)
HARD = Cell(length=60, dimension=20, lower=17, upper=20)


def tools(cell: Cell = EASY, budget: int = 20_000, board: Board | None = None) -> CodeTools:
    return CodeTools(cell=cell, budget=budget, board=board)


def test_the_target_states_what_a_new_entry_needs() -> None:
    text = tools().describe_target()
    assert "[24,8]" in text
    assert "d >= 5" in text
    assert "3 generator" in text


def test_the_cell_key_identifies_the_entry() -> None:
    assert tools().cell_key == "[24,8]"


def test_a_commissioned_search_spends_and_reports() -> None:
    surface = tools()
    outcome = surface.commission_search(3, restarts=200, steps=30, seed=1)
    assert outcome.evaluations > 0
    assert surface.used == outcome.evaluations
    assert surface.best_distance >= outcome.best_distance or outcome.best_distance == 0


def test_a_commissioned_search_that_clears_the_bound_banks_a_verified_code() -> None:
    surface = tools()
    surface.commission_search(3, restarts=500, steps=60, seed=2)
    banked = surface.banked
    assert banked is not None
    assert banked.distance >= EASY.target
    assert banked.dimension == 8
    assert "beats the published" in banked.render()


def test_a_mismatched_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="must equal the length"):
        tools().commission_search(5, restarts=10, steps=2)


def test_one_search_cannot_consume_the_whole_budget() -> None:
    surface = tools(cell=HARD, budget=1000)
    surface.commission_search(3, restarts=100_000, steps=500, seed=4)
    assert surface.remaining > 0


def test_an_explicit_evaluation_reports_the_measurement() -> None:
    surface = tools(cell=HARD)
    text = surface.evaluate((1, 0b101, 0b1011))
    assert "[60," in text or "dimension is" in text
    assert surface.used == 1


def test_an_out_of_range_polynomial_is_rejected() -> None:
    assert "degree below" in tools().evaluate((1 << 30, 1, 1))


def test_a_wrong_number_of_polynomials_is_rejected() -> None:
    assert "not 24" in tools().evaluate((1, 1))


def test_a_rank_deficient_construction_is_explained() -> None:
    """All-ones generators share x^k - 1's factors, so the dimension collapses."""
    text = tools().evaluate((0b11111111, 0b11111111, 0b11111111))
    assert "dimension is" in text
    assert "non-trivial factor" in text


def test_the_evaluation_budget_is_enforced() -> None:
    surface = tools(budget=1)
    surface.evaluate((1, 1, 1))
    with pytest.raises(BudgetExhaustedError, match="budget of 1"):
        surface.evaluate((1, 1, 1))


def test_status_reports_budget_and_best() -> None:
    surface = tools()
    assert "banked: none" in surface.status()


def test_attempts_are_published_to_the_board(tmp_path: Path) -> None:
    board = Board(path=tmp_path / "b.jsonl")
    surface = tools(board=board)
    surface.commission_search(3, restarts=50, steps=10, seed=7)
    assert len(board.attempts("[24,8]")) == 1
    assert "attempt(s) on this cell" in surface.read_board()


def test_without_a_board_the_history_says_so() -> None:
    assert "No shared history" in tools().read_board()


def test_the_board_round_trips_and_orders_by_shortfall(tmp_path: Path) -> None:
    board = Board(path=tmp_path / "b.jsonl")
    board.record(
        Attempt(cell="[60,20]", detail="a", best_distance=16, target_distance=18, evaluations=10)
    )
    board.record(
        Attempt(cell="[60,20]", detail="b", best_distance=17, target_distance=18, evaluations=10)
    )
    rendered = board.render("[60,20]")
    assert rendered.index("d=17") < rendered.index("d=16")
    assert "Best reached anywhere: d=17" in rendered


def test_a_torn_board_line_is_skipped(tmp_path: Path) -> None:
    board = Board(path=tmp_path / "b.jsonl")
    board.record(Attempt(cell="c", detail="d", best_distance=1, target_distance=2, evaluations=1))
    with board.path.open("a") as handle:
        handle.write("{not json\n")
    assert len(board.attempts()) == 1


def test_an_absent_board_is_empty(tmp_path: Path) -> None:
    assert Board(path=tmp_path / "none.jsonl").attempts() == ()
    assert "No other arm" in Board(path=tmp_path / "none.jsonl").render("x")


def test_the_briefing_states_the_facts_arms_should_not_rediscover() -> None:
    assert "LOWER bound by one is a new table" in BRIEFING
    assert "cyclotomic cosets" in BRIEFING
    assert "reaches the PUBLISHED distance" in BRIEFING


# --- router and session ---


def router(budget: int = 20_000, max_calls: int = 20) -> Router:
    return Router(tools=tools(budget=budget), max_tool_calls=max_calls)


def test_router_describes_and_reads_history() -> None:
    bus = router()
    assert "[24,8]" in bus.handle("describe_target", {})[0]
    assert "No shared history" in bus.handle("read_board", {})[0]
    assert "evaluated" in bus.handle("status", {})[0]


def test_router_commissions_a_search() -> None:
    content, ok = router().handle(
        "commission_search", {"index": 3, "restarts": 100, "steps": 20, "seed": 1}
    )
    assert ok
    assert "codes evaluated" in content


def test_router_evaluates_explicit_polynomials() -> None:
    content, ok = router().handle("evaluate", {"polynomials": [1, 5, 11]})
    assert ok or "rejected" in content


def test_router_reports_an_unknown_tool() -> None:
    content, ok = router().handle("nope", {})
    assert not ok
    assert "unknown tool" in content


def test_router_reports_a_malformed_payload() -> None:
    content, ok = router().handle("commission_search", {"index": 3})
    assert not ok
    assert "KeyError" in content


def test_router_enforces_its_call_cap() -> None:
    bus = router(max_calls=1)
    bus.handle("status", {})
    content, ok = bus.handle("status", {})
    assert not ok
    assert "tool call budget exhausted" in content


def _result_message(tokens: int = 11) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        stop_reason="end_turn",
        usage={"output_tokens": tokens},
    )


def test_session_rejects_an_empty_instruction() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CodeSession(cell=EASY).run("  ")


def test_session_reports_an_incomplete_run() -> None:
    async def fake(**_: object) -> AsyncIterator[object]:
        yield AssistantMessage(content=[], model="m")
        yield _result_message()

    session = CodeSession(cell=EASY, settings=Settings(budget=100), query_fn=fake)
    result = session.run("find one")
    assert result.status == "incomplete"
    assert not result.improved
    assert result.turns == 1
    assert "status=incomplete" in result.render()


def test_session_reports_a_banked_code() -> None:
    session = CodeSession(cell=EASY, settings=Settings(budget=20_000))

    async def fake(**_: object) -> AsyncIterator[object]:
        session.tools.commission_search(3, restarts=500, steps=60, seed=2)
        yield _result_message()

    session._query = fake  # noqa: SLF001
    result = session.run("find one")
    assert result.improved
    assert result.status == "banked"
    assert "beats the published" in result.render()


def test_session_reports_budget_exhaustion() -> None:
    session = CodeSession(cell=EASY, settings=Settings(max_tool_calls=1))

    async def fake(**_: object) -> AsyncIterator[object]:
        session._router.handle("status", {})  # noqa: SLF001
        session._router.handle("status", {})  # noqa: SLF001
        yield _result_message()

    session._query = fake  # noqa: SLF001
    assert session.run("x").status == "budget"


def test_the_board_survives_concurrent_writers(tmp_path: Path) -> None:
    board = Board(path=tmp_path / "b.jsonl")
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda i: board.record(
                    Attempt(cell="c", detail="d", best_distance=i, target_distance=9, evaluations=1)
                ),
                range(48),
            )
        )
    assert len(board.attempts()) == 48
    assert all(json.loads(line) for line in board.path.read_text().splitlines())


def test_asyncio_is_available_for_the_thread_offload() -> None:
    assert asyncio.get_event_loop_policy() is not None
