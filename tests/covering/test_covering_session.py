from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage

from covering.board import BRIEFING, Attempt, Board
from covering.design import Parameters
from covering.session import CoveringSession, Router, Settings
from covering.tools import CoveringTools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

FREE_CALL_ALLOWANCE = 4

SMALL = Parameters(points=7, block_size=3, strength=2)
CELL = "C(7,3,2)"


def tools(budget: int = 200_000, published: int = 8) -> CoveringTools:
    return CoveringTools(parameters=SMALL, published=published, lower_bound=5, budget=budget)


def router(max_calls: int = 20, board: Board | None = None) -> Router:
    return Router(tools=tools(), cell=CELL, max_tool_calls=max_calls, board=board)


# --- board ---


def test_the_board_round_trips_and_orders_by_shortfall(tmp_path: Path) -> None:
    board = Board(path=tmp_path / "b.jsonl")
    board.record(Attempt(cell=CELL, group="Z7", base_blocks=1, blocks=7, uncovered=4, moves=10))
    board.record(Attempt(cell=CELL, group="1", base_blocks=7, blocks=7, uncovered=1, moves=90))
    board.record(Attempt(cell="other", group="Z9", base_blocks=1, blocks=9, uncovered=0, moves=5))
    assert len(board.attempts()) == 3
    assert len(board.attempts(CELL)) == 2
    text = board.render(CELL)
    assert "Closest anywhere: 1 uncovered" in text
    assert text.index("left 1 uncovered") < text.index("left 4 uncovered")


def test_a_torn_board_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "b.jsonl"
    board = Board(path=path)
    board.record(Attempt(cell=CELL, group="Z7", base_blocks=1, blocks=7, uncovered=2, moves=1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n\n")
    assert len(board.attempts()) == 1


def test_an_absent_board_is_empty(tmp_path: Path) -> None:
    board = Board(path=tmp_path / "missing.jsonl")
    assert board.attempts() == ()
    assert "No other arm" in board.render(CELL)


def test_the_briefing_states_the_facts_arms_should_not_rediscover() -> None:
    assert "stabiliser" in BRIEFING
    assert "v/gcd(v,k)" in BRIEFING
    assert "trivial group is a legitimate choice" in BRIEFING


# --- router ---


def test_router_describes_and_reads_history(tmp_path: Path) -> None:
    bare = router()
    assert "NEW REPOSITORY ENTRY" in bare.handle("describe_target", {})[0]
    assert "No board is attached" in bare.handle("read_board", {})[0]
    wired = router(board=Board(path=tmp_path / "b.jsonl"))
    assert "No other arm" in wired.handle("read_board", {})[0]


def test_router_checks_feasibility_without_spending(tmp_path: Path) -> None:
    only = router(board=Board(path=tmp_path / "b.jsonl"))
    content, ok = only.handle("check_feasible", {"family": "cyclic", "target": 7})
    assert ok
    assert "achievable orbit sizes" in content
    assert only.tools.used == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"family": "trivial"},
        {"family": "cyclic"},
        {"family": "cyclic_fixed", "argument": 1},
        {"family": "authored", "cycles": [[0, 1, 2]], "name": "mine"},
    ],
)
def test_router_builds_every_way_of_naming_a_group(payload: dict[str, object]) -> None:
    content, ok = router().handle("check_feasible", dict(payload) | {"target": 7})
    assert ok
    assert "orbit sizes" in content


def test_router_builds_a_grid_group() -> None:
    surface = CoveringTools(
        parameters=Parameters(points=25, block_size=15, strength=5),
        published=40,
        lower_bound=30,
        budget=1000,
    )
    node = Router(tools=surface, cell="C(25,15,5)", max_tool_calls=5)
    content, ok = node.handle(
        "check_feasible", {"family": "grid", "rows": 5, "columns": 5, "target": 40}
    )
    assert ok
    assert "order 25" in content


def test_router_searches_and_publishes_to_the_board(tmp_path: Path) -> None:
    board = Board(path=tmp_path / "b.jsonl")
    node = router(board=board)
    content, ok = node.handle(
        "search", {"family": "cyclic", "base_count": 1, "moves": 2000, "seed": 1}
    )
    assert ok
    assert "Uncovered:" in content
    assert len(board.attempts(CELL)) == 1


def test_router_adopts_authored_blocks_and_inspects_them(tmp_path: Path) -> None:
    node = router(board=Board(path=tmp_path / "b.jsonl"))
    blocks = [list(subset) for subset in combinations(range(7), 3)]
    content, ok = node.handle("adopt", {"family": "trivial", "base_blocks": blocks})
    assert ok
    assert "Uncovered: 0" in content
    assert "nothing uncovered" in node.handle("inspect", {})[0]


def test_router_resumes_from_the_incumbent() -> None:
    node = router()
    node.handle("search", {"family": "cyclic", "base_count": 1, "moves": 2000, "seed": 1})
    spent = node.tools.used
    resumed, ok = node.handle("resume", {"moves": 2000, "seed": 2})
    assert ok
    assert "Uncovered:" in resumed
    assert node.tools.used > spent


def test_router_reports_an_unknown_tool() -> None:
    content, ok = router().handle("nonesuch", {})
    assert not ok
    assert "unknown tool" in content


def test_router_reports_a_malformed_payload() -> None:
    content, ok = router().handle("search", {"family": "cyclic"})
    assert not ok
    assert "KeyError" in content


def test_router_enforces_its_cap_on_calls_that_spend() -> None:
    node = router(max_calls=1)
    node.handle("search", {"family": "cyclic", "base_count": 1, "moves": 100, "seed": 1})
    content, ok = node.handle(
        "search", {"family": "cyclic", "base_count": 1, "moves": 100, "seed": 2}
    )
    assert not ok
    assert "search budget exhausted" in content
    assert node.exhausted


def test_a_tool_that_spends_nothing_spends_no_allowance_either() -> None:
    """check_feasible is advertised as free, so it must not consume the call cap.

    A live session took the advertisement at its word, spent sixteen of its
    thirty calls ruling groups in and out, and was left with too few searches
    to reach a design that works while most of its move budget went unspent.
    """
    node = router(max_calls=10)
    for _ in range(20):
        content, ok = node.handle("check_feasible", {"family": "cyclic", "target": 7})
        assert ok, content
    assert not node.exhausted
    assert node.calls == 0
    _, ok = node.handle("search", {"family": "cyclic", "base_count": 1, "moves": 100})
    assert ok


def test_free_calls_are_still_bounded() -> None:
    """Unbounded free calls would let a session loop for ever without searching."""
    node = router(max_calls=1)
    outcomes = [node.handle("status", {})[1] for _ in range(1 + FREE_CALL_ALLOWANCE)]
    assert outcomes[0]
    assert not outcomes[-1]
    assert not node.exhausted


def test_router_surfaces_an_exhausted_move_budget() -> None:
    node = Router(tools=tools(budget=1), cell=CELL, max_tool_calls=9)
    node.handle("search", {"family": "cyclic", "base_count": 1, "moves": 50, "seed": 1})
    content, ok = node.handle(
        "search", {"family": "cyclic", "base_count": 1, "moves": 50, "seed": 2}
    )
    assert not ok
    assert "budget of 1" in content
    assert node.exhausted


# --- session ---


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


def session(**kwargs: object) -> CoveringSession:
    return CoveringSession(tools=tools(**kwargs), cell=CELL)  # type: ignore[arg-type]


def test_session_rejects_an_empty_instruction() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        session().run("  ")


def test_session_reports_an_incomplete_run() -> None:
    async def fake(**_: object) -> AsyncIterator[object]:
        yield AssistantMessage(content=[], model="m")
        yield _result_message()

    node = CoveringSession(tools=tools(), cell=CELL, query_fn=fake)
    result = node.run("find one")
    assert result.status == "incomplete"
    assert not result.improved
    assert result.turns == 1
    assert "status=incomplete" in result.render()
    assert "best_uncovered=n/a" in result.render()


def test_session_reports_a_banked_design() -> None:
    node = CoveringSession(tools=tools(published=8), cell=CELL)

    async def fake(**_: object) -> AsyncIterator[object]:
        lines = ((0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2))
        blocks = tuple(sum(1 << point for point in line) for line in lines)
        node.tools.adopt(node.tools.trivial_group(), blocks)
        yield _result_message()

    node._query = fake  # noqa: SLF001
    result = node.run("find one")
    assert result.improved
    assert result.status == "banked"
    assert "beating the published 8" in result.render()


def test_session_reports_budget_exhaustion() -> None:
    node = CoveringSession(tools=tools(), cell=CELL, settings=Settings(max_tool_calls=1))

    async def fake(**_: object) -> AsyncIterator[object]:
        spend = {"family": "trivial", "base_count": 1, "moves": 100}
        node._router.handle("search", spend)  # noqa: SLF001
        node._router.handle("search", spend)  # noqa: SLF001
        yield _result_message()

    node._query = fake  # noqa: SLF001
    assert node.run("x").status == "budget"


def test_the_board_records_what_was_tried_not_the_best_so_far(tmp_path: Path) -> None:
    """An arm's entry must describe its own design, not the incumbent's.

    Recording the incumbent's block count against whatever group was named
    last is how the shared history came to claim that 24 base blocks under the
    trivial group develop to 312 blocks.
    """
    board = Board(path=tmp_path / "b.jsonl")
    node = router(board=board)
    blocks = [list(subset) for subset in combinations(range(7), 3)]
    node.handle("adopt", {"family": "trivial", "base_blocks": blocks})
    node.handle("adopt", {"family": "trivial", "base_blocks": [[0, 1, 2]]})
    entries = board.attempts(CELL)
    assert len(entries) == 2
    assert entries[0].blocks == len(blocks)
    assert entries[1].blocks == 1
    assert entries[1].base_blocks == 1


def test_the_history_does_not_call_an_oversized_design_the_closest(tmp_path: Path) -> None:
    """A perfect covering of the wrong size is not close to anything."""
    board = Board(path=tmp_path / "b.jsonl")
    board.record(Attempt(cell=CELL, group="big", base_blocks=9, blocks=99, uncovered=0, moves=1))
    board.record(Attempt(cell=CELL, group="right", base_blocks=7, blocks=7, uncovered=3, moves=1))
    text = board.render(CELL, target=7)
    assert text.index("group right" if "group right" in text else "right") < text.index("big")
    assert "Closest anywhere: 3 uncovered" in text
    untargeted = board.render(CELL)
    assert "Closest anywhere: 0 uncovered" in untargeted


def test_the_history_flags_a_leader_that_is_too_large(tmp_path: Path) -> None:
    """If the only attempts are oversized, say so rather than implying success."""
    board = Board(path=tmp_path / "b.jsonl")
    board.record(Attempt(cell=CELL, group="big", base_blocks=9, blocks=99, uncovered=0, moves=1))
    assert "a new entry needs 7" in board.render(CELL, target=7)
