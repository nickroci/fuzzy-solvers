from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage

from ramsey.cayley import circulant
from ramsey.graph import to_graph6
from ramsey.session import (
    RamseyAgentSession,
    RamseySessionSettings,
    RamseyToolRouter,
    render_result,
)
from ramsey.tools import RamseyTarget, RamseyTools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

CLEBSCH_TARGET = RamseyTarget(second_colour=6, record=16, credit="test")
CLEBSCH = {"moduli": [2, 2, 2, 2], "generators": [1, 2, 4, 8, 15]}


def router(budget: int = 20, max_tool_calls: int = 20) -> RamseyToolRouter:
    return RamseyToolRouter(
        tools=RamseyTools(target=CLEBSCH_TARGET, kernel_budget=budget),
        max_tool_calls=max_tool_calls,
    )


def _result_message(output_tokens: int = 14) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="session",
        stop_reason="end_turn",
        usage={"output_tokens": output_tokens},
    )


# --- router ---


def test_router_describes_the_target() -> None:
    content, ok = router().handle("describe_target", {})
    assert ok
    assert "16 vertices" in content


def test_router_evaluates_and_banks_a_genuine_witness() -> None:
    bus = router()
    content, ok = bus.handle("evaluate_cayley", CLEBSCH)
    assert ok
    assert "ACCEPTED" in content
    handle = content.split("]")[0].lstrip("[")
    content, ok = bus.handle("bank", {"handle": handle})
    assert ok
    assert "R(3,6) > 16" in content


def test_router_reports_a_failed_construction_with_its_obstruction() -> None:
    content, ok = router().handle("evaluate_cayley", {"moduli": [16], "generators": [1, 6]})
    assert ok
    assert "independent set of size" in content


def test_router_reports_a_non_sum_free_set() -> None:
    content, ok = router().handle("evaluate_cayley", {"moduli": [16], "generators": [1, 2]})
    assert "NOT triangle-free" in content
    assert ok


def test_router_marks_a_rejected_proposal_as_not_ok() -> None:
    content, ok = router().handle("evaluate_cayley", {"moduli": [15], "generators": [1, 4]})
    assert not ok
    assert "rejected" in content


def test_router_accepts_an_explicit_graph6() -> None:
    payload = {"graph6": to_graph6(circulant(16, (1, 6))), "label": "explicit"}
    content, ok = router().handle("evaluate_graph6", payload)
    assert ok
    assert "independent set of size" in content


def test_router_surfaces_the_kernel_budget_as_an_error() -> None:
    bus = router(budget=1)
    bus.handle("evaluate_cayley", {"moduli": [16], "generators": [1, 6]})
    content, ok = bus.handle("evaluate_cayley", {"moduli": [16], "generators": [1, 7]})
    assert not ok
    assert "budget of 1" in content
    assert bus.budget_exhausted


def test_router_enforces_its_own_tool_call_cap() -> None:
    bus = router(max_tool_calls=1)
    bus.handle("status", {})
    content, ok = bus.handle("status", {})
    assert not ok
    assert "tool call budget exhausted" in content


def test_router_reports_an_unknown_tool() -> None:
    content, ok = router().handle("nope", {})
    assert not ok
    assert "unknown tool" in content


def test_router_reports_a_malformed_payload() -> None:
    content, ok = router().handle("evaluate_cayley", {"moduli": [16]})
    assert not ok
    assert "KeyError" in content


def test_router_logs_every_call() -> None:
    bus = router()
    bus.handle("status", {})
    bus.handle("describe_target", {})
    assert [entry.name for entry in bus.log] == ["status", "describe_target"]


# --- session ---


def test_session_rejects_an_empty_instruction() -> None:
    session = RamseyAgentSession(target=CLEBSCH_TARGET)
    with pytest.raises(ValueError, match="must contain 1-"):
        session.run("   ")


def test_session_reports_an_incomplete_run() -> None:
    async def fake_query(**_: object) -> AsyncIterator[object]:
        yield AssistantMessage(content=[], model="claude-haiku-test")
        yield _result_message()

    session = RamseyAgentSession(
        target=CLEBSCH_TARGET,
        settings=RamseySessionSettings(max_turns=3, max_tool_calls=5, kernel_budget=5),
        query_fn=fake_query,
    )
    result = session.run("find a witness")
    assert result.status == "incomplete"
    assert not result.improved
    assert result.turns == 1
    assert result.output_tokens == 14
    assert result.witness_graph6 == ""
    assert "status=incomplete" in render_result(result)


def test_session_reports_a_banked_witness() -> None:
    session = RamseyAgentSession(target=CLEBSCH_TARGET, query_fn=None)

    async def fake_query(**_: object) -> AsyncIterator[object]:
        evaluation = session.tools.evaluate_cayley((2, 2, 2, 2), (1, 2, 4, 8, 15))
        session.tools.bank(evaluation.handle)
        yield AssistantMessage(content=[], model="claude-haiku-test")
        yield _result_message()

    session._query = fake_query  # noqa: SLF001
    result = session.run("find a witness")
    assert result.improved
    assert result.status == "banked"
    assert result.independence_number == 5
    assert result.witness_graph6
    report = render_result(result)
    assert "R(3,6) > 16" in report
    assert "witness graph6" in report


def test_session_reports_budget_exhaustion() -> None:
    session = RamseyAgentSession(
        target=CLEBSCH_TARGET,
        settings=RamseySessionSettings(max_turns=3, max_tool_calls=1, kernel_budget=1),
    )

    async def fake_query(**_: object) -> AsyncIterator[object]:
        session._router.handle("status", {})  # noqa: SLF001
        session._router.handle("status", {})  # noqa: SLF001
        yield _result_message()

    session._query = fake_query  # noqa: SLF001
    result = session.run("find a witness")
    assert result.status == "budget"
