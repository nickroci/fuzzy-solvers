from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from claude_agent_sdk import ResultMessage

from covering.campaign import (
    ArmOutcome,
    CampaignReport,
    Target,
    agent_arm,
    control_arm,
    naive_arm,
    run_campaign_async,
    trivial_group,
)
from covering.design import check_covering

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# The Fano plane covers all 21 pairs with 7 triples, so a control arm aimed at
# 7 is aimed at something that exists - which is what makes a miss meaningful.
FANO = Target(points=7, block_size=3, strength=2, published=8, lower_bound=5)
HARD = Target(points=13, block_size=3, strength=2, published=14, lower_bound=11)


def _result_message() -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        stop_reason="end_turn",
        usage={"output_tokens": 1},
    )


def test_a_target_renders_its_cell_and_objective() -> None:
    assert FANO.label == "C(7,3,2)"
    assert FANO.target_blocks == 7
    assert "7 blocks or fewer" in FANO.instruction()
    assert "lower bound is 5" in FANO.instruction()


def test_the_trivial_group_develops_a_design_into_itself() -> None:
    group = trivial_group(7)
    assert group.order == 1


def test_a_control_arm_finds_a_design_that_exists() -> None:
    """Seven triples covering every pair of seven points is reachable mechanically."""
    outcome = control_arm(FANO, 400_000, seed=3)
    assert outcome.kind == "control"
    assert outcome.improved
    assert len(outcome.blocks) <= FANO.target_blocks
    assert check_covering(FANO.parameters, outcome.blocks).accepted


def test_a_control_arm_reports_a_miss_rather_than_inventing_one() -> None:
    """Thirteen triples cannot cover 78 pairs, so the arm must come back short."""
    outcome = control_arm(HARD, 200_000, seed=1)
    assert not outcome.improved
    assert outcome.blocks == ()
    assert outcome.uncovered > 0
    assert "left" in outcome.render()


def test_a_control_arm_spends_no_more_than_its_budget() -> None:
    outcome = control_arm(HARD, 120_000, seed=2)
    assert outcome.moves <= 120_000


def test_an_agent_arm_reports_what_the_model_banked() -> None:
    lines = ((0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2))
    captured: dict[str, object] = {}

    async def fake(**kwargs: object) -> AsyncIterator[object]:
        captured.update(kwargs)
        yield _result_message()

    async def drive() -> ArmOutcome:
        return await agent_arm(FANO, 50_000, query_fn=fake)

    outcome = asyncio.run(drive())
    assert outcome.kind == "agent"
    assert not outcome.improved
    assert "objective" in str(captured["prompt"])
    assert lines


def test_a_campaign_runs_both_halves_and_compares_them() -> None:
    async def fake(**_: object) -> AsyncIterator[object]:
        yield _result_message()

    async def drive() -> CampaignReport:
        return await run_campaign_async((FANO,), 400_000, concurrency=2, query_fn=fake)

    report = asyncio.run(drive())
    assert len(report.of("agent")) == 1
    assert len(report.of("control")) == 1
    text = report.render()
    assert "1 agent arms, 1 control arms" in text
    assert "control moves" in text
    # the control reaches a design the do-nothing agent stub cannot
    assert report.of("control")[0].improved
    assert not report.of("agent")[0].improved


def test_a_dropped_transport_is_not_recorded_as_a_search_result() -> None:
    """An arm that died on the wire has measured nothing and must not cancel the rest."""

    async def broken(**_: object) -> AsyncIterator[object]:
        yield _result_message()
        message = "connection reset"
        raise RuntimeError(message)

    async def drive() -> CampaignReport:
        return await run_campaign_async((FANO,), 200_000, query_fn=broken)

    report = asyncio.run(drive())
    agent = report.of("agent")[0]
    assert agent.moves == 0
    assert not agent.improved
    assert report.of("control")[0].improved


def test_a_campaign_rejects_a_meaningless_concurrency() -> None:
    async def drive() -> CampaignReport:
        return await run_campaign_async((FANO,), 1000, concurrency=0)

    with pytest.raises(ValueError, match="concurrency must be positive"):
        asyncio.run(drive())


def test_an_outcome_marks_an_improvement_in_its_rendering() -> None:
    won = ArmOutcome(label="C(7,3,2)", kind="agent", moves=5, uncovered=0, blocks=(1, 2))
    lost = ArmOutcome(label="C(7,3,2)", kind="agent", moves=5, uncovered=3)
    assert won.render().startswith("***")
    assert not lost.improved
    assert lost.render().startswith("   ")


def test_the_naive_arm_uses_none_of_this_project_s_ideas() -> None:
    """The honest baseline: uniform swaps, random restarts, nothing else.

    It must still be able to find something reachable, or it is a strawman
    rather than a baseline.
    """
    outcome = naive_arm(FANO, 400_000, seed=5)
    assert outcome.kind == "naive"
    assert outcome.improved
    assert check_covering(FANO.parameters, outcome.blocks).accepted


def test_the_naive_arm_reports_a_miss_rather_than_inventing_one() -> None:
    outcome = naive_arm(HARD, 200_000, seed=2)
    assert not outcome.improved
    assert outcome.blocks == ()
    assert outcome.uncovered > 0


def test_the_naive_arm_stays_inside_its_budget() -> None:
    outcome = naive_arm(HARD, 150_000, seed=4)
    assert outcome.moves <= 150_000
