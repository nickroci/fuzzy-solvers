from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Literal

import pytest

from ramsey.campaign import (
    OPEN_TARGETS,
    ArmOutcome,
    CampaignArm,
    CampaignReport,
    RetryPolicy,
    completed_labels,
    frontier_arms,
    run_campaign,
)
from ramsey.cayley import AbelianGroup, abelian_groups
from ramsey.session import RamseySessionResult, RamseySessionSettings
from ramsey.tools import RamseyTarget

if TYPE_CHECKING:
    from pathlib import Path

TARGET = RamseyTarget(second_colour=18, record=99, credit="Ex16")


def test_open_targets_match_the_published_table() -> None:
    assert [(t.second_colour, t.record) for t in OPEN_TARGETS] == [
        (15, 74),
        (16, 82),
        (17, 92),
        (18, 99),
        (19, 106),
        (20, 111),
        (21, 122),
        (22, 131),
        (23, 139),
    ]


def test_arm_label_identifies_target_group_and_seed() -> None:
    arm = CampaignArm(target=TARGET, group=AbelianGroup(moduli=(3, 3, 11)), seed=2)
    assert arm.label == "R(3,18)>99/Z3 x Z3 x Z11#2"
    assert not arm.is_cyclic


def test_a_cyclic_arm_below_the_exhaustion_order_is_flagged() -> None:
    arm = CampaignArm(target=RamseyTarget(second_colour=15, record=74), group=AbelianGroup((74,)))
    assert arm.is_cyclic
    assert arm.covered_by_cyclic_exhaustion
    assert "Harborth and Krause" in arm.instruction()


def test_a_cyclic_arm_above_the_exhaustion_order_is_not_flagged() -> None:
    arm = CampaignArm(target=RamseyTarget(second_colour=22, record=131), group=AbelianGroup((131,)))
    assert arm.is_cyclic
    assert not arm.covered_by_cyclic_exhaustion


def test_instruction_names_the_alternative_groups() -> None:
    arm = CampaignArm(target=TARGET, group=AbelianGroup(moduli=(3, 3, 11)))
    text = arm.instruction()
    assert "Z3 x Z3 x Z11" in text
    assert "Z9 x Z11" in text
    assert "analyse()" in text


def test_instruction_says_when_a_group_is_the_only_one() -> None:
    arm = CampaignArm(target=RamseyTarget(second_colour=22, record=131), group=AbelianGroup((131,)))
    assert "only abelian group" in arm.instruction()


def test_frontier_covers_every_group_of_every_target() -> None:
    arms = frontier_arms()
    assert len(arms) == sum(len(abelian_groups(t.record)) for t in OPEN_TARGETS)


def test_frontier_seeds_multiply_the_arms() -> None:
    assert len(frontier_arms(seeds=3)) == 3 * len(frontier_arms(seeds=1))


def test_non_cyclic_only_keeps_the_arms_that_could_still_be_new() -> None:
    arms = frontier_arms(non_cyclic_only=True)
    assert all(not arm.covered_by_cyclic_exhaustion for arm in arms)
    labels = {arm.group.name for arm in arms}
    assert "Z2 x Z2 x Z23" in labels
    assert "Z3 x Z3 x Z11" in labels


def _result(
    status: Literal["banked", "budget", "incomplete"] = "incomplete",
) -> RamseySessionResult:
    return RamseySessionResult(
        status=status,
        target=TARGET,
        kernel_calls=7,
        tool_calls=3,
        turns=2,
        output_tokens=5,
        witness_graph6="G?" if status == "banked" else "",
        independence_number=17 if status == "banked" else None,
    )


def test_report_separates_improvements() -> None:
    good = ArmOutcome(arm=CampaignArm(TARGET, AbelianGroup((3, 3, 11))), result=_result("banked"))
    bad = ArmOutcome(arm=CampaignArm(TARGET, AbelianGroup((99,))), result=_result())
    report = CampaignReport(outcomes=(good, bad))
    assert report.improvements == (good,)
    assert report.kernel_calls == 14
    rendered = report.render()
    assert "1 improvements" in rendered
    assert "***" in rendered


def test_report_renders_failures() -> None:
    broken = ArmOutcome(arm=CampaignArm(TARGET, AbelianGroup((99,))), failure="RuntimeError: x")
    report = CampaignReport(outcomes=(broken,))
    assert not broken.improved
    assert "err" in report.render()
    assert report.kernel_calls == 0


def test_campaign_runs_arms_concurrently_and_isolates_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[str] = []

    class FakeSession:
        def __init__(self, *, target: RamseyTarget, settings: object, **_: object) -> None:
            self.target = target

        async def run_async(self, instruction: str) -> RamseySessionResult:
            started.append(instruction[:20])
            await asyncio.sleep(0)
            if self.target.second_colour == 16:
                message = "boom"
                raise RuntimeError(message)
            return _result()

    monkeypatch.setattr("ramsey.campaign.RamseyAgentSession", FakeSession)
    arms = (
        CampaignArm(TARGET, AbelianGroup((3, 3, 11))),
        CampaignArm(RamseyTarget(second_colour=16, record=82), AbelianGroup((2, 41))),
    )
    report = run_campaign(
        arms,
        RamseySessionSettings(),
        concurrency=2,
        retry=RetryPolicy(attempts=1, backoff_seconds=0.0),
    )
    assert len(report.outcomes) == 2
    assert len(started) == 2
    assert any(outcome.failure.startswith("RuntimeError") for outcome in report.outcomes)


def test_campaign_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency must be positive"):
        run_campaign((), RamseySessionSettings(), concurrency=0)


# --- durability: retry, timeout, journal, resume ---


def test_retry_policy_rejects_a_zero_attempt_budget() -> None:
    with pytest.raises(ValueError, match="attempts must be positive"):
        RetryPolicy(attempts=0)


def test_retry_policy_rejects_negative_backoff() -> None:
    with pytest.raises(ValueError, match="backoff_seconds must be non-negative"):
        RetryPolicy(backoff_seconds=-1.0)


def test_a_transient_transport_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class FlakySession:
        def __init__(self, *, target: RamseyTarget, settings: object, **_: object) -> None:
            self.target = target

        async def run_async(self, instruction: str) -> RamseySessionResult:
            calls["n"] += 1
            if calls["n"] == 1:
                message = "Connection lost while your computer was asleep"
                raise RuntimeError(message)
            return _result()

    monkeypatch.setattr("ramsey.campaign.RamseyAgentSession", FlakySession)
    report = run_campaign(
        (CampaignArm(TARGET, AbelianGroup((3, 3, 11))),),
        RamseySessionSettings(),
        retry=RetryPolicy(attempts=3, backoff_seconds=0.0),
    )
    assert calls["n"] == 2
    assert report.outcomes[0].result is not None
    assert report.outcomes[0].attempts == 2
    assert not report.outcomes[0].failure


def test_an_arm_that_never_succeeds_records_its_last_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeadSession:
        def __init__(self, *, target: RamseyTarget, settings: object, **_: object) -> None:
            pass

        async def run_async(self, instruction: str) -> RamseySessionResult:
            message = "401 revoked"
            raise RuntimeError(message)

    monkeypatch.setattr("ramsey.campaign.RamseyAgentSession", DeadSession)
    report = run_campaign(
        (CampaignArm(TARGET, AbelianGroup((3, 3, 11))),),
        RamseySessionSettings(),
        retry=RetryPolicy(attempts=2, backoff_seconds=0.0),
    )
    outcome = report.outcomes[0]
    assert outcome.attempts == 2
    assert "401 revoked" in outcome.failure


def test_a_hanging_arm_is_cut_off_by_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class HangingSession:
        def __init__(self, *, target: RamseyTarget, settings: object, **_: object) -> None:
            pass

        async def run_async(self, instruction: str) -> RamseySessionResult:
            await asyncio.sleep(10)
            return _result()

    monkeypatch.setattr("ramsey.campaign.RamseyAgentSession", HangingSession)
    report = run_campaign(
        (CampaignArm(TARGET, AbelianGroup((3, 3, 11))),),
        RamseySessionSettings(),
        retry=RetryPolicy(attempts=1, backoff_seconds=0.0, timeout_seconds=0.05),
    )
    assert "TimeoutError" in report.outcomes[0].failure


def test_each_finished_arm_is_journalled_immediately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class OkSession:
        def __init__(self, *, target: RamseyTarget, settings: object, **_: object) -> None:
            pass

        async def run_async(self, instruction: str) -> RamseySessionResult:
            return _result("banked")

    monkeypatch.setattr("ramsey.campaign.RamseyAgentSession", OkSession)
    journal = tmp_path / "arms.jsonl"
    arms = (
        CampaignArm(TARGET, AbelianGroup((3, 3, 11)), seed=0),
        CampaignArm(TARGET, AbelianGroup((3, 3, 11)), seed=1),
    )
    run_campaign(arms, RamseySessionSettings(), journal=journal)
    lines = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert {entry["seed"] for entry in lines} == {0, 1}
    assert all(entry["status"] == "banked" for entry in lines)


def test_completed_labels_skips_failed_arms(tmp_path: Path) -> None:
    journal = tmp_path / "arms.jsonl"
    good = ArmOutcome(arm=CampaignArm(TARGET, AbelianGroup((3, 3, 11)), seed=0), result=_result())
    bad = ArmOutcome(
        arm=CampaignArm(TARGET, AbelianGroup((3, 3, 11)), seed=1), failure="RuntimeError: x"
    )
    journal.write_text(good.journal_entry() + "\n" + bad.journal_entry() + "\n")
    labels = completed_labels(journal)
    assert good.arm.label in labels
    assert bad.arm.label not in labels


def test_completed_labels_is_empty_without_a_journal(tmp_path: Path) -> None:
    assert completed_labels(tmp_path / "absent.jsonl") == frozenset()
