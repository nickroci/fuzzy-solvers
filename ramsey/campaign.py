"""Parallel fan-out of model-driven searches across targets and groups.

Each search is an independent session against its own kernel ledger, so the
whole open frontier can run at once rather than one target at a time.  Arms
share nothing: a session's archive, budget and incumbent are its own, which
keeps every banked witness attributable to the session that authored it.

The frontier is the ``R(3, k)`` lower-bound row of Table IIa in Radziszowski's
dynamic survey.  Every entry there is an explicit graph, and the survey's own
commentary is the reason to attack this row rather than another: *"some of
them should not be that hard to improve, in contrast to the bounds in
Table Ia."*

Which groups are worth trying is decided by the structure theorem, not by
taste.  ``abelian_groups`` enumerates them, and for a squarefree order there
is only the cyclic one — which is why the published exhaustive cyclic searches
settle those orders, and why the orders admitting a non-cyclic group are the
interesting arms.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ramsey.blackboard import DOMAIN_BRIEFING
from ramsey.cayley import AbelianGroup, abelian_groups
from ramsey.session import (
    RamseyAgentSession,
    RamseySessionResult,
    RamseySessionSettings,
)
from ramsey.tools import RamseyTarget

if TYPE_CHECKING:
    from pathlib import Path

    from ramsey.blackboard import Blackboard

OPEN_TARGETS: tuple[RamseyTarget, ...] = (
    RamseyTarget(second_colour=15, record=74, credit="Kol2"),
    RamseyTarget(second_colour=16, record=82, credit="Ex21"),
    RamseyTarget(second_colour=17, record=92, credit="WWY1"),
    RamseyTarget(second_colour=18, record=99, credit="Ex16"),
    RamseyTarget(second_colour=19, record=106, credit="WWY1"),
    RamseyTarget(second_colour=20, record=111, credit="Ex16"),
    RamseyTarget(second_colour=21, record=122, credit="WWY1"),
    RamseyTarget(second_colour=22, record=131, credit="WSLX2"),
    RamseyTarget(second_colour=23, record=139, credit="XWCS"),
)
"""The R(3,k) lower bounds of Table IIa, DS1.17 (2024), as of this campaign."""

CYCLIC_EXHAUSTED_BELOW = 102
"""Harborth and Krause searched cyclic graphs exhaustively below this order."""


@dataclass(frozen=True, slots=True)
class CampaignArm:
    """One independent search: a target, a group to work in, and a seed."""

    target: RamseyTarget
    group: AbelianGroup
    seed: int = 0

    @property
    def label(self) -> str:
        """Render a stable identifier for the arm."""
        return (
            f"R(3,{self.target.second_colour})>{self.target.record}/{self.group.name}#{self.seed}"
        )

    @property
    def is_cyclic(self) -> bool:
        """Report whether the arm works in a cyclic group."""
        return self.group.is_cyclic

    @property
    def covered_by_cyclic_exhaustion(self) -> bool:
        """Report whether published exhaustive cyclic search already settles this arm.

        A cyclic group below the exhaustion order has been swept, so such an
        arm is reconnaissance rather than a real attempt, and its result should
        be read as a check on the referee rather than as a discovery.
        """
        return self.is_cyclic and self.target.order < CYCLIC_EXHAUSTED_BELOW

    def instruction(self) -> str:
        """Render the arm's objective, stating only what is established."""
        others = [group.name for group in abelian_groups(self.target.order) if group != self.group]
        objective = (
            f"Find a graph proving R(3,{self.target.second_colour}) > {self.target.order}, "
            f"improving the published lower bound (credited to {self.target.credit})."
        )
        lines = [
            objective,
            f"Work in the abelian group {self.group.name} of order {self.target.order}.",
        ]
        if others:
            lines.append(f"The other abelian groups of this order are: {', '.join(others)}.")
        else:
            lines.append("This is the only abelian group of this order.")
        if self.covered_by_cyclic_exhaustion:
            lines.append(
                "Note: Harborth and Krause exhaustively searched cyclic graphs below "
                f"{CYCLIC_EXHAUSTED_BELOW} vertices, so a cyclic construction here is "
                "very unlikely to be new."
            )
        lines.append(
            "Use analyse() to see why a construction fails before refining it, and "
            "the 'break' move class to spend calls only on generators that can "
            "actually destroy the current obstruction."
        )
        return " ".join(lines) + "\n\n" + DOMAIN_BRIEFING


def frontier_arms(
    targets: tuple[RamseyTarget, ...] = OPEN_TARGETS,
    *,
    seeds: int = 1,
    non_cyclic_only: bool = False,
) -> tuple[CampaignArm, ...]:
    """Build one arm per (target, abelian group, seed) over the frontier.

    ``non_cyclic_only`` drops the arms that published cyclic exhaustions
    already settle, which is the cheap way to spend a budget where a result
    could still be new.
    """
    arms = [
        CampaignArm(target=target, group=group, seed=seed)
        for target in targets
        for group in abelian_groups(target.order)
        for seed in range(seeds)
    ]
    if non_cyclic_only:
        arms = [arm for arm in arms if not arm.covered_by_cyclic_exhaustion]
    return tuple(arms)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How hard to try an arm before recording it as failed.

    A dropped connection is not a search result.  The transport fails for
    reasons that have nothing to do with the mathematics - a sleeping laptop,
    an expired token, a timeout - and an arm that dies that way has measured
    nothing, so it deserves another attempt rather than a place in the report.
    """

    attempts: int = 3
    backoff_seconds: float = 10.0
    timeout_seconds: float | None = 1800.0

    def __post_init__(self) -> None:
        """Reject a policy that would never run an arm."""
        if self.attempts < 1:
            message = f"attempts must be positive, got {self.attempts}"
            raise ValueError(message)
        if self.backoff_seconds < 0:
            message = f"backoff_seconds must be non-negative, got {self.backoff_seconds}"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """One arm's result, or the failure that replaced it."""

    arm: CampaignArm
    result: RamseySessionResult | None = None
    failure: str = ""
    attempts: int = 1

    @property
    def improved(self) -> bool:
        """Report whether this arm banked a referee-accepted witness."""
        return self.result is not None and self.result.improved

    def journal_entry(self) -> str:
        """Render one JSON line, so a finished arm survives the process."""
        payload = {
            "label": self.arm.label,
            "second_colour": self.arm.target.second_colour,
            "record": self.arm.target.record,
            "group": self.arm.group.name,
            "seed": self.arm.seed,
            "attempts": self.attempts,
            "failure": self.failure,
            "status": self.result.status if self.result is not None else "failed",
            "kernel_calls": self.result.kernel_calls if self.result is not None else 0,
            "witness_graph6": self.result.witness_graph6 if self.result is not None else "",
            "independence_number": (
                self.result.independence_number if self.result is not None else None
            ),
            "best_independence": (
                self.result.best_independence if self.result is not None else None
            ),
            "best_generators": (
                list(self.result.best_generators) if self.result is not None else []
            ),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def completed_labels(journal: Path) -> frozenset[str]:
    """Return the arm labels already recorded, so a rerun resumes.

    Only arms that actually reached the model count as done: an arm recorded
    with a transport failure is left for the next run to retry.
    """
    if not journal.exists():
        return frozenset()
    labels: set[str] = set()
    for line in journal.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if not entry.get("failure"):
            labels.add(str(entry["label"]))
    return frozenset(labels)


@dataclass(frozen=True, slots=True)
class CampaignReport:
    """Every arm's outcome, with the improvements pulled out."""

    outcomes: tuple[ArmOutcome, ...] = field(default_factory=tuple)

    @property
    def improvements(self) -> tuple[ArmOutcome, ...]:
        """Return only the arms that produced a new witness."""
        return tuple(outcome for outcome in self.outcomes if outcome.improved)

    @property
    def kernel_calls(self) -> int:
        """Return total kernel calls spent across the campaign."""
        return sum(
            outcome.result.kernel_calls for outcome in self.outcomes if outcome.result is not None
        )

    def render(self) -> str:
        """Render one line per arm, improvements first."""
        headline = (
            f"{len(self.outcomes)} arms, {len(self.improvements)} improvements, "
            f"{self.kernel_calls} kernel calls"
        )
        lines = [headline]
        lines.extend(
            f"  *** {outcome.arm.label}: R(3,{outcome.arm.target.second_colour}) > "
            f"{outcome.arm.target.order}  alpha={outcome.result.independence_number}  "
            f"{outcome.result.witness_graph6}"
            for outcome in self.improvements
            if outcome.result is not None
        )
        for outcome in self.outcomes:
            if outcome.improved:
                continue
            if outcome.failure:
                lines.append(f"  err {outcome.arm.label}: {outcome.failure}")
            elif outcome.result is not None:
                best = outcome.result.best_independence
                cap = outcome.arm.target.independence_cap
                reach = f"best alpha={best} (cap {cap}, gap {best - cap})" if best else "no search"
                lines.append(
                    f"  --  {outcome.arm.label}: {outcome.result.status}, "
                    f"{outcome.result.kernel_calls} calls, {reach}"
                )
        return "\n".join(lines)


async def run_campaign_async(
    arms: tuple[CampaignArm, ...],
    settings: RamseySessionSettings,
    *,
    concurrency: int = 4,
    retry: RetryPolicy | None = None,
    journal: Path | None = None,
    blackboard: Blackboard | None = None,
) -> CampaignReport:
    """Run every arm concurrently, bounded by ``concurrency``.

    Each arm is retried through transport failures, bounded by a per-attempt
    timeout, and appended to ``journal`` the moment it finishes.  That is what
    makes a long campaign survivable: results exist on disk as they happen
    rather than only when the last arm returns, so an interrupted run loses at
    most the arms still in flight.

    One arm's failure never cancels the fan-out.
    """
    if concurrency < 1:
        message = f"concurrency must be positive, got {concurrency}"
        raise ValueError(message)
    policy = retry or RetryPolicy()
    limit = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def attempt(arm: CampaignArm) -> ArmOutcome:
        last = ""
        for number in range(1, policy.attempts + 1):
            session = RamseyAgentSession(
                target=arm.target, settings=settings, blackboard=blackboard
            )
            try:
                coroutine = session.run_async(arm.instruction())
                result = (
                    await asyncio.wait_for(coroutine, policy.timeout_seconds)
                    if policy.timeout_seconds is not None
                    else await coroutine
                )
            except Exception as error:  # noqa: BLE001 - transport failure is not a result
                last = f"{type(error).__name__}: {error}"
                if number < policy.attempts:
                    await asyncio.sleep(policy.backoff_seconds * number)
                continue
            return ArmOutcome(arm=arm, result=result, attempts=number)
        return ArmOutcome(arm=arm, failure=last, attempts=policy.attempts)

    async def run_one(arm: CampaignArm) -> ArmOutcome:
        async with limit:
            outcome = await attempt(arm)
        if journal is not None:
            async with lock:
                with journal.open("a", encoding="utf-8") as handle:
                    handle.write(outcome.journal_entry() + "\n")
        return outcome

    outcomes = await asyncio.gather(*(run_one(arm) for arm in arms))
    return CampaignReport(outcomes=tuple(outcomes))


def run_campaign(
    arms: tuple[CampaignArm, ...],
    settings: RamseySessionSettings,
    *,
    concurrency: int = 4,
    retry: RetryPolicy | None = None,
    journal: Path | None = None,
    blackboard: Blackboard | None = None,
) -> CampaignReport:
    """Run a campaign synchronously."""
    return asyncio.run(
        run_campaign_async(
            arms,
            settings,
            concurrency=concurrency,
            retry=retry,
            journal=journal,
            blackboard=blackboard,
        )
    )
