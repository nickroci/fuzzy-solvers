"""Parallel fan-out of model-driven searches, against a matched mechanical control.

A fan-out of agents is only worth its cost if it beats the same compute spent
without them.  That is not rhetoric: the mechanical portfolio here is a real
search that has found a repository entry on its own, so an agent arm that ties
with it has demonstrated nothing, however interesting its reasoning was.

So every campaign runs both.  Each cell gets one agent arm and one control arm
with the *same move budget*, and the report puts them side by side.  The agent
arms share a board; the control arms share nothing, because there is nothing to
share -- that asymmetry is the hypothesis under test.

The control is deliberately the strongest mechanical thing we have rather than
a strawman: random restarts at the target count, and solve-then-delete-a-block
repair, which starts from a valid design instead of from noise.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from covering.design import Parameters, check_covering
from covering.kernel import Schedule, anneal
from covering.orbit import PointGroup, develop
from covering.orbit_search import random_block
from covering.session import CoveringSession, Settings
from covering.tools import CoveringTools

if TYPE_CHECKING:
    from collections.abc import Callable

    from covering.board import Board

_SOLVE_SHARE = 0.25
_REPAIR_ATTEMPTS = 4


@dataclass(frozen=True, slots=True)
class Target:
    """One repository cell worth attacking."""

    points: int
    block_size: int
    strength: int
    published: int
    lower_bound: int
    provenance: str = ""

    @property
    def label(self) -> str:
        """Render the cell in repository notation."""
        return f"C({self.points},{self.block_size},{self.strength})"

    @property
    def parameters(self) -> Parameters:
        """Return the shape of the covering problem."""
        return Parameters(points=self.points, block_size=self.block_size, strength=self.strength)

    @property
    def target_blocks(self) -> int:
        """Return the block count a new entry needs."""
        return self.published - 1

    def instruction(self) -> str:
        """Render the arm's objective."""
        return (
            f"Find a ({self.points},{self.block_size},{self.strength})-covering design "
            f"with {self.target_blocks} blocks or fewer. The published best is "
            f"{self.published} and the best known lower bound is {self.lower_bound}."
        )


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """What one arm reached, and what it spent getting there."""

    label: str
    kind: str
    moves: int
    uncovered: int
    blocks: tuple[int, ...] = ()

    @property
    def improved(self) -> bool:
        """Report whether the arm produced a design that beats the entry."""
        return self.uncovered == 0 and bool(self.blocks)

    def render(self) -> str:
        """Render one line of outcome."""
        mark = "***" if self.improved else "   "
        return (
            f"{mark} {self.kind:<8} {self.label:<14} left {self.uncovered:>4} "
            f"uncovered in {self.moves:,} moves"
        )


def trivial_group(points: int) -> PointGroup:
    """Return the one-element group, which develops a design into itself."""
    return PointGroup(points=points, generators=(tuple(range(points)),), name="1")


def naive_arm(target: Target, budget: int, *, seed: int = 0) -> ArmOutcome:
    """Textbook simulated annealing, with none of this project's ideas in it.

    This is the honest baseline. :func:`control_arm` is not one: its guided
    neighbourhood, its solve-then-delete-a-block descent and its temperatures
    were all designed by a model during this work, so measuring an agent
    against it compares per-cell deliberation with a model-authored fixed
    strategy -- not with the absence of a model.

    What is left here is what anyone would write from the textbook: start from
    a random design of the target size, swap a uniformly random point of a
    uniformly random block for a uniformly random outsider, accept by
    Metropolis on the count of uncovered subsets, cool geometrically, restart
    when the budget allows.

    It does keep the fast kernel -- incremental coverage rather than a full
    rescore -- because the alternative is measuring implementation speed
    instead of search strategy. That concession favours the baseline: the
    incremental scheme is itself one of the ideas under test, and without it
    this arm would run some three orders of magnitude slower.
    """
    parameters = target.parameters
    group = trivial_group(target.points)
    rng = random.Random(seed)
    spent = 0
    best = parameters.requirements
    found: tuple[int, ...] = ()
    slice_size = max(budget // 10, 1)

    while spent < budget and not found:
        base = tuple(random_block(parameters, rng) for _ in range(target.target_blocks))
        result = anneal(
            parameters,
            group,
            base,
            Schedule(
                steps=slice_size,
                seed=rng.randrange(10**6),
                start_temp=1.2,
                end_temp=0.02,
                guided=0.0,
            ),
        )
        spent += result.moves or slice_size
        best = min(best, result.uncovered)
        if result.uncovered == 0:
            found = develop(group, result.base_blocks)

    if found and not (
        check_covering(parameters, found).accepted and len(found) <= target.target_blocks
    ):
        found = ()
    return ArmOutcome(
        label=target.label,
        kind="naive",
        moves=spent,
        uncovered=0 if found else best,
        blocks=found,
    )


def control_arm(target: Target, budget: int, *, seed: int = 0) -> ArmOutcome:
    """Spend a move budget mechanically: restarts, then solve-and-repair.

    This is what an agent arm has to beat. It gets the same budget and the same
    kernel, and differs only in that nothing chooses for it.
    """
    parameters = target.parameters
    group = trivial_group(target.points)
    rng = random.Random(seed)
    spent = 0
    best = parameters.requirements
    found: tuple[int, ...] = ()
    solve_budget = int(budget * _SOLVE_SHARE)
    slice_size = max(budget // 10, 1)

    while spent < solve_budget and not found:
        base = tuple(random_block(parameters, rng) for _ in range(target.published))
        result = anneal(
            parameters, group, base, Schedule(steps=slice_size, seed=rng.randrange(10**6))
        )
        spent += result.moves or slice_size
        if result.uncovered:
            continue
        for drop in rng.sample(range(target.published), min(_REPAIR_ATTEMPTS, target.published)):
            if spent >= budget:
                break
            seeded = result.base_blocks[:drop] + result.base_blocks[drop + 1 :]
            repaired = anneal(
                parameters,
                group,
                seeded,
                Schedule(steps=slice_size, seed=rng.randrange(10**6), start_temp=0.5),
            )
            spent += repaired.moves or slice_size
            best = min(best, repaired.uncovered)
            if repaired.uncovered == 0:
                found = develop(group, repaired.base_blocks)
                break

    while spent < budget and not found:
        base = tuple(random_block(parameters, rng) for _ in range(target.target_blocks))
        result = anneal(
            parameters, group, base, Schedule(steps=slice_size, seed=rng.randrange(10**6))
        )
        spent += result.moves or slice_size
        best = min(best, result.uncovered)
        if result.uncovered == 0:
            found = develop(group, result.base_blocks)

    if found and not (
        check_covering(parameters, found).accepted and len(found) <= target.target_blocks
    ):
        found = ()
    return ArmOutcome(
        label=target.label,
        kind="control",
        moves=spent,
        uncovered=0 if found else best,
        blocks=found,
    )


async def agent_arm(
    target: Target,
    budget: int,
    *,
    settings: Settings | None = None,
    board: Board | None = None,
    query_fn: Callable[..., object] | None = None,
) -> ArmOutcome:
    """Let a model spend the same budget, choosing for itself."""
    tools = CoveringTools(
        parameters=target.parameters,
        published=target.published,
        lower_bound=target.lower_bound,
        budget=budget,
        provenance=target.provenance,
    )
    session = CoveringSession(
        tools=tools,
        cell=target.label,
        settings=settings or Settings(budget=budget),
        board=board,
        query_fn=query_fn,
    )
    result = await session.run_async(target.instruction())
    banked = result.banked
    return ArmOutcome(
        label=target.label,
        kind="agent",
        moves=tools.used,
        uncovered=max(result.best_uncovered, 0),
        blocks=banked.blocks if banked is not None else (),
    )


@dataclass(frozen=True, slots=True)
class CampaignReport:
    """Both halves of the comparison, side by side."""

    outcomes: tuple[ArmOutcome, ...] = field(default_factory=tuple)

    def of(self, kind: str) -> tuple[ArmOutcome, ...]:
        """Return the outcomes of one arm kind."""
        return tuple(outcome for outcome in self.outcomes if outcome.kind == kind)

    def render(self) -> str:
        """Render the comparison, improvements first."""
        agents, controls = self.of("agent"), self.of("control")
        wins = tuple(outcome for outcome in self.outcomes if outcome.improved)
        agent_line = (
            f"  agent   moves {sum(o.moves for o in agents):,}, "
            f"improvements {sum(1 for o in agents if o.improved)}"
        )
        control_line = (
            f"  control moves {sum(o.moves for o in controls):,}, "
            f"improvements {sum(1 for o in controls if o.improved)}"
        )
        headline = (
            f"{len(agents)} agent arms, {len(controls)} control arms, {len(wins)} improvement(s)"
        )
        lines = [headline, agent_line, control_line]
        lines.extend(
            outcome.render()
            for outcome in sorted(self.outcomes, key=lambda o: (not o.improved, o.label, o.kind))
        )
        return "\n".join(lines)


async def run_campaign_async(
    targets: tuple[Target, ...],
    budget: int,
    *,
    concurrency: int = 4,
    settings: Settings | None = None,
    board: Board | None = None,
    query_fn: Callable[..., object] | None = None,
) -> CampaignReport:
    """Run an agent arm and a matched control arm for every target.

    One arm's failure never cancels the fan-out: a transport error is not a
    search result, and a campaign that discards every other arm because one
    connection dropped has measured nothing.
    """
    if concurrency < 1:
        message = f"concurrency must be positive, got {concurrency}"
        raise ValueError(message)
    limit = asyncio.Semaphore(concurrency)

    async def one_agent(target: Target) -> ArmOutcome:
        async with limit:
            try:
                return await agent_arm(
                    target, budget, settings=settings, board=board, query_fn=query_fn
                )
            except Exception:  # noqa: BLE001 - a dropped transport is not a result
                return ArmOutcome(
                    label=target.label,
                    kind="agent",
                    moves=0,
                    uncovered=target.parameters.requirements,
                )

    async def one_control(target: Target, seed: int) -> ArmOutcome:
        async with limit:
            return await asyncio.to_thread(control_arm, target, budget, seed=seed)

    jobs = [one_agent(target) for target in targets]
    jobs += [one_control(target, index) for index, target in enumerate(targets)]
    return CampaignReport(outcomes=tuple(await asyncio.gather(*jobs)))
