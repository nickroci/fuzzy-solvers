"""A model-driven search session for ``R(3, k)`` lower-bound constructions.

This is the fuzzy half.  A Claude session holds the search: it picks which
group to build over, which connection set to try, and what to do with the
exact feedback the referee returns.  Nothing in this package proposes a
construction on the model's behalf, so a banked witness is model-authored by
construction, and the kernel ledger in :class:`~ramsey.tools.RamseyTools`
makes the cost directly comparable to a mechanical sweep.

The session runs over the Claude Agent SDK against the Claude Code
subscription, matching :mod:`evals.subscription_backend`: use it for
development iteration, and the metered backend for measured runs.

Feedback is deliberately exact and neutral.  When a construction fails, the
model is told the size of the independent set that beat it and which vertices
form it — never a score, a ranking, or a hint about what to try next.  What to
do with an obstruction is the search judgement being measured.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

from ramsey.graph import parse_graph6
from ramsey.tools import BudgetExhaustedError, RamseyTarget, RamseyTools

if TYPE_CHECKING:
    from collections.abc import Callable

    from ramsey.blackboard import Blackboard

_SERVER_NAME = "ramsey"
_MAX_INSTRUCTION_CHARS = 8_000

SYSTEM_PROMPT = """\
You are searching for an explicit graph that improves a published lower bound \
on a classical Ramsey number R(3,k). The entire deliverable is one graph.

Rules that are enforced mechanically, not by you:
- A referee measures every construction exactly. You never assert that a graph \
works; you build it and read the measurement.
- Every evaluation costs one kernel call from a fixed budget. Spend it on \
structural judgement, not on enumeration you could have reasoned about.
- A construction is triangle-free exactly when its connection set is sum-free \
in the group: no two members, equal members allowed, may sum to a member. \
Check this yourself before spending a call; the referee will reject it \
otherwise and you will have paid for nothing.
- In a triangle-free graph every neighbourhood is independent, so the degree \
can never exceed k-1. For a Cayley graph the degree is exactly the connection \
set size.

When a construction is triangle-free but fails, you are told one independent \
set that exceeded the cap. That set is the obstruction: read its structure and \
change the construction to break it. Repeating a near-identical connection set \
wastes budget.

A set is independent exactly when none of its pairwise differences lies in \
the connection set. So the only generators that can destroy an obstruction are \
the differences within it: analyse() computes that set and tells you which of \
them keep the construction legal. Adding anything else leaves the obstruction \
standing and wastes the call. When analyse reports no legal repair, the \
connection set is trapped and you must drop a generator or change group.

Call read_blackboard() before anything else. Other arms are searching the \
same target in parallel and record what they tried and what it reached. Your \
job is to try something they have not, or to press where they got closest - \
not to rediscover their dead ends.

Your budget is thousands of graph evaluations, not dozens. Spend it through \
commission_search: name a group and a structural hypothesis, charge it tens of \
thousands of restarts, and read back the best independence number reached. \
Evaluating one graph at a time wastes the budget by three orders of magnitude. \
Use evaluate_cayley only to check one specific construction you have a reason \
to name, and analyse/refine only to interrogate a construction that a search \
already brought close to the cap.

Three construction languages are available, in increasing generality. \
commission_search builds Cayley graphs: maximally symmetric, fastest to \
evaluate, and the family that produced most published bounds. commission_lift \
builds voltage-graph lifts, which contain Cayley graphs as the base_size=1 \
case and reach graphs no Cayley construction can. commission_repair edits \
individual edges of the best construction so far, leaving symmetry behind \
entirely - this is the only way to reach a witness that is not algebraic at \
all, and it needs a seed that is already close.

A known case worth learning from: R(3,6) > 17 is true, but no Cayley graph on \
17 vertices witnesses it - the best circulant reaches independence 6 against a \
cap of 5. Repairing that circulant seed edge by edge reaches a witness in \
seconds. When a Cayley search plateaus one or two above the cap, that is the \
signal to repair, not to keep searching the same family.

The gap the search reports - how far the best independence number sits above \
the cap - is your signal. If a family is far above, abandon it and try a \
different group, degree or hypothesis. If it is close, press there.

You have several ways to spend the budget and the choice is yours. A fresh \
evaluate_cayley is a cold shot at a new region. analyse() spends one call to \
tell you why the current construction fails and what could fix it. refine() \
walks the neighbourhood, one generator per step. All three cost one call per \
evaluation, so they are priced against each other: analyse before refining a \
construction you intend to keep, refine when one is close, and start fresh \
when it is not.

Groups are given as a tuple of moduli: (111,) is cyclic Z111, (3,37) is \
Z3 x Z37, (2,2,2,2) is the elementary abelian group of order 16. The product \
of the moduli must equal the required vertex count. Non-cyclic groups are \
often where the good constructions live, because the published cyclic ones \
have largely been exhausted.

Call bank() the moment a construction is accepted. Then stop.\
"""


@dataclass(frozen=True, slots=True)
class RamseySessionSettings:
    """Transport and budget settings for one search session."""

    model: str = "claude-sonnet-5"
    max_turns: int = 60
    max_tool_calls: int = 80
    kernel_budget: int = 60


@dataclass(frozen=True, slots=True)
class SessionToolCall:
    """One recorded model tool call, for provenance."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RamseySessionResult:
    """The outcome of one model-driven search session."""

    status: Literal["banked", "budget", "incomplete"]
    target: RamseyTarget
    kernel_calls: int
    tool_calls: int
    turns: int
    output_tokens: int
    witness_graph6: str = ""
    independence_number: int | None = None
    best_independence: int | None = None
    best_generators: tuple[int, ...] = ()
    tool_log: tuple[SessionToolCall, ...] = ()
    stop_reason: str | None = None

    @property
    def improved(self) -> bool:
        """Report whether the session banked a referee-accepted new witness."""
        return self.status == "banked"


@dataclass(slots=True)
class RamseyToolRouter:
    """Route model tool calls into the exact tool surface."""

    tools: RamseyTools
    max_tool_calls: int
    calls: int = field(default=0, init=False)
    log: list[SessionToolCall] = field(default_factory=list, init=False)
    budget_exhausted: bool = field(default=False, init=False)

    def handle(self, name: str, payload: dict[str, Any]) -> tuple[str, bool]:
        """Dispatch one tool call, returning rendered content and an ok flag."""
        if self.calls >= self.max_tool_calls:
            self.budget_exhausted = True
            return "tool call budget exhausted; stop now", False
        self.calls += 1
        try:
            content, ok = self._dispatch(name, payload)
        except BudgetExhaustedError as error:
            self.budget_exhausted = True
            content, ok = str(error), False
        except (KeyError, ValueError, TypeError) as error:
            content, ok = f"{type(error).__name__}: {error}", False
        self.log.append(SessionToolCall(name=name, ok=ok, detail=content[:400]))
        return content, ok

    def _dispatch(self, name: str, payload: dict[str, Any]) -> tuple[str, bool]:
        handlers: dict[str, Callable[[dict[str, Any]], tuple[str, bool]]] = {
            "describe_target": lambda _: (self.tools.describe_target(), True),
            "status": lambda _: (self.tools.status(), True),
            "read_blackboard": lambda _: (self.tools.read_blackboard(), True),
            "commission_search": self._commission,
            "commission_lift": self._commission_lift,
            "commission_repair": self._commission_repair,
            "analyse": self._analyse,
            "evaluate_cayley": self._evaluate_cayley,
            "evaluate_graph6": self._evaluate_graph6,
            "refine": self._refine,
            "bank": lambda p: (self.tools.bank(str(p["handle"])), True),
        }
        handler = handlers.get(name)
        if handler is None:
            message = f"unknown tool {name!r}"
            raise KeyError(message)
        return handler(payload)

    def _commission(self, payload: dict[str, Any]) -> tuple[str, bool]:
        outcome = self.tools.commission_search(
            tuple(int(v) for v in payload["moduli"]),
            int(payload["degree"]),
            int(payload["restarts"]),
            int(payload["steps"]),
            seed=int(payload.get("seed", 0)),
            required=tuple(int(v) for v in payload.get("required", ())),
            forbidden=tuple(int(v) for v in payload.get("forbidden", ())),
        )
        return outcome.render() + f"\n{self.tools.status()}", True

    def _commission_lift(self, payload: dict[str, Any]) -> tuple[str, bool]:
        outcome = self.tools.commission_lift(
            int(payload["base_size"]),
            tuple(int(v) for v in payload["moduli"]),
            int(payload["degree"]),
            int(payload["restarts"]),
            seed=int(payload.get("seed", 0)),
        )
        return outcome.render() + f"\n{self.tools.status()}", True

    def _commission_repair(self, payload: dict[str, Any]) -> tuple[str, bool]:
        outcome = self.tools.commission_repair(
            int(payload["steps"]),
            int(payload["restarts"]),
            seed=int(payload.get("seed", 0)),
        )
        return outcome.render() + f"\n{self.tools.status()}", True

    def _analyse(self, payload: dict[str, Any]) -> tuple[str, bool]:
        analysis = self.tools.analyse(str(payload["handle"]))
        return analysis.render() + f"\n{self.tools.status()}", True

    def _evaluate_cayley(self, payload: dict[str, Any]) -> tuple[str, bool]:
        evaluation = self.tools.evaluate_cayley(
            tuple(int(v) for v in payload["moduli"]),
            tuple(int(v) for v in payload["generators"]),
        )
        return evaluation.render() + f"\n{self.tools.status()}", not evaluation.rejected_reason

    def _evaluate_graph6(self, payload: dict[str, Any]) -> tuple[str, bool]:
        graph = parse_graph6(str(payload["graph6"]))
        evaluation = self.tools.evaluate_graph(str(payload.get("label", "explicit")), graph)
        return evaluation.render() + f"\n{self.tools.status()}", not evaluation.rejected_reason

    def _refine(self, payload: dict[str, Any]) -> tuple[str, bool]:
        refinement = self.tools.refine(
            str(payload["handle"]),
            str(payload["move_class"]),
            int(payload["steps"]),
            seed=int(payload.get("seed", 0)),
        )
        return refinement.render() + f"\n{self.tools.status()}", True


_TOOL_SPECS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("describe_target", "State the target bound and its structural constraints.", {}),
    ("status", "Report kernel calls spent, candidates archived, and the incumbent.", {}),
    (
        "evaluate_cayley",
        (
            "Build and exactly measure a Cayley graph. Costs one kernel call. "
            "moduli is the group as a list of factors whose product is the required "
            "vertex count; generators is a list of element indices, closed under "
            "negation automatically."
        ),
        {"moduli": list, "generators": list},
    ),
    (
        "evaluate_graph6",
        "Exactly measure an explicit graph given in graph6. Costs one kernel call.",
        {"graph6": str, "label": str},
    ),
    (
        "read_blackboard",
        (
            "Report every attempt any arm has made on this target: which family, "
            "which parameters, how far above the cap it reached, and at what cost. "
            "READ THIS FIRST. Other arms are working the same target in parallel, "
            "and repeating a family that already plateaued wastes your entire budget."
        ),
        {},
    ),
    (
        "commission_search",
        (
            "Commission a mechanical search and read what it found. THIS IS WHERE "
            "THE BUDGET SHOULD GO. You choose the group (moduli), the degree, how "
            "many restarts and hill-climbing steps, and optionally generators to "
            "require or forbid - that is your structural hypothesis. The kernel then "
            "runs random-restart hill-climbing at full speed and reports the best "
            "independence number it reached and the generators that achieved it. One "
            "call buys thousands of evaluations; proposing graphs one at a time buys "
            "one. Charge a large restart and step count. A single call is capped at "
            "about a third of your remaining budget, so you get several distinct "
            "hypotheses - use them on different groups, degrees and constraints "
            "rather than repeating one."
        ),
        {
            "moduli": list,
            "degree": int,
            "restarts": int,
            "steps": int,
            "seed": int,
            "required": list,
            "forbidden": list,
        },
    ),
    (
        "commission_lift",
        (
            "Commission a search over voltage-graph LIFTS, which strictly contain "
            "Cayley graphs. A lift takes a base graph on base_size vertices, a group, "
            "and voltages on the base edges; base_size times the group order must "
            "equal the target vertex count. base_size=1 is exactly a Cayley graph; "
            "larger bases reach graphs NO Cayley construction can; base_size equal to "
            "the order with moduli [1] is an unconstrained graph search. Use this when "
            "a Cayley search has plateaued."
        ),
        {"base_size": int, "moduli": list, "degree": int, "restarts": int, "seed": int},
    ),
    (
        "commission_repair",
        (
            "Repair the best construction found so far by editing individual edges, "
            "leaving the symmetric space entirely. An oversized independent set can "
            "only be destroyed by adding an edge inside it, so that is what the repair "
            "does, deleting elsewhere when the degree cap blocks it. This is how you "
            "reach a witness that is not a Cayley graph or a lift at all. Commission a "
            "search first to supply the seed; repair works best from a seed already "
            "within a few of the cap."
        ),
        {"steps": int, "restarts": int, "seed": int},
    ),
    (
        "analyse",
        (
            "Measure the structure of the independent set defeating a construction: "
            "its size, whether it is a coset, and exactly which generators would "
            "destroy it while keeping the connection set sum-free and within the "
            "degree cap. Costs one kernel call."
        ),
        {"handle": str},
    ),
    (
        "refine",
        (
            "Hill-climb from an archived candidate. You choose the starting handle, "
            "the move class and how many steps to walk. Move classes: 'swap' one "
            "generator, 'add' one, 'drop' one, or 'break', which tries only the "
            "generators that provably destroy the current obstruction. Each step costs "
            "one kernel call, so a 10-step refine costs "
            "the same as 10 fresh evaluations: spend refines where a construction is "
            "already close, and fresh evaluations where it is not."
        ),
        {"handle": str, "move_class": str, "steps": int, "seed": int},
    ),
    ("bank", "Promote a referee-accepted candidate to the incumbent witness.", {"handle": str}),
)


def _bind(name: str, description: str, schema: dict[str, Any], router: RamseyToolRouter) -> Any:  # noqa: ANN401
    """Bind one router method as an in-process MCP tool.

    The kernel work is CPU-bound and can run for minutes, so it is pushed to a
    worker thread.  Running it inline would block the event loop, which in a
    fan-out means one arm's search starves every other arm's session until
    they time out - the concurrency would be nominal only.
    """

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        content, ok = await asyncio.to_thread(router.handle, name, args)
        return {"content": [{"type": "text", "text": content}], "isError": not ok}

    return tool(name, description, schema)(handler)


class RamseyAgentSession:
    """Drive one model-authored search over the exact tool surface."""

    def __init__(
        self,
        *,
        target: RamseyTarget,
        settings: RamseySessionSettings | None = None,
        query_fn: Callable[..., Any] | None = None,
        blackboard: Blackboard | None = None,
    ) -> None:
        self._settings = settings or RamseySessionSettings()
        self._tools = RamseyTools(
            target=target, kernel_budget=self._settings.kernel_budget, blackboard=blackboard
        )
        self._router = RamseyToolRouter(
            tools=self._tools, max_tool_calls=self._settings.max_tool_calls
        )
        self._query = query_fn or query

    @property
    def tools(self) -> RamseyTools:
        """Expose the tool surface for inspection after a run."""
        return self._tools

    def run(self, instruction: str) -> RamseySessionResult:
        """Run the session synchronously and return its outcome."""
        return asyncio.run(self.run_async(instruction))

    async def run_async(self, instruction: str) -> RamseySessionResult:
        """Run the session on the caller's event loop, for parallel fan-out."""
        if not instruction.strip() or len(instruction) > _MAX_INSTRUCTION_CHARS:
            message = f"instruction must contain 1-{_MAX_INSTRUCTION_CHARS} characters"
            raise ValueError(message)
        return await self._run(instruction)

    async def _run(self, instruction: str) -> RamseySessionResult:
        sdk_tools = [_bind(name, doc, schema, self._router) for name, doc, schema in _TOOL_SPECS]
        server = create_sdk_mcp_server(name=_SERVER_NAME, tools=sdk_tools)
        options = ClaudeAgentOptions(
            model=self._settings.model,
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            mcp_servers={_SERVER_NAME: server},
            strict_mcp_config=True,
            allowed_tools=[f"mcp__{_SERVER_NAME}__{name}" for name, _, _ in _TOOL_SPECS],
            max_turns=self._settings.max_turns,
            permission_mode="dontAsk",
            setting_sources=[],
            skills=[],
        )
        prompt = (
            f"<objective>\n{instruction}\n</objective>\n\n"
            f"<target>\n{self._tools.describe_target()}\n</target>"
        )
        turns = 0
        outcome: ResultMessage | None = None
        async for message in self._query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                turns += 1
            elif isinstance(message, ResultMessage):
                outcome = message
        return self._result(turns=turns, outcome=outcome)

    def _result(self, *, turns: int, outcome: ResultMessage | None) -> RamseySessionResult:
        banked = self._tools.banked
        if banked is not None:
            status: Literal["banked", "budget", "incomplete"] = "banked"
        elif self._router.budget_exhausted:
            status = "budget"
        else:
            status = "incomplete"
        usage = cast("dict[str, Any]", getattr(outcome, "usage", None) or {})
        return RamseySessionResult(
            status=status,
            target=self._tools.target,
            kernel_calls=self._tools.calls_used,
            tool_calls=self._router.calls,
            turns=turns,
            output_tokens=int(usage.get("output_tokens", 0)),
            witness_graph6=self._tools.witness_graph6() if banked is not None else "",
            independence_number=(None if banked is None else banked.evaluation.independence_number),
            tool_log=tuple(self._router.log),
            stop_reason=getattr(outcome, "stop_reason", None),
        )


def render_result(result: RamseySessionResult) -> str:
    """Render a session outcome as a short report."""
    headline = (
        f"status={result.status} kernel_calls={result.kernel_calls}, "
        f"{result.tool_calls} tool calls, {result.turns} turns, "
        f"{result.output_tokens} output tokens"
    )
    lines = [headline]
    if result.best_independence is not None:
        lines.append(
            f"best independence reached: {result.best_independence} "
            f"(cap {result.target.independence_cap}) via {result.best_generators}"
        )
    if result.improved:
        lines.append(
            f"R(3,{result.target.second_colour}) > {result.target.order} "
            f"(published bound was {result.target.record}); "
            f"independence number {result.independence_number}"
        )
        lines.append(f"witness graph6: {result.witness_graph6}")
    return "\n".join(lines)
