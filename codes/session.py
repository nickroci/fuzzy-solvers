"""A model-driven search for a new best-known binary linear code.

A Claude session holds the search: it reads what other arms found, chooses
which quasi-cyclic shapes and structural constraints to commission, and
decides where to press. Nothing here proposes a construction on the model's
behalf, so a banked code is model-authored, and the ledger makes its cost
directly comparable to a matched mechanical arm.
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

from codes.board import BRIEFING
from codes.tools import Banked, BudgetExhaustedError, CodeTools

if TYPE_CHECKING:
    from collections.abc import Callable

    from codes.board import Board
    from codes.tables import Cell

_SERVER = "codes"

SYSTEM_PROMPT = f"""\
You are searching for a binary linear code that beats a published bound. The
entire deliverable is a handful of generator polynomials.

Rules enforced mechanically, not by you:
- A referee measures every code exactly. You never assert that a code works.
- Every code evaluated costs one call from a fixed budget. A commissioned
  search buys thousands per call; evaluating candidates one at a time buys one.
- A single commissioned search is capped at about a third of your remaining
  budget, so you get several distinct hypotheses. Use them on genuinely
  different structures, not on the same shape at a larger size.

Call read_board() first. Other arms are searching the same cell and record
what they tried and what distance it reached. Try what they have not.

{BRIEFING}
Bank happens automatically when a code clears the bound. Then stop.\
"""


@dataclass(frozen=True, slots=True)
class Settings:
    """Transport and budget settings for one session."""

    model: str = "claude-sonnet-5"
    max_turns: int = 60
    max_tool_calls: int = 45
    budget: int = 400_000


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One recorded model tool call, for provenance."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SessionResult:
    """The outcome of one model-driven search."""

    status: Literal["banked", "budget", "incomplete"]
    cell: Cell
    evaluations: int
    tool_calls: int
    turns: int
    output_tokens: int
    best_distance: int = 0
    banked: Banked | None = None
    log: tuple[ToolCall, ...] = ()

    @property
    def improved(self) -> bool:
        """Report whether the session banked a verified new entry."""
        return self.banked is not None

    def render(self) -> str:
        """Render the outcome as a short report."""
        head = (
            f"status={self.status} evaluations={self.evaluations:,} "
            f"tool_calls={self.tool_calls} turns={self.turns} "
            f"tokens={self.output_tokens} best_d={self.best_distance} "
            f"(published {self.cell.lower}, need {self.cell.target})"
        )
        if self.banked is not None:
            return f"{head}\n*** {self.banked.render()}"
        return head


@dataclass(slots=True)
class Router:
    """Route model tool calls into the exact tool surface."""

    tools: CodeTools
    max_tool_calls: int
    calls: int = field(default=0, init=False)
    log: list[ToolCall] = field(default_factory=list, init=False)
    exhausted: bool = field(default=False, init=False)

    def handle(self, name: str, payload: dict[str, Any]) -> tuple[str, bool]:
        """Dispatch one tool call, returning rendered content and an ok flag."""
        if self.calls >= self.max_tool_calls:
            self.exhausted = True
            return "tool call budget exhausted; stop now", False
        self.calls += 1
        try:
            content, ok = self._dispatch(name, payload)
        except BudgetExhaustedError as error:
            self.exhausted = True
            content, ok = str(error), False
        except (KeyError, ValueError, TypeError) as error:
            content, ok = f"{type(error).__name__}: {error}", False
        self.log.append(ToolCall(name=name, ok=ok, detail=content[:400]))
        return content, ok

    def _dispatch(self, name: str, payload: dict[str, Any]) -> tuple[str, bool]:
        handlers: dict[str, Callable[[dict[str, Any]], tuple[str, bool]]] = {
            "describe_target": lambda _: (self.tools.describe_target(), True),
            "read_board": lambda _: (self.tools.read_board(), True),
            "status": lambda _: (self.tools.status(), True),
            "commission_search": self._commission,
            "evaluate": self._evaluate,
        }
        handler = handlers.get(name)
        if handler is None:
            message = f"unknown tool {name!r}"
            raise KeyError(message)
        return handler(payload)

    def _commission(self, payload: dict[str, Any]) -> tuple[str, bool]:
        outcome = self.tools.commission_search(
            int(payload["index"]),
            int(payload["restarts"]),
            int(payload["steps"]),
            seed=int(payload.get("seed", 0)),
            weight=int(payload.get("weight", 0)),
        )
        return outcome.render() + f"\n{self.tools.status()}", True

    def _evaluate(self, payload: dict[str, Any]) -> tuple[str, bool]:
        content = self.tools.evaluate(tuple(int(v) for v in payload["polynomials"]))
        return content + f"\n{self.tools.status()}", not content.startswith("rejected")


_SPECS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("describe_target", "State the cell, its published bounds and what a new entry needs.", {}),
    (
        "read_board",
        (
            "Report every attempt any arm has made on this cell: the shape tried, the "
            "distance reached, and the cost. READ THIS FIRST - repeating a shape that "
            "already plateaued wastes your whole budget."
        ),
        {},
    ),
    ("status", "Report budget used and the best distance reached.", {}),
    (
        "commission_search",
        (
            "Commission a mechanical search over quasi-cyclic codes and read what it "
            "found. THIS IS WHERE THE BUDGET GOES. You choose the index (number of "
            "generator polynomials), how many restarts and hill-climbing steps, a seed, "
            "and optionally a fixed weight so every polynomial is drawn with exactly "
            "that many terms. One call buys thousands of codes."
        ),
        {"index": int, "restarts": int, "steps": int, "seed": int, "weight": int},
    ),
    (
        "evaluate",
        (
            "Measure one specific quasi-cyclic code exactly. Pass the generator "
            "polynomials as integers, one per block, each a bitmask whose bit i means "
            "the term x^i. Costs one evaluation. Use this for a construction you have "
            "a specific algebraic reason to name."
        ),
        {"polynomials": list},
    ),
)


def _bind(name: str, description: str, schema: dict[str, Any], router: Router) -> Any:  # noqa: ANN401
    """Bind one router method as an in-process MCP tool.

    The kernel work is CPU-bound and can run for minutes, so it is pushed to a
    worker thread; running it inline would block the event loop and starve
    every other arm in a fan-out.
    """

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        content, ok = await asyncio.to_thread(router.handle, name, args)
        return {"content": [{"type": "text", "text": content}], "isError": not ok}

    return tool(name, description, schema)(handler)


class CodeSession:
    """Drive one model-authored search over the exact tool surface."""

    def __init__(
        self,
        *,
        cell: Cell,
        settings: Settings | None = None,
        board: Board | None = None,
        query_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._tools = CodeTools(cell=cell, budget=self._settings.budget, board=board)
        self._router = Router(tools=self._tools, max_tool_calls=self._settings.max_tool_calls)
        self._query = query_fn or query

    @property
    def tools(self) -> CodeTools:
        """Expose the tool surface for inspection after a run."""
        return self._tools

    def run(self, instruction: str) -> SessionResult:
        """Run the session synchronously."""
        return asyncio.run(self.run_async(instruction))

    async def run_async(self, instruction: str) -> SessionResult:
        """Run on the caller's event loop, for parallel fan-out."""
        if not instruction.strip():
            message = "instruction must not be empty"
            raise ValueError(message)
        sdk_tools = [_bind(n, d, s, self._router) for n, d, s in _SPECS]
        options = ClaudeAgentOptions(
            model=self._settings.model,
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            mcp_servers={_SERVER: create_sdk_mcp_server(name=_SERVER, tools=sdk_tools)},
            strict_mcp_config=True,
            allowed_tools=[f"mcp__{_SERVER}__{n}" for n, _, _ in _SPECS],
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
        stream = cast("Any", self._query(prompt=prompt, options=options))
        async for message in stream:
            if isinstance(message, AssistantMessage):
                turns += 1
            elif isinstance(message, ResultMessage):
                outcome = message
        return self._result(turns=turns, outcome=outcome)

    def _result(self, *, turns: int, outcome: ResultMessage | None) -> SessionResult:
        banked = self._tools.banked
        if banked is not None:
            status: Literal["banked", "budget", "incomplete"] = "banked"
        elif self._router.exhausted:
            status = "budget"
        else:
            status = "incomplete"
        usage = getattr(outcome, "usage", None) or {}
        return SessionResult(
            status=status,
            cell=self._tools.cell,
            evaluations=self._tools.used,
            tool_calls=self._router.calls,
            turns=turns,
            output_tokens=int(usage.get("output_tokens", 0)),
            best_distance=self._tools.best_distance,
            banked=banked,
            log=tuple(self._router.log),
        )
