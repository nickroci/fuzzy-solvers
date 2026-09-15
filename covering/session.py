"""A model-driven search for a covering design that beats a published entry.

A Claude session holds the search: it reads what other arms found, names the
group to build in, checks the arithmetic before spending anything, and reads
the structure of what its design is missing before deciding where to press.
Nothing here proposes a construction on the model's behalf, so a banked design
is model-authored, and the ledger makes its cost directly comparable to a
matched mechanical arm.
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

from covering.board import BRIEFING, Attempt
from covering.kernel import mask
from covering.tools import Banked, BudgetExhaustedError, CoveringTools

if TYPE_CHECKING:
    from collections.abc import Callable

    from covering.board import Board
    from covering.orbit import PointGroup

_SERVER = "covering"
_LOG_WIDTH = 400
_FREE_CALL_MULTIPLE = 4
FREE_TOOLS = frozenset(
    {"describe_target", "read_board", "status", "inspect", "check_feasible", "adopt"}
)
"""Tools that spend no moves, so they must not spend the scarcer allowance either."""

SYSTEM_PROMPT = f"""\
You are searching for a covering design that beats a published entry. The
entire deliverable is a handful of base blocks and the group to develop them
under.

Rules enforced mechanically, not by you:
- A referee checks every t-subset. You never assert that a design works.
- check_feasible costs nothing. Call it before every search. A group that
  cannot express your target block count will never produce it.
- Search spends moves from a fixed budget, and one call is capped at about a
  third of what remains, so you get several distinct hypotheses. Spend them on
  genuinely different algebra, not on the same group with another seed.
- When a search falls short, call inspect() before searching again. It tells
  you whether the deficit is one orbit (wrong stabiliser) or scattered (short
  of blocks). Choosing without reading it is guessing.

Call read_board() first. Other arms are working the same cell and record what
they tried and how close they got. Try what they have not.

{BRIEFING}
Bank happens automatically when a design clears the published count. Then stop.\
"""


@dataclass(frozen=True, slots=True)
class Settings:
    """Transport and budget settings for one session."""

    model: str = "claude-sonnet-5"
    max_turns: int = 60
    max_tool_calls: int = 45
    budget: int = 400_000_000


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
    cell: str
    published: int
    moves: int
    tool_calls: int
    turns: int
    output_tokens: int
    best_uncovered: int = -1
    banked: Banked | None = None
    log: tuple[ToolCall, ...] = ()

    @property
    def improved(self) -> bool:
        """Report whether the session banked a verified new entry."""
        return self.banked is not None

    def render(self) -> str:
        """Render the outcome as a short report."""
        reached = "n/a" if self.best_uncovered < 0 else str(self.best_uncovered)
        head = (
            f"status={self.status} cell={self.cell} moves={self.moves:,} "
            f"tool_calls={self.tool_calls} turns={self.turns} "
            f"tokens={self.output_tokens} best_uncovered={reached} "
            f"(published {self.published})"
        )
        if self.banked is not None:
            return f"{head}\n*** {self.banked.render()}"
        return head


@dataclass(slots=True)
class Router:
    """Route model tool calls into the exact tool surface."""

    tools: CoveringTools
    cell: str
    max_tool_calls: int
    board: Board | None = None
    calls: int = field(default=0, init=False)
    free_calls: int = field(default=0, init=False)
    log: list[ToolCall] = field(default_factory=list, init=False)
    exhausted: bool = field(default=False, init=False)

    def handle(self, name: str, payload: dict[str, Any]) -> tuple[str, bool]:
        """Dispatch one tool call, returning rendered content and an ok flag.

        A tool that spends no moves does not spend the call allowance either.
        check_feasible is advertised as costing nothing and a live session took
        that at its word: it spent sixteen of its thirty calls ruling groups in
        and out, was left with eight searches, never reached the design that
        works, and finished with most of its move budget unspent. Charging the
        scarcest resource for a call described as free is the surface lying.
        """
        free = name in FREE_TOOLS
        if free:
            if self.free_calls >= self.max_tool_calls * _FREE_CALL_MULTIPLE:
                return "free-call allowance exhausted; search or stop", False
            self.free_calls += 1
        else:
            if self.calls >= self.max_tool_calls:
                self.exhausted = True
                return "search budget exhausted; stop now", False
            self.calls += 1
        try:
            content, ok = self._dispatch(name, payload)
        except BudgetExhaustedError as error:
            self.exhausted = True
            content, ok = str(error), False
        except (KeyError, ValueError, TypeError) as error:
            content, ok = f"{type(error).__name__}: {error}", False
        self.log.append(ToolCall(name=name, ok=ok, detail=content[:_LOG_WIDTH]))
        return content, ok

    def _dispatch(self, name: str, payload: dict[str, Any]) -> tuple[str, bool]:
        handlers: dict[str, Callable[[dict[str, Any]], tuple[str, bool]]] = {
            "describe_target": lambda _: (self.tools.describe_target(), True),
            "read_board": lambda _: (self._read_board(), True),
            "status": lambda _: (self.tools.status(), True),
            "inspect": lambda _: (self.tools.inspect(), True),
            "check_feasible": self._feasible,
            "search": self._search,
            "resume": self._resume,
            "adopt": self._adopt,
        }
        handler = handlers.get(name)
        if handler is None:
            message = f"unknown tool {name!r}"
            raise KeyError(message)
        return handler(payload)

    def _read_board(self) -> str:
        if self.board is None:
            return "No board is attached to this run."
        return self.board.render(self.cell, target=self.tools.target_blocks)

    def _group(self, payload: dict[str, Any]) -> PointGroup:
        """Build the group the model named, in whichever way it named it."""
        family = str(payload.get("family", "cyclic"))
        if family == "trivial":
            return self.tools.trivial_group()
        if family == "grid":
            return self.tools.grid_group(int(payload["rows"]), int(payload["columns"]))
        if family == "authored":
            cycles = tuple(tuple(int(p) for p in cycle) for cycle in payload["cycles"])
            return self.tools.authored_group(cycles, str(payload.get("name", "authored")))
        return self.tools.named_group(family, int(payload.get("argument", 0)))

    def _feasible(self, payload: dict[str, Any]) -> tuple[str, bool]:
        group = self._group(payload)
        target = int(payload.get("target", self.tools.target_blocks))
        return self.tools.check_feasible(group, target).render(), True

    def _note(self, group: PointGroup, base_blocks: int) -> None:
        """Record what was tried, so the next arm does not repeat it."""
        outcome = self.tools.last_outcome
        if self.board is None or outcome is None:
            return
        self.board.record(
            Attempt(
                cell=self.cell,
                group=group.name,
                base_blocks=base_blocks,
                blocks=outcome.block_count,
                uncovered=outcome.uncovered,
                moves=self.tools.used,
            )
        )

    def _search(self, payload: dict[str, Any]) -> tuple[str, bool]:
        group = self._group(payload)
        base_count = int(payload["base_count"])
        moves = int(payload["moves"])
        content = self.tools.search(group, base_count, moves, seed=int(payload.get("seed", 0)))
        self._note(group, base_count)
        return content + f"\n{self.tools.status()}", True

    def _resume(self, payload: dict[str, Any]) -> tuple[str, bool]:
        content = self.tools.resume(int(payload["moves"]), seed=int(payload.get("seed", 0)))
        return content + f"\n{self.tools.status()}", True

    def _adopt(self, payload: dict[str, Any]) -> tuple[str, bool]:
        group = self._group(payload)
        blocks = tuple(
            mask(tuple(int(point) for point in block)) for block in payload["base_blocks"]
        )
        content = self.tools.adopt(group, blocks)
        self._note(group, len(blocks))
        return content + f"\n{self.tools.status()}", True


_GROUP_SCHEMA: dict[str, Any] = {
    "family": str,
    "argument": int,
    "rows": int,
    "columns": int,
    "cycles": list,
    "name": str,
}

_SPECS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("describe_target", "State the cell, its published entry and what a new one needs.", {}),
    (
        "read_board",
        (
            "Report every attempt any arm has made on this cell: the group tried, how "
            "many base blocks, and how many requirements were left uncovered. READ "
            "THIS FIRST - repeating a group that already stalled wastes your budget."
        ),
        {},
    ),
    ("status", "Report the moves spent and the best coverage reached.", {}),
    (
        "check_feasible",
        (
            "Report whether a group can express a block count at all. COSTS NOTHING, "
            "so call it before every search. Name the group by family: 'cyclic', "
            "'cyclic_fixed' (argument = how many points to hold fixed), 'multiplier' "
            "(argument = a unit mod v), 'frobenius' (argument = multiplier order), "
            "'grid' (rows and columns), 'trivial', or 'authored' with cycles as a list "
            "of lists of points."
        ),
        _GROUP_SCHEMA | {"target": int},
    ),
    (
        "search",
        (
            "Develop base_count random base blocks under a group and anneal them for "
            "the given number of moves, `restarts` times from different random "
            "starts. THIS IS WHERE THE BUDGET GOES. Set restarts high: annealing "
            "finds a design on a minority of attempts however long each runs, so ten "
            "restarts of ten million moves beats one of a hundred million. The group "
            "is named exactly as in check_feasible."
        ),
        _GROUP_SCHEMA | {"base_count": int, "moves": int, "seed": int, "restarts": int},
    ),
    (
        "resume",
        (
            "Anneal further from the best design reached so far, at a lower starting "
            "temperature. Use this to finish a design that is close, rather than "
            "starting a fresh search that throws it away."
        ),
        {"moves": int, "seed": int},
    ),
    (
        "inspect",
        (
            "Report WHAT the best design is missing: how many orbits the uncovered "
            "requirements form under your group, a representative one, and which "
            "points carry the shortfall. A single orbit means the algebra is wrong, "
            "not the luck. Costs nothing."
        ),
        {},
    ),
    (
        "adopt",
        (
            "Take base blocks you wrote yourself as the starting design, given as a "
            "list of point lists. Costs nothing. Use this for a construction you have "
            "a specific algebraic reason to name."
        ),
        _GROUP_SCHEMA | {"base_blocks": list},
    ),
)


def _bind(name: str, description: str, schema: dict[str, Any], router: Router) -> Any:  # noqa: ANN401
    """Bind one router method as an in-process MCP tool.

    The kernel work is CPU-bound and runs for minutes, so it is pushed to a
    worker thread; running it inline would block the event loop and starve
    every other arm in a fan-out.
    """

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        content, ok = await asyncio.to_thread(router.handle, name, args)
        return {"content": [{"type": "text", "text": content}], "isError": not ok}

    return tool(name, description, schema)(handler)


class CoveringSession:
    """Drive one model-authored search over the exact tool surface."""

    def __init__(
        self,
        *,
        tools: CoveringTools,
        cell: str,
        settings: Settings | None = None,
        board: Board | None = None,
        query_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._tools = tools
        self._cell = cell
        self._router = Router(
            tools=tools,
            cell=cell,
            max_tool_calls=self._settings.max_tool_calls,
            board=board,
        )
        self._query = query_fn or query

    @property
    def tools(self) -> CoveringTools:
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
            cell=self._cell,
            published=self._tools.published,
            moves=self._tools.used,
            tool_calls=self._router.calls,
            turns=turns,
            output_tokens=int(usage.get("output_tokens", 0)),
            best_uncovered=self._tools.incumbent_uncovered,
            banked=banked,
            log=tuple(self._router.log),
        )
