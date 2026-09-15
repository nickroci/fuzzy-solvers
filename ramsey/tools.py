"""The model-facing tool surface for authoring ``R(3, k)`` lower bounds.

This is the solver boundary.  The model chooses *what to construct* — which
group, which connection set, at which order — and this module does exactly
what it is told, measures the result with the referee, and reports the
measurement.  It contains no construction heuristic, no ranking, and no
search: anything the model gets credit for, the model authored.

A lower bound is a single graph.  ``R(3, k) > n`` holds exactly when some
triangle-free graph on ``n`` vertices has independence number at most
``k - 1``, so a successful run's entire output is one adjacency matrix that
anyone can recheck in milliseconds.  Two structural facts bound the search
and are reported to the model rather than hidden in the code: in a
triangle-free graph every neighbourhood is independent, so the degree can
never exceed ``k - 1``; and for a Cayley graph the degree is exactly the size
of the connection set, so the connection set size is capped before anything
is built.

Every evaluation costs exactly one kernel call against one global budget, so
the model arm can be compared against a mechanical sweep at matched cost.
Candidates are immutable and content-addressed; only :meth:`RamseyTools.bank`
moves the incumbent, and only on a strict improvement that the referee has
accepted.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from ramsey.analysis import ObstructionAnalysis, analyse_obstruction
from ramsey.blackboard import Attempt, Blackboard
from ramsey.cayley import AbelianGroup, cayley_graph, sum_violation, symmetric_closure
from ramsey.graph import (
    RamseyGraph,
    capped_independence_number,
    check_ramsey_graph,
    independent_set_within,
    maximum_independent_set,
    to_graph6,
)
from ramsey.lift import VoltageGraph, lift_graph
from ramsey.program import (
    LiftOutcome,
    LiftProgram,
    SearchOutcome,
    SearchProgram,
    run_lift_program,
    run_program,
)
from ramsey.repair import RepairOutcome, repair_search

_GRADIENT_HEADROOM = 6
_MAX_SEARCH_SHARE = 0.35
MOVE_CLASSES = ("swap", "add", "drop", "break")


@dataclass(frozen=True, slots=True)
class RamseyTarget:
    """One lower-bound target: beat ``record`` for ``R(3, second_colour)``.

    ``record`` is the published bound, so a witness on ``record`` vertices
    proves ``R(3, k) > record`` and improves the table by one.
    """

    second_colour: int
    record: int
    credit: str = ""

    @property
    def order(self) -> int:
        """Return the vertex count a new witness must have."""
        return self.record

    @property
    def independence_cap(self) -> int:
        """Return the largest independence number a witness may have."""
        return self.second_colour - 1

    @property
    def degree_cap(self) -> int:
        """Return the largest degree a witness may have."""
        return self.second_colour - 1

    def describe(self) -> str:
        """Render the target and its structural constraints for the model."""
        return (
            f"Target: improve R(3,{self.second_colour}) >= {self.record}"
            f"{f' [{self.credit}]' if self.credit else ''} to >= {self.record + 1}.\n"
            f"You must exhibit a graph on exactly {self.order} vertices that is "
            f"triangle-free and has independence number at most {self.independence_cap}.\n"
            f"Because every neighbourhood of a triangle-free graph is independent, no "
            f"vertex may have degree above {self.degree_cap}. For a Cayley graph the "
            f"degree equals the connection set size, so the symmetric connection set "
            f"may hold at most {self.degree_cap} elements.\n"
            f"A construction is triangle-free exactly when its connection set is "
            f"sum-free: no two members (equal members allowed) may sum to a member."
        )


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The exact, neutral measurement of one proposed construction."""

    handle: str
    group: str
    order: int
    connection: tuple[int, ...]
    degree: int
    triangle_free: bool
    sum_violation: tuple[int, int] | None
    independence_at_most_cap: bool
    independence_number: int | None
    obstruction: tuple[int, ...]
    accepted: bool
    rejected_reason: str = ""

    def render(self) -> str:
        """Render the measurement as feedback, stating only measured facts."""
        if self.rejected_reason:
            return f"[{self.handle}] rejected: {self.rejected_reason}"
        if not self.triangle_free:
            pair = self.sum_violation
            return (
                f"[{self.handle}] {self.group}, degree {self.degree}: NOT triangle-free. "
                f"Connection elements {pair[0]} and {pair[1]} sum to a connection element."
                if pair
                else f"[{self.handle}] not triangle-free."
            )
        if self.accepted:
            return (
                f"[{self.handle}] {self.group}, degree {self.degree}: ACCEPTED. "
                f"Triangle-free on {self.order} vertices with independence number "
                f"{self.independence_number}."
            )
        return (
            f"[{self.handle}] {self.group}, degree {self.degree}: triangle-free, but an "
            f"independent set of size {len(self.obstruction)} exists, which exceeds the "
            f"cap. One such set: {self.obstruction}."
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """An immutable, content-addressed evaluated construction."""

    handle: str
    graph: RamseyGraph
    evaluation: Evaluation
    moduli: tuple[int, ...] = ()
    generators: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Refinement:
    """The outcome of one bounded hill-climb from an archived candidate.

    The model chooses the starting candidate, the move class and the step
    budget; the kernel executes the walk.  Every trial costs one ledger call,
    so a refinement is priced exactly like the cold evaluations it replaces
    and the two strategies stay comparable.
    """

    handle: str
    move_class: str
    steps_spent: int
    start_independence: int
    best_independence: int
    best_handle: str
    accepted: bool
    improved: bool
    generators: tuple[int, ...]

    def render(self) -> str:
        """Render the walk as feedback, stating only measured facts."""
        headline = (
            f"refine[{self.move_class}] from {self.handle}: {self.steps_spent} trials, "
            f"independence {self.start_independence} -> {self.best_independence}"
        )
        if self.accepted:
            return (
                f"{headline}. ACCEPTED as [{self.best_handle}] with generators {self.generators}."
            )
        if self.improved:
            return (
                f"{headline}. Best is [{self.best_handle}] with generators "
                f"{self.generators}; still above the cap."
            )
        return f"{headline}. No move in this class improved the incumbent."


class BudgetExhaustedError(RuntimeError):
    """Raised when a kernel call is requested past the global budget."""

    def __init__(self, budget: int) -> None:
        super().__init__(f"kernel budget of {budget} calls is exhausted")
        self.budget = budget


@dataclass(slots=True)
class RamseyTools:
    """Crash-resilient tool surface over one target and one kernel budget."""

    target: RamseyTarget
    kernel_budget: int
    blackboard: Blackboard | None = None
    _calls: int = field(default=0, init=False)
    _archive: dict[str, Candidate] = field(default_factory=dict, init=False)
    _banked: str | None = field(default=None, init=False)
    _best_independence: int | None = field(default=None, init=False)
    _best_generators: tuple[int, ...] = field(default=(), init=False)
    _best_moduli: tuple[int, ...] = field(default=(), init=False)
    _seed_graph: RamseyGraph | None = field(default=None, init=False)
    _seed_score: int | None = field(default=None, init=False)

    @property
    def calls_used(self) -> int:
        """Return the number of kernel calls spent."""
        return self._calls

    @property
    def calls_remaining(self) -> int:
        """Return the number of kernel calls still available."""
        return self.kernel_budget - self._calls

    @property
    def best_independence(self) -> int | None:
        """Return the lowest independence number any construction reached.

        This is the campaign's real diagnostic.  A run that banks nothing is
        uninterpretable without it: landing one above the cap and landing
        fifteen above mean entirely different things about the approach.
        """
        return self._best_independence

    @property
    def best_generators(self) -> tuple[int, ...]:
        """Return the generators that achieved the best independence number."""
        return self._best_generators

    def _record_best(
        self, score: int, generators: tuple[int, ...], moduli: tuple[int, ...]
    ) -> None:
        """Keep the lowest independence number seen anywhere in this session."""
        if self._best_independence is None or score < self._best_independence:
            self._best_independence = score
            self._best_generators = generators
            self._best_moduli = moduli

    def commission_search(
        self,
        moduli: tuple[int, ...],
        degree: int,
        restarts: int,
        steps: int,
        *,
        seed: int = 0,
        required: tuple[int, ...] = (),
        forbidden: tuple[int, ...] = (),
    ) -> SearchOutcome:
        """Run a mechanical search the model specifies, and report what it found.

        This is where the budget goes.  One call buys thousands of evaluations
        instead of one, so the model's judgement is spent choosing *where* to
        point a large search rather than proposing graphs one at a time.  Every
        evaluation the kernel performs is charged to the same ledger, so a
        commissioned search and an equal number of hand-proposed graphs cost
        exactly the same.

        A single call is capped at a share of what remains, so one over-large
        request cannot consume the whole session.  Testing several structural
        hypotheses is the point; spending everything on the first one is not a
        decision the surface should let the model make by accident.
        """
        group = AbelianGroup(moduli=tuple(moduli))
        if group.order != self.target.order:
            message = f"group order {group.order} is not the required {self.target.order}"
            raise ValueError(message)
        if degree > self.target.degree_cap:
            message = f"degree {degree} exceeds the cap {self.target.degree_cap}"
            raise ValueError(message)
        program = SearchProgram(
            moduli=tuple(moduli),
            degree=degree,
            restarts=restarts,
            steps=steps,
            seed=seed,
            required=tuple(required),
            forbidden=tuple(forbidden),
        )
        share = max(int(self.calls_remaining * _MAX_SEARCH_SHARE), 1)
        outcome = run_program(program, self.target.independence_cap, budget=share)
        self._calls += outcome.evaluations
        self._publish("cayley", program.describe(), outcome.best_independence, outcome.evaluations)
        if outcome.best_generators:
            self._record_best(outcome.best_independence, outcome.best_generators, program.moduli)
        if outcome.accepted:
            connection = symmetric_closure(group, outcome.best_generators)
            handle = _handle(program.moduli, connection)
            graph = cayley_graph(group, connection)
            self._measure(handle, group.name, connection, graph, program.moduli)
        return outcome

    def commission_lift(
        self,
        base_size: int,
        moduli: tuple[int, ...],
        degree: int,
        restarts: int,
        *,
        seed: int = 0,
    ) -> LiftOutcome:
        """Commission a search over voltage-graph lifts.

        A base size of one is exactly the Cayley case, so this strictly
        contains the Cayley search; a larger base reaches graphs no Cayley
        construction can, and a base equal to the order with the trivial group
        is an unconstrained graph search.  The model chooses where on that
        spectrum to spend the budget.
        """
        program = LiftProgram(
            base_size=base_size,
            moduli=tuple(moduli),
            degree=degree,
            restarts=restarts,
            steps=0,
            seed=seed,
        )
        if base_size * AbelianGroup(moduli=tuple(moduli)).order != self.target.order:
            message = f"base size {base_size} times the group order must equal {self.target.order}"
            raise ValueError(message)
        if degree > self.target.degree_cap:
            message = f"degree {degree} exceeds the cap {self.target.degree_cap}"
            raise ValueError(message)
        share = max(int(self.calls_remaining * _MAX_SEARCH_SHARE), 1)
        outcome = run_lift_program(program, self.target.independence_cap, budget=share)
        self._calls += outcome.evaluations
        self._publish("lift", program.describe(), outcome.best_independence, outcome.evaluations)
        if outcome.best_voltages:
            construction = VoltageGraph(
                base_size=base_size, moduli=tuple(moduli), voltages=outcome.best_voltages
            )
            graph = lift_graph(construction)
            self._record_seed(outcome.best_independence, graph)
            if outcome.accepted:
                handle = _handle((base_size, *moduli), tuple(graph.adjacency))
                self._measure(handle, construction.name, (), graph, ())
        return outcome

    def commission_repair(self, steps: int, restarts: int, *, seed: int = 0) -> RepairOutcome:
        """Repair the best construction found so far by editing its edges.

        Structured search supplies a seed; this leaves the structure behind.
        An oversized independent set can only be destroyed by an edge inside
        it, so that is what the repair adds, freeing degree capacity by
        deletion when no legal addition remains.
        """
        if self._seed_graph is None:
            message = "no seed to repair: commission a search first"
            raise ValueError(message)
        if restarts < 1:
            message = f"restarts must be positive, got {restarts}"
            raise ValueError(message)
        share = max(int(self.calls_remaining * _MAX_SEARCH_SHARE), 1)
        generator = random.Random(seed)
        best: RepairOutcome | None = None
        spent = 0
        for _ in range(restarts):
            if spent >= share:
                break
            outcome = repair_search(
                self._seed_graph,
                self.target.independence_cap,
                self.target.degree_cap,
                steps=steps,
                rng=generator,
                budget=share - spent,
            )
            spent += outcome.evaluations
            if best is None or outcome.best_independence < best.best_independence:
                best = outcome
            if outcome.accepted:
                break
        self._calls += spent
        if best is not None:
            self._publish(
                "repair",
                f"{restarts} restarts x {steps} steps from alpha={best.start_independence}",
                best.best_independence,
                spent,
            )
        if best is None:  # pragma: no cover - restarts is positive
            message = "the repair performed no work"
            raise ValueError(message)
        self._record_best_score(best.best_independence)
        if best.accepted:
            handle = _handle((self.target.order,), tuple(best.best_graph.adjacency))
            self._measure(handle, "repaired", (), best.best_graph, ())
        return best

    def _record_seed(self, score: int, graph: RamseyGraph) -> None:
        """Keep the best graph seen, so a repair has something to start from."""
        self._record_best_score(score)
        if self._seed_score is None or score <= self._seed_score:
            self._seed_score = score
            self._seed_graph = graph

    def _record_best_score(self, score: int) -> None:
        """Keep the lowest independence number reached anywhere in this session."""
        if self._best_independence is None or score < self._best_independence:
            self._best_independence = score

    @property
    def banked(self) -> Candidate | None:
        """Return the banked witness, if the model has secured one."""
        return None if self._banked is None else self._archive[self._banked]

    def describe_target(self) -> str:
        """Return the target statement and its structural constraints."""
        return self.target.describe()

    @property
    def target_key(self) -> str:
        """Return the stable identifier this target is recorded under."""
        return f"R(3,{self.target.second_colour})>{self.target.record}"

    def read_blackboard(self) -> str:
        """Report what every other arm has already tried on this target.

        Reading this before committing a budget is the difference between a
        fan-out that compounds and thirty-five arms independently rediscovering
        the same dead ends.
        """
        if self.blackboard is None:
            return "No shared history is available for this run."
        return self.blackboard.render(self.target_key)

    def _publish(self, family: str, detail: str, best: int, evaluations: int) -> None:
        """Record an attempt so later arms inherit it."""
        if self.blackboard is None or not evaluations:
            return
        self.blackboard.record(
            Attempt(
                target=self.target_key,
                family=family,
                detail=detail,
                best_independence=best,
                independence_cap=self.target.independence_cap,
                evaluations=evaluations,
            )
        )

    def evaluate_cayley(self, moduli: tuple[int, ...], generators: tuple[int, ...]) -> Evaluation:
        """Build and exactly measure one Cayley construction.

        Costs one kernel call.  The generator list is closed under negation
        first, so the model may pass either a full symmetric set or one
        representative per inverse pair.
        """
        self._spend()
        handle = _handle(moduli, generators)
        group = AbelianGroup(moduli=tuple(moduli))
        if group.order != self.target.order:
            return self._rejected(
                handle,
                group,
                (),
                f"group order {group.order} is not the required {self.target.order}",
            )
        connection = symmetric_closure(group, tuple(generators))
        if not connection:
            return self._rejected(handle, group, connection, "the connection set is empty")
        if len(connection) > self.target.degree_cap:
            return self._rejected(
                handle,
                group,
                connection,
                f"connection set size {len(connection)} exceeds the degree cap "
                f"{self.target.degree_cap}",
            )
        violation = sum_violation(group, connection)
        if violation is not None:
            evaluation = Evaluation(
                handle=handle,
                group=group.name,
                order=group.order,
                connection=connection,
                degree=len(connection),
                triangle_free=False,
                sum_violation=violation,
                independence_at_most_cap=False,
                independence_number=None,
                obstruction=(),
                accepted=False,
            )
            self._archive[handle] = Candidate(
                handle=handle,
                graph=cayley_graph(group, connection),
                evaluation=evaluation,
                moduli=group.moduli,
                generators=connection,
            )
            return evaluation
        graph = cayley_graph(group, connection)
        return self._measure(handle, group.name, connection, graph, group.moduli)

    def evaluate_graph(self, handle_hint: str, graph: RamseyGraph) -> Evaluation:
        """Exactly measure an explicit graph, whatever produced it.

        Costs one kernel call.  This is the escape hatch for constructions
        that are not Cayley graphs.
        """
        self._spend()
        handle = _handle((graph.order,), tuple(graph.adjacency))
        if graph.order != self.target.order:
            return Evaluation(
                handle=handle,
                group=handle_hint,
                order=graph.order,
                connection=(),
                degree=max(graph.degrees(), default=0),
                triangle_free=False,
                sum_violation=None,
                independence_at_most_cap=False,
                independence_number=None,
                obstruction=(),
                accepted=False,
                rejected_reason=(
                    f"graph order {graph.order} is not the required {self.target.order}"
                ),
            )
        return self._measure(handle, handle_hint, (), graph, ())

    def refine(self, handle: str, move_class: str, steps: int, *, seed: int = 0) -> Refinement:
        """Hill-climb from an archived Cayley candidate under one move class.

        The model picks the starting point, the neighbourhood and the step
        budget; the kernel walks it.  Each trial spends one ledger call, and
        the walk stops early when the budget runs out, so a refinement never
        costs more than the cold evaluations it displaces.
        """
        candidate, group = self._refinable(handle, move_class, steps)
        ceiling = self.target.independence_cap + _GRADIENT_HEADROOM
        if move_class == "break":
            return self._break_walk(handle, candidate, group, steps, ceiling)
        start = capped_independence_number(candidate.graph, ceiling)
        best_score = start
        best_reps = _representatives_of(group, candidate.generators)
        best_handle = handle
        pool = _representative_pool(group)
        spent = 0
        for offset in range(steps):
            if self.calls_remaining <= 0:
                break
            connection = self._legal_neighbour(group, best_reps, pool, move_class, seed + offset)
            if connection is None:
                continue
            self._spend()
            spent += 1
            trial_handle = _handle(group.moduli, connection)
            graph = cayley_graph(group, connection)
            score = capped_independence_number(graph, ceiling)
            if score < best_score:
                best_score = score
                best_reps = _representatives_of(group, connection)
                best_handle = trial_handle
                self._measure(trial_handle, group.name, connection, graph, group.moduli)
        best = self._archive.get(best_handle)
        return Refinement(
            handle=handle,
            move_class=move_class,
            steps_spent=spent,
            start_independence=start,
            best_independence=best_score,
            best_handle=best_handle,
            accepted=best is not None and best.evaluation.accepted,
            improved=best_score < start,
            generators=tuple(best_reps),
        )

    def analyse(self, handle: str) -> ObstructionAnalysis:
        """Measure the structure of what defeats a construction.

        Costs one kernel call, because it computes an exact maximum
        independent set.  What it returns is measurement: the obstruction, its
        symmetry, and which generators would destroy it while keeping the
        connection set legal.  It never says which to pick.
        """
        candidate, group = self._cayley_candidate(handle)
        self._spend()
        return analyse_obstruction(
            group,
            candidate.generators,
            candidate.graph,
            degree_cap=self.target.degree_cap,
        )

    def _break_walk(
        self,
        handle: str,
        candidate: Candidate,
        group: AbelianGroup,
        steps: int,
        ceiling: int,
    ) -> Refinement:
        """Walk only the generators that provably destroy the obstruction.

        Every element outside the obstruction's difference set leaves it
        independent, so trying one is a wasted call.  This restricts the walk
        to the legal repairs and spends the budget where it can move.
        """
        start = capped_independence_number(candidate.graph, ceiling)
        analysis = analyse_obstruction(
            group, candidate.generators, candidate.graph, degree_cap=self.target.degree_cap
        )
        self._spend()
        best_score, best_handle = start, handle
        best_reps = _representatives_of(group, candidate.generators)
        spent = 1
        for repair in analysis.repairs[: max(steps - 1, 0)]:
            if self.calls_remaining <= 0:
                break
            connection = symmetric_closure(group, (*candidate.generators, repair))
            self._spend()
            spent += 1
            trial_handle = _handle(group.moduli, connection)
            graph = cayley_graph(group, connection)
            score = capped_independence_number(graph, ceiling)
            if score < best_score:
                best_score = score
                best_reps = _representatives_of(group, connection)
                best_handle = trial_handle
                self._measure(trial_handle, group.name, connection, graph, group.moduli)
        best = self._archive.get(best_handle)
        return Refinement(
            handle=handle,
            move_class="break",
            steps_spent=spent,
            start_independence=start,
            best_independence=best_score,
            best_handle=best_handle,
            accepted=best is not None and best.evaluation.accepted,
            improved=best_score < start,
            generators=tuple(best_reps),
        )

    def _cayley_candidate(self, handle: str) -> tuple[Candidate, AbelianGroup]:
        """Return an archived Cayley candidate and its group."""
        candidate = self._archive.get(handle)
        if candidate is None:
            message = f"no candidate with handle {handle}"
            raise KeyError(message)
        if not candidate.moduli:
            message = f"candidate {handle} is not a Cayley construction and cannot be refined"
            raise ValueError(message)
        return candidate, AbelianGroup(moduli=candidate.moduli)

    def _refinable(
        self, handle: str, move_class: str, steps: int
    ) -> tuple[Candidate, AbelianGroup]:
        """Validate a refinement request and return its starting point."""
        if move_class not in MOVE_CLASSES:
            message = f"move_class must be one of {MOVE_CLASSES}, got {move_class!r}"
            raise ValueError(message)
        if steps < 1:
            message = f"steps must be positive, got {steps}"
            raise ValueError(message)
        return self._cayley_candidate(handle)

    def _legal_neighbour(
        self,
        group: AbelianGroup,
        current: tuple[int, ...],
        pool: tuple[int, ...],
        move_class: str,
        nonce: int,
    ) -> tuple[int, ...] | None:
        """Return a neighbour's symmetric connection set when it is admissible."""
        proposal = _neighbour(current, pool, move_class, nonce)
        if proposal is None:
            return None
        connection = symmetric_closure(group, proposal)
        if not connection or len(connection) > self.target.degree_cap:
            return None
        if sum_violation(group, connection) is not None:
            return None
        return connection

    def bank(self, handle: str) -> str:
        """Promote an accepted candidate to the incumbent witness.

        Only a candidate the referee accepted may be banked, so the incumbent
        can never rest on the search's own opinion of itself.
        """
        candidate = self._archive.get(handle)
        if candidate is None:
            message = f"no candidate with handle {handle}"
            raise KeyError(message)
        if not candidate.evaluation.accepted:
            message = f"candidate {handle} was not accepted by the referee"
            raise ValueError(message)
        self._banked = handle
        return (
            f"banked {handle}: R(3,{self.target.second_colour}) > {self.target.order}, "
            f"improving the published bound {self.target.record}"
        )

    def status(self) -> str:
        """Render budget use, archive size, and the incumbent."""
        banked = self.banked
        incumbent = "none" if banked is None else banked.handle
        return (
            f"kernel calls {self._calls}/{self.kernel_budget}, "
            f"{len(self._archive)} candidates archived, incumbent: {incumbent}"
        )

    def witness_graph6(self) -> str:
        """Return the banked witness in graph6, the publishable certificate."""
        banked = self.banked
        if banked is None:
            message = "no witness has been banked"
            raise ValueError(message)
        return to_graph6(banked.graph)

    def _spend(self) -> None:
        if self._calls >= self.kernel_budget:
            raise BudgetExhaustedError(self.kernel_budget)
        self._calls += 1

    def _rejected(
        self, handle: str, group: AbelianGroup, connection: tuple[int, ...], reason: str
    ) -> Evaluation:
        return Evaluation(
            handle=handle,
            group=group.name,
            order=group.order,
            connection=connection,
            degree=len(connection),
            triangle_free=False,
            sum_violation=None,
            independence_at_most_cap=False,
            independence_number=None,
            obstruction=(),
            accepted=False,
            rejected_reason=reason,
        )

    def _measure(
        self,
        handle: str,
        label: str,
        connection: tuple[int, ...],
        graph: RamseyGraph,
        moduli: tuple[int, ...],
    ) -> Evaluation:
        """Decide the independence cap first, and only then pay for an exact value."""
        cap = self.target.independence_cap
        witness = independent_set_within(graph, graph.full_mask, cap + 1)
        if witness is None:
            exact = maximum_independent_set(graph).size
            self._record_best(
                exact,
                _representatives_of(AbelianGroup(moduli), connection) if moduli else (),
                moduli,
            )
            check = check_ramsey_graph(graph, self.target.second_colour)
            evaluation = Evaluation(
                handle=handle,
                group=label,
                order=graph.order,
                connection=connection,
                degree=max(graph.degrees(), default=0),
                triangle_free=True,
                sum_violation=None,
                independence_at_most_cap=True,
                independence_number=exact,
                obstruction=(),
                accepted=check.accepted,
            )
        else:
            self._record_best(
                len(witness.vertices),
                _representatives_of(AbelianGroup(moduli), connection) if moduli else (),
                moduli,
            )
            evaluation = Evaluation(
                handle=handle,
                group=label,
                order=graph.order,
                connection=connection,
                degree=max(graph.degrees(), default=0),
                triangle_free=True,
                sum_violation=None,
                independence_at_most_cap=False,
                independence_number=None,
                obstruction=witness.vertices,
                accepted=False,
            )
        self._archive[handle] = Candidate(
            handle=handle,
            graph=graph,
            evaluation=evaluation,
            moduli=moduli,
            generators=connection,
        )
        return evaluation


def _representative_pool(group: AbelianGroup) -> tuple[int, ...]:
    """Return one representative per inverse pair, excluding the identity."""
    return tuple(sorted({min(e, group.negate(e)) for e in range(1, group.order)}))


def _representatives_of(group: AbelianGroup, connection: tuple[int, ...]) -> tuple[int, ...]:
    """Collapse a symmetric connection set to one representative per pair."""
    return tuple(sorted({min(e, group.negate(e)) for e in connection}))


def _neighbour(
    current: tuple[int, ...],
    pool: tuple[int, ...],
    move_class: str,
    nonce: int,
) -> tuple[int, ...] | None:
    """Return one deterministic neighbour of a generator set, or ``None``.

    Determinism keeps a refinement reproducible from its seed, which is what
    lets a banked witness be replayed rather than merely believed.
    """
    outside = tuple(element for element in pool if element not in set(current))
    if move_class == "drop":
        if len(current) <= 1:
            return None
        index = nonce % len(current)
        return tuple(e for position, e in enumerate(current) if position != index)
    if move_class == "add":
        if not outside:
            return None
        return tuple(sorted((*current, outside[nonce % len(outside)])))
    if not current or not outside:
        return None
    index = nonce % len(current)
    incoming = outside[(nonce // max(len(current), 1)) % len(outside)]
    kept = tuple(e for position, e in enumerate(current) if position != index)
    return tuple(sorted((*kept, incoming)))


def _handle(moduli: tuple[int, ...], payload: tuple[int, ...]) -> str:
    """Content-address a construction so equal proposals share one handle."""
    digest = hashlib.sha256(repr((tuple(moduli), tuple(payload))).encode()).hexdigest()
    return f"c_{digest[:16]}"
