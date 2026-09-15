"""CLI: run one bounded batch of the R(3,k) frontier, journalling as it goes."""

from __future__ import annotations

import argparse
from pathlib import Path

from ramsey.blackboard import Blackboard
from ramsey.campaign import (
    RetryPolicy,
    completed_labels,
    frontier_arms,
    run_campaign,
)
from ramsey.session import RamseySessionSettings


def main(argv: list[str] | None = None) -> int:
    """Run the frontier, resuming from any arms the journal already records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--blackboard", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--kernel-budget", type=int, default=40_000)
    parser.add_argument("--tool-calls", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument(
        "--shard",
        default="0/1",
        help="i/n: take only the arms whose index is congruent to i modulo n. "
        "The kernel is CPU-bound and Python threads share a GIL, so real "
        "parallelism means one process per shard.",
    )
    arguments = parser.parse_args(argv)
    index, _, count = arguments.shard.partition("/")
    shard, shards = int(index), int(count or 1)

    done = completed_labels(arguments.journal)
    arms = tuple(
        arm
        for position, arm in enumerate(frontier_arms(seeds=arguments.seeds, non_cyclic_only=True))
        if arm.label not in done and position % shards == shard
    )
    print(f"shard {shard}/{shards}: {len(arms)} arms ({len(done)} already complete)", flush=True)
    report = run_campaign(
        arms,
        RamseySessionSettings(
            model="claude-sonnet-5",
            max_turns=arguments.tool_calls + 20,
            max_tool_calls=arguments.tool_calls,
            kernel_budget=arguments.kernel_budget,
        ),
        concurrency=arguments.concurrency,
        retry=RetryPolicy(attempts=3, backoff_seconds=15.0, timeout_seconds=arguments.timeout),
        journal=arguments.journal,
        blackboard=Blackboard(path=arguments.blackboard),
    )
    print("\n" + report.render(), flush=True)
    return 0
