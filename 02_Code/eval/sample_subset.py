"""
eval/sample_subset.py
---------------------
Deterministic stratified subset of a SWE-bench split.

Ablations are reported on a subset rather than all 300 instances, so the
subset has to be a stated object rather than "whatever ``--max_tasks 100``
happened to take." Taking the first N is not a sample: SWE-bench splits are
grouped by repository, so a prefix over-represents whichever projects sort
first and can omit others entirely.

This module allocates instances across repositories in proportion to their
share of the split, using the largest-remainder method so the counts sum
exactly to N, then draws within each repository with a fixed seed. The result
is written to JSON with the seed and per-repository counts recorded, so the
same subset can be regenerated and checked by a reviewer.

Selection reads only ``instance_id`` and ``repo``. It never inspects a
reference patch, so it cannot bias the subset toward instances of any
particular difficulty.

Usage:
    python -m eval.sample_subset --n 100 --out eval/subsets/lite100.json
    python -m eval.swebench_runner --config no_critic --run_id x \\
        --instance_file eval/subsets/lite100.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

DEFAULT_SEED = 20260801


def allocate(repo_counts: dict[str, int], n: int) -> dict[str, int]:
    """
    Proportionally allocate *n* slots across repositories.

    Uses the largest-remainder (Hamilton) method: floor each exact share, then
    hand out the remaining slots to the largest fractional parts. Guarantees
    the allocation sums to *n* and never asks a repository for more instances
    than it has.

    Ties on the fractional part are broken by repository name so the result is
    reproducible across Python versions and dict orderings.
    """
    total = sum(repo_counts.values())
    if n >= total:
        return dict(repo_counts)
    if n <= 0:
        return {repo: 0 for repo in repo_counts}

    exact = {repo: count * n / total for repo, count in repo_counts.items()}
    floors = {repo: int(value) for repo, value in exact.items()}
    remaining = n - sum(floors.values())

    order = sorted(
        repo_counts,
        key=lambda repo: (-(exact[repo] - floors[repo]), repo),
    )
    for repo in order:
        if remaining <= 0:
            break
        if floors[repo] < repo_counts[repo]:
            floors[repo] += 1
            remaining -= 1

    # If capacity constraints blocked some slots, redistribute to whoever has
    # room, still in a deterministic order.
    while remaining > 0:
        progressed = False
        for repo in sorted(repo_counts):
            if remaining <= 0:
                break
            if floors[repo] < repo_counts[repo]:
                floors[repo] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return floors


def stratified_sample(
    instances: list[dict], n: int, seed: int = DEFAULT_SEED
) -> list[str]:
    """Return *n* instance ids, stratified by repository, deterministically."""
    by_repo: dict[str, list[str]] = {}
    for instance in instances:
        by_repo.setdefault(instance["repo"], []).append(instance["instance_id"])

    quota = allocate({repo: len(ids) for repo, ids in by_repo.items()}, n)

    rng = random.Random(seed)
    chosen: list[str] = []
    for repo in sorted(by_repo):
        pool = sorted(by_repo[repo])          # sort first: dataset order is not guaranteed
        chosen.extend(rng.sample(pool, quota[repo]))

    return sorted(chosen)


def load_instance_ids(path: str | Path) -> list[str]:
    """
    Read a subset file written by this module.

    Also accepts a plain newline-delimited list, so a hand-written instance
    list works without ceremony.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        return list(json.loads(text)["instance_ids"])
    return [line.strip() for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified SWE-bench subset")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--split", default="lite",
                        choices=["lite", "verified", "full"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset_name = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }[args.split]
    instances = [
        {"instance_id": row["instance_id"], "repo": row["repo"]}
        for row in load_dataset(dataset_name, split="test")
    ]

    ids = stratified_sample(instances, args.n, args.seed)
    per_repo = Counter(
        i["repo"] for i in instances if i["instance_id"] in set(ids)
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "dataset": dataset_name,
        "split": args.split,
        "n": len(ids),
        "seed": args.seed,
        "per_repo": dict(sorted(per_repo.items())),
        "instance_ids": ids,
    }, indent=2), encoding="utf-8")

    print(f"\n  {len(ids)} instances across {len(per_repo)} repositories "
          f"(seed {args.seed})")
    for repo, count in sorted(per_repo.items()):
        print(f"    {count:>4}  {repo}")
    print(f"\n  Written to {out_path}\n")


if __name__ == "__main__":
    main()
