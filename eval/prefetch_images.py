"""
eval/prefetch_images.py
-----------------------
Pull SWE-bench instance images ahead of a run.

Image pulls dominate wall-clock on a slow link — a single instance image is
~4GB — and pulling them inline means the agent loop sits idle behind the
network. Prefetching separates the two so a run can start as soon as its
images are local, and so a broken connection costs a retry rather than a
half-finished evaluation.

Names come from :func:`eval.instance_env.instance_image_key`, deliberately,
rather than from ``swebench.harness.prepare_images``. That module builds
images locally with ``namespace=None``, which produces un-namespaced names;
the runner looks up namespaced ones, so every image would be fetched twice.
Pulling the exact names the runner will ask for is the only way to guarantee
the cache actually hits.

Usage:
    python -m eval.prefetch_images --split lite
    python -m eval.prefetch_images --split lite --workers 3
    python -m eval.prefetch_images --instance_file eval/subsets/lite100.json
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import docker
import docker.errors

from eval.instance_env import instance_image_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# This run prints one line per image over many hours. httpx logs every
# request at INFO and the Hub repeats an auth warning per call, which buries
# the progress lines it is the whole point of this module to emit.
for _noisy in ("httpx", "httpcore", "huggingface_hub", "datasets",
               "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}

_print_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefetch SWE-bench instance images")
    p.add_argument("--split", default="lite", choices=list(DATASETS))
    p.add_argument("--instance_file", help="subset file from eval.sample_subset")
    # Pulls are network-bound. A few concurrent pulls help when the limit is
    # per-connection, and hurt nothing when it is total bandwidth; more than a
    # handful mostly multiplies the chance of a registry-side throttle.
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()


def disk_free_gb(client: docker.DockerClient) -> float | None:
    try:
        import shutil

        root = client.info().get("DockerRootDir", "/var/lib/docker")
        return shutil.disk_usage(root).free / 1e9
    except Exception:
        return None


def pull_one(client: docker.DockerClient, image: str, retries: int) -> tuple[str, str, float]:
    """Return ``(image, status, seconds)``. Never raises."""
    started = time.monotonic()
    try:
        client.images.get(image)
        return image, "cached", 0.0
    except docker.errors.ImageNotFound:
        pass
    except docker.errors.APIError as exc:
        return image, f"error: {exc}", time.monotonic() - started

    last = ""
    for attempt in range(1, retries + 1):
        try:
            client.images.pull(image)
            return image, "pulled", time.monotonic() - started
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                # A dropped pull on a slow link is ordinary; back off and retry
                # rather than failing the whole prefetch.
                time.sleep(min(30 * attempt, 120))
    return image, f"failed after {retries}: {last}", time.monotonic() - started


def main() -> int:
    args = parse_args()

    from datasets import load_dataset

    # Full rows, not a projection. ``make_test_spec`` reads base_commit,
    # version and test_patch; handing it {instance_id, repo} makes it raise
    # KeyError on every instance and silently demotes all 300 lookups to the
    # fallback convention. This is a launcher, so reading the full instance is
    # allowed — the agent-visible projection is built in eval.task_adapter.
    rows = [dict(r) for r in load_dataset(DATASETS[args.split], split="test")]

    if args.instance_file:
        from eval.sample_subset import load_instance_ids

        wanted = set(load_instance_ids(args.instance_file))
        rows = [r for r in rows if r["instance_id"] in wanted]

    # Deterministic order so an interrupted prefetch resumes predictably.
    images = sorted({instance_image_key(r) for r in rows})
    client = docker.from_env()

    free = disk_free_gb(client)
    logger.info("%d instances -> %d distinct images (%d workers)",
                len(rows), len(images), args.workers)
    if free is not None:
        # ~4GB per image measured, but instances of the same repo+version
        # share base and env layers, so on-disk cost lands far below the
        # naive product. Warn on the realistic figure, not the upper bound,
        # or the warning fires on every healthy run and stops being read.
        upper_gb = len(images) * 4
        logger.info("disk free: %.0f GB (naive upper bound %d GB before "
                    "layer sharing)", free, upper_gb)
        if free < upper_gb * 0.35:
            logger.warning("free space may not cover this run; watch for "
                           "ENOSPC and prune with `docker image prune`")

    done = pulled = cached = failed = 0
    bytes_before = sum(i.attrs.get("Size", 0) for i in client.images.list())
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(pull_one, client, img, args.retries): img
                   for img in images}
        for future in as_completed(futures):
            image, status, seconds = future.result()
            done += 1
            if status == "pulled":
                pulled += 1
            elif status == "cached":
                cached += 1
            else:
                failed += 1

            elapsed = time.monotonic() - started
            # Rate over *pulled* images only; cached ones are free and would
            # make the estimate optimistic.
            eta = ""
            if pulled:
                per = elapsed / pulled
                remaining = len(images) - done
                eta = f"  eta {per * remaining / 3600:.1f}h"

            with _print_lock:
                mark = "ok " if status in ("pulled", "cached") else "FAIL"
                logger.info("[%3d/%3d] %s %-12s %s%s",
                            done, len(images), mark, status.split(":")[0],
                            image.split("/")[-1], eta)
                if status.startswith(("failed", "error")):
                    logger.error("        %s", status)

    bytes_after = sum(i.attrs.get("Size", 0) for i in client.images.list())
    logger.info("-" * 60)
    logger.info("pulled %d, cached %d, failed %d in %.1fh",
                pulled, cached, failed, (time.monotonic() - started) / 3600)
    logger.info("image store grew by %.1f GB",
                (bytes_after - bytes_before) / 1e9)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
