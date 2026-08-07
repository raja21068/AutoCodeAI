"""
eval/instance_env.py
--------------------
Repository-backed execution environment for a single SWE-bench instance.

This replaces the generic ``python:3.10-slim`` sandbox for benchmark runs.
Agents interact with the *actual* repository at the *actual* base commit,
inside the *same* instance image family the official harness uses, so that
commands such as ``pytest``, ``./tests/runtests.py`` or ``bin/test`` behave
the way they do during grading.

Environment layout (SWE-bench convention):
    /testbed                      repository checkout at ``base_commit``
    /opt/miniconda3/envs/testbed  preinstalled project dependencies

Information boundary
--------------------
This module is part of the *launcher*, which is trusted and may read the
full instance dict (it needs ``instance_id`` to resolve the image). Nothing
here is exposed to an agent. The agent-visible projection is built
separately by :mod:`eval.task_adapter`.

Note on ``test_patch``: official instance images ship the repository at the
base commit *without* the test patch applied — the harness applies it at
grading time. Using these images is therefore leakage-free: the agent never
sees the tests it will be graded on.
"""

from __future__ import annotations

import io
import logging
import shlex
import tarfile
from typing import Tuple

import docker
import docker.errors

logger = logging.getLogger(__name__)

WORKDIR = "/testbed"
CONDA_ACTIVATE = "source /opt/miniconda3/bin/activate && conda activate testbed"


#: Docker Hub org publishing the prebuilt instance images. ``make_test_spec``
#: defaults this to ``None``, which yields a *locally built* image name rather
#: than a pullable one, so it has to be passed explicitly — the official
#: harness does the same (``run_evaluation.py``).
IMAGE_NAMESPACE = "swebench"


def instance_image_key(instance: dict) -> str:
    """
    Resolve the official instance image tag for *instance*.

    Prefers SWE-bench's own resolution so that image naming stays correct
    across harness versions; falls back to the documented tag convention.

    The import path moved when ``test_spec`` became a package (swebench 2.x):
    its ``__init__`` re-exports the submodules, not ``make_test_spec``, so
    both spellings are tried. Getting this wrong is quiet and expensive — a
    bad name fails at pull time on every instance, which the official report
    records as ``error_instances`` and is easily misread as a low resolve
    rate rather than a broken run.
    """
    make_test_spec = None
    for module in ("swebench.harness.test_spec.test_spec",
                   "swebench.harness.test_spec"):
        try:
            make_test_spec = __import__(module, fromlist=["make_test_spec"]).make_test_spec
            break
        except (ImportError, AttributeError):
            continue

    if make_test_spec is not None:
        try:
            return make_test_spec(instance, IMAGE_NAMESPACE).instance_image_key
        except Exception as exc:  # pragma: no cover - unexpected harness change
            logger.warning("make_test_spec failed for %s (%s); using fallback naming",
                           instance.get("instance_id"), exc)

    # Documented convention, mirroring TestSpec.instance_image_key: the
    # separator swap is applied to the whole namespaced string, and the
    # instance id is lowercased.
    key = f"sweb.eval.x86_64.{instance['instance_id'].lower()}:latest"
    return f"{IMAGE_NAMESPACE}/{key}".replace("__", "_1776_")


class InstanceEnv:
    """
    A running container holding one SWE-bench repository.

    Use as a context manager::

        with InstanceEnv(instance) as env:
            code, out, err = env.exec("python -m pytest tests/ -x -q")
            patch = env.diff()
    """

    def __init__(
        self,
        instance: dict,
        *,
        timeout: int = 300,
        mem_limit: str = "4g",
        nano_cpus: int = 2_000_000_000,
        network_disabled: bool = False,
    ) -> None:
        self.instance = instance
        self.instance_id = instance["instance_id"]
        self.base_commit = instance["base_commit"]
        self.image = instance_image_key(instance)
        self.timeout = timeout
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        # Dependencies are preinstalled in the image, so the network can stay
        # off during interaction. Kept configurable for environments that need
        # to resolve an extra package at setup time.
        self.network_disabled = network_disabled

        self.client = docker.from_env()
        self.container = None
        self.exec_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "InstanceEnv":
        try:
            self.client.images.get(self.image)
        except docker.errors.ImageNotFound:
            logger.info("Pulling instance image %s …", self.image)
            self.client.images.pull(self.image)

        self.container = self.client.containers.run(
            self.image,
            command="sleep infinity",
            detach=True,
            remove=False,
            working_dir=WORKDIR,
            mem_limit=self.mem_limit,
            nano_cpus=self.nano_cpus,
            pids_limit=512,
            network_disabled=self.network_disabled,
        )
        # Guarantee a pristine checkout even if the image was left dirty.
        self.reset()
        logger.info("Started %s for %s", self.image, self.instance_id)
        return self

    def stop(self) -> None:
        if self.container is not None:
            try:
                self.container.remove(force=True)
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("Container cleanup failed for %s", self.instance_id)
            self.container = None

    def __enter__(self) -> "InstanceEnv":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def exec(self, command: str, timeout: int | None = None) -> Tuple[int, str, str]:
        """
        Run *command* in the project environment at ``/testbed``.

        Returns ``(exit_code, stdout, stderr)``. A non-zero exit code is a
        normal outcome, not an exception — it is the feedback signal.
        """
        if self.container is None:
            raise RuntimeError("InstanceEnv.exec() called before start()")

        self.exec_count += 1
        wrapped = f"{CONDA_ACTIVATE} && cd {WORKDIR} && {command}"
        # ``timeout`` guards against tests that hang; the coreutils binary is
        # present in every instance image.
        limit = timeout or self.timeout
        guarded = f"timeout {limit} bash -lc {shlex.quote(wrapped)}"

        try:
            exit_code, streams = self.container.exec_run(guarded, demux=True)
        except docker.errors.APIError as exc:
            return 1, "", f"DockerAPIError: {exc}"

        stdout_b, stderr_b = streams if streams else (None, None)
        stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
        stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""

        if exit_code == 124:
            stderr += f"\n[timeout after {limit}s]"
        return exit_code, stdout, stderr

    # ------------------------------------------------------------------
    # Filesystem
    # ------------------------------------------------------------------

    def read_file(self, path: str, max_bytes: int = 200_000) -> str:
        code, out, err = self.exec(f"head -c {max_bytes} {shlex.quote(path)}")
        return out if code == 0 else ""

    def write_file(self, path: str, content: str) -> None:
        if self.container is None:
            raise RuntimeError("InstanceEnv.write_file() called before start()")

        buf = io.BytesIO()
        encoded = content.encode("utf-8")
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=path.lstrip("/").split("/")[-1])
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))

        dest_dir = "/".join(f"{WORKDIR}/{path.lstrip('/')}".split("/")[:-1]) or WORKDIR
        self.exec(f"mkdir -p {shlex.quote(dest_dir)}")
        self.container.put_archive(dest_dir, buf.getvalue())

    # ------------------------------------------------------------------
    # Patch handling
    # ------------------------------------------------------------------

    def apply_patch(self, diff: str) -> Tuple[bool, str]:
        """
        Apply a unified diff to the working tree.

        Tries ``git apply`` first, then falls back to ``patch -p1``, which
        tolerates slightly off context lines that models often produce.
        """
        if not diff or not diff.strip():
            return False, "empty patch"

        if not diff.endswith("\n"):
            diff += "\n"
        self.write_file(".agent.patch", diff)

        code, out, err = self.exec("git apply -v .agent.patch")
        if code == 0:
            return True, "applied with git apply"

        code2, out2, err2 = self.exec("patch -p1 --fuzz=5 -i .agent.patch")
        if code2 == 0:
            return True, "applied with patch -p1"

        return False, f"git apply: {err.strip()}\npatch: {(err2 or out2).strip()}"

    def diff(self) -> str:
        """Return the model patch: working tree vs. the base commit."""
        # Stage new files so they appear in the diff, but never ship our own
        # scratch file to the grader.
        self.exec("rm -f .agent.patch")
        self.exec("git add -A")
        self.exec(f"git reset -q -- .agent.patch")
        code, out, _ = self.exec(f"git diff --cached {self.base_commit}")
        return out if code == 0 else ""

    def reset(self) -> None:
        """Discard all working-tree changes and return to the base commit."""
        self.exec("git reset -q --hard")
        self.exec("git clean -qfd")
        self.exec(f"git checkout -q {self.base_commit}")
