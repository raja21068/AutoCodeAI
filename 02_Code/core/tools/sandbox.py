"""
core/tools/sandbox.py — Isolated Docker sandbox for the interactive app.

Injects code via an in-memory tar archive, runs it inside a resource-capped
container with networking disabled, and force-removes the container on exit
regardless of outcome.

Scope: this sandbox executes *standalone* generated code for the web
application. It is deliberately not used for benchmark runs, where the agent
must act on a real repository with project dependencies installed — see
``eval/instance_env.py``. Using this image for SWE-bench tasks was the reason
earlier benchmark runs produced import errors instead of test results.

The image must already contain pytest. The previous version attempted
``pip install pytest`` inside a container created with
``network_disabled=True``, which cannot succeed; the resulting
"No module named pytest" was then surfaced as though it were a test failure.
Availability is now checked up front and reported as an environment problem.
Override the image with ``SANDBOX_IMAGE``.
"""

import io
import logging
import os
import tarfile
from typing import Tuple

import docker
import docker.errors

logger = logging.getLogger(__name__)


class DockerSandbox:
    def __init__(self) -> None:
        self.client  = docker.from_env()
        # Build with: docker build -f Dockerfile.sandbox -t agentforge-sandbox .
        self.image   = os.getenv("SANDBOX_IMAGE", "agentforge-sandbox:latest")
        self.timeout = int(os.getenv("SANDBOX_TIMEOUT", "30"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tar(filename: str, content: str) -> bytes:
        """Return an in-memory tar archive containing *content* as *filename*."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            encoded = content.encode("utf-8")
            info    = tarfile.TarInfo(name=filename)
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
        return buf.getvalue()

    def _copy_to_container(
        self,
        container,
        filename: str,
        content: str,
        dest_dir: str = "/tmp",
    ) -> None:
        tar_bytes = self._make_tar(filename, content)
        container.put_archive(dest_dir, tar_bytes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_code(self, code: str, test_code: str = "") -> Tuple[str, str]:
        """
        Execute *code* inside a sandboxed container.

        If *test_code* is provided, pytest is run after the main script.
        Returns (stdout, stderr).
        """
        container = None
        try:
            container = self.client.containers.run(
                self.image,
                command="sleep 60",          # idle — code injected via exec_run
                detach=True,
                remove=False,
                mem_limit="512m",
                nano_cpus=500_000_000,       # 0.5 CPU
                pids_limit=64,
                network_disabled=True,
                tmpfs={"/workspace": "size=128m,exec"},
            )

            self._copy_to_container(container, "code.py", code)

            _, output = container.exec_run(
                "bash -c 'cd /workspace && python /tmp/code.py'",
                demux=False,
            )
            stdout = output.decode("utf-8", errors="replace") if output else ""
            stderr = ""

            if test_code:
                probe, _ = container.exec_run("python -m pytest --version")
                if probe != 0:
                    return stdout, (
                        f"pytest is not installed in sandbox image "
                        f"{self.image!r}, and networking is disabled so it "
                        f"cannot be installed at run time. Set SANDBOX_IMAGE "
                        f"to an image that includes pytest."
                    )

                self._copy_to_container(container, "test_code.py", test_code)
                _, t_out = container.exec_run(
                    "bash -c 'python -m pytest /tmp/test_code.py -v --tb=short'",
                    demux=False,
                )
                pytest_output = (
                    t_out.decode("utf-8", errors="replace") if t_out else ""
                )
                stdout += f"\n\n--- pytest ---\n{pytest_output}"

            return stdout, stderr

        except docker.errors.DockerException as exc:
            return "", f"DockerException: {exc}"
        except Exception as exc:
            return "", str(exc)
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
