"""tests/test_sandbox.py — Unit tests for DockerSandbox."""

import io
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from core.tools.sandbox import DockerSandbox


class TestMakeTar:
    def test_returns_valid_tar(self):
        tar_bytes = DockerSandbox._make_tar("hello.py", "print('hello')")
        buf = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=buf) as tar:
            names = tar.getnames()
        assert "hello.py" in names

    def test_content_is_preserved(self):
        content   = "x = 42\nprint(x)\n"
        tar_bytes = DockerSandbox._make_tar("code.py", content)
        buf = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=buf) as tar:
            member = tar.extractfile("code.py")
            assert member.read().decode() == content


class TestDockerSandbox:
    @patch("core.tools.sandbox.docker.from_env")
    def test_run_code_returns_stdout(self, mock_docker):
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, b"Hello, world!\n")
        mock_docker.return_value.containers.run.return_value = mock_container

        sandbox = DockerSandbox()
        stdout, stderr = sandbox.run_code("print('Hello, world!')")

        assert "Hello" in stdout
        assert stderr == ""

    @patch("core.tools.sandbox.docker.from_env")
    def test_container_always_removed(self, mock_docker):
        mock_container = MagicMock()
        mock_container.exec_run.side_effect = RuntimeError("crash")
        mock_docker.return_value.containers.run.return_value = mock_container

        sandbox = DockerSandbox()
        sandbox.run_code("raise ValueError('oops')")

        mock_container.remove.assert_called_once_with(force=True)

    @patch("core.tools.sandbox.docker.from_env")
    def test_docker_exception_returns_stderr(self, mock_docker):
        import docker.errors
        mock_docker.return_value.containers.run.side_effect = (
            docker.errors.DockerException("daemon not running")
        )
        sandbox = DockerSandbox()
        stdout, stderr = sandbox.run_code("pass")
        assert stdout == ""
        assert "DockerException" in stderr
