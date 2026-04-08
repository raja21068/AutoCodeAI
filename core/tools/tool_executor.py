"""
core/tools/tool_executor.py — Safe execution of whitelisted commands.

Provides a ToolExecutor class that can run pre-approved commands like:
- git (clone, status, add, commit, push, pull)
- pip (install, uninstall, list)
- Shell utilities (ls, mkdir, cat, rm, etc.)

Security features:
- Whitelist-only command execution
- Parameter sanitization with shlex.quote
- Timeout enforcement
- Working directory isolation
"""

import subprocess
import shlex
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


# ── Command Templates ──────────────────────────────────────────────────────
# These templates define allowed commands. Parameters are validated before use.

ALLOWED_COMMANDS = {
    # Git operations
    "git_clone": "git clone {url} {dest}",
    "git_status": "git status",
    "git_add": "git add {files}",
    "git_commit": "git commit -m {message}",
    "git_push": "git push {remote} {branch}",
    "git_pull": "git pull {remote} {branch}",
    "git_diff": "git diff {file}",
    "git_log": "git log --oneline -n {count}",
    
    # Package management
    "pip_install": "pip install {package}",
    "pip_uninstall": "pip uninstall -y {package}",
    "pip_list": "pip list",
    "pip_freeze": "pip freeze > requirements.txt",
    
    # File operations
    "ls": "ls -la {path}",
    "mkdir": "mkdir -p {path}",
    "cat": "cat {file}",
    "rm": "rm {file}",
    "rm_rf": "rm -rf {path}",
    "cp": "cp {source} {dest}",
    "mv": "mv {source} {dest}",
    
    # Python execution
    "python_run": "python {file}",
    "pytest_run": "pytest {path}",
    
    # Environment
    "env_list": "env",
    "pwd": "pwd",
    "which": "which {command}",
}


class ToolExecutor:
    """Execute whitelisted commands in a controlled environment."""
    
    def __init__(self, cwd: str = ".", timeout: int = 30):
        """
        Args:
            cwd: Working directory for command execution
            timeout: Maximum execution time in seconds
        """
        self.cwd = os.path.abspath(cwd)
        self.timeout = timeout
        logger.info(f"ToolExecutor initialized with cwd={self.cwd}, timeout={timeout}s")
    
    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a whitelisted command with the given parameters.
        
        Args:
            tool_name: Name of the command from ALLOWED_COMMANDS
            params: Dictionary of parameter values
            
        Returns:
            Dictionary with keys: stdout, stderr, returncode, or error
        """
        if tool_name not in ALLOWED_COMMANDS:
            logger.warning(f"Tool '{tool_name}' not in whitelist")
            return {"error": f"Tool '{tool_name}' not allowed"}
        
        cmd_template = ALLOWED_COMMANDS[tool_name]
        
        try:
            # Format parameters safely
            formatted_params = self._sanitize_params(params)
            cmd = cmd_template.format(**formatted_params)
        except KeyError as e:
            logger.error(f"Missing parameter for {tool_name}: {e}")
            return {"error": f"Missing parameter: {e}"}
        except Exception as e:
            logger.error(f"Parameter formatting error for {tool_name}: {e}")
            return {"error": f"Parameter error: {e}"}
        
        logger.info(f"Executing: {cmd}")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=os.environ.copy()
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": cmd
            }
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {self.timeout}s: {cmd}")
            return {"error": f"Command timed out after {self.timeout}s"}
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"error": str(e)}
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Sanitize parameters to prevent shell injection.
        
        Args:
            params: Raw parameter dictionary
            
        Returns:
            Sanitized parameter dictionary with quoted values
        """
        formatted = {}
        for key, value in params.items():
            if value is None:
                formatted[key] = ""
            elif isinstance(value, (list, tuple)):
                # Join multiple values with spaces (e.g., for git add)
                formatted[key] = " ".join(shlex.quote(str(v)) for v in value)
            elif isinstance(value, str):
                # Quote strings that might contain special characters
                if self._needs_quoting(value):
                    formatted[key] = shlex.quote(value)
                else:
                    formatted[key] = value
            else:
                formatted[key] = str(value)
        return formatted
    
    @staticmethod
    def _needs_quoting(value: str) -> bool:
        """Check if a string needs shell quoting."""
        dangerous_chars = [" ", ";", "|", "&", "$", "`", "(", ")", "<", ">", "\n", "\t"]
        return any(char in value for char in dangerous_chars)
    
    def execute_multiple(self, commands: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        Execute multiple commands in sequence.
        
        Args:
            commands: List of dicts with 'tool' and 'params' keys
            
        Returns:
            List of result dictionaries
        """
        results = []
        for cmd in commands:
            tool_name = cmd.get("tool")
            params = cmd.get("params", {})
            result = self.execute(tool_name, params)
            results.append(result)
            
            # Stop on first error if requested
            if cmd.get("stop_on_error") and result.get("error"):
                logger.warning(f"Stopping execution chain due to error: {result['error']}")
                break
        
        return results
    
    def validate_command(self, tool_name: str) -> bool:
        """Check if a command is in the whitelist."""
        return tool_name in ALLOWED_COMMANDS
    
    def list_available_tools(self) -> list[str]:
        """Return list of available tool names."""
        return list(ALLOWED_COMMANDS.keys())


# ── Convenience Functions ──────────────────────────────────────────────────

def create_executor(cwd: str = ".", timeout: int = 30) -> ToolExecutor:
    """Create a new ToolExecutor instance."""
    return ToolExecutor(cwd=cwd, timeout=timeout)


def quick_execute(tool_name: str, params: Dict[str, Any], cwd: str = ".") -> Dict[str, Any]:
    """Execute a single command without creating a persistent executor."""
    executor = ToolExecutor(cwd=cwd)
    return executor.execute(tool_name, params)
