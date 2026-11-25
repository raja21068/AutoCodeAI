# Changelog

All notable changes to AutoCodeAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-04-08

### 🎉 Major Release - Parallel Execution, Tool Integration, and Flexible LLM Support

### Added

#### Tool Integration
- **Tool Executor System** (`core/tools/tool_executor.py`)
  - Whitelist-based command execution for security
  - Support for git operations (clone, status, add, commit, push, pull, diff, log)
  - Package management (pip install, uninstall, list, freeze)
  - File operations (ls, mkdir, cat, rm, cp, mv)
  - Python execution (run scripts, pytest)
  - Environment utilities (env, pwd, which)
  - Parameter sanitization with shlex.quote
  - Timeout enforcement and error handling
  - Working directory isolation

#### Parallel Execution
- **Parallel Agent Execution** in orchestrator
  - Independent tasks run concurrently
  - Automatic dependency management via `parallel_group` in plan steps
  - 3-5x faster completion for multi-step tasks
  - `run_parallel()` method for explicit parallel execution
  - `run_parallel_coders()` helper for multiple coder agents
- **New API Endpoint**: `POST /api/agent/run_parallel`
  - Execute multiple steps concurrently
  - Supports mixed agent types (coder, tester, tool)
  - Returns results array preserving order

#### Flexible LLM Support
- **Multi-Mode LLM Client** (`core/utils/llm.py`)
  - **LiteLLM mode** (default): unified interface to 100+ providers
  - **OpenAI mode**: direct OpenAI API access
  - **DeepSeek mode**: direct DeepSeek API integration
  - **Local mode**: Ollama and compatible local endpoints
  - Per-agent model routing maintained across all modes
  - Seamless mode switching via environment variables
- **Configuration Variables**:
  - `LLM_MODE`: Choose provider mode
  - `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`: Provider-specific keys
  - `LOCAL_LLM_URL`, `LOCAL_MODEL`: Local model configuration
  - `LLM_MODEL`: Global model override

#### Web User Interface
- **Modern Web UI** (`static/index.html`)
  - Beautiful, responsive design with dark theme
  - Real-time output streaming via SSE
  - Task submission with file context
  - Status indicators (idle, running, success, error)
  - Clear output functionality
  - Keyboard shortcuts (Shift+Enter to submit)
  - Feature showcase cards
- **Static File Serving** in main.py
  - Root redirect to web UI
  - FastAPI StaticFiles mount

#### Enhanced Planning
- **Updated PlannerAgent**
  - Support for `tool` agent type in plans
  - `parallel_group` field for concurrent execution
  - `tool_name` and `tool_params` for tool steps
  - Improved JSON schema documentation

### Changed

- **Orchestrator** (`services/orchestrator.py`)
  - Refactored `run()` method to support parallel execution
  - Added `_execute_step()` helper for unified step execution
  - Integrated ToolExecutor for tool steps
  - Groups steps by `parallel_group` for concurrent execution
  - Enhanced error handling for parallel tasks
  - Tool output stored in memory for downstream agents

- **API Routes** (`api/routes.py`)
  - Added `ParallelRequest` model
  - Enhanced documentation with examples
  - Updated route descriptions

- **Main Application** (`main.py`)
  - Updated title to "AutoCodeAI"
  - Version bumped to 2.0.0
  - Added static file serving
  - Root endpoint redirects to web UI
  - Enhanced health endpoint with version info

- **Requirements** (`requirements.txt`)
  - Added explicit `openai>=1.30.0` dependency
  - Updated LiteLLM dependency documentation

- **Environment Configuration** (`.env.example`)
  - Added LLM_MODE configuration section
  - Added tool executor settings
  - Added parallel execution settings
  - Reorganized for better clarity
  - Added inline documentation

### Improved

- **Documentation** (`README.md`)
  - Added "What's New in v2.0" section
  - Updated project name to AutoCodeAI
  - Enhanced architecture description
  - Added tool executor to agents table
  - Updated project structure with new files
  - Comprehensive configuration table with sections
  - Web UI usage examples
  - Parallel execution examples
  - LLM mode switching examples
  - Updated API reference with new endpoints

- **Error Handling**
  - Better error messages in LLM client
  - Tool executor validation and sanitization
  - Parallel execution exception handling

- **Performance**
  - Concurrent task execution for independent steps
  - Reduced latency for multi-step workflows

### Security

- **Tool Executor**
  - Whitelist-only command execution
  - Parameter sanitization prevents shell injection
  - Timeout enforcement prevents runaway processes
  - Working directory isolation

### Dependencies

- `openai>=1.30.0` - Direct API access for non-LiteLLM modes
- All existing dependencies maintained

---

## [1.0.0] - 2024-03-XX

### Initial Release

- Multi-agent orchestration (Planner, Coder, Tester, Debugger, Critic)
- Docker sandbox execution
- ChromaDB semantic memory
- Real-time streaming (SSE, WebSocket)
- Diff-based file editing
- Repository awareness and indexing
- LiteLLM integration for multiple providers
- REST API endpoints
- Comprehensive test suite

---

## Upgrade Guide: 1.0 → 2.0

### Breaking Changes

None! Version 2.0 is fully backward compatible.

### Recommended Actions

1. **Update environment configuration**:
   ```bash
   cp .env.example .env
   # Add new variables: LLM_MODE, TOOL_EXECUTOR_TIMEOUT, etc.
   ```

2. **Try the web UI**:
   ```bash
   uvicorn main:app --reload
   open http://localhost:8000
   ```

3. **Test parallel execution**:
   ```python
   # Plans can now include parallel_group
   {
     "steps": [
       {"agent": "coder", "description": "Task 1", "parallel_group": 1},
       {"agent": "coder", "description": "Task 2", "parallel_group": 1},
       {"agent": "tester", "description": "Test", "parallel_group": 2}
     ]
   }
   ```

4. **Try tool integration**:
   ```python
   # Plans can now include tool steps
   {
     "steps": [
       {"agent": "tool", "tool_name": "git_clone", 
        "tool_params": {"url": "...", "dest": "lib"}},
       {"agent": "coder", "description": "Use the library"}
     ]
   }
   ```

5. **Experiment with different LLM modes**:
   ```bash
   # Try DeepSeek
   export LLM_MODE=deepseek
   export DEEPSEEK_API_KEY=sk-...

   # Or local models
   export LLM_MODE=local
   export LOCAL_MODEL=deepseek-coder
   ```

### Migration Notes

- All existing API endpoints remain unchanged
- Existing plans without `parallel_group` execute sequentially (backward compatible)
- Default `LLM_MODE` is `litellm` (existing behavior)
- Tool execution is opt-in via plan steps
