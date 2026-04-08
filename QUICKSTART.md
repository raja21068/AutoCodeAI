# 🚀 Quick Start Guide - AutoCodeAI v2.0

Get up and running with AutoCodeAI in 5 minutes!

## Prerequisites

- **Docker** (for sandboxed code execution)
- **Python 3.11+**
- **API Key** for your chosen LLM provider (OpenAI, DeepSeek, or local Ollama)

## Installation

### Option 1: Web UI (Recommended for First-Time Users)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/autocodeai.git
cd autocodeai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API key:
# For OpenAI:
#   LLM_MODE=openai
#   OPENAI_API_KEY=sk-your-key-here
# For DeepSeek:
#   LLM_MODE=deepseek
#   DEEPSEEK_API_KEY=your-key-here

# 4. Start the server
uvicorn main:app --reload

# 5. Open the web UI
open http://localhost:8000
```

You should see the AutoCodeAI web interface! 🎉

### Option 2: Docker Compose (Full Stack)

```bash
# 1. Clone and configure
git clone https://github.com/yourusername/autocodeai.git
cd autocodeai
cp .env.example .env
# Edit .env with your API keys

# 2. Start everything
docker compose up --build

# 3. Access the UI
open http://localhost:8000
```

This starts:
- AutoCodeAI backend (port 8000)
- ChromaDB vector database (port 8001)
- Web UI

## Your First Task

### Using the Web UI

1. Open `http://localhost:8000`
2. Type a task in the text area:
   ```
   Create a REST API endpoint for user registration with email validation,
   password hashing, and proper error handling. Include unit tests.
   ```
3. Click **"🚀 Run Agents"**
4. Watch the agents work in real-time!

### Using the API (curl)

```bash
curl -X POST http://localhost:8000/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Write a binary search function with comprehensive pytest tests",
    "context_files": []
  }'
```

### Using Python

```python
import asyncio
from services.orchestrator import Orchestrator

async def main():
    orch = Orchestrator()
    
    results = await orch.run(
        task="Create a FastAPI endpoint for file upload with size validation",
        context_files=[],
        callback=lambda msg: print(msg, end="", flush=True)
    )
    
    print("\n\n✅ Task complete!")
    for result in results:
        print(f"- {result['type']}: {result['step']}")

asyncio.run(main())
```

## Example Tasks

Try these examples to see AutoCodeAI in action:

### 1. Code Generation
```
Create a Python class for managing a TODO list with add, remove, complete, 
and list methods. Include full pytest coverage with fixtures.
```

### 2. API Development
```
Build a FastAPI endpoint for user authentication using JWT tokens. 
Include login, register, and protected routes. Add unit tests.
```

### 3. Data Processing
```
Write a function to parse CSV files and convert them to JSON with 
error handling for malformed data. Include edge case tests.
```

### 4. Bug Fixing
```
File: buggy_sort.py
Task: Fix the sorting algorithm and add comprehensive tests
```

## Advanced Features

### 🔧 Tool Integration

Create tasks that use git, pip, and shell commands:

```json
{
  "task": "Clone the FastAPI repo and create a hello world example",
  "context_files": []
}
```

The planner can include tool steps:
```json
{
  "steps": [
    {
      "agent": "tool",
      "tool_name": "git_clone",
      "tool_params": {
        "url": "https://github.com/tiangolo/fastapi",
        "dest": "fastapi_lib"
      }
    },
    {
      "agent": "coder",
      "description": "Create hello world using FastAPI"
    }
  ]
}
```

### ⚡ Parallel Execution

For tasks with independent subtasks, the planner automatically groups them:

```
Create three microservices: user service, auth service, and notification service.
Each should have its own API endpoint and tests.
```

The planner creates parallel groups:
```json
{
  "steps": [
    {"agent": "coder", "description": "User service", "parallel_group": 1},
    {"agent": "coder", "description": "Auth service", "parallel_group": 1},
    {"agent": "coder", "description": "Notification service", "parallel_group": 1},
    {"agent": "tester", "description": "Integration tests", "parallel_group": 2}
  ]
}
```

All three services are coded in parallel, then tested together!

### 🌐 Switching LLM Providers

#### Use DeepSeek (Faster, Cheaper for Coding)

```bash
# Edit .env
LLM_MODE=deepseek
DEEPSEEK_API_KEY=sk-your-deepseek-key

# Or set environment variables
export LLM_MODE=deepseek
export DEEPSEEK_API_KEY=sk-...
```

#### Use Local Models (Free, Private)

```bash
# 1. Install and start Ollama
# Download from https://ollama.ai

# 2. Pull a model
ollama pull deepseek-coder

# 3. Configure AutoCodeAI
export LLM_MODE=local
export LOCAL_MODEL=deepseek-coder
export LOCAL_LLM_URL=http://localhost:11434/v1

# 4. Run!
uvicorn main:app --reload
```

#### Mix and Match Models

Use different models for different agents:

```bash
# .env
LLM_MODE=litellm
PLANNER_MODEL=gpt-4o              # Best for planning
CODER_MODEL=deepseek/deepseek-chat # Fast for coding
TESTER_MODEL=groq/llama-3.3-70b-versatile  # Free, fast
DEBUGGER_MODEL=anthropic/claude-sonnet-4-5 # Best for debugging
CRITIC_MODEL=anthropic/claude-sonnet-4-5   # Best for review
```

## Configuration Tips

### Minimum Config (OpenAI)

```bash
LLM_MODE=openai
OPENAI_API_KEY=sk-your-key-here
```

### Cost-Optimized Config

```bash
LLM_MODE=litellm
PLANNER_MODEL=gpt-4o-mini           # Cheaper planner
CODER_MODEL=deepseek/deepseek-chat  # Very cheap coding
TESTER_MODEL=groq/llama-3.3-70b-versatile  # Free!
DEBUGGER_MODEL=deepseek/deepseek-chat
CRITIC_MODEL=gpt-4o-mini
```

### Privacy-First Config (All Local)

```bash
LLM_MODE=local
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_MODEL=deepseek-coder
ENABLE_PARALLEL_EXECUTION=true
```

## Troubleshooting

### "API key not found"
- Check your `.env` file exists
- Verify the API key variable matches your `LLM_MODE`
- Restart the server after changing `.env`

### "Docker not found"
- Install Docker Desktop
- Ensure Docker daemon is running
- Test: `docker ps`

### Slow response
- Try using DeepSeek (`LLM_MODE=deepseek`) - it's faster
- Enable parallel execution in `.env`
- Use local models for maximum speed

### Web UI not loading
- Check the server is running: `curl http://localhost:8000/health`
- Look for errors in the terminal
- Try clearing browser cache

## Next Steps

1. **Read the full [README.md](README.md)** for architecture details
2. **Check [CHANGELOG.md](CHANGELOG.md)** for all v2.0 features
3. **Explore the API** at `http://localhost:8000/docs`
4. **Join the community** (link to Discord/GitHub Discussions)
5. **Contribute** - see [CONTRIBUTING.md](CONTRIBUTING.md)

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/autocodeai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/autocodeai/discussions)
- **Documentation**: Full docs at `/docs`

---

**Happy Coding!** 🚀 If AutoCodeAI helps you, please star the repo ⭐
