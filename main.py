"""
main.py — FastAPI application entry point.

Run:
    uvicorn main:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from api.routes import router, get_orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_orchestrator()          # warm up on startup
    yield
    from api.routes import _orchestrator
    if _orchestrator:
        _orchestrator.shutdown()


app = FastAPI(
    title="AutoCodeAI",
    description="Multi-agent autonomous coding system with parallel execution, tool integration, and flexible LLM support.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", tags=["root"])
async def root():
    """Redirect to web UI."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "version": "2.0.0"}
