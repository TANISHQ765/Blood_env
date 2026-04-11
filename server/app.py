"""
BloodEnv FastAPI Server
=======================
POST /reset  — start a new episode
POST /step   — take one action  
GET  /state  — get current state
GET  /health — liveness check
main() is REQUIRED by openenv validate for multi-mode deployment.
"""

import threading
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from blood_env import BloodEnv
from models import Action, Observation, State

# ── Global state ───────────────────────────────────────────────────────────────
_envs: dict = {}
_lock = threading.Lock()
VALID_TASKS = ["easy_routing", "emergency_response", "hard_optimization"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for task in VALID_TASKS:
        _envs[task] = BloodEnv(task_id=task)
        _envs[task].reset()
    yield


app = FastAPI(
    title="BloodEnv",
    description="AI-driven emergency blood logistics environment for OpenEnv",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="easy_routing")


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


class ResetResponse(BaseModel):
    observation: Observation
    task_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "tasks": VALID_TASKS}


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = None):
    task_id = (request.task_id if request else None) or "easy_routing"
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")
    with _lock:
        if task_id not in _envs:
            _envs[task_id] = BloodEnv(task_id=task_id)
        obs = _envs[task_id].reset()
    return ResetResponse(observation=obs, task_id=task_id)


@app.post("/step", response_model=StepResponse)
async def step(action: Action, task_id: str = "easy_routing"):
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    with _lock:
        if task_id not in _envs:
            raise HTTPException(status_code=400, detail="Call /reset first.")
        obs, reward, done, info = _envs[task_id].step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=State)
async def state(task_id: str = "easy_routing"):
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    with _lock:
        if task_id not in _envs:
            raise HTTPException(status_code=400, detail="Call /reset first.")
        return _envs[task_id].state()


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"id": "easy_routing",        "difficulty": "easy",   "description": "Route O+ blood to a single emergency node."},
            {"id": "emergency_response",  "difficulty": "medium", "description": "Handle multiple emergencies with mixed blood types."},
            {"id": "hard_optimization",   "difficulty": "hard",   "description": "Scarce inventory, rare blood types, critical emergencies."},
        ]
    }


# ── REQUIRED for openenv validate (multi-mode deployment check) ────────────────

def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )


if __name__ == "__main__":
    main()
