"""
Week 4: FastAPI wrapper around the agent pipeline.

This turns the orchestrator from "a script I run in my terminal" into "a
service other things can call over HTTP" — the actual requirement for
deploying it anywhere (Render, a Docker container, etc).

Two endpoints:
  - GET  /health  — a simple check that the service is alive, used by
    uptime monitors and Render's own health checks.
  - POST /run     — runs the full agent pipeline (research -> write ->
    review -> send) for a given task and email.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.orchestrator import run_pipeline

app = FastAPI(
    title="Autonomous Agent API",
    description="Runs a multi-agent research -> write -> review -> send pipeline.",
)


class RunRequest(BaseModel):
    task: str
    to_email: str


class RunResponse(BaseModel):
    status: str
    summary: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest):
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty.")
    if not request.to_email.strip():
        raise HTTPException(status_code=400, detail="to_email cannot be empty.")

    try:
        final_state = run_pipeline(request.task, request.to_email)
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail="The agent pipeline failed to complete.")

    return RunResponse(
        status=final_state.get("final_status", "unknown"),
        summary=final_state.get("summary"),
    )
