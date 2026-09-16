"""
Week 5 (extension): FastAPI wrapper around the agent pipeline.

Two endpoints:
  - GET  /health  — a simple check that the service is alive, used by
    uptime monitors and Render's own health checks. No auth required.
  - POST /run     — runs the full agent pipeline. Requires an API key
    and is rate-limited, since each call costs real API usage (Groq,
    Tavily, Brevo) and shouldn't be open to anyone with the URL.
"""

import os

from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from agents.orchestrator import run_pipeline

app = FastAPI(
    title="Autonomous Agent API",
    description="Runs a multi-agent research -> write -> review -> send pipeline.",
)

# Rate limiting: tracked per client IP, using an in-memory store. Fine for
# a single-instance deployment like this one — a multi-instance production
# setup would use a shared Redis backend instead so all instances agree on
# the count.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API key auth: a single shared secret, set as an environment variable and
# never committed to the repo. The client sends it in the "X-API-Key"
# header.
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided_key: str = Security(api_key_header)):
    if not API_KEY:
        # Fails safe: if the server itself has no key configured, refuse
        # every request rather than silently allowing everyone through.
        raise HTTPException(status_code=500, detail="Server is not configured with an API key.")
    if provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return provided_key


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
@limiter.limit("5/hour")
def run(request: Request, body: RunRequest, api_key: str = Security(require_api_key)):
    if not body.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty.")
    if not body.to_email.strip():
        raise HTTPException(status_code=400, detail="to_email cannot be empty.")

    try:
        final_state = run_pipeline(body.task, body.to_email)
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail="The agent pipeline failed to complete.")

    return RunResponse(
        status=final_state.get("final_status", "unknown"),
        summary=final_state.get("summary"),
    )
