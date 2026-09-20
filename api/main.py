"""
FastAPI wrapper around the agent pipeline.

Endpoints:
  - GET  /            — serves the frontend page (api/static/index.html).
  - GET  /health       — liveness check, no auth required.
  - POST /run          — runs the full pipeline, returns the final result
    as JSON. Requires an API key (sent as the X-API-Key header), is
    rate-limited per visitor, and is capped on total emails sent per day —
    since each call costs real API usage (Groq, Tavily, Brevo) and
    shouldn't be open to unlimited abuse even by someone holding a valid
    key.

Security notes (found and fixed before sharing this publicly):
  - Render sits behind its own proxy, so a rate limiter keyed on the raw
    socket address (slowapi's default) sees Render's internal proxy IP
    for every request, not the real visitor — meaning the limiter would
    silently apply to everyone as one shared bucket instead of per
    visitor. Fixed by reading the X-Forwarded-For header instead.
  - "to_email" is fully caller-controlled, so a held API key could be used
    to direct output to arbitrary third-party addresses. Mitigated with
    real email format validation and a small daily cap on total emails
    sent, so a leaked key has a bounded blast radius rather than an
    unlimited one.
"""

import os
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from agents.orchestrator import run_pipeline

app = FastAPI(
    title="Autonomous Agent API",
    description="Runs a multi-agent research -> write -> review -> send pipeline.",
)


def get_real_client_ip(request: Request) -> str:
    """
    Render (and most hosts) terminate the connection at their own edge
    and forward requests to the app over an internal network — so
    request.client.host is always Render's proxy, not the visitor. The
    real visitor IP is in the X-Forwarded-For header instead, as the
    first address in that list.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=get_real_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided_key: str = Security(api_key_header)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server is not configured with an API key.")
    if provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return provided_key


# A simple daily cap on total emails sent, shared across everyone using
# the API key. Resets at midnight (or on redeploy — an in-memory counter
# isn't perfectly durable, but it's a reasonable, low-effort backstop for
# a demo-scale project, not a high-traffic production service).
DAILY_EMAIL_LIMIT = 20
_email_count_today = 0
_email_count_date = date.today()


def check_and_increment_daily_email_cap():
    global _email_count_today, _email_count_date
    if date.today() != _email_count_date:
        _email_count_date = date.today()
        _email_count_today = 0
    if _email_count_today >= DAILY_EMAIL_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Daily email limit reached for this demo. Please try again tomorrow.",
        )
    _email_count_today += 1


class RunRequest(BaseModel):
    task: str
    to_email: EmailStr


class RunResponse(BaseModel):
    status: str
    summary: str | None = None


FRONTEND_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def frontend():
    return FRONTEND_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
@limiter.limit("5/hour")
def run(request: Request, body: RunRequest, api_key: str = Security(require_api_key)):
    if not body.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty.")

    check_and_increment_daily_email_cap()

    try:
        final_state = run_pipeline(body.task, body.to_email)
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail="The agent pipeline failed to complete.")

    return RunResponse(
        status=final_state.get("final_status", "unknown"),
        summary=final_state.get("summary"),
    )
