"""
WEEK 1 — the agent's first (and only, for week 1) tool.

This file will hold a thin wrapper around the Tavily search API:
  - takes a query string
  - calls Tavily
  - returns clean, LLM-readable results (not raw JSON)

Every tool in this project follows the same shape so the agent loop in
agents/react_agent.py can call any of them the same way:
    def run(args: dict) -> str:
        ...
        return "plain text result the LLM can read"

Later weeks will add more files here (email.py, code_exec.py, etc.) —
each one self-contained, same shape, easy to plug into the orchestrator.
"""
