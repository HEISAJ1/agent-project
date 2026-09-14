# Autonomous Multi-Agent System

Status: Week 1 in progress.

See PROJECT_OVERVIEW.pdf (or the project overview doc) for what this project
is and why it's built this way. This README will grow into a proper
engineering doc (architecture diagram, eval results, known failure modes) by
the end of week 4 — it stays minimal until there's something real to report.

## Folder layout

- agents/   the core agent loop(s). Week 1: one hand-rolled ReAct agent.
            Week 2: ported into a LangGraph multi-agent graph.
- tools/    one file per real tool the agent can call (search, email, etc).
            Every tool has the same shape so the agent loop can call any of
            them the same way.
- evals/    the test-case suite that proves the system works reliably.
            Stays empty until the agent works end-to-end (week 3).
- api/      FastAPI wrapper that exposes the finished system over HTTP.
            Stays empty until week 4 (deployment).
