# Autonomous Multi-Agent System

Status: Week 3 in progress (observability + evals done).

See PROJECT_OVERVIEW.pdf (or the project overview doc) for what this project
is and why it's built this way. This README will grow into a full
engineering doc (architecture diagram, final eval results) by the end of
week 4.

## Known limitations (found via the eval suite)

- **The Researcher doesn't clearly distinguish "no relevant information
  exists" from "here's somewhat related information instead."** Caught by
  the eval suite: a test asking about a topic with no real answer (Brevo's
  nonexistent "opening hours") got a 0.0 relevance score because the agent
  substituted adjacent info (pricing, features) rather than saying no
  relevant answer was found. Not yet fixed — documented here as a known
  gap. A real fix would have the Researcher's prompt explicitly instruct
  it to say "no relevant information found" when search results don't
  actually answer the question, rather than answering with the closest
  available information.

- **The Researcher sometimes lists options instead of committing to one
  direct answer.** A test asking "what is the latest stable version of
  Python" scored 0.2 — the agent found and listed multiple version numbers
  and pre-releases but never clearly stated which one is currently stable.
  Not yet fixed.

- **The Writer occasionally embellished beyond its source material** —
    Fixed: tightened the Writer's system prompt to explicitly
  forbid adding information not present in the findings and to preserve
  all key points. Confirmed fixed — re-running the full eval suite after
  this change brought the pass rate from 83% to 100% (12/12).Status: Complete. Weeks 1-4 done — agent, multi-agent orchestration,
evals + observability, deployment + CI all working.

**Live**: https://agent-project-izh1.onrender.com
(free tier — cold starts after ~15 min idle can take 30-50s on first request)

**Repo**: https://github.com/HEISAJ1/agent-project