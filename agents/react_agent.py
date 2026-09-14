"""
WEEK 1 — the core of the whole project lives here first.

This file will hold a hand-rolled ReAct loop (Reason -> Act -> Observe -> repeat),
written in plain Python, with NO framework (no LangGraph yet).

Why hand-roll it before using LangGraph:
  - You need to be able to explain the mechanics in an interview, not just say
    "the framework did it."
  - It's the same loop LangGraph runs internally, just visible.

What this file will eventually contain:
  1. A function that sends the current conversation + a list of available tools
     to the LLM (Groq).
  2. A parser that reads the LLM's response and checks: did it ask to call a tool,
     or is it giving a final answer?
  3. If it asked for a tool: execute that tool, take the result, and feed it back
     into the conversation as an "observation" message. Then loop back to step 1.
  4. If it gave a final answer: stop and return that answer.
  5. Basic retry logic for when a tool call fails or returns garbage.
  6. A hard step limit so the agent can't loop forever.

Nothing here yet — this is just the skeleton. We fill this in step by step
once the repo layout is settled.
"""
