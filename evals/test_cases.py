"""
Week 3: the eval suite.

This is what proves the system works reliably, instead of "I tried it a
few times and it seemed fine." Each test case runs a real agent function
and scores its output using GEval (an LLM-as-a-judge metric) — graded by
our own Groq judge model, not OpenAI, to keep the whole project free.

Two things get tested here:
  1. The Researcher: given a task, does it come back with findings that
     are actually relevant to that task?
  2. The Writer: given research findings, does it write a summary that
     accurately reflects them (no invented claims) and reads clearly?

This is a starting suite (12 cases) covering the core behavior of both
agents. A production version of this would grow to 20-30+ cases,
including deliberately tricky/edge cases — this is the foundation that
pattern extends from.

Run with: deepeval test run evals/test_cases.py
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents import react_agent, writer_agent
from evals.groq_judge import GroqJudge

judge = GroqJudge()

research_relevance = GEval(
    name="Research Relevance",
    criteria=(
        "Determine whether the ACTUAL_OUTPUT contains information that is "
        "directly relevant to answering the question in INPUT. It does not "
        "need to be a complete final answer — just relevant findings."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT,
                       LLMTestCaseParams.ACTUAL_OUTPUT],
    model=judge,
    threshold=0.5,
)

writer_accuracy = GEval(
    name="Writer Accuracy And Clarity",
    criteria=(
        "Determine whether the ACTUAL_OUTPUT is a clear, well-organized "
        "summary that accurately reflects the information given in "
        "EXPECTED_OUTPUT, without introducing claims that are not "
        "supported by it."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=judge,
    threshold=0.5,
)


RESEARCH_TASKS = [
    "What is the current Groq free tier rate limit?",
    "What is the latest stable version of Python?",
    "What is the capital of France?",
    "What is 12 multiplied by 8?",
    "What is Brevo's daily email sending limit on its free plan?",
    "What model does LangGraph recommend for orchestration in 2026?",
]


@pytest.mark.parametrize("task", RESEARCH_TASKS)
def test_researcher_produces_relevant_findings(task):
    findings = react_agent.research(task)
    assert findings is not None, "Researcher failed to produce any findings at all"

    test_case = LLMTestCase(input=task, actual_output=findings)
    assert_test(test_case, [research_relevance])


WRITER_CASES = [
    {
        "task": "Summarize Groq's free tier limits",
        "research": (
            "Groq's free tier: no credit card required, ~30 requests/min, "
            "~14,400 requests/day, all models available. Per-model token "
            "limits apply, e.g. ~12K tokens/min on Llama 3.3 70B."
        ),
    },
    {
        "task": "Summarize what LangGraph is used for",
        "research": (
            "LangGraph is a library for building stateful, multi-agent LLM "
            "applications as graphs. It supports checkpointing, conditional "
            "routing between nodes, and is commonly used to orchestrate "
            "multiple specialized agents."
        ),
    },
    {
        "task": "Summarize Brevo's free email plan",
        "research": (
            "Brevo's free plan allows sending up to 300 emails per day at "
            "no cost, with no credit card required to sign up. It includes "
            "transactional email sending via API."
        ),
    },
    {
        "task": "Summarize what DeepEval is",
        "research": (
            "DeepEval is an open-source Python framework for unit testing "
            "LLM applications, built on top of pytest. It provides metrics "
            "like GEval, answer relevancy, and faithfulness, most of which "
            "use an LLM as a judge to score outputs."
        ),
    },
    {
        "task": "Summarize the ReAct agent pattern",
        "research": (
            "ReAct stands for Reason + Act. It is a pattern where an LLM "
            "alternates between reasoning about what to do next and taking "
            "an action (like calling a tool), observing the result, and "
            "repeating until it has enough information to answer."
        ),
    },
    {
        "task": "Summarize why observability matters for LLM agents",
        "research": (
            "Observability for LLM agents means tracing every decision, "
            "tool call, and cost per run. Without it, debugging a "
            "misbehaving agent is guesswork. Tools like Langfuse capture "
            "this automatically via tracing decorators."
        ),
    },
]


@pytest.mark.parametrize("case", WRITER_CASES)
def test_writer_reflects_research_accurately(case):
    summary = writer_agent.write_summary(case["task"], case["research"])

    test_case = LLMTestCase(
        input=case["task"],
        actual_output=summary,
        expected_output=case["research"],
    )
    assert_test(test_case, [writer_accuracy])
