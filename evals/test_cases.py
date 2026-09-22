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

Attribution addition: after live testing surfaced a real bug class — a
vague research task producing an ambiguous search query, which pulled
back mixed-entity results (men's vs women's team, different
competitions) that a summary then misattributed to the wrong entity —
two more tests were added.

The fix has three layers, and the tests reflect what each layer can
actually guarantee:
  1. Researcher query disambiguation (react_agent.py's system prompt):
     nudges the model to phrase a more specific query. This is inherent
     LLM behavior and can't be guaranteed every run, so
     test_researcher_disambiguates_ambiguous_entity WARNS rather than
     fails the suite when it doesn't happen — useful signal for tuning,
     not a hard requirement.
  2. Writer's own caution (writer_agent.py's system prompt): also
     improved, but proven insufficient alone (see project history).
  3. Review step attribution check (orchestrator.py's review_node): the
     real, reliable fix. It's a fresh, independent pass that catches
     misattribution for ANY entity, not just ones already seen, and
     doesn't depend on the Researcher or Writer getting it right on the
     first try. test_review_catches_misattributed_fact tests this
     directly and deterministically — this is the test that actually
     matters, since review is the last checkpoint before anything is
     sent.

An earlier version of this suite also had a test holding the Writer
ALONE to the attribution bar (test_writer_attributes_facts_correctly).
It was removed: since the real fix relies on review catching what the
Writer's first pass misses, that test was checking the wrong layer.

Run with: deepeval test run evals/test_cases.py
"""

import warnings
import pytest
from unittest.mock import patch
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents import react_agent, writer_agent, orchestrator
from evals.groq_judge import GroqJudge
from tools import search

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


# Soft check: a real bug where a vague research task ("summary chelsea")
# led the Researcher to search with a bare, ambiguous query, which pulled
# back a mix of results (men's team, women's team, different
# competitions/seasons). The prompt now nudges the Researcher to
# disambiguate its query, but since that's LLM phrasing behavior, it
# can't be guaranteed every run — so this WARNS instead of failing the
# suite. It's useful signal (are we still drifting toward vague
# queries?) without blocking CI on something the review step already
# covers as a hard backstop.
def test_researcher_disambiguates_ambiguous_entity():
    captured_queries = []
    original_search = search.run

    def spy_search(args):
        captured_queries.append(args.get("query", ""))
        return original_search(args)

    with patch("agents.react_agent.search.run", side_effect=spy_search):
        react_agent.research("summary chelsea")

    assert captured_queries, "Researcher did not perform a search at all"
    query = captured_queries[0].lower()
    disambiguating_terms = ["men's", "men ", "premier league", "first team"]
    if not any(term in query for term in disambiguating_terms):
        warnings.warn(
            f"Researcher's query ('{captured_queries[0]}') didn't include "
            "an explicit disambiguating term. Not a hard failure — the "
            "review step (test_review_catches_misattributed_fact) is the "
            "actual backstop against misattributed facts reaching output "
            "— but worth watching if this keeps happening."
        )


# Exact noisy research findings from the run that produced the "5-0 win
# over Bournemouth" mix-up, where a summary presented a Chelsea Women's
# result (tied to Sonia Bompastor, Chelsea Women's manager) as if it were
# a men's first-team Premier League result. Used below to directly test
# the review step's attribution-checking criterion.
ATTRIBUTION_CASES = [
    {
        "task": "summary chelsea",
        "research": (
            "1. 2026–27 Chelsea F.C. season\n"
            "https://en.wikipedia.org/wiki/2026%E2%80%9327_Chelsea_F.C._season\n"
            "| Competition | First match | Last match | Starting round | "
            "Final position | Record |\n"
            "| Premier League | 24 August 2026 | 30 May 2027 | Matchday 1 "
            "| TBD | 4 | 2 | 1 | 1 | 10 | 9 | +1 | 050.00 |\n"
            "| FA Cup | 8–11 January 2027 | TBD | T\n\n"
            "2. Key dates for Chelsea's 2026/27 season | Official Site\n"
            "Saturday 9 January – Chelsea begin their FA Cup campaign in "
            "the third round. Saturday 13 March - Arsenal visit Stamford "
            "Bridge. Sunday 21 March – Carabao Cup final at Wembley.\n\n"
            "3. Chelsea FC: Fixtures, Squad & Results | worldfootball.net\n"
            "| from | Name | Pos | Team |\n"
            "| 08/2026 | Emiliano Martínez | GK | Aston Villa |\n"
            "| 08/2026 | Pep Chavarría | DF | Rayo Vallecano |\n"
            "| 08/2026 | Jordan Henderson | MF |\n\n"
            "4. Chelsea Fixtures, Results & Standings 2026/27 | Premier "
            "League\n"
            "Premier League report: Chelsea 5-0 Bournemouth Sonia "
            "Bompastor\n\n"
            "5. Chelsea season review: Sackings, disobedience and "
            "'huddlegate' define a woeful 2025-26 - The Athletic\n"
            "Chelsea finished bottom of the Premier League fair play "
            "table for the second time in three seasons, accumulating "
            "eight red cards, double the next-worst, Tottenham (11 "
            "total across all competitions)."
        ),
    },
]


# Direct test of the review step's attribution-checking criterion. Feeds
# review_node a summary we know contains a real misattribution and
# asserts review correctly rejects it, rather than relying on the
# Researcher happening to reproduce the same ambiguous search results
# again (which is non-deterministic). This is the fix that matters: a
# fresh, independent pass that catches misattribution for any entity —
# and the one test in this file that actually gates whether a bad output
# could reach a real email.
def test_review_catches_misattributed_fact():
    bad_state = {
        "task": "summary chelsea",
        "to_email": "test@example.com",
        "research_findings": ATTRIBUTION_CASES[0]["research"],
        "summary": (
            "Chelsea's recent results include a 5-0 win over Bournemouth. "
            "The club is also managed by Sonia Bompastor and had a strong "
            "showing in the Premier League this season, alongside squad "
            "additions including Emiliano Martinez and Jordan Henderson."
        ),
        "review_passed": False,
        "review_feedback": None,
        "write_attempts": 1,
        "final_status": None,
    }

    result_state = orchestrator.review_node(bad_state)

    assert result_state["review_passed"] is False, (
        "Review incorrectly PASSED a summary that misattributes a "
        "Chelsea Women's result (tied to Sonia Bompastor) to the men's "
        f"team. Verdict was: {result_state['review_feedback']}"
    )