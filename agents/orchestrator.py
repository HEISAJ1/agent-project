"""
Week 2: the orchestrator — wires Researcher, Writer, and Sender into one
actual multi-agent system using LangGraph.

The flow:
  research -> write -> review -> (pass: send) or (fail: back to write)

The "review" step is the important part: instead of blindly handing the
Writer's output to the Sender, the orchestrator checks it against a simple
rubric first. If it fails, the graph loops back to the Writer WITH the
feedback, instead of shipping something bad. This loop-back is capped at a
few attempts so a stuck review can't loop forever.
"""

import os
from typing import TypedDict, Optional
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agents import react_agent, writer_agent, sender_agent

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"
MAX_WRITE_ATTEMPTS = 3


class AgentState(TypedDict):
    task: str
    to_email: str
    research_findings: Optional[str]
    summary: Optional[str]
    review_passed: bool
    review_feedback: Optional[str]
    write_attempts: int
    final_status: Optional[str]


def research_node(state: AgentState) -> AgentState:
    findings = react_agent.research(state["task"])
    state["research_findings"] = findings or "No search results were found."
    return state


def write_node(state: AgentState) -> AgentState:
    state["write_attempts"] += 1

    task_for_writer = state["task"]
    if state.get("review_feedback"):
        task_for_writer += (
            f"\n\n(Note: a previous draft was rejected for this reason: "
            f"{state['review_feedback']}. Please fix that in this version.)"
        )

    state["summary"] = writer_agent.write_summary(
        task_for_writer, state["research_findings"]
    )
    return state


def review_node(state: AgentState) -> AgentState:
    print("\n--- Reviewing the summary before sending ---")

    review_prompt = [
        {
            "role": "system",
            "content": (
                "You are a strict editor. Judge whether the SUMMARY "
                "accurately reflects the RESEARCH and is clearly written. "
                "Respond with exactly one line: either 'PASS' or "
                "'FAIL: <short reason>'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"RESEARCH:\n{state['research_findings']}\n\n"
                f"SUMMARY:\n{state['summary']}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=MODEL, messages=review_prompt)
    verdict = response.choices[0].message.content.strip()
    print(f"Review verdict: {verdict}")

    if verdict.upper().startswith("PASS"):
        state["review_passed"] = True
        state["review_feedback"] = None
    else:
        state["review_passed"] = False
        state["review_feedback"] = verdict

    return state


def send_node(state: AgentState) -> AgentState:
    result = sender_agent.send_email(
        to_email=state["to_email"],
        subject=f"Summary: {state['task']}",
        body=state["summary"],
    )
    print(result)
    state["final_status"] = result
    return state


def after_review(state: AgentState) -> str:
    if state["review_passed"]:
        return "send"
    if state["write_attempts"] >= MAX_WRITE_ATTEMPTS:
        print("Max write attempts reached — sending best available draft.")
        return "send"
    return "rewrite"


graph = StateGraph(AgentState)

graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_node("review", review_node)
graph.add_node("send", send_node)

graph.set_entry_point("research")
graph.add_edge("research", "write")
graph.add_edge("write", "review")
graph.add_conditional_edges(
    "review", after_review, {"send": "send", "rewrite": "write"}
)
graph.add_edge("send", END)

app = graph.compile()


def run_pipeline(task: str, to_email: str) -> AgentState:
    initial_state: AgentState = {
        "task": task,
        "to_email": to_email,
        "research_findings": None,
        "summary": None,
        "review_passed": False,
        "review_feedback": None,
        "write_attempts": 0,
        "final_status": None,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    task_input = input("Enter a task: ")
    email_input = input("Enter your email to send the result to: ")
    final_state = run_pipeline(task_input, email_input)
    print("\n=== DONE ===")
    print(final_state["final_status"])
