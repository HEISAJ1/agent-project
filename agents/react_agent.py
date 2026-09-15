"""
Week 1 core (now reused as the Researcher agent in week 2): a hand-rolled
agent loop, no framework.

Two functions:
  - research(task): decides whether a search is needed, runs it if so, and
    returns the RAW findings (no writing/polishing). This is what the
    orchestrator's Researcher node calls in week 2.
  - run_agent(task): calls research(), then also writes a final answer
    itself. Kept only so this file can still be run standalone/manually
    tested on its own, the way it was in week 1.

Why a fresh prompt for the final-answer step instead of continuing the
conversation: once this model's history contains a tool call/result, it
kept trying to emit tool-call syntax again even when no tools were offered
in that request — a real reliability quirk. Starting fresh sidesteps it.

Week 3 addition: every LLM call here is wrapped with Langfuse's @observe()
decorator so it shows up as a trace, and we manually attach token usage so
cost per run is visible too.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
from langfuse import observe, get_client

from tools import search

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
langfuse = get_client()

MODEL = "openai/gpt-oss-120b"
MAX_SEARCH_ATTEMPTS = 3

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


@observe(as_type="generation", name="researcher")
def research(user_task: str) -> str | None:
    """
    Decides whether the task needs a web search, runs it if so, and returns
    the raw findings as plain text. Returns None if no search was needed
    or if every search attempt failed.
    """
    print("\n--- Researcher: deciding whether to search ---")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful research agent. Use the web_search tool "
                "if you need current information you don't already know."
            ),
        },
        {"role": "user", "content": user_task},
    ]

    for attempt in range(1, MAX_SEARCH_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMA,
            )
        except Exception as e:
            print(f"API call failed (attempt {attempt}): {e}")
            continue

        reply = response.choices[0].message

        if reply.tool_calls:
            tool_call = reply.tool_calls[0]

            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if "query" not in args:
                print(f"Malformed tool call on attempt {attempt}, retrying...")
                continue

            print(f"Researcher is searching for: {args['query']}")
            try:
                result = search.run(args)
            except Exception as e:
                result = f"Error running search: {e}"

            print(f"Tool result: {result[:200]}...")

            langfuse.update_current_generation(
                model=MODEL,
                input=user_task,
                output=result,
                usage_details={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                },
                metadata={"searched_for": args["query"]},
            )
            return result
        else:
            print("Researcher decided no search was needed.")
            langfuse.update_current_generation(
                model=MODEL,
                input=user_task,
                output=reply.content,
                usage_details={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                },
                metadata={"searched": False},
            )
            return reply.content

    print("Researcher could not complete a valid search after retries.")
    return None


@observe()
def run_agent(user_task: str) -> str:
    """Standalone use: research + write a final answer in one call."""
    search_result = research(user_task)

    print("\n--- Writing the final answer ---")

    if search_result is None:
        synthesis_messages = [
            {"role": "system", "content": "Answer the user's question as best you can."},
            {"role": "user", "content": user_task},
        ]
    else:
        synthesis_messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question directly and concisely, "
                    "using the search results provided. Do not mention tools "
                    "or the search process — just answer the question."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {user_task}\n\nSearch results:\n{search_result}",
            },
        ]

    response = client.chat.completions.create(
        model=MODEL, messages=synthesis_messages)
    return response.choices[0].message.content


if __name__ == "__main__":
    task = input("Enter a task for the agent: ")
    answer = run_agent(task)
    print("\n=== FINAL ANSWER ===")
    print(answer)
    langfuse.flush()
