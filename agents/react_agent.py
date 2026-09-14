
"""
Week 1 core: a hand-rolled agent loop, no framework.

Design, in plain terms:
  1. Ask the LLM: does this task need a web search, or can you answer directly?
  2. If it wants to search: run the search tool for real, get the results.
  3. Hand those results to the model in a FRESH, clean prompt (not a
     continuation of the same conversation) and ask for a final answer.
  4. If it can answer directly with no search: just return that.

Why a fresh prompt for step 3 instead of continuing the conversation:
once this model's conversation history contains a tool call/result, it kept
trying to emit tool-call syntax again even when no tools were offered in
that request — a real reliability quirk. Starting the final answer as a
clean, new request sidesteps that entirely instead of fighting it.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

from tools import search

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

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


def run_agent(user_task: str) -> str:
    print("\n--- Step 1: deciding whether to search ---")

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

    search_result = None

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

            print(f"Agent is searching for: {args['query']}")
            try:
                search_result = search.run(args)
            except Exception as e:
                search_result = f"Error running search: {e}"

            print(f"Tool result: {search_result[:200]}...")
            break
        else:
            print("Agent answered directly, no search needed.")
            return reply.content

    print("\n--- Step 2: writing the final answer ---")

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
