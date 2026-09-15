"""
Week 2: the Writer agent.

Its only job: take raw research findings and turn them into a clean, well-
written summary. It has no tools and does no searching — pure synthesis.
Keeping it single-purpose (instead of one agent that both researches AND
writes) is the actual point of "multi-agent": each agent stays simple and
good at one thing, instead of one prompt trying to juggle everything.

Week 3 addition: wrapped with Langfuse's @observe() so this call shows up
in traces, with token usage attached for cost tracking.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from langfuse import observe, get_client

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
langfuse = get_client()

MODEL = "openai/gpt-oss-120b"


@observe(as_type="generation", name="writer")
def write_summary(original_task: str, research_findings: str) -> str:
    """
    Takes the original task and the Researcher agent's findings, and returns
    a clean, well-written summary suitable for sending to someone.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clear, concise writer. Given research findings, "
                "write a well-organized summary suitable for emailing to "
                "someone. Use plain language, short paragraphs, and no "
                "mention of 'tools' or the research process itself."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original task: {original_task}\n\n"
                f"Research findings:\n{research_findings}\n\n"
                "Write a clean summary based on this."
            ),
        },
    ]

    response = client.chat.completions.create(model=MODEL, messages=messages)
    output = response.choices[0].message.content

    langfuse.update_current_generation(
        model=MODEL,
        input=original_task,
        output=output,
        usage_details={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
    )

    return output


if __name__ == "__main__":
    fake_research = (
        "Groq's free tier: no credit card required, ~30 requests/min, "
        "~14,400 requests/day, all models available including Llama 3.3 "
        "and GPT-OSS. Per-model token limits also apply, e.g. ~12K tokens/"
        "min on Llama 3.3 70B."
    )
    summary = write_summary("Summarize Groq's free tier limits", fake_research)
    print(summary)
    langfuse.flush()
