"""
Week 2: the Writer agent.

Its only job: take raw research findings and turn them into a clean, well-
written summary. It has no tools and does no searching — pure synthesis.
Keeping it single-purpose (instead of one agent that both researches AND
writes) is the actual point of "multi-agent": each agent stays simple and
good at one thing, instead of one prompt trying to juggle everything.

Week 3 addition: wrapped with Langfuse's @observe() so this call shows up
in traces, with token usage attached for cost tracking.

Freshness addition: told the real current date and instructed to carry
over any dates present in the research findings (e.g. "as of March 2026")
rather than presenting search results as timeless facts.

Attribution-fix addition: after live testing showed the Writer could pull
a real fact from noisy multi-source search results but attach it to the
wrong team/competition/date (e.g. crediting a goal to the wrong club, or
reporting a Chelsea Women's result as if it were the men's team), the
prompt now requires every specific fact to stay tied to its full
disambiguating context, or be dropped/flagged instead of stated plainly.
"""

import os
from datetime import date
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
    today = date.today().strftime("%B %d, %Y")

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a clear, concise writer. Today's date is {today}. "
                "Given research findings, write a well-organized summary "
                "suitable for emailing to someone. Use plain language, "
                "short paragraphs, and no mention of 'tools' or the "
                "research process itself. Only include facts that are "
                "actually present in the research findings provided — do "
                "not add information, even if you believe it to be true, "
                "and do not omit any of the findings' key points. If the "
                "findings include a publication date, reflect that "
                "recency (e.g. 'as of <date>') instead of stating things "
                "as permanent, unchanging facts. If the findings are "
                "ambiguous, incomplete, or don't clearly state a specific "
                "detail, say that plainly rather than filling the gap "
                "with a confident-sounding guess. Every specific fact you "
                "include (a score, a scorer, a signing, a quote) must stay "
                "attached to the exact context that identifies it in the "
                "source — which team, which competition (e.g. men's vs "
                "women's, league vs cup), and which date. If that context "
                "is missing, unclear, or conflicts across sources, either "
                "state the ambiguity explicitly or leave the fact out "
                "entirely — never present a fact as if it applies to a "
                "different match, team, or competition than the source "
                "actually specifies, even if it seems like a safe "
                "assumption."
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