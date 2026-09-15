"""
DeepEval's metrics (like GEval) are "LLM-as-a-judge" — they need a model to
grade outputs. By default DeepEval expects OpenAI, which would break the
"entire project costs $0" goal.

This file wraps our existing Groq client so DeepEval uses Groq as the judge
instead — everything, including the evals themselves, stays on the free
tier we already set up.

DeepEval requires any custom judge model to subclass DeepEvalBaseLLM and
implement: generate(), a_generate(), and get_model_name().
"""

import os
from groq import Groq
from dotenv import load_dotenv
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # DeepEval sometimes calls the async version — since we don't have
        # an async Groq client wired up, just reuse the sync one. Fine for
        # a project this size; a truly async version would use Groq's
        # AsyncGroq client instead.
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Groq ({JUDGE_MODEL})"


if __name__ == "__main__":
    judge = GroqJudge()
    result = judge.generate("Say 'judge is working' and nothing else.")
    print(result)
