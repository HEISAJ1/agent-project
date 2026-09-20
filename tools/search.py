"""
The agent's first tool: web search, via Tavily.

Every tool in this project follows the same shape: takes a dict of arguments,
returns a plain string the LLM can read. That consistency is what lets the
agent loop call any tool the same way, without special-casing each one.

Freshness note: Tavily includes a "published_date" on results when the
source page provides one - not every page does, so it's included only
when present rather than guessed. Surfacing it is what lets the agent (and
the final summary) ground claims to a real date instead of presenting
search results as timeless facts.
"""

import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def run(args: dict) -> str:
    """
    args expected shape: {"query": "some search text"}
    Returns a plain-text summary of the top search results, including each
    result's published date where Tavily provides one.
    """
    query = args.get("query", "")
    if not query:
        return "Error: no search query was provided."

    try:
        results = client.search(query=query, max_results=5)
    except Exception as e:
        return f"Error: search failed ({e})"

    entries = results.get("results", [])
    if not entries:
        return f"No results found for '{query}'."

    formatted = []
    for i, entry in enumerate(entries, start=1):
        title = entry.get("title", "Untitled")
        url = entry.get("url", "")
        snippet = entry.get("content", "")[:300]
        published = entry.get("published_date")
        date_line = f"   Published: {published}\n" if published else ""
        formatted.append(f"{i}. {title}\n{date_line}   {url}\n   {snippet}")

    return "\n\n".join(formatted)


if __name__ == "__main__":
    test_result = run({"query": "current Groq API rate limits free tier"})
    print(test_result)
