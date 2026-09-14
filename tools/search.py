from dotenv import load_dotenv
import os
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def run(args: dict) -> str:
    """
    args expected shape: {"query": "some search text"}
    Returns a plain-text summary of the top search results.
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
        formatted.append(f"{i}. {title}\n   {url}\n   {snippet}")

    return "\n\n".join(formatted)


# Quick manual test — run this file directly to check the tool works on its own,
# before the agent loop ever touches it.
if __name__ == "__main__":
    test_result = run({"query": "current Groq API rate limits free tier"})
    print(test_result)
