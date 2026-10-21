import json
import re
import urllib.parse
import urllib.request
import warnings
from duckduckgo_search import DDGS


def _fetch_live_weather(query: str) -> str | None:
    query_lower = query.lower()
    if not any(w in query_lower for w in ["weather", "temperature", "temp", "forecast", "rain", "climate"]):
        return None

    noise = {
        "what", "is", "the", "current", "today", "now", "weather", "temperature",
        "temp", "in", "of", "for", "celsius", "c", "f", "fahrenheit", "forecast",
        "degree", "degrees", "show", "tell", "me", "live", "report", "at",
    }
    tokens = [w for w in re.split(r"\W+", query) if w and w.lower() not in noise]
    if not tokens:
        return None
    city = " ".join(tokens)

    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            cur = data["current_condition"][0]
            temp = cur["temp_C"]
            desc = cur["weatherDesc"][0]["value"]
            humidity = cur["humidity"]
            feels = cur["FeelsLikeC"]
            return (
                f"**Live Weather Report for {city.title()}:**\n"
                f"- Temperature: {temp}°C (Feels like: {feels}°C)\n"
                f"- Condition: {desc}\n"
                f"- Humidity: {humidity}%\n"
                f"- Source: https://wttr.in/{urllib.parse.quote(city)}"
            )
    except Exception:
        return None


def _search_wikipedia(query: str, max_results: int = 3) -> list[dict]:
    try:
        url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
            f"{urllib.parse.quote(query)}&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SimpleChatbot/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("query", {}).get("search", [])
            results = []
            for item in items[:max_results]:
                title = item.get("title", "")
                raw_snippet = item.get("snippet", "")
                clean_snippet = re.sub(r"<.*?>", "", raw_snippet)
                link = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                results.append({"title": title, "body": clean_snippet, "href": link})
            return results
    except Exception:
        return []


def web_search(query: str, max_results: int = 5) -> str:
    """Searches the live web to find up-to-date information, weather, news, or factual knowledge.

    Args:
        query: The search term or question to search the web for.
        max_results: Maximum number of search results to retrieve (default is 5).

    Returns:
        A formatted summary of top search results containing titles, snippets, and URLs.
    """
    clean_query = query.strip()
    if not clean_query:
        return "Error: Search query cannot be empty."

    raw_results = []
    ddgs_error = None

    # 1. Primary: DuckDuckGo search
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ddgs = DDGS()
            try:
                raw_results = list(ddgs.text(clean_query, backend="lite", max_results=max_results))
            except Exception as e:
                ddgs_error = str(e)

            if not raw_results and not ddgs_error:
                try:
                    raw_results = list(ddgs.text(clean_query, backend="auto", max_results=max_results))
                except Exception as e:
                    ddgs_error = str(e)
    except Exception as e:
        ddgs_error = str(e)

    if ddgs_error:
        return f"Error: Web search failed ({ddgs_error})."

    # If DDGS found results, format and return them directly
    if raw_results:
        formatted_entries = []
        for i, item in enumerate(raw_results, 1):
            title = item.get("title", "No Title").strip()
            snippet = item.get("body", "No Snippet").strip()
            url = item.get("href", "").strip()
            formatted_entries.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}")
        return "\n\n".join(formatted_entries)

    # 2. Fallback A: Specialized live weather report for weather/temperature queries
    weather_result = _fetch_live_weather(clean_query)
    if weather_result:
        return weather_result

    # 3. Fallback B: Encyclopedic knowledge via Wikipedia
    wiki_results = _search_wikipedia(clean_query, max_results=max_results)
    if wiki_results:
        formatted_entries = []
        for i, item in enumerate(wiki_results, 1):
            title = item.get("title", "No Title").strip()
            snippet = item.get("body", "No Snippet").strip()
            url = item.get("href", "").strip()
            formatted_entries.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}")
        return "\n\n".join(formatted_entries)

    # 4. Final no-results report
    return f"No web search results found for query: '{clean_query}'."
