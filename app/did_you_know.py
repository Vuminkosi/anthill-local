"""Category catalog and bounded Wikipedia research for Did You Know videos."""

from __future__ import annotations

from collections.abc import Callable

import requests


RESEARCH_API = "https://en.wikipedia.org/w/api.php"
RESEARCH_USER_AGENT = "AntHillStudio/1.0 (local Did You Know research)"
RESEARCH_LICENSE = "CC BY-SA 4.0"

CATEGORIES = {
    "science": {
        "label": "Science",
        "subcategories": {
            "space": {"label": "Space and astronomy", "query": "space astronomy"},
            "human_body": {"label": "The human body", "query": "human body biology"},
            "physics": {"label": "Physics and chemistry", "query": "physics chemistry"},
            "inventions": {"label": "Scientific inventions", "query": "scientific inventions"},
        },
    },
    "nature": {
        "label": "Nature",
        "subcategories": {
            "animals": {"label": "Animals", "query": "animal biology"},
            "ocean": {"label": "Oceans and marine life", "query": "ocean marine life"},
            "plants": {"label": "Plants and forests", "query": "plants forests botany"},
            "weather": {"label": "Weather and climate", "query": "weather climate"},
        },
    },
    "history": {
        "label": "History",
        "subcategories": {
            "ancient": {"label": "The ancient world", "query": "ancient civilizations"},
            "inventions": {"label": "Historical inventions", "query": "history inventions"},
            "exploration": {"label": "Exploration and discoveries", "query": "exploration discoveries history"},
            "events": {"label": "World events", "query": "world history events"},
        },
    },
    "geography": {
        "label": "Geography",
        "subcategories": {
            "countries": {"label": "Countries and places", "query": "countries geography"},
            "landmarks": {"label": "Landmarks", "query": "world landmarks"},
            "earth": {"label": "Earth and geology", "query": "earth geology"},
            "cultures": {"label": "Cultures and traditions", "query": "world cultures traditions"},
        },
    },
    "technology": {
        "label": "Technology",
        "subcategories": {
            "computing": {"label": "Computers and AI", "query": "computing artificial intelligence"},
            "engineering": {"label": "Engineering and machines", "query": "engineering machines"},
            "everyday": {"label": "Everyday technology", "query": "everyday technology inventions"},
            "transport": {"label": "Transport", "query": "transport technology"},
        },
    },
    "human_relations": {
        "label": "Human Relations",
        "subcategories": {
            "communication": {"label": "Communication", "query": "human communication interpersonal relationships"},
            "friendship": {"label": "Friendship", "query": "friendship social relationships"},
            "conflict": {"label": "Conflict and cooperation", "query": "interpersonal conflict cooperation"},
            "family": {"label": "Family relationships", "query": "family relationships sociology"},
            "empathy": {"label": "Empathy and social behavior", "query": "empathy social behavior psychology"},
        },
    },
    "politics": {
        "label": "Politics",
        "subcategories": {
            "systems": {"label": "Political systems", "query": "political systems government"},
            "leadership": {"label": "Leadership and governance", "query": "political leadership governance"},
            "elections": {"label": "Elections", "query": "elections electoral systems"},
            "diplomacy": {"label": "Diplomacy and international relations", "query": "diplomacy international relations"},
            "movements": {"label": "Political movements", "query": "political movements history"},
        },
    },
    "romantic_relationships": {
        "label": "Romantic Relationships",
        "subcategories": {
            "history": {"label": "History and culture of love", "query": "romantic love history culture"},
            "communication": {"label": "Communication and attachment", "query": "romantic relationships communication attachment"},
            "courtship": {"label": "Courtship and dating", "query": "courtship dating history"},
            "separation": {"label": "Breakups and reconciliation", "query": "romantic relationship breakup reconciliation"},
        },
    },
    "bible_stories": {
        "label": "Bible Stories",
        "subcategories": {
            "old_testament": {"label": "Old Testament stories", "query": "Hebrew Bible Old Testament narratives"},
            "new_testament": {"label": "New Testament stories", "query": "New Testament narratives Bible"},
            "book_of_enoch": {"label": "The Book of Enoch", "query": "Book of Enoch"},
            "angels": {"label": "Angels (The Heavenly, the Fallen, and the Holy Ones Among Us)", "query": "angels Bible fallen angels heavenly beings"},
            "heavenly_help": {"label": "Angels and heavenly help available to us", "query": "angels divine assistance heavenly help religious traditions"},
            "heavenly_angels": {"label": "Heavenly angels and messengers", "query": "heavenly angels messengers Bible"},
            "fallen_angels": {"label": "Fallen angels", "query": "fallen angels Bible religious traditions"},
            "holy_ones": {"label": "Holy ones among us", "query": "holy ones angels among humans religious traditions"},
        },
    },
    "marriages": {
        "label": "Marriages",
        "subcategories": {
            "traditions": {"label": "Wedding traditions around the world", "query": "wedding traditions cultures"},
            "history": {"label": "Marriage through history", "query": "marriage history social institution"},
            "partnership": {"label": "Partnership and family life", "query": "marriage partnership family sociology"},
            "communication": {"label": "Communication in marriage", "query": "marital communication relationships"},
        },
    },
    "marriage_relationship_dynamics": {
        "label": "Marriage and Relationship Dynamics",
        "subcategories": {
            "communication": {"label": "Communication and listening", "query": "marriage relationship communication listening"},
            "conflict_repair": {"label": "Conflict and repair", "query": "marital conflict relationship repair"},
            "shared_roles": {"label": "Shared roles and decisions", "query": "marriage household roles decision making"},
            "intimacy": {"label": "Intimacy and connection", "query": "marital intimacy relationship connection"},
            "change": {"label": "Relationships through life changes", "query": "marriage relationship life transitions"},
        },
    },
    "business": {
        "label": "Business",
        "subcategories": {
            "entrepreneurship": {"label": "Entrepreneurship", "query": "entrepreneurship business founders"},
            "companies": {"label": "Companies and industries", "query": "companies business industry history"},
            "leadership": {"label": "Business leadership", "query": "business leadership management"},
            "economics": {"label": "Markets and economics", "query": "business markets economics"},
            "workplace": {"label": "Work and the workplace", "query": "workplace business organizational behavior"},
        },
    },
}


def category_selection(category: str, subcategory: str) -> tuple[str, str, str]:
    """Return validated category labels and research terms for stable keys."""
    selected_category = CATEGORIES.get(category)
    if selected_category is None:
        raise ValueError("Choose a supported Did You Know category.")
    selected_subcategory = selected_category["subcategories"].get(subcategory)
    if selected_subcategory is None:
        raise ValueError("Choose a subcategory from the selected category.")
    return (
        selected_category["label"],
        selected_subcategory["label"],
        selected_subcategory["query"],
    )


def fetch_research_sources(
    category: str,
    subcategory: str,
    cancelled: Callable[[], bool] | None = None,
) -> list[dict[str, str]]:
    """Fetch a bounded set of article lead extracts for the selected topic only."""
    category_label, subcategory_label, search_terms = category_selection(category, subcategory)
    if cancelled is not None and cancelled():
        raise RuntimeError("Did You Know research was cancelled.")

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{search_terms} facts",
        "gsrnamespace": 0,
        "gsrlimit": 5,
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "exchars": 1800,
        "inprop": "url",
        "format": "json",
        "formatversion": 2,
        "maxlag": 5,
        "maxage": 86400,
    }
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            RESEARCH_API,
            params=params,
            headers={"User-Agent": RESEARCH_USER_AGENT, "Accept": "application/json"},
            timeout=(10, 25),
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise RuntimeError("Wikipedia research returned an unexpected redirect.")
        response.raise_for_status()
        payload = response.json()

    if cancelled is not None and cancelled():
        raise RuntimeError("Did You Know research was cancelled.")
    pages = payload.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    sources: list[dict[str, str]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        title = page.get("title")
        extract = page.get("extract")
        url = page.get("fullurl")
        if not isinstance(title, str) or not isinstance(extract, str) or len(extract.strip()) < 180:
            continue
        if not isinstance(url, str) or not url.startswith("https://en.wikipedia.org/wiki/"):
            continue
        sources.append(
            {
                "id": f"S{len(sources) + 1}",
                "title": title[:180],
                "url": url,
                "extract": extract.strip()[:1800],
                "category": category_label,
                "subcategory": subcategory_label,
                "license": RESEARCH_LICENSE,
            }
        )
        if len(sources) == 4:
            break
    if not sources:
        raise RuntimeError("Wikipedia did not return usable source material for that selection.")
    return sources
