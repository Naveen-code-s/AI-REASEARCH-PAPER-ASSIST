import requests
import feedparser
from urllib.parse import quote


def search_semantic_scholar(query, limit=5):
    """
    Search research papers using Semantic Scholar API.
    Returns normalized paper metadata.
    """

    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
    )

    params = {
        "query": query,
        "limit": limit,
        "fields": (
            "title,authors,year,abstract,"
            "url,externalIds,citationCount"
        )
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        papers = []

        for paper in data.get("data", []):

            papers.append({
                "title": paper.get("title", "Unknown Title"),

                "authors": [
                    author.get("name")
                    for author in paper.get("authors", [])
                ],

                "year": paper.get("year"),

                "abstract": paper.get("abstract"),

                "url": paper.get("url"),

                "doi": (
                    paper.get("externalIds", {})
                    .get("DOI")
                ),

                "citation_count": paper.get(
                    "citationCount", 0
                ),

                "source": "Semantic Scholar"
            })

        return papers

    except Exception as error:

        print(
            f"Semantic Scholar search error: {error}"
        )

        return []


def search_arxiv(query, limit=5):
    """
    Search research papers using arXiv.
    """

    encoded_query = quote(query)

    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded_query}"
        f"&start=0&max_results={limit}"
    )

    try:

        response = requests.get(
            url,
            timeout=15
        )

        response.raise_for_status()

        feed = feedparser.parse(
            response.text
        )

        papers = []

        for entry in feed.entries:

            papers.append({
                "title": entry.title.replace(
                    "\n", " "
                ).strip(),

                "authors": [
                    author.name
                    for author in entry.authors
                ],

                "year": (
                    entry.published[:4]
                    if hasattr(entry, "published")
                    else None
                ),

                "abstract": entry.summary.replace(
                    "\n", " "
                ).strip(),

                "url": entry.link,

                "doi": None,

                "citation_count": None,

                "source": "arXiv"
            })

        return papers

    except Exception as error:

        print(
            f"arXiv search error: {error}"
        )

        return []


def search_global_papers(query, limit=5):

    semantic_papers = (
        search_semantic_scholar(query, limit)
    )

    arxiv_papers = (
        search_arxiv(query, limit)
    )

    all_papers = (
        semantic_papers + arxiv_papers
    )

    return remove_duplicates(all_papers)


def remove_duplicates(papers):

    unique_papers = []
    seen_titles = set()

    for paper in papers:

        title = paper.get(
            "title", ""
        ).lower().strip()

        if title and title not in seen_titles:

            seen_titles.add(title)

            unique_papers.append(paper)

    return unique_papers