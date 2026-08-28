def global_papers_to_context(papers, max_papers=5):
    """
    Convert global academic search results into
    structured context for the AI.
    """

    context_parts = []

    for index, paper in enumerate(
        papers[:max_papers],
        start=1
    ):

        title = paper.get("title", "Unknown Title")
        authors = ", ".join(
            paper.get("authors", [])[:5]
        )
        year = paper.get("year", "Unknown")
        abstract = paper.get("abstract") or "No abstract available."
        source = paper.get("source", "Global Source")
        url = paper.get("url", "")

        paper_context = f"""
[GLOBAL SOURCE {index}]
Title: {title}
Authors: {authors}
Year: {year}
Database: {source}
URL: {url}

Abstract / Available Evidence:
{abstract}
"""

        context_parts.append(paper_context)

    return "\n\n".join(context_parts)