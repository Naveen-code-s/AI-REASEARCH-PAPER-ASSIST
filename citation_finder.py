from global_search import search_global_papers


def find_citations(claim, limit=5):
    """
    Find global academic papers that may be relevant
    to a user's research claim.
    """

    papers = search_global_papers(
        claim,
        limit=limit
    )

    recommendations = []

    for paper in papers:

        # Basic evidence check:
        # A paper without an abstract is still shown,
        # but clearly marked as a recommendation rather
        # than proof of the claim.
        has_abstract = bool(
            paper.get("abstract")
        )

        recommendation = {
            "title": paper.get(
                "title",
                "Unknown Title"
            ),
            "authors": paper.get(
                "authors",
                []
            ),
            "year": paper.get(
                "year",
                "Unknown"
            ),
            "source": paper.get(
                "source",
                "Unknown"
            ),
            "abstract": paper.get(
                "abstract"
            ),
            "url": paper.get(
                "url"
            ),
            "doi": paper.get(
                "doi"
            ),
            "evidence_status": (
                "Abstract available"
                if has_abstract
                else "Metadata only — verify before citing"
            )
        }

        recommendations.append(
            recommendation
        )

    return recommendations