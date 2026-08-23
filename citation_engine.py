def generate_citations(documents):

    citations = []

    seen = set()

    for doc in documents:

        source = doc.metadata.get(
            "source",
            "Unknown Paper"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        key = f"{source}-{page}"

        if key in seen:
            continue

        seen.add(key)

        citations.append({
            "paper": source,
            "page": page
        })

    return citations
