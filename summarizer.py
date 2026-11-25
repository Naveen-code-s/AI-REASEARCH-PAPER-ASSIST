
from rag_pipeline import get_llm


def summarize_paper(text):
    if not text or not text.strip():
        raise ValueError("No research-paper text provided.")

    prompt = f"""
You are ResearchAI, an academic research analyst.

Summarize the provided research paper.

Use these sections:

1. Research Problem
2. Research Objectives
3. Methodology
4. Dataset and Participants
5. Tools and Technologies
6. Main Findings
7. Limitations
8. Conclusion
9. Future Research Directions

Rules:
- Use only the supplied paper text.
- Do not invent datasets, results or statistics.
- Mark unavailable information as "Not reported".
- Preserve important technical terminology.

Research Paper:
{text}
"""

    response = get_llm().invoke(prompt)

    return response.content
