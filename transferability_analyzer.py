
import json
import re
from typing import Any

from rag_pipeline import get_llm, retrieve_documents


ANALYSIS_DIMENSIONS = [
    "research_design",
    "population",
    "sample_and_dataset",
    "research_environment",
    "tools_and_resources",
    "methodology",
    "evaluation_metrics",
    "limitations",
]


def _extract_json(text: str) -> dict[str, Any]:
    """Safely extract a JSON object from an LLM response."""
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if not match:
        raise ValueError("The AI did not return a valid JSON report.")

    result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError("Unexpected analysis response format.")

    return result


def _format_evidence(documents) -> str:
    """Format retrieved PDF passages with source and page metadata."""
    if not documents:
        return "No supporting passages were retrieved."

    sections = []

    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown paper")
        page = doc.metadata.get("page", "Unknown")

        sections.append(
            f"""
[EVIDENCE {index}]
Paper: {source}
Page: {page}
Passage:
{doc.page_content}
"""
        )

    return "\n".join(sections)


def analyze_transferability(
    vector_db,
    source_paper: str,
    target_context: str,
) -> dict[str, Any]:
    """
    Evaluate whether a methodology from an uploaded source paper
    may be adapted to a specified target research context.

    The result is an evidence-based assessment, not proof that
    a method will work in the target environment.
    """

    if vector_db is None:
        raise ValueError("Index at least one research paper first.")

    if not source_paper.strip():
        raise ValueError("Select a source research paper.")

    if not target_context.strip():
        raise ValueError("Describe the target research context.")

    retrieval_query = (
        f"Research methodology, research design, dataset, "
        f"population, sample, experimental setup, tools, "
        f"evaluation metrics and limitations in {source_paper}"
    )

    documents = retrieve_documents(
        vector_db,
        retrieval_query,
    )

    # Prefer evidence from the selected source paper.
    source_documents = [
        doc for doc in documents
        if doc.metadata.get("source") == source_paper
    ]

    if source_documents:
        documents = source_documents

    evidence = _format_evidence(documents)

    prompt = f"""
You are an academic research methodology analyst.

Your task is to evaluate whether a methodology from a source
research paper can be transferred or adapted to a target
research context.

SOURCE PAPER:
{source_paper}

TARGET RESEARCH CONTEXT:
{target_context}

RETRIEVED SOURCE-PAPER EVIDENCE:
{evidence}

STRICT EVIDENCE RULES:
1. Use only the supplied evidence to describe the source paper.
2. Never invent a dataset, sample size, research procedure,
   result, limitation, or evaluation metric.
3. If a detail is absent, write "Not reported in retrieved evidence".
4. Distinguish direct reuse from adaptation.
5. Do not claim that transferability guarantees successful results.
6. Every evidence-backed observation must cite its evidence
   number and page when available.
7. If the retrieved passages are insufficient, clearly state that.
8. Do not manufacture a numerical transferability score.

Return ONLY valid JSON with this structure:

{{
  "source_paper": "{source_paper}",
  "target_context": "{target_context}",
  "overall_assessment": "Potentially transferable | Transferable with adaptation | Limited transferability | Insufficient evidence",
  "executive_summary": "Short explanation",
  "dimensions": [
    {{
      "dimension": "Research design",
      "status": "Reusable | Adaptation required | Not directly transferable | Unknown",
      "source_observation": "What the evidence says",
      "target_requirement": "What the target context requires",
      "recommendation": "Practical recommendation",
      "evidence": ["EVIDENCE 1, page 3"]
    }}
  ],
  "reusable_components": ["..."],
  "required_adaptations": ["..."],
  "potential_risks": ["..."],
  "missing_information": ["..."],
  "validation_plan": ["..."],
  "limitations_of_this_assessment": ["..."]
}}

Include all eight dimensions:
{", ".join(ANALYSIS_DIMENSIONS)}

Return a cautious, academically responsible analysis.
"""

    response = get_llm().invoke(prompt)

    result = _extract_json(response.content)

    # Preserve the retrieved evidence separately for UI citations.
    result["retrieved_evidence"] = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown"),
            "passage": doc.page_content,
        }
        for doc in documents
    ]

    return result
