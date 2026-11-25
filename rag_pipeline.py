
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    TEMPERATURE,
)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_vector_store(documents):
    if not documents:
        raise ValueError("No valid PDF text was extracted.")

    texts = []
    metadatas = []

    for doc in documents:
        text = doc.get("text", "").strip()

        if not text:
            continue

        texts.append(text)

        metadatas.append({
            "source": doc.get("source", "Unknown paper"),
            "page": doc.get("page", "Unknown"),
        })

    if not texts:
        raise ValueError("No readable text found in uploaded papers.")

    return FAISS.from_texts(
        texts=texts,
        embedding=get_embeddings(),
        metadatas=metadatas,
    )


def get_llm():
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Configure it in your local .env file."
        )

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=TEMPERATURE,
    )


def retrieve_documents(vector_db, question, k=None):
    if vector_db is None:
        raise ValueError("Research papers have not been indexed.")

    if not question or not question.strip():
        raise ValueError("Enter a research question.")

    return vector_db.similarity_search(
        question,
        k=k or TOP_K,
    )


def _build_local_context(documents):
    sections = []

    for index, doc in enumerate(documents, start=1):
        sections.append(
            f"""
[LOCAL SOURCE {index}]
Paper: {doc.metadata.get("source", "Unknown")}
Page: {doc.metadata.get("page", "Unknown")}

Evidence:
{doc.page_content}
"""
        )

    return "\n".join(sections)


def answer_question(vector_db, question):
    documents = retrieve_documents(
        vector_db,
        question,
    )

    context = _build_local_context(documents)

    prompt = f"""
You are ResearchAI, an academic research assistant.

Answer using ONLY the uploaded research-paper evidence below.

Rules:
- Do not invent facts or citations.
- Clearly identify unsupported information.
- Reference source numbers and page numbers.
- Distinguish facts from interpretation.

Structure:
1. Direct Answer
2. Detailed Explanation
3. Supporting Evidence
4. Limitations or Missing Information

Question:
{question}

Research Evidence:
{context}
"""

    response = get_llm().invoke(prompt)

    return response.content, documents


def answer_question_hybrid(vector_db, question):
    """
    Combine uploaded PDF evidence with global academic
    paper metadata and abstracts.
    """

    local_documents = retrieve_documents(
        vector_db,
        question,
    )

    from global_search import search_global_papers
    from global_context import global_papers_to_context

    try:
        global_papers = search_global_papers(
            question,
            limit=5,
        )
    except Exception:
        global_papers = []

    local_context = _build_local_context(
        local_documents
    )

    global_context = global_papers_to_context(
        global_papers,
        max_papers=5,
    )

    prompt = f"""
You are ResearchAI, an evidence-focused academic assistant.

Answer the user's question using the available evidence.

LOCAL PDF EVIDENCE:
{local_context}

GLOBAL ACADEMIC EVIDENCE:
{global_context}

IMPORTANT RULES:
1. Local evidence comes from uploaded PDF passages.
2. Global evidence may contain only metadata or abstracts.
3. Never claim to have read a full global paper unless its
   full text is actually available.
4. Never invent results, authors, citations or statistics.
5. Clearly identify missing or insufficient evidence.
6. Reference source labels when making factual claims.
7. Distinguish established evidence from suggestions.

ANSWER FORMAT:
1. Direct Answer
2. Detailed Explanation
3. Uploaded Paper Evidence
4. Global Research References
5. Limitations and Research Gaps

Question:
{question}
"""

    response = get_llm().invoke(prompt)

    return (
        response.content,
        local_documents,
        global_papers,
    )
