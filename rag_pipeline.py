import os
from dotenv import load_dotenv
load_dotenv()  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K
)


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_vector_store(documents):

    embeddings = get_embeddings()

    texts = []
    metadatas = []

    for doc in documents:

        texts.append(doc["text"])

        metadatas.append({
            "source": doc["source"],
            "page": doc["page"]
        })

    vector_db = FAISS.from_texts(
        texts,
        embeddings,
        metadatas=metadatas
    )

    return vector_db


def get_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    return ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-120b",
        temperature=0.1
    )


def retrieve_documents(vector_db, question):

    return vector_db.similarity_search(
        question,
        k=TOP_K
    )


def answer_question(vector_db, question):

    documents = retrieve_documents(
        vector_db,
        question
    )

    context = ""

    for index, doc in enumerate(documents, start=1):

        context += f"""
SOURCE {index}
Paper: {doc.metadata.get("source")}
Page: {doc.metadata.get("page")}

{doc.page_content}

-------------------------
"""

    prompt = f"""
You are an AI Research Paper Assistant.

Answer the question using ONLY the provided
research-paper context.

Do not invent facts or citations.

If the answer is not supported by the context,
say:

"The uploaded research papers do not provide
enough evidence to answer this question."

Give:

1. Direct answer
2. Explanation
3. Supporting sources

Question:
{question}

Research Context:
{context}
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content, documents
def answer_question_hybrid(vector_db, question):
    """
    Answer using both uploaded PDFs and
    global academic paper search results.
    """

    # 1. Search uploaded PDFs
    local_documents = retrieve_documents(
        vector_db,
        question
    )

    # 2. Search global academic databases
    from global_search import search_global_papers
    from global_context import global_papers_to_context

    global_papers = search_global_papers(
        question,
        limit=5
    )

    # 3. Build Local PDF context
    local_context = ""

    for index, doc in enumerate(
        local_documents,
        start=1
    ):
        local_context += f"""
[LOCAL SOURCE {index}]
Paper: {doc.metadata.get("source")}
Page: {doc.metadata.get("page")}

Evidence:
{doc.page_content}

-------------------------
"""

    # 4. Convert Global papers to AI context
    global_context = global_papers_to_context(
        global_papers,
        max_papers=5
    )

    # 5. Combine both sources
    combined_context = f"""
=========================
UPLOADED PDF EVIDENCE
=========================

{local_context}

=========================
GLOBAL ACADEMIC EVIDENCE
=========================

{global_context}
"""

    # 6. Create strict research prompt
    prompt = f"""
You are an AI Research Paper Assistant.

Answer the user's question using ONLY the evidence
provided below.

SOURCE RULES:

1. LOCAL SOURCE evidence comes from uploaded PDF pages.
2. GLOBAL SOURCE evidence may come from paper metadata
   and abstracts. Do NOT claim you read the full paper
   unless full paper text is explicitly provided.
3. Do not invent facts, authors, results, or citations.
4. If evidence is insufficient, clearly say so.
5. When making a factual statement, mention the relevant
   source label where possible, for example:
   [LOCAL SOURCE 1] or [GLOBAL SOURCE 2].

Answer format:

1. Direct Answer
2. Explanation
3. Evidence-Based Sources

Question:
{question}

Available Evidence:
{combined_context}
"""

    # 7. Generate answer
    llm = get_llm()
    response = llm.invoke(prompt)

    # Return answer + both source types
    return (
        response.content,
        local_documents,
        global_papers
    )
    def answer_question_hybrid(vector_db, question):
     local_docs = vector_db.similarity_search(question)
     global_papers = search_semantic_scholar(question)
     return answer, local_docs, global_papers
    