import os

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
