from langchain_groq import ChatGroq

from config import LLM_MODEL


def summarize_paper(text):

    llm = ChatGroq(
       model="llama-3.1-8b-instant",
        temperature=0.1
    )

    prompt = f"""
You are an academic research assistant.

Summarize the following research paper.

Include:

- Research problem
- Objective
- Methodology
- Dataset
- Main findings
- Limitations
- Conclusion

Do not invent information.

Research Paper:

{text}
"""

    response = llm.invoke(prompt)

    return response.content
