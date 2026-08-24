import streamlit as st

from pdf_processor import (
    extract_pdf,
    create_chunks
)

from rag_pipeline import (
    create_vector_store,
    answer_question,
    answer_question_hybrid
)


from citation_engine import generate_citations
from global_search import search_global_papers
from citation_finder import find_citations
st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)


st.title("📚 AI Research Paper Assistant")

st.caption(
    "RAG-powered academic research assistant"
)


if "vector_db" not in st.session_state:

    st.session_state.vector_db = None


if "papers" not in st.session_state:

    st.session_state.papers = []


with st.sidebar:

    st.header("📄 Research Papers")

    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🔎 Index Papers"):

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF."
            )

        else:

            all_documents = []

            with st.spinner(
                "Processing research papers..."
            ):

                for pdf in uploaded_files:

                    pages = extract_pdf(pdf)

                    documents = create_chunks(
                        pages,
                        pdf.name
                    )

                    all_documents.extend(
                        documents
                    )

            st.session_state.vector_db = (
                create_vector_store(
                    all_documents
                )
            )

            st.session_state.papers = [
                pdf.name
                for pdf in uploaded_files
            ]

            st.success(
                f"{len(uploaded_files)} paper(s) indexed."
            )


st.subheader("💬 Ask Your Research Question")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What methodology was used "
        "in the research?"
    )
)


if st.button("🤖 Ask AI"):

    if st.session_state.vector_db is None:

        st.warning(
            "Please upload and index papers first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a research question."
        )

    else:

        with st.spinner(
            "Searching papers and generating answer..."
        ):

            answer, documents, global_papers = answer_question_hybrid(
    st.session_state.vector_db,
    question
)
            

        st.subheader("🤖 Answer")

        st.write(answer)

        st.divider()

        st.subheader("📚 Supporting Sources")

        citations = generate_citations(
            documents
        )

        for citation in citations:

            st.markdown(
                f"""
**📄 {citation["paper"]}**

Page: {citation["page"]}
"""
            )

            st.caption(
                "Retrieved passage:"
            )

        st.divider()

        st.subheader("🔎 Retrieved Evidence")

        for index, doc in enumerate(
            documents,
            start=1
        ):

            with st.expander(
                f"Source {index} — "
                f"{doc.metadata.get('source')}"
            ):

                st.write(
                    f"Page: "
                    f"{doc.metadata.get('page')}"
                )

                st.write(
                    doc.page_content
                )
st.divider()
               
st.subheader("🌍 Global Academic Paper Search")

global_query = st.text_input(
    "Enter a research topic",
    placeholder="Example: Retrieval Augmented Generation"
)

if st.button("🌍 Search Global Papers"):

    if not global_query.strip():
        st.warning("Please enter a research topic.")

    else:
        with st.spinner("Searching global academic sources..."):

            papers = search_global_papers(global_query)

        if not papers:
            st.warning("No papers found. Try another topic.")

        else:
            st.success(f"Found {len(papers)} research papers.")

            for index, paper in enumerate(papers, start=1):

                with st.expander(
                    f"📄 {index}. {paper['title']}"
                ):

                    st.write(f"**Source:** {paper['source']}")
                    st.write(f"**Year:** {paper['year']}")

                    authors = ", ".join(paper["authors"][:5])
                    st.write(f"**Authors:** {authors}")

                    if paper.get("abstract"):
                        st.write("**Abstract:**")
                        st.write(paper["abstract"])

                    if paper.get("url"):
                        st.link_button(
                            "🔗 Open Original Source",
                            paper["url"]
                        ) 
st.write("✅ GLOBAL SEARCH TEST - MAIN.PY UPDATED")  
st.divider()

st.subheader("📝 Intelligent Citation Finder")

st.caption(
    "Paste a research claim or statement to find relevant academic papers."
)

claim = st.text_area(
    "Enter your research claim",
    placeholder=(
        "Example: Retrieval-Augmented Generation "
        "can improve factual accuracy in language models."
    ),
    key="citation_claim"
)

if st.button(
    "🔍 Find Supporting Papers",
    key="find_citations_button"
):

    if not claim.strip():

        st.warning(
            "Please enter a research claim first."
        )

    else:

        with st.spinner(
            "Searching academic sources for relevant papers..."
        ):

            recommendations = find_citations(
                claim
            )

        if not recommendations:

            st.warning(
                "No relevant papers found. Try rephrasing your claim."
            )

        else:

            st.success(
                f"Found {len(recommendations)} citation recommendations."
            )

            for index, paper in enumerate(
                recommendations,
                start=1
            ):

                with st.expander(
                    f"📚 Citation {index}: {paper['title']}"
                ):

                    authors = ", ".join(
                        paper.get("authors", [])[:5]
                    )

                    st.write(
                        f"**Authors:** {authors or 'Unknown'}"
                    )

                    st.write(
                        f"**Year:** "
                        f"{paper.get('year', 'Unknown')}"
                    )

                    st.write(
                        f"**Database:** "
                        f"{paper.get('source', 'Unknown')}"
                    )

                    st.info(
                        paper.get(
                            "evidence_status",
                            "Verify source before citing"
                        )
                    )

                    if paper.get("abstract"):

                        st.write(
                            "**Available Abstract:**"
                        )

                        st.write(
                            paper["abstract"]
                        )

                    if paper.get("doi"):

                        st.write(
                            f"**DOI:** {paper['doi']}"
                        )

                    if paper.get("url"):

                        st.link_button(
                            "🔗 Verify / Open Original Paper",
                            paper["url"],
                            key=f"citation_link_{index}"
                        )