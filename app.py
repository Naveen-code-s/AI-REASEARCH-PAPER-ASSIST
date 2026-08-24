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


if st.button("🤖 "):

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