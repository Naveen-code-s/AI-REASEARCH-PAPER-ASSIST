
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from pdf_processor import extract_pdf, create_chunks
from rag_pipeline import (
    create_vector_store,
    answer_question_hybrid,
)
from citation_engine import generate_citations
from global_search import search_global_papers
from citation_finder import find_citations
from summarizer import summarize_paper
from transferability_analyzer import analyze_transferability


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="ResearchAI | Academic Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #292344;
        --muted: #77718d;
        --purple: #7352c7;
        --purple-dark: #4d348d;
        --lavender: #f1edff;
        --border: #e8e2f5;
        --surface: #ffffff;
        --background: #f8f7fc;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 0%, #eee8ff 0, transparent 28%),
            linear-gradient(135deg, #faf9ff 0%, #f7f6fc 55%, #f4f1fb 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(248,247,252,0.88);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #211b3c 0%, #302451 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f7f3ff !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c9c0e5 !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding: 9px 12px;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.09);
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.5px;
    }

    p, label, .stMarkdown {
        color: var(--ink);
    }

    .hero {
        padding: 30px 32px;
        border-radius: 22px;
        color: white;
        background: linear-gradient(115deg, #35245f 0%, #6042a5 55%, #9278dc 100%);
        box-shadow: 0 16px 40px rgba(77,52,141,0.18);
        margin: 8px 0 24px 0;
    }

    .hero h1 {
        color: white !important;
        font-size: 32px;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #eee8ff !important;
        margin: 0;
        font-size: 15px;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 800;
        color: #d9ccff;
        margin-bottom: 10px;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 19px;
        min-height: 112px;
        box-shadow: 0 5px 18px rgba(48,35,79,0.04);
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .metric-value {
        color: var(--ink);
        font-size: 27px;
        font-weight: 800;
        font-family: 'Manrope', sans-serif;
    }

    .panel {
        background: white;
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 20px;
        box-shadow: 0 5px 18px rgba(48,35,79,0.035);
        margin-bottom: 14px;
    }

    .paper-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .paper-title {
        font-weight: 800;
        color: var(--ink);
        font-size: 15px;
    }

    .muted {
        color: var(--muted);
        font-size: 13px;
    }

    .stTextInput input,
    .stTextArea textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        background: #ffffff !important;
        color: #292344 !important;
        -webkit-text-fill-color: #292344 !important;
        caret-color: #7352c7 !important;
        border: 1px solid #dcd3f0 !important;
        border-radius: 12px !important;
        font-size: 15px !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #8b849e !important;
        -webkit-text-fill-color: #8b849e !important;
        opacity: 1 !important;
    }

    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stFileUploader label {
        color: #30254e !important;
        font-weight: 700 !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border: 1px solid #8062d1 !important;
        box-shadow: 0 0 0 3px rgba(128,98,209,0.13) !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(110deg, #6546b5, #8767d5);
        color: white !important;
        border: 0;
        border-radius: 11px;
        padding: 10px 18px;
        font-weight: 700;
        transition: all .2s ease;
        box-shadow: 0 5px 13px rgba(101,70,181,0.16);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 9px 20px rgba(101,70,181,0.24);
        border: 0;
    }

    [data-testid="stExpander"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 14px;
        border: 1px solid var(--border);
        border-radius: 14px;
    }

    hr {
        border-color: #e9e3f4;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

DEFAULTS = {
    "vector_db": None,
    "papers": [],
    "history": [],
    "global_results": [],
    "last_answer": None,
    "last_documents": [],
    "indexed_at": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# REUSABLE HELPERS
# ---------------------------------------------------------

def add_history(category, query, result):
    st.session_state.history.insert(
        0,
        {
            "category": category,
            "query": query,
            "result": str(result)[:1200],
            "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        },
    )
    st.session_state.history = st.session_state.history[:30]


def page_header(title, subtitle):
    st.markdown(f"## {title}")
    st.caption(subtitle)


def render_citations(documents):
    if not documents:
        st.info("No supporting passages were returned for this response.")
        return

    citations = generate_citations(documents)

    st.markdown("### Supporting sources")

    for citation in citations:
        paper = citation.get("paper", "Unknown paper")
        page = citation.get("page", "Unknown")

        with st.expander(f"📄 {paper} · Page {page}"):
            matching = [
                doc for doc in documents
                if doc.metadata.get("source") == paper
                and str(doc.metadata.get("page", "Unknown")) == str(page)
            ]

            for doc in matching:
                st.write(doc.page_content)


def render_evidence(documents):
    if not documents:
        return

    with st.expander("🔎 Inspect retrieved evidence", expanded=False):
        for index, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown paper")
            page = doc.metadata.get("page", "Unknown")

            st.markdown(f"**Evidence {index} — {source}**")
            st.caption(f"Page: {page}")
            st.write(doc.page_content)
            st.divider()


def get_page_text(pages):
    parts = []

    for page in pages:
        if isinstance(page, dict):
            text = page.get("text", page.get("page_content", ""))
        else:
            text = str(page)

        if text:
            parts.append(text)

    return "\n\n".join(parts)


def require_index():
    if st.session_state.vector_db is None:
        st.warning("Upload and index at least one PDF paper first.")
        return False
    return True


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding:14px 4px 22px 4px;">
            <div style="font-size:27px;font-weight:800;letter-spacing:-1px;">
                ✦ ResearchAI
            </div>
            <div style="font-size:11px;color:#c9c0e5;letter-spacing:1.5px;">
                ACADEMIC INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**WORKSPACE**")

    navigation = st.radio(
        "Navigation",
        [
            "Overview",
            "Ask AI",
            "Research Library",
            "Global Paper Search",
            "Citation Finder",
            "Paper Summarizer",
            "Method Transferability",
            "Research History",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**YOUR RESEARCH**")

    st.caption(f"Indexed papers: {len(st.session_state.papers)}")

    if st.session_state.indexed_at:
        st.caption(f"Last indexed: {st.session_state.indexed_at}")

    st.markdown("---")
    st.caption("ResearchAI 2.0 · Academic Research Workspace")


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------

if navigation == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">YOUR AI RESEARCH COMPANION</div>
            <h1>Research deeper. Discover smarter.</h1>
            <p>Explore papers, ask evidence-backed questions, discover academic work,
            and evaluate whether research methods can transfer to new domains.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    metrics = [
        ("RESEARCH PAPERS", len(st.session_state.papers), "📚"),
        ("RESEARCH ACTIVITY", len(st.session_state.history), "⚡"),
        ("WORKSPACE STATUS", "Ready" if st.session_state.vector_db else "Awaiting PDFs", "✦"),
    ]

    for col, (label, value, symbol) in zip([c1, c2, c3], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{symbol} {label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("### What would you like to research today?")

    action_cols = st.columns(3)
    actions = [
        ("💬", "Ask your papers", "Get answers grounded in uploaded research."),
        ("🌐", "Discover research", "Search academic papers across external sources."),
        ("🧬", "Transfer methodology", "Evaluate methods in a different research context."),
        ("📖", "Summarize a paper", "Extract the important ideas from a paper."),
        ("🔗", "Find citations", "Discover papers relevant to a research claim."),
        ("🗂️", "Manage library", "Upload and organize your research papers."),
    ]

    for index, (icon, title, description) in enumerate(actions):
        with action_cols[index % 3]:
            st.markdown(
                f"""
                <div class="panel">
                    <div style="font-size:25px">{icon}</div>
                    <div style="font-weight:800;margin:7px 0">{title}</div>
                    <div class="muted">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Start with your research papers")

    uploads = st.file_uploader(
        "Add PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
        key="overview_upload",
    )

    if st.button("✨ Index uploaded papers", key="overview_index"):
        if not uploads:
            st.warning("Choose one or more PDF files first.")
        else:
            all_documents = []
            successful_names = []

            try:
                progress = st.progress(0, text="Preparing your research library...")

                for index, pdf in enumerate(uploads):
                    pages = extract_pdf(pdf)
                    chunks = create_chunks(pages, pdf.name)

                    if chunks:
                        all_documents.extend(chunks)
                        successful_names.append(pdf.name)

                    progress.progress(
                        (index + 1) / len(uploads),
                        text=f"Processing {pdf.name}...",
                    )

                if not all_documents:
                    st.error("No readable text was extracted from these PDFs.")
                else:
                    st.session_state.vector_db = create_vector_store(all_documents)
                    st.session_state.papers = list(
                        dict.fromkeys(st.session_state.papers + successful_names)
                    )
                    st.session_state.indexed_at = datetime.now().strftime(
                        "%d %b %Y, %I:%M %p"
                    )
                    st.success(
                        f"Successfully indexed {len(successful_names)} paper(s)."
                    )
                    st.rerun()

            except Exception as exc:
                st.error(f"PDF indexing failed: {exc}")


# ---------------------------------------------------------
# ASK AI
# ---------------------------------------------------------

elif navigation == "Ask AI":
    page_header(
        "Research AI Workspace",
        "Ask questions across your uploaded research papers and inspect the supporting evidence.",
    )

    if require_index():
        question = st.text_area(
            "Your research question",
            placeholder="Example: Compare the research methodologies used in my uploaded papers...",
            height=125,
            key="ask_question",
        )

        if st.button("✦ Generate research answer", key="ask_submit"):
            if not question.strip():
                st.warning("Enter a research question.")
            else:
                try:
                    with st.spinner("Reading research evidence and preparing your answer..."):
                        answer, documents, global_papers = answer_question_hybrid(
                            st.session_state.vector_db,
                            question,
                        )

                    st.session_state.last_answer = answer
                    st.session_state.last_documents = documents
                    add_history("Ask AI", question, answer)

                    st.markdown("### ✨ AI Research Response")
                    st.markdown(answer)

                    render_citations(documents)
                    render_evidence(documents)

                    if global_papers:
                        st.markdown("### Related global research")
                        for paper in global_papers:
                            with st.expander(paper.get("title", "Research paper")):
                                st.write(paper.get("abstract", "No abstract available."))
                                if paper.get("url"):
                                    st.link_button("Open paper", paper["url"])

                except Exception as exc:
                    st.error(f"Unable to generate the answer: {exc}")

        if st.session_state.last_answer:
            st.divider()
            st.markdown("### Previous answer in this session")
            st.write(st.session_state.last_answer)


# ---------------------------------------------------------
# RESEARCH LIBRARY
# ---------------------------------------------------------

elif navigation == "Research Library":
    page_header(
        "Research Library",
        "Upload, index, and manage the papers available to your AI research workspace.",
    )

    uploads = st.file_uploader(
        "Upload research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="library_upload",
    )

    if st.button("📚 Index selected papers", key="library_index"):
        if not uploads:
            st.warning("Select PDF files first.")
        else:
            all_documents = []
            successful_names = []

            try:
                progress = st.progress(0)

                for index, pdf in enumerate(uploads):
                    pages = extract_pdf(pdf)
                    chunks = create_chunks(pages, pdf.name)

                    if chunks:
                        all_documents.extend(chunks)
                        successful_names.append(pdf.name)

                    progress.progress((index + 1) / len(uploads))

                if not all_documents:
                    st.error("No readable text was found in the selected PDFs.")
                else:
                    st.session_state.vector_db = create_vector_store(all_documents)
                    st.session_state.papers = list(
                        dict.fromkeys(st.session_state.papers + successful_names)
                    )
                    st.session_state.indexed_at = datetime.now().strftime(
                        "%d %b %Y, %I:%M %p"
                    )
                    st.success(f"Indexed {len(successful_names)} paper(s).")
                    st.rerun()

            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.markdown("### Indexed papers")

    if not st.session_state.papers:
        st.info("Your research library is empty. Upload PDFs to get started.")
    else:
        for index, paper in enumerate(st.session_state.papers, start=1):
            st.markdown(
                f"""
                <div class="paper-card">
                    <div class="paper-title">📄 {paper}</div>
                    <div class="muted">Indexed research document · {index}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------
# GLOBAL ACADEMIC SEARCH
# ---------------------------------------------------------

elif navigation == "Global Paper Search":
    page_header(
        "Global Academic Discovery",
        "Search for related academic research using the connected Semantic Scholar and arXiv sources.",
    )

    query = st.text_input(
        "Research topic",
        placeholder="Example: Retrieval augmented generation in healthcare",
        key="global_query",
    )

    limit = st.slider("Maximum papers to display", 3, 15, 5)

    if st.button("🌐 Search academic papers", key="global_submit"):
        if not query.strip():
            st.warning("Enter a research topic.")
        else:
            try:
                with st.spinner("Searching academic sources..."):
                    results = search_global_papers(query, limit=limit)

                st.session_state.global_results = results
                add_history("Global Search", query, f"{len(results)} papers found")

                if not results:
                    st.info("No papers found. Try a broader research topic.")
                else:
                    st.success(f"Found {len(results)} research papers.")

            except Exception as exc:
                st.error(f"Academic search failed: {exc}")

    for index, paper in enumerate(st.session_state.global_results, start=1):
        with st.expander(f"📄 {index}. {paper.get('title', 'Untitled paper')}"):
            st.write(f"**Source:** {paper.get('source', 'Unknown')}")
            st.write(f"**Year:** {paper.get('year', 'Unknown')}")
            authors = paper.get("authors", [])
            st.write(f"**Authors:** {', '.join(authors[:6]) if authors else 'Unknown'}")
            st.write(paper.get("abstract") or "No abstract available.")

            if paper.get("url"):
                st.link_button("↗ Open original paper", paper["url"])


# ---------------------------------------------------------
# CITATION FINDER
# ---------------------------------------------------------

elif navigation == "Citation Finder":
    page_header(
        "Intelligent Citation Finder",
        "Find academic papers that may support a research claim. Verify the original source before citing.",
    )

    claim = st.text_area(
        "Research claim",
        placeholder="Example: Deep learning can improve automated medical image classification.",
        height=115,
        key="citation_claim",
    )

    if st.button("🔗 Find supporting papers", key="citation_submit"):
        if not claim.strip():
            st.warning("Enter a research claim.")
        else:
            try:
                with st.spinner("Finding relevant academic papers..."):
                    recommendations = find_citations(claim)

                add_history("Citation Finder", claim, f"{len(recommendations)} papers found")

                if not recommendations:
                    st.info("No relevant recommendations found.")
                else:
                    st.success(f"Found {len(recommendations)} recommendations.")

                    for index, paper in enumerate(recommendations, start=1):
                        with st.expander(
                            f"📚 {index}. {paper.get('title', 'Untitled paper')}"
                        ):
                            authors = paper.get("authors", [])
                            st.write(
                                f"**Authors:** {', '.join(authors[:5]) if authors else 'Unknown'}"
                            )
                            st.write(f"**Year:** {paper.get('year', 'Unknown')}")
                            st.write(f"**Database:** {paper.get('source', 'Unknown')}")
                            st.info(
                                paper.get(
                                    "evidence_status",
                                    "Verify the original source before citing.",
                                )
                            )

                            if paper.get("abstract"):
                                st.write(paper["abstract"])

                            if paper.get("doi"):
                                st.write(f"**DOI:** {paper['doi']}")

                            if paper.get("url"):
                                st.link_button(
                                    "↗ Verify original paper",
                                    paper["url"],
                                    key=f"citation_result_{index}",
                                )

            except Exception as exc:
                st.error(f"Citation search failed: {exc}")


# ---------------------------------------------------------
# PAPER SUMMARIZER
# ---------------------------------------------------------

elif navigation == "Paper Summarizer":
    page_header(
        "Research Paper Summarizer",
        "Create a structured summary from an uploaded and indexed research paper.",
    )

    if require_index():
        selected_paper = st.selectbox(
            "Select source paper",
            st.session_state.papers,
            key="summary_paper",
        )

        if st.button("📖 Summarize selected paper", key="summary_submit"):
            try:
                with st.spinner("Preparing paper summary..."):
                    pages = extract_pdf_from_library_placeholder = None

                    # Retrieve relevant indexed passages for the selected paper.
                    from rag_pipeline import retrieve_documents

                    docs = retrieve_documents(
                        st.session_state.vector_db,
                        f"Abstract, research objective, methodology, results, conclusion and limitations of {selected_paper}",
                        k=12,
                    )

                    docs = [
                        doc for doc in docs
                        if doc.metadata.get("source") == selected_paper
                    ]

                    paper_text = "\n\n".join(doc.page_content for doc in docs)

                    if not paper_text.strip():
                        st.warning(
                            "No passages for this paper were retrieved. "
                            "Try re-indexing the PDF."
                        )
                    else:
                        summary = summarize_paper(paper_text)
                        st.markdown("### ✨ Research summary")
                        st.write(summary)
                        render_evidence(docs)
                        add_history("Summarizer", selected_paper, summary)

            except Exception as exc:
                st.error(f"Summarization failed: {exc}")


# ---------------------------------------------------------
# METHODOLOGY TRANSFERABILITY
# ---------------------------------------------------------

elif navigation == "Method Transferability":
    page_header(
        "Research Method Transferability Analyzer",
        "Evaluate which parts of a research methodology can be reused in a different research context.",
    )

    if require_index():
        selected_paper = st.selectbox(
            "Source research paper",
            st.session_state.papers,
            key="transfer_source",
        )

        target_context = st.text_area(
            "Describe your target research context",
            placeholder=(
                "Example: Apply the plant leaf disease image-classification "
                "methodology to detect manufacturing surface defects."
            ),
            height=145,
            key="transfer_target",
        )

        st.caption(
            "The analyzer distinguishes reusable methods, required adaptations, "
            "non-transferable elements, and unknown information. It does not invent a transferability percentage."
        )

        if st.button("🧬 Analyze method transferability", key="transfer_submit"):
            if not target_context.strip():
                st.warning("Describe the target research context.")
            else:
                try:
                    with st.spinner(
                        "Examining methodology, evidence, limitations and adaptation requirements..."
                    ):
                        report = analyze_transferability(
                            st.session_state.vector_db,
                            selected_paper,
                            target_context,
                        )

                    add_history(
                        "Transferability",
                        target_context,
                        report.get("executive_summary", str(report)),
                    )

                    st.markdown("### ✨ Transferability assessment")
                    st.write(report.get("overall_assessment", "Assessment unavailable."))

                    st.markdown("### Executive summary")
                    st.write(report.get("executive_summary", "Not provided."))

                    dimensions = report.get("dimensions", [])

                    if dimensions:
                        st.markdown("### Methodology comparison")

                        for item in dimensions:
                            if not isinstance(item, dict):
                                continue

                            name = item.get("dimension", "Research dimension")
                            status = item.get("status", "Unknown")

                            with st.expander(f"{name} · {status}"):
                                st.write(
                                    item.get("assessment", item.get("analysis", "No assessment provided."))
                                )
                                if item.get("source_evidence"):
                                    st.write("**Source evidence:**", item["source_evidence"])
                                if item.get("recommendation"):
                                    st.write("**Recommendation:**", item["recommendation"])

                    sections = [
                        ("Reusable components", "reusable_components"),
                        ("Required adaptations", "required_adaptations"),
                        ("Potential risks", "potential_risks"),
                        ("Missing information", "missing_information"),
                        ("Validation plan", "validation_plan"),
                        ("Assessment limitations", "limitations_of_this_assessment"),
                    ]

                    for title, key in sections:
                        st.markdown(f"### {title}")
                        value = report.get(key, [])

                        if isinstance(value, list):
                            if value:
                                for entry in value:
                                    if isinstance(entry, dict):
                                        st.markdown(
                                            f"- **{entry.get('item', entry.get('title', 'Item'))}:** "
                                            f"{entry.get('description', entry.get('reason', str(entry)))}"
                                        )
                                    else:
                                        st.markdown(f"- {entry}")
                            else:
                                st.caption("No items returned.")
                        else:
                            st.write(value or "Not provided.")

                    evidence = report.get("retrieved_evidence", [])
                    if evidence:
                        with st.expander("🔎 Source evidence used"):
                            for index, item in enumerate(evidence, start=1):
                                st.markdown(
                                    f"**Evidence {index}: {item.get('source', 'Unknown')} "
                                    f"· Page {item.get('page', 'Unknown')}**"
                                )
                                st.write(item.get("passage", ""))

                except Exception as exc:
                    st.error(f"Transferability analysis failed: {exc}")


# ---------------------------------------------------------
# RESEARCH HISTORY
# ---------------------------------------------------------

elif navigation == "Research History":
    page_header(
        "Research Activity",
        "Review the searches and AI research operations performed during this session.",
    )

    if not st.session_state.history:
        st.info("Your research activity will appear here.")
    else:
        if st.button("Clear session history", key="clear_history"):
            st.session_state.history = []
            st.rerun()

        for index, item in enumerate(st.session_state.history, start=1):
            with st.expander(
                f"{item['category']} · {item['time']} · {item['query'][:75]}"
            ):
                st.write("**Query / source:**")
                st.write(item["query"])
                st.write("**Result preview:**")
                st.write(item["result"])


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center;padding:8px 0 18px 0;">
        <div style="font-weight:800;color:#59428f;">✦ ResearchAI 2.0</div>
        <div style="font-size:12px;color:#89839d;">
            AI-powered Academic Research Intelligence Workspace
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
