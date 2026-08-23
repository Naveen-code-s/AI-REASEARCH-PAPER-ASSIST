from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def create_chunks(pages, filename):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    documents = []

    for page in pages:

        chunks = splitter.split_text(page["text"])

        for chunk in chunks:

            documents.append({
                "text": chunk,
                "page": page["page"],
                "source": filename
            })

    return documents


