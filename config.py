
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama-3.3-70b-versatile"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOP_K = 5
GLOBAL_SEARCH_LIMIT = 5

TEMPERATURE = 0.1
