"""Embed note chunks and store / retrieve them with ChromaDB."""

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from loader import DEFAULT_NOTES_DIR, load_and_chunk_notes

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_DIR = PROJECT_ROOT / "db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "college_notes"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Local embedding model — no API key required."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _db_has_index(db_dir: Path) -> bool:
    """True if Chroma has already persisted an index in ``db_dir``."""
    if not db_dir.is_dir():
        return False
    indexed_files = [p for p in db_dir.iterdir() if p.name != ".gitkeep"]
    return bool(indexed_files)


def build_vectorstore(
    chunks: list[Document],
    db_dir: Path | str | None = None,
    *,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Embed ``chunks`` and persist them to ChromaDB under ``db_dir``."""
    directory = Path(db_dir) if db_dir is not None else DEFAULT_DB_DIR
    directory.mkdir(parents=True, exist_ok=True)

    if not chunks:
        raise ValueError("No chunks to index. Add PDF/DOCX files to the notes/ folder.")

    embedder = embeddings or get_embeddings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=str(directory),
        collection_name=COLLECTION_NAME,
    )


def load_vectorstore(
    db_dir: Path | str | None = None,
    *,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Load an existing Chroma index from ``db_dir``."""
    directory = Path(db_dir) if db_dir is not None else DEFAULT_DB_DIR
    if not _db_has_index(directory):
        raise FileNotFoundError(
            f"No vector index found in {directory}. Run build_vectorstore() first."
        )

    embedder = embeddings or get_embeddings()
    return Chroma(
        persist_directory=str(directory),
        embedding_function=embedder,
        collection_name=COLLECTION_NAME,
    )


def get_or_create_vectorstore(
    notes_dir: Path | str | None = None,
    db_dir: Path | str | None = None,
) -> VectorStore:
    """Build a new index when ``db/`` is empty; otherwise load the saved one."""
    directory = Path(db_dir) if db_dir is not None else DEFAULT_DB_DIR

    if _db_has_index(directory):
        print(f"Loading existing vector store from: {directory}")
        return load_vectorstore(directory)

    print(f"Building new vector store from notes in: {notes_dir or DEFAULT_NOTES_DIR}")
    chunks = load_and_chunk_notes(notes_dir)
    print(f"Indexed {len(chunks)} chunk(s).")
    return build_vectorstore(chunks, directory)


def similarity_search(
    vectorstore: VectorStore,
    query: str,
    *,
    k: int = 1,
) -> list[Document]:
    """Return the top ``k`` chunks most similar to ``query``."""
    return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    TEST_QUERY = "What is Python used for?"

    store = get_or_create_vectorstore()
    top_matches = similarity_search(store, TEST_QUERY, k=1)

    print(f'\nTest query: "{TEST_QUERY}"')
    if top_matches:
        best = top_matches[0]
        print("\n--- Top matching chunk ---")
        print(f"Source filename: {best.metadata.get('filename', best.metadata.get('source', '?'))}")
        print(best.page_content)
    else:
        print("\nNo matches returned. Check that notes/ contains indexed documents.")
