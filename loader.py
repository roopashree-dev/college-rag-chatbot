"""Load PDF/DOCX notes from disk and split them into retrieval-sized chunks."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_NOTES_DIR = PROJECT_ROOT / "notes"


def _load_single_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = UnstructuredWordDocumentLoader(str(path))
    else:
        return []

    documents = loader.load()
    for doc in documents:
        doc.metadata["filename"] = path.name
        doc.metadata["source"] = path.name
    return documents


def load_and_chunk_notes(
    notes_dir: Path | str | None = None,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Load every PDF and DOCX in ``notes_dir``, split into chunks, and attach
    ``filename`` / ``source`` metadata on each chunk for citations.
    """
    directory = Path(notes_dir) if notes_dir is not None else DEFAULT_NOTES_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Notes folder not found: {directory}")

    paths = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx"}
    )
    if not paths:
        return []

    raw_documents: list[Document] = []
    for path in paths:
        raw_documents.extend(_load_single_file(path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(raw_documents)


if __name__ == "__main__":
    chunks = load_and_chunk_notes()
    print(f"Created {len(chunks)} chunk(s) from notes in: {DEFAULT_NOTES_DIR}")

    if chunks:
        first = chunks[0]
        print("\n--- First chunk ---")
        print(f"Source filename: {first.metadata.get('filename', first.metadata.get('source', '?'))}")
        print(first.page_content)
    else:
        print("\nNo chunks yet. Add at least one .pdf or .docx file to the notes/ folder.")
