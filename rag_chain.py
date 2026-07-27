"""Retrieve relevant note chunks and generate answers with Groq."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from vectorstore import get_or_create_vectorstore, similarity_search

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 3

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful study assistant. Answer the user's question using ONLY "
            "the context below from their course notes. Do not use outside knowledge. "
            "If the context does not contain enough information to answer, say clearly: "
            '"I could not find that in your notes." Do not guess or make up facts.',
        ),
        (
            "human",
            "Context from notes:\n\n{context}\n\nQuestion: {question}",
        ),
    ]
)


def _get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file in the project root."
        )
    return api_key


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, groq_api_key=_get_groq_api_key())


@lru_cache(maxsize=1)
def _get_vectorstore():
    return get_or_create_vectorstore()


def _chunk_source(chunk: Document) -> str:
    return chunk.metadata.get("filename", chunk.metadata.get("source", "unknown"))


def _extract_sources(chunks: list[Document]) -> list[str]:
    """Return unique source filenames in retrieval order."""
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        source = _chunk_source(chunk)
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def _format_context(chunks: list[Document]) -> str:
    if not chunks:
        return "(No matching notes found.)"

    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source = _chunk_source(chunk)
        parts.append(f"[Excerpt {index} — {source}]\n{chunk.page_content}")
    return "\n\n".join(parts)


def answer_question(query: str, *, k: int = TOP_K) -> tuple[str, list[str]]:
    """
    Retrieve the top ``k`` note chunks, build a grounded prompt, and return
    the Groq LLM answer plus deduplicated source filenames used for context.
    """
    store = _get_vectorstore()
    chunks = similarity_search(store, query, k=k)
    sources = _extract_sources(chunks)
    context = _format_context(chunks)

    chain = PROMPT | _get_llm()
    response = chain.invoke({"context": context, "question": query})
    return response.content, sources


if __name__ == "__main__":
    test_query = "What is Python used for?"
    answer, sources = answer_question(test_query)

    print(f'Question: "{test_query}"\n')
    print(f"Answer: {answer}")
    print(f"Sources: {', '.join(sources) if sources else '(none)'}")
