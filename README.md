# College Notes RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from my own college subject notes (PDF and Word documents), with source citation showing exactly which file each answer came from.

Built as a portfolio project to demonstrate practical GenAI/RAG skills for Python/AI engineering roles.

## Demo



![Chatbot Screenshot](screenshot.png)



*Add a screenshot of your chatbot answering a question here — save it as `screenshot.png` in the repo root.*

## How It Works

This chatbot works like an open-book exam — it doesn't memorize notes, it looks them up fresh for every question:

1. **Load** — PDF and DOCX notes are loaded from the `/notes` folder
2. **Chunk** — Each document is split into smaller pieces (~1000 characters, with overlap) so context fits within model limits
3. **Embed** — Each chunk is converted into a vector (numerical representation of its meaning) using a HuggingFace sentence-transformer model
4. **Store** — Chunks and their embeddings are stored in ChromaDB, a local vector database
5. **Retrieve** — When a question is asked, it's embedded the same way, and ChromaDB finds the most similar chunks by meaning (not keyword matching)
6. **Generate** — The retrieved chunks + question are sent to Groq's free cloud LLM (Llama 3.1), which writes an answer using only that context
7. **Cite** — Since each chunk's source file is tracked throughout, the answer displays which document it came from

## Tech Stack

- **Python** — core language
- **LangChain** — orchestration for loading, chunking, retrieval, and prompting
- **ChromaDB** — local vector database for storing and searching embeddings
- **HuggingFace sentence-transformers** — free, local embedding model (`all-MiniLM-L6-v2`)
- **Groq API** — free-tier cloud LLM (`llama-3.1-8b-instant`) for answer generation
- **Streamlit** — web UI for the chat interface

## Setup

**1. Clone the repo:**