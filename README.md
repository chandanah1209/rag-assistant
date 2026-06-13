# RAG Documentation Assistant

A RAG-based technical documentation assistant built with LangGraph and FastAPI.

## Architecture
- **Query Analysis Node**: Rewrites and classifies queries
- **Retrieval Node**: Searches ChromaDB vector store
- **Grading Node**: Filters relevant documents
- **Generation Node**: Generates answers with citations

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add GROQ_API_KEY to .env file
4. Run: `python -m uvicorn app.main:app --reload`

## API Endpoints
- POST /query - Ask a question
- POST /ingest - Upload documents
- GET /documents - List documents
- POST /feedback - Submit feedback

## Tech Stack
- LangGraph, FastAPI, ChromaDB, Groq, SentenceTransformers