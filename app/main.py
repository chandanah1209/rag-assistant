from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import os
from app.graph import build_graph
from app.ingestion import ingest_documents, get_all_documents

app = FastAPI(title="RAG Documentation Assistant")
graph = build_graph()

# ─── Request Models ───────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    answer_id: str
    rating: str        # "up" or "down"
    comment: Optional[str] = None


# ─── Routes ───────────────────────────────────────────────────

@app.post("/query")
async def query(req: QueryRequest):
    """Submit a question and get an answer with sources"""
    initial_state = {
        "question": req.question,
        "rewritten_query": None,
        "documents": [],
        "relevant_docs": [],
        "answer": None,
        "retry_count": 0,
        "query_type": None
    }
    result = graph.invoke(initial_state)

    sources = list(set([
        doc["source"] for doc in result["relevant_docs"]
    ]))

    return {
        "answer": result["answer"],
        "sources": sources,
        "query_type": result["query_type"],
        "retries": result["retry_count"]
    }


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    """Upload and ingest new documents"""
    os.makedirs("./docs", exist_ok=True)

    for file in files:
        content = await file.read()
        with open(f"./docs/{file.filename}", "wb") as f:
            f.write(content)

    ingest_documents()
    return {"message": "Documents ingested successfully!"}


@app.get("/documents")
async def list_documents():
    """List all indexed documents"""
    data = get_all_documents()
    sources = list(set([
        m.get("source", "unknown")
        for m in data["metadatas"]
    ]))
    return {
        "documents": sources,
        "total_chunks": len(data["ids"])
    }


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """Submit feedback on an answer"""
    # Can be saved to a DB later
    print(f"Feedback received: {req.rating} for {req.answer_id}")
    return {
        "status": "received",
        "rating": req.rating,
        "comment": req.comment
    }


@app.get("/")
async def root():
    return {"message": "RAG Documentation Assistant is running!"}