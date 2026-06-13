import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from app.state import GraphState

load_dotenv()

# Initialize clients
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="docs")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_analysis_node(state: GraphState) -> GraphState:
    prompt = f"""Rewrite this question to improve search results.
Also classify it as: conceptual, how-to, troubleshooting, or api-reference.

Question: {state['question']}

Respond in exactly this format:
REWRITTEN: <rewritten question>
TYPE: <type>"""

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    lines = response.choices[0].message.content.strip().split("\n")
    rewritten = lines[0].replace("REWRITTEN:", "").strip()
    qtype = lines[1].replace("TYPE:", "").strip() if len(lines) > 1 else "conceptual"

    return {**state, "rewritten_query": rewritten, "query_type": qtype}

def retrieval_node(state: GraphState) -> GraphState:
    query = state.get("rewritten_query") or state["question"]
    embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=4
    )

    documents = []
    for i, doc in enumerate(results["documents"][0]):
        documents.append({
            "content": doc,
            "source": results["metadatas"][0][i].get("source", "unknown")
        })

    return {**state, "documents": documents}

def grading_node(state: GraphState) -> GraphState:
    relevant_docs = []

    for doc in state["documents"]:
        prompt = f"""Is this document relevant to the question? Answer ONLY 'relevant' or 'irrelevant'.

Question: {state['question']}
Document: {doc['content']}"""

        response = llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        verdict = response.choices[0].message.content.strip().lower()
        if "relevant" in verdict:
            relevant_docs.append(doc)

    return {**state, "relevant_docs": relevant_docs}

def generation_node(state: GraphState) -> GraphState:
    context = "\n\n".join([
        f"[Source: {doc['source']}]\n{doc['content']}"
        for doc in state["relevant_docs"]
    ])

    prompt = f"""Answer the question using ONLY the context below.
Include [Source: filename] citations in your answer.

Context:
{context}

Question: {state['question']}"""

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {**state, "answer": response.choices[0].message.content}

def no_answer_node(state: GraphState) -> GraphState:
    return {**state, "answer": "I don't have enough information in the documentation to answer this question."}