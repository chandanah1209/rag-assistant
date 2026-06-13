import os
import chromadb
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="docs")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def ingest_documents(docs_path: str = "./docs"):
    """Load, chunk, embed and store documents in ChromaDB"""
    
    # Step 1: Load documents
    loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("No documents found in docs folder!")
        return
    
    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    # Step 3: Embed and store
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk.page_content).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk.page_content],
            metadatas=[{"source": chunk.metadata.get("source", "unknown")}]
        )
    
    print(f"Successfully ingested {len(chunks)} chunks from {len(documents)} documents!")

def get_all_documents():
    """Return all documents in the collection"""
    return collection.get()

if __name__ == "__main__":
    ingest_documents()