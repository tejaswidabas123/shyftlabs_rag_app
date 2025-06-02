# main.py

# Imports
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles  # Added for serving static frontend
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF parsing
from bs4 import BeautifulSoup
import time

# LLM imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Serve static files (frontend.html) from the /static route
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Chroma vector DB (local)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_rag_collection")

# Initialize embedding model for semantic search
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small and fast model

# Initialize LLM model (local HuggingFace model)
model_name = "tiiuae/falcon-rw-1b"  # Free, local model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --- Endpoints ---

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Hello from RAG App!"}

# Upload endpoint: allows uploading PDF or HTML files
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()

    # Extract text depending on file type
    if file.filename.endswith(".pdf"):
        pdf = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()

    elif file.filename.endswith(".html") or file.filename.endswith(".htm"):
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text()

    else:
        return {"error": "Unsupported file type. Please upload PDF or HTML."}

    # Split extracted text into chunks for embedding
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    embeddings = embed_model.encode(chunks).tolist()

    # Store chunks and their embeddings in Chroma
    for idx, emb in enumerate(embeddings):
        collection.add(
            documents=[chunks[idx]],
            embeddings=[emb],
            ids=[f"{file.filename}-chunk-{idx}"]
        )

    return {"status": "Uploaded and indexed", "chunks": len(chunks)}

# Semantic search endpoint
@app.get("/search")
def search(query: str, top_k: int = 5):
    # Embed the query
    query_embedding = embed_model.encode([query]).tolist()[0]

    # Query Chroma for top_k relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )

    # Prepare search results
    hits = []
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        hits.append({
            "document": doc,
            "distance": dist
        })

    return {"query": query, "results": hits}

# Stream Answer endpoint: generates answer using LLM based on retrieved context
@app.get("/stream_answer")
def stream_answer(query: str, top_k: int = 5):
    # Embed the query
    query_embedding = embed_model.encode([query]).tolist()[0]

    # Retrieve top_k relevant chunks from Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents"]
    )

    # Concatenate retrieved documents as context
    context = "\n\n".join(results['documents'][0])

    # Build prompt for LLM
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Generate answer using the LLM
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Stream the LLM output token by token
    def llm_stream():
        for word in generated_text.split():
            yield word + " "
            time.sleep(0.02)

    return StreamingResponse(llm_stream(), media_type="text/plain")
