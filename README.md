---

```markdown
# Retrieval-Augmented Generation (RAG) App — ShyftLabs AI Engineer Take-Home Assignment

---

## What This App Does

- Upload PDFs or HTML files  
- Embed documents and store in Chroma vector DB  
- Perform semantic search over documents  
- Generate streamed answers using a local free LLM  
- Serve interactive frontend via FastAPI static route  
- Fully local & free — no OpenAI keys, no paid APIs

---

## Tech Stack Used

| Component            | Tool |
|----------------------|------|
| API Server            | FastAPI |
| Vector Database       | Chroma (local, free) |
| Embedding Model       | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM Model (FREE)      | HuggingFace Transformers → `tiiuae/falcon-rw-1b` |
| PDF Parsing           | PyMuPDF (`fitz`) |
| HTML Parsing          | BeautifulSoup |
| Frontend              | Vanilla HTML + JS (served via FastAPI static route) |
| Deployment            | Localhost → Fully Free |

---

## Why This App is 100% Free

- All tools used are open-source  
- No OpenAI keys, no API costs  
- Models run locally on CPU  
- Suitable for offline use  
- Reproducible on any machine

---

## Key Skills Demonstrated

- LLM orchestration with custom RAG pipeline  
- Semantic search using vector embeddings  
- Streaming API for token-level generation  
- Document parsing and chunking for optimal retrieval  
- Serving static frontend professionally via FastAPI  
- Experience integrating HuggingFace models locally  
- Efficient use of ChromaDB with latest API  
- Frontend + Backend full-stack integration  

---

## Project Structure

```

shyftlabs\_rag\_app/
├── static/
│   └── frontend.html       # Interactive UI served by FastAPI
├── chroma\_db/              # Chroma vector DB 
├── main.py                 # Full FastAPI app
├── requirements.txt        # Dependencies
├── .gitignore              # Git ignore file
├── README.md               
└── venv/                   # Local virtual environment

````

---

## How to Run This App

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/shyftlabs_rag_app.git
cd shyftlabs_rag_app
````

### 2. Create and activate virtual env

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the backend server

```bash
uvicorn main:app --reload
```

Backend will be available at:

```
http://127.0.0.1:8000
```

API docs (Swagger UI):

```
http://127.0.0.1:8000/docs
```

### 5. Open the Frontend

Frontend is served via static route:

```
http://127.0.0.1:8000/static/frontend.html
```

---

## Future Improvements

* Add React + Tailwind frontend
* Add hybrid keyword + semantic search
* Add LLM streaming using llama.cpp for faster response

---

## Final Notes

* This project demonstrates a complete end-to-end free RAG pipeline
* All models are open-source and run locally
* No proprietary APIs or paid services used
* Designed with scalability and clean architecture in mind

---

Built by **Tejaswi** for **ShyftLabs AI Engineer Take-Home Assignment**

```
