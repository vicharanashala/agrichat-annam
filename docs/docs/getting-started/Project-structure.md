---
sidebar_position: 2
---

# 📁 Project Structure

AgriChat-Annam is structured in a modular format to separate the retrieval pipelines, backend API, and static frontend. Below is the top-level layout:

---

```bash
agrichat-annam/
├── .env                        # Environment config (used in backend)
├── docker-compose.yml         # Runs frontend and backend together
├── render.yaml                # Render deployment settings
├── README.md
│
├── agrichat-backend/          # FastAPI Backend with RAG pipelines
│   ├── app.py
│   ├── backendRequirements.txt
│   └── RAGpipelinev3/         # Main RAG pipeline (embedding, retrieval, LLM)
│       ├── main.py
│       ├── creating_database.py
│       ├── ollama_embedding.py
│       ├── chromaDb/
│       ├── Data/
│       └── Gemini_based_processing/   # Gemini-based RAG variant
│           ├── main.py
│           ├── creating_database.py
│           └── chromaDb/
│
├── agrichat-frontend/         # Static frontend (HTML, JS, CSS)
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── Dockerfile
│
├── nginx/                     # NGINX config for containerized frontend
│   └── default.conf
````

---

🔧 **Key Components**

* **`Pipelines/`**
  The core RAG logic is modularized for experimentation and easy upgrades:

  * `RAG pipeline v1/`: Initial version with markdown parsing and basic retrieval.
  * `RAG pipeline v2/`: Improved with SentenceTransformer-based embeddings.
  * `RAGpipelinev3/`: Default pipeline using Ollama + optimized Chroma workflows.
  * `Gemini_based_processing/`: Cloud-based variant using Google Gemini APIs.

* **`Data/`**
  Each pipeline contains its own `data/` folder, typically with a CSV knowledge base (`sample_data.csv`).

* **`chromaDb/`**
  Stores Chroma vector databases for semantic search, unique to each pipeline.

* **`creating_database.py`**
  Converts raw CSV/markdown into vector embeddings and stores them in ChromaDB.

* **`main.py`**
  Executes core RAG flow: vector search + LLM response.

* **`app.py`**
  FastAPI server. The top-level `agrichat-backend/app.py` launches the app and imports the active pipeline. Pipelines may also include their own test endpoints.

* **`agrichat-frontend/`**
  Static frontend served via NGINX:

  * `index.html`: Chat UI
  * `script.js`: API interactions
  * `style.css`: Visual styling

---

🔍 **Notes**

* **Pipeline Flexibility**
  Supports multiple RAG pipelines — developers can swap or upgrade by changing imports/config.

* **Modular & Extensible Design**
  Each pipeline is fully encapsulated, making experimentation and switching straightforward.

* **Frontend-Backend Separation**
  Clear decoupling between UI and backend via REST API improves scalability and deployment.

* **LLM-Agnostic Architecture**
  Compatible with both local (Ollama) and cloud (Gemini) LLMs.

* **Developer Docs**
  Internal developer documentation is maintained in the `/docs` folder using [Docusaurus](https://docusaurus.io).

* **Local-First with Cloud Support**
  Designed for smooth local dev (FastAPI + static files) and easy Docker/Render/cloud deployment.